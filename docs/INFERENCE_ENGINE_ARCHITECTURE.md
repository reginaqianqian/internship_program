# tactile_sdk 推理引擎架构说明（供初学者）

本文档说明 tactile_sdk 中**推理引擎**部分的整体架构、数据流和关键类型，便于初学者理解与二次开发。

---

## 一、整体架构概览

### 1.1 推理在触觉链路中的位置

触觉数据从设备到应用的大致链路是：

```
[触觉设备] --UDP--> [Receiver] --Frame--> [InferEngine] --Frame+推理结果--> [回调 / dequeue]
                         ^                        ^
                         |                        |
                    StreamParser             多后端引擎
                    (组包/解析)              (TensorRT/CoreML/Dummy)
```

- **Receiver**：在主机上绑定 UDP 端口，接收原始报文，用 `StreamParser` 组包并解析成 `Frame::Ptr`。
- **InferEngine**：接收 `Frame`，对需要推理的帧调用神经网络，得到 F6、形变图、接触点等，再通过回调或 `dequeue` 交给上层。
- **InferOffline**：不接设备，仅对「参考图 + 实时图」做单次推理，用于调试或离线工具。

### 1.2 相关文件结构（仅列推理相关）

```
tactile_sdk/module/tactile_sdk/
├── def.h                    # Frame、LoggerBase 等公共定义
├── touch.h / touch.cpp      # Touch 主 API，内部创建 InferEngine + Receiver
├── infer_offline.h / infer_offline.cpp   # 离线推理 API
└── src/
    ├── engine_base.h        # 推理引擎抽象基类
    ├── infer.h / infer.cpp  # 在线推理引擎（核心）
    ├── trt.h / trt.cpp      # TensorRT 实现
    ├── coreml.h / coreml.cpp # ONNX Runtime + CoreML 实现
    ├── dummy_engine.h / dummy_engine.cpp # 占位实现（返回空）
    ├── receiver.h / receiver.cpp         # UDP 接收并 enqueue 到 InferEngine
    ├── stream_protocol.h / stream_protocol.cpp  # 流解析、ContentParser
    ├── content_protocol.h / content_protocol.cpp # 协议解析、Frame 构造、is_to_infer()
    ├── drift_detect.h / drift_detect.cpp # 漂移检测与自动校准（可选，依赖 OpenCV）
    ├── nms.h / nms.cpp      # 接触点 NMS 后处理
    ├── util.h / util.cpp    # SafeQueue、ThreadPool、日志等
    └── tensor.h / tensor.cpp # DataBlock 运算（add_db_float、uint8_to_float 等）
```

---

## 二、推理引擎抽象：EngineBase

所有「真正做前向计算」的后端都继承自同一抽象接口，便于编译期切换或扩展。

### 2.1 接口定义（engine_base.h）

```cpp
class EngineBase {
public:
    virtual std::vector<std::vector<float>> infer(const float* input, int size) = 0;
};
```

- **input**：主机上的 float 数组，布局为 `[ref_image, realtime_image]` 拼在一起；  
  - 单样本时：`size = 240*320` 表示一张图元素个数，整块 input 长度为 `2 * 240 * 320`（参考 + 实时）。  
  - 批量时：多帧的 ref 连续存放，再接着多帧的 realtime，`size` 为**单张图元素个数**（即 240*320），总长度为 `batch_size * 2 * 240 * 320`。
- **返回**：`vector<vector<float>>`，长度固定为 3，依次为：
  - `[0]`：F6 力/力矩（6 维 × batch）
  - `[1]`：形变图（deform，240×240 × batch）
  - `[2]`：接触点网格（如 15×15×3 × batch），供后续 NMS。

### 2.2 三种实现

| 实现类 | 文件 | 说明 | 编译开关 |
|--------|------|------|----------|
| **TensorRTInference** | trt.h / trt.cpp | 加载 ONNX/engine，GPU 推理，支持动态 batch | `INFER_ENGINE == 1` |
| **CoremlInference** | coreml.h / coreml.cpp | ONNX Runtime + CoreML（如 macOS/Apple Silicon） | `INFER_ENGINE == 2` |
| **DummyEngine** | dummy_engine.h / .cpp | 直接返回空 `{}`，用于无模型或占位 | 默认 |

在 `infer.cpp` 和 `infer_offline.cpp` 中通过 `#if (INFER_ENGINE == 2) ... #elif (INFER_ENGINE == 1) ... #else ... #endif` 选择包含哪个头并 `make_shared` 对应引擎。初学者只需改 CMake 中定义的 `INFER_ENGINE` 即可切换后端。

### 2.3 TensorRT 实现要点（trt.cpp）

- **输入**：两个 tensor，通常名为 `initial_image`、`realtime_image`，形状 `[N, 1, 240, 320]`。
- **流程**：Host 拷贝到 pinned memory → 异步拷贝到 GPU → `enqueueV3` 执行 → 结果拷回 Host → 按 batch 拆成 `vector<vector<float>>`。
- **动态 batch**：通过 `setupDynamicShapes(batchSize)` 按当前 batch 重设输入/输出尺寸并重绑 binding。
- **序列化**：若存在同名的 `.engine` 文件则反序列化加载，否则从 ONNX 构建并写入 `.engine`。

### 2.4 CoreML 实现要点（coreml.cpp）

- 使用 ONNX Runtime，执行 provider 为 CoreML，适合在 Mac 上跑。
- 输入/输出名和维度从 Session 读取，推理接口同样满足 `infer(input, size)` 的约定。

---

## 三、在线推理核心：InferEngine

### 3.1 职责概述

- **输入**：来自 Receiver 的 `Frame::Ptr`（通过 `enqueue`）。
- **输出**：  
  - 带推理结果的 `Frame` 通过**回调** `std::function<void(Frame::Ptr)>` 交给上层；  
  - 和/或通过 **per-channel 的 `dequeue(ch, timeout)`** 被拉取。
- **内部**：一个**推理线程**从 `queue_in_` 取帧，按 `model_path` 配置的「模型路径 → channel 列表」组成 batch，调用对应 `EngineBase::infer`，写回 F6/DEFORM/CONTACT_POINT，再做校准与 NMS，最后放入 `queue_out_[ch]` 并触发回调。

### 3.2 构造函数与配置（infer.h / infer.cpp）

```cpp
InferEngine(
    std::map<std::string, std::vector<int>> model_path,  // 模型路径 -> 使用该模型的 channel 列表
    std::vector<int> channels,                          // 所有 channel
    int batch_size,
    int buffer_size,
    const std::function<void(Frame::Ptr)>* callback,
    std::shared_ptr<ThreadPool> pool,
    bool infer_from_device
);
```

- **model_path**：例如 `{"model/a.onnx", {0,1}}, {"model/b.onnx", {2}}` 表示 0、1 用 a.onnx，2 用 b.onnx。  
  Impl 里会为每个 path 创建一个 `EngineBase`（TensorRT/CoreML/Dummy 之一），存于 `engine_[path]`。
- **channels**：所有通道号，用于校准状态、`queue_out_` 按 channel 创建等。
- **batch_size**：一次推理最多凑多少帧；推理线程会攒够一批（或超时）再调用 `engine->infer(...)`。
- **buffer_size**：输入队列 `queue_in_` 的最大长度（按帧计）。
- **callback**：每帧推理完成后调用；在 Impl 中与线程池配合，在池里执行回调并 `enqueue` 到 `queue_out_[ch]`。
- **infer_from_device**：为 false 时会在输出前对 F6/DEFORM 做「减 offset」的校准（见下）。

### 3.3 核心数据结构（Impl）

- **queue_in_**：`SafeQueue<Frame::Ptr>`，Receiver 解析出的帧入队。
- **queue_out_**：`map<int, SafeQueue<Frame::Ptr>>`，按 channel 的出队队列，供 `dequeue(ch, timeout)` 取结果。
- **engine_**：`map<std::string, shared_ptr<EngineBase>>`，key 为模型路径。
- **calib_stat_**：`CalibStat`，管理每通道的参考图、F6/DEFORM 的 offset、校准状态机（见下）。
- **drift_**：可选 `drift_detect`，在启用 OpenCV 时做漂移检测并触发 `to_calib(ch)`。
- **nms_ret_**：NMS 输出缓冲区；推理得到的是网格形式的 contact point，经 `nms_execute` 后写入此缓冲再封装成 `Frame::content["CONTACT_POINT"]`。

### 3.4 推理线程主循环（infer_()）简化为 4 步

1. **组 batch**  
   从 `queue_in_` 里取帧（超时 0.1s 以便响应 quit）。  
   若 `!frame->is_to_infer()`（例如已有 F6/DEFORM，来自设备端推理），则直接 `add_output_job_(frame)` 不再推理。  
   否则按 `model_path` 把该帧归到对应 path 的 `jobs[path]`，直到总帧数达到 `batch_size_` 或超时。

2. **批量推理**  
   对每个 path，若 `jobs[path]` 非空：  
   - 从每帧取 `content["RAW"]`，配合 `calib_stat_->check_ref_image` 更新参考图；  
   - 拼成 `[ref_1,...,ref_N, rt_1,...,rt_N]` 的 float 数组，调用 `engine_[path]->infer(to_infer, frames.size() * kImgSize_)`；  
   - 将返回的 3 个 vector 按帧拆成 F6、DEFORM、CONTACT_POINT，写入各 `frame->content`；  
   - 调用 `calib_stat_->check_offset` / `check_calib_finished` 做校准状态更新。

3. **后处理与输出**  
   对这批帧逐个：  
   - 若启用 `drift_`，则 `drift_->auto_calib(frame->copy())`；  
   - `add_output_job_(frame)`：在线程池中执行「可选 apply_offset → NMS → 回调 → 入队 queue_out_[ch]」。

4. **add_output_job_**  
   在线程池里：  
   - 若 `!infer_from_device_`，对 F6/DEFORM 做 `calib_stat_->apply_offset(f)`；  
   - 若有 `CONTACT_POINT`，则 `nms_execute(...)` 写 `nms_ret_`，再替换 `f->content["CONTACT_POINT"]`；  
   - 若设置了 `callback_` 则调用；  
   - 将 `f` 放入 `queue_out_[f->channel]`。

因此，**一帧从入队到可 dequeue/回调**：经过「组 batch → 引擎推理 → 校准/NMS → 回调 + 出队队列」。

---

## 四、校准（CalibStat）与 is_to_infer

### 4.1 为什么需要校准

触觉的「零力」会随温度/时间漂移。做法是：  
- 维护每通道的**参考图**（无接触时的 RAW）；  
- 模型输入为「参考图 + 当前帧 RAW」；  
- 在线时还会维护 F6/DEFORM 的 **offset**（校零时的均值），输出时减去，使静止时接近 0。

### 4.2 CalibStat 状态机（简化）

- **calib_num_[ch]**：0 = 正常；1 = 等待更新参考图；2 = 等待用当前推理结果更新 offset。
- **to_calib(ch)**：将 ch 置为 1（开始一次校准）。
- **check_ref_image(ch, raw)**：若 ref 为空或当前处于「更新参考图」状态，则用当前 raw 更新 ref（并可能从 1 进到 2）。
- **check_offset(f)**：若该通道处于「更新 offset」状态，则用本帧的 F6/DEFORM 累加进 offset，并置回 0。
- **calib_zero(num_frames * max_retry)**：对所有 channel 调用 `to_calib`，然后等待所有通道走完「更新 ref → 推理 → 更新 offset」，用条件变量 `calib_cv_` 等待完成或失败。

参考图也可由上层通过 `set_ref_image(ch, raw)` 直接设置（例如 Touch 启动时从设备拉取初始图）。

### 4.3 Frame::is_to_infer()（content_protocol.cpp）

```cpp
// 需要推理：有 RAW，且（没有 DEFORM 或没有 F6）
return has_raw && (!has_deform || !has_f6);
```

即：只有 RAW、没有设备端给的 DEFORM/F6 的帧才会进入「组 batch → 推理」；若设备已算过 F6/DEFORM，则直接走输出分支，不再调用引擎。

---

## 五、漂移检测（drift_detect，可选）

在 `#ifdef USE_OPENCV` 下，Impl 会创建 `drift_detect`：

- 输入：推理后的 `Frame::Ptr`（如 `frame->copy()`）。
- 逻辑：根据图像/时序判断是否发生漂移，若需要则调用传入的 `func_calib(ch)`，这里即 `calib_stat_->to_calib(ch)`，从而触发一次该通道的「更新 ref → 更新 offset」流程。
- 在 `infer_()` 里，每帧推理完成后若存在 `drift_`，会执行 `drift_->auto_calib(frame->copy())`。

不编 OpenCV 时，`drift_` 为 nullptr，不影响其余流程。

---

## 六、NMS（接触点后处理）

- **输入**：模型输出的 contact point 网格（如 15×15×3，每个格点 3 个 float）。
- **nms_init**：初始化内部参数（网格尺寸、阈值等）。
- **nms_execute**：在网格上做非极大值抑制，得到少量接触点，写入 `ptr_ret`，返回数量。
- InferEngine 在 `add_output_job_` 里把结果写回 `frame->content["CONTACT_POINT"]`，形状变为 `[ret_size, 3]`。

---

## 七、离线推理：InferOffline

- **用途**：不接设备，只对两张图（initial_image、realtime_image）做一次推理，例如脚本或工具里验证模型。
- **接口**：`Frame::Ptr infer(uint8_t* initial_image, uint8_t* realtime_image, int size);`  
  `size` 需等于 240×320。
- **实现**：内部一个 `EngineBase`（同样由 `INFER_ENGINE` 决定 TensorRT/CoreML/Dummy），构造 `to_infer` 后调用 `engine_->infer(...)`，再把返回的 3 个 vector 填进一个新建的 `Frame`（含 F6、DEFORM、CONTACT_POINT），并对 CONTACT_POINT 做 NMS。  
  不做校准（无 ref/offset 状态），F6/DEFORM 也不减 offset。

---

## 八、与 Touch、Receiver 的衔接

- **Touch**（touch.cpp）：  
  - 根据 `TouchSetting`（如 `model_path`、`batch_size`、`buffer_size`、`infer_from_device`）创建 `InferEngine`；  
  - 用同一 `InferEngine` 和 `Receiver` 的构造函数把 `Receiver` 与引擎绑在一起；  
  - `Touch::start()` 先 `infer_engine->start()`（拉起推理线程），再 `recv->start()`（开始 UDP 收包）。  
- **Receiver**（receiver.cpp）：  
  - 收包后由 `stream_parser_->receive_multi(buffer_)` 得到 `RawFrame`（即 `ContentParser`）；  
  - 在线程池中执行 `raw_frame->parse()` 得到 `Frame::Ptr`，然后 `engine_->enqueue(content)`。  

因此，**数据流**是：UDP → StreamParser → ContentParser::parse() → Frame → InferEngine::enqueue → 推理线程 → 回调 + queue_out_ → 应用层 dequeue 或回调处理。

---

## 九、关键类型与常量小结

| 类型/常量 | 含义 |
|-----------|------|
| **Frame** | 一帧触觉数据；含 frame_id、channel、ts、content（map<string, DataBlock::Ptr>）；content 常见 key：RAW、DEFORM、F6、CONTACT_POINT。 |
| **DataBlock::Ptr** | 多维数据块，有 shape、unit_size、data()。 |
| **Frame::is_to_infer()** | 是否有 RAW 且需要主机推理（无设备端 F6/DEFORM）。 |
| **kImgHeight_ / kImgWidth_** | 240、320。 |
| **kDeformHeight_ / kDeformWidth_** | 240、240。 |
| **kF6Size_** | 6。 |
| **kGridHeight_ / kGridWidth_** | 15、15，接触点网格。 |
| **SafeQueue<T>** | 带最大长度、超时 dequeue 的线程安全队列。 |
| **EngineBase::infer(input, size)** | 统一推理接口；input 为 [ref, realtime] 拼成的 float，size 为单图元素数；返回 3 个 vector<float>。 |

---

## 十、初学者阅读顺序建议

1. **engine_base.h** → **dummy_engine** → **trt.h / trt.cpp**：理解「接口 + 一个真实后端」的输入输出与数据布局。  
2. **def.h**（Frame、DataBlock）→ **content_protocol**（parse、is_to_infer）：理解一帧从报文到 `Frame` 的构成。  
3. **infer.h** → **infer.cpp** 的 Impl 构造与 **infer_()**：理解组 batch、调引擎、写回 content、校准、NMS、回调与 queue_out_。  
4. **CalibStat** 与 **set_ref_image / calib_zero**：理解校零流程。  
5. **receiver.cpp**（enqueue）→ **touch.cpp**（创建 InferEngine + Receiver、start 顺序）：理解整条链路。  
6. **infer_offline.cpp**：对比在线流程，理解单次推理、无校准的简化版。

按上述顺序阅读并对照本文档，即可掌握 tactile_sdk 推理引擎的架构与内容，便于调试、扩展新引擎或修改前后处理。
