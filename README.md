# internship_program
Sort out the project completed during the internship


leapc-python-bindings/
├── leapc-cffi/                          # 底层：C → Python CFFI 绑定
│   ├── setup.py                         # 构建脚本（查找SDK、编译C扩展）
│   ├── README.md
│   ├── MANIFEST.in
│   └── src/
│       ├── leapc_cffi/
│       │   ├── __init__.py              # 导出 ffi + libleapc
│       │   ├── LeapC.h                  # LeapC API 头文件（从SDK拷贝）
│       │   ├── LeapC.dll / LeapC.lib    # Windows 动态库（从SDK拷贝/链接）
│       └── scripts/
│           ├── cffi_build.py            # CFFI 编译脚本
│           └── cffi_src.h               # CFFI 源头文件
│
├── leapc-python-api/                    # 上层：Pythonic API 封装
│   ├── setup.py
│   ├── README.md
│   ├── MANIFEST.in
│   └── src/leap/
│       ├── __init__.py                  # 包入口，查找 leapc_cffi 并导出公共 API
│       ├── cstruct.py                   # C 结构体基类
│       ├── connection.py                # 连接管理
│       ├── event_listener.py            # 事件监听器
│       ├── events.py                    # 事件类型定义
│       ├── datatypes.py                 # 手部/骨骼等数据类型
│       ├── enums.py                     # 枚举类型（自动从 LeapC 生成）
│       ├── device.py                    # 设备管理
│       ├── functions.py                 # 全局函数
│       ├── exceptions.py               # 异常类型
│       └── recording.py                # 录制/回放
│
├── examples/                            # 示例代码
│   ├── tracking_event_example.py
│   ├── multi_device_example.py
│   ├── simple_pinching_example.py
│   ├── interpolation_example.py
│   ├── visualiser.py
│   └── print_current_time.py
│
├── requirements.txt
├── pyproject.toml
├── README.md / CONTRIBUTING.md / LICENSE.md
└── .gitlab-ci.yml


## 实习经验
如果当前代码分支没有群里提到的需求，问一下是哪个分支的代码。

项目确认需求，例如代码放在哪个目录下，以及