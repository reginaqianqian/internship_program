
#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <vector>

#include <def.h>
#include "engine_base.h"

namespace nv = nvinfer1;

namespace sharpa {
namespace tactile {

class Logger : public nv::ILogger {
public:
    explicit Logger(std::shared_ptr<LoggerBase> logger,
                    Severity severity = Severity::kERROR)
        : logger_(logger), severity_(severity) {}
    void log(Severity severity, const char* msg) noexcept override;

private:
    Severity severity_;
    std::shared_ptr<LoggerBase> logger_{nullptr};
};

class TensorRTInference : public EngineBase {
private:
    class HostDeviceMem {
    public:
        void* host;
        void* device;
        std::vector<int> shape;
        size_t size;
        nv::DataType dtype;

        HostDeviceMem(size_t size, nv::DataType dtype);
        ~HostDeviceMem();

    private:
        size_t getTypeSize();
    };

    Logger logger_;
    std::unique_ptr<nv::IRuntime> runtime_;
    std::unique_ptr<nv::ICudaEngine> engine_;
    std::unique_ptr<nv::IExecutionContext> context_;
    std::vector<std::unique_ptr<HostDeviceMem>> inputs_;
    std::vector<std::unique_ptr<HostDeviceMem>> outputs_;
    std::vector<void*> bindings_;
    cudaStream_t stream_;

public:
    TensorRTInference(const std::string& onnxPath, int defaultBatchSize = 1);
    std::vector<std::vector<float>> infer(const float* input, int size) override;

private:
    nv::ICudaEngine* loadOrBuildEngine(const std::string& onnxPath);
    nv::ICudaEngine* buildEngine(const std::string& onnxPath,
                                 const std::string& enginePath);
    void setupDynamicShapes(int batchSize);
};

}  // namespace tactile
}  // namespace sharpa
