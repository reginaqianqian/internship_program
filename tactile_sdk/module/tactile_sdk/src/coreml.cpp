
#include "coreml.h"

#include <onnxruntime_cxx_api.h>
#include <memory>

namespace sharpa {
namespace tactile {

struct CoremlInference::Impl {
    std::shared_ptr<Ort::Env> env;
    std::vector<Ort::AllocatedStringPtr> input_names_ptr;
    std::vector<const char*> inputNames;
    std::vector<std::vector<int64_t>> inputShapes;
    std::vector<Ort::AllocatedStringPtr> output_names_ptr;
    std::vector<const char*> outputNames;
    std::shared_ptr<Ort::Session> session;
};

void logFunc(
    void* param
    , OrtLoggingLevel severity
    , const char* category
    , const char* logid
    , const char* code_location
    , const char* message) {
    auto logger = (LoggerBase *)param;
    switch (severity)
    {
    case ORT_LOGGING_LEVEL_VERBOSE:
        logger->debug("[coreml] " + std::string(message));
        break;
    case ORT_LOGGING_LEVEL_INFO:
        logger->debug("[coreml] " + std::string(message));  /* make info msg also print only during debug */
        break;
    case ORT_LOGGING_LEVEL_WARNING:
        logger->warn("[coreml] " + std::string(message));
        break;
    case ORT_LOGGING_LEVEL_ERROR:
    case ORT_LOGGING_LEVEL_FATAL:
        logger->error("[coreml] " + std::string(message));
        break;
    default:
        break;
    }
}

CoremlInference::CoremlInference(const std::string& onnxPath, int defaultBatchSize) {
    impl_ = std::make_shared<Impl>();
    impl_->env =
        std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_INFO, "", &logFunc,
                                   LoggerSingleton::getInstance().getLogger().get());

    Ort::SessionOptions so;
    std::unordered_map<std::string, std::string> providerOptions;
    providerOptions["ModelFormat"] = "MLProgram";
    providerOptions["MLComputeUnits"] = "ALL";
    providerOptions["RequireStaticInputShapes"] = "0";
    providerOptions["EnableOnSubgraphs"] = "0";
    so.AppendExecutionProvider("CoreML", providerOptions);

    try {
        impl_->session = std::make_shared<Ort::Session>(*(impl_->env), onnxPath.c_str(), so);
    } catch (const Ort::Exception& e) {
        LOG_ERROR(e.what());
        throw e;
    }
    
    Ort::AllocatorWithDefaultOptions allocator;
    /* get input names and shapes - MODERN API */
    size_t num_input_nodes = impl_->session->GetInputCount();
    for (size_t i = 0; i < num_input_nodes; i++) {
        /* get input name */
        auto name = impl_->session->GetInputNameAllocated(i, allocator);
        impl_->input_names_ptr.push_back(std::move(name));
        impl_->inputNames.push_back(impl_->input_names_ptr.back().get());
        
        /* get input type and shape */
        Ort::TypeInfo type_info = impl_->session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        impl_->inputShapes.push_back(shape);
    }
    
    /* get output names - MODERN API */
    size_t num_output_nodes = impl_->session->GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto name = impl_->session->GetOutputNameAllocated(i, allocator);
        impl_->output_names_ptr.push_back(std::move(name));
        impl_->outputNames.push_back(impl_->output_names_ptr.back().get());
    }
}

std::vector<std::vector<float>> CoremlInference::infer(const float *input, int size) {
    const int batchSize = size / (1 * 240 * 320);
    impl_->inputShapes[0][0] = batchSize;

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    std::vector<Ort::Value> inputTensors;
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, 
        const_cast<float*>(input), 
        size, 
        impl_->inputShapes[0].data(), 
        impl_->inputShapes[0].size()
    ));

    std::vector<Ort::Value> outputTensors;
    try {
        outputTensors = impl_->session->Run(
            Ort::RunOptions{nullptr}, 
            impl_->inputNames.data(), 
            inputTensors.data(), 
            impl_->inputNames.size(), 
            impl_->outputNames.data(), 
            impl_->outputNames.size()
        );
    } catch (const Ort::Exception& e) {
        LOG_ERROR(e.what());
        throw e;
    }

    /* prepare results */
    std::vector<std::vector<float>> results;
    for (auto& output : outputTensors) {
        float *arr = output.GetTensorMutableData<float>();
        auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
        size_t size = 1;
        for(auto var : shape) size *= var;
        results.emplace_back(arr, arr + size);
    }
    return results;
}

}
}
