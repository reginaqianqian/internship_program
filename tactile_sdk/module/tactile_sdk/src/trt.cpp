#include "trt.h"

// #include <iostream>
#include <fstream>
#include <string>
#include <numeric>

#include <cstring>
#include "util.h"

namespace sharpa {
namespace tactile {

/* member function of Logger */
void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= severity_)
        logger_->error(msg);
}

/* member function of HostDeviceMem */
TensorRTInference::HostDeviceMem::HostDeviceMem(size_t size, nv::DataType dtype) : size(size), dtype(dtype) {
    cudaMallocHost(&host, size * getTypeSize());
    cudaMalloc(&device, size * getTypeSize());
}

TensorRTInference::HostDeviceMem::~HostDeviceMem() {
    cudaFreeHost(host);
    cudaFree(device);
}

size_t TensorRTInference::HostDeviceMem::getTypeSize() {
    switch(dtype) {
        case nv::DataType::kFLOAT: return sizeof(float);
        case nv::DataType::kHALF: return sizeof(uint16_t);
        default: throw std::runtime_error("Unsupported data type");
    }
}

/* member function of TensorRTInference */

TensorRTInference::TensorRTInference(const std::string& onnxPath, int defaultBatchSize)
    : logger_(LoggerSingleton::getInstance().getLogger(), Logger::Severity::kERROR) {
    runtime_.reset(nv::createInferRuntime(logger_));
    engine_.reset(loadOrBuildEngine(onnxPath));
    context_.reset(engine_->createExecutionContext());
    cudaStreamCreate(&stream_);
    setupDynamicShapes(defaultBatchSize);
}

std::vector<std::vector<float>> TensorRTInference::infer(const float *input, int size) {
    const int batchSize = size / (1 * 240 * 320);
    
    const char* inputName1 = engine_->getIOTensorName(0);  // 第一个输入
    const char* inputName2 = engine_->getIOTensorName(1);  // 第二个输入

    // 动态更新批次大小（若需要）
    nv::Dims inputDims1 = context_->getTensorShape(inputName1);
    nv::Dims inputDims2 = context_->getTensorShape(inputName2);
    
    if (inputDims1.d[0] != batchSize || inputDims2.d[0] != batchSize) {
        setupDynamicShapes(batchSize);
    }


    /* copy input and execute inference */
    memcpy(inputs_[0]->host, input, size * sizeof(float));
    cudaMemcpyAsync(inputs_[0]->device, inputs_[0]->host, 
                    size * sizeof(float), cudaMemcpyHostToDevice, stream_);
    memcpy(inputs_[1]->host, input+size, size * sizeof(float));
    cudaMemcpyAsync(inputs_[1]->device, inputs_[1]->host, 
                    size * sizeof(float), cudaMemcpyHostToDevice, stream_);
    
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        std::string tensor_name = engine_->getIOTensorName(i); /* get the tensor name */
        void* binding = bindings_[i]; /* get the binding address */
        context_->setTensorAddress(tensor_name.c_str(), binding); /* set the tensor address */
    }
    
    // context_->enqueueV2(bindings_.data(), stream_, nullptr);
    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("Execution failed");
    }
    
    /* copy outputs back */
    for (auto& output : outputs_) {
        cudaMemcpyAsync(output->host, output->device,
                      output->size * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    }
    
    cudaStreamSynchronize(stream_);
    
    /* prepare results */
    std::vector<std::vector<float>> results;
    for (auto& output : outputs_) {
        float* hostData = static_cast<float*>(output->host);
        results.emplace_back(hostData, hostData + output->size);
    }
    return results;
}

nv::ICudaEngine* TensorRTInference::loadOrBuildEngine(const std::string& onnxPath) {
    const std::string enginePath = onnxPath.substr(0, onnxPath.find_last_of('.')) + ".engine";
    std::ifstream engineFile(enginePath, std::ios::binary);
    
    if (engineFile.good()) {
        engineFile.seekg(0, std::ios::end);
        const size_t size = engineFile.tellg();
        engineFile.seekg(0, std::ios::beg);
        std::vector<char> engineData(size);
        engineFile.read(engineData.data(), size);
        return runtime_->deserializeCudaEngine(engineData.data(), size);
    }
    return buildEngine(onnxPath, enginePath);
}

nv::ICudaEngine* TensorRTInference::buildEngine(const std::string& onnxPath, const std::string& enginePath) {
    LOG_INFO("building trt engine");
    auto builder = std::unique_ptr<nv::IBuilder>(nv::createInferBuilder(logger_));
    const auto networkFlags = 1U << static_cast<uint32_t>(nv::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nv::INetworkDefinition>(builder->createNetworkV2(networkFlags));
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
    
    /* parse ONNX */
    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nv::ILogger::Severity::kERROR))) {
        throw std::runtime_error("Failed to parse ONNX file");
    }

    /* configure builder */
    auto config = std::unique_ptr<nv::IBuilderConfig>(builder->createBuilderConfig());
    config->setFlag(nv::BuilderFlag::kFP16);
    config->setMemoryPoolLimit(nv::MemoryPoolType::kWORKSPACE, 1 << 30);

    /* set optimization profile */
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("initial_image", nv::OptProfileSelector::kMIN, nv::Dims4{1, 1, 240, 320});
    profile->setDimensions("initial_image", nv::OptProfileSelector::kOPT, nv::Dims4{5, 1, 240, 320});
    profile->setDimensions("initial_image", nv::OptProfileSelector::kMAX, nv::Dims4{10, 1, 240, 320});
    profile->setDimensions("realtime_image", nv::OptProfileSelector::kMIN, nv::Dims4{1, 1, 240, 320});
    profile->setDimensions("realtime_image", nv::OptProfileSelector::kOPT, nv::Dims4{5, 1, 240, 320});
    profile->setDimensions("realtime_image", nv::OptProfileSelector::kMAX, nv::Dims4{10, 1, 240, 320});
    config->addOptimizationProfile(profile);

    /* build engine */
    auto serializedEngine = std::unique_ptr<nv::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!serializedEngine) {
        throw std::runtime_error("Failed to build TensorRT engine");
    }

    /* save engine */
    std::ofstream engineFile(enginePath, std::ios::binary);
    engineFile.write(static_cast<const char*>(serializedEngine->data()), serializedEngine->size());

    LOG_INFO("building finished");
    return runtime_->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size());
}

void TensorRTInference::setupDynamicShapes(int batchSize) {

    /* reallocate buffers */
    inputs_.clear();
    outputs_.clear();
    bindings_.clear();

    int numBindings = engine_->getNbIOTensors();
    for (int i = 0; i < numBindings; ++i) {
        const char* tensorName = engine_->getIOTensorName(i);
        nv::Dims dims = context_->getTensorShape(tensorName);
        dims.d[0] = batchSize; 
        const nv::DataType dtype = engine_->getTensorDataType(tensorName);

        if (engine_->getTensorIOMode(tensorName) == nv::TensorIOMode::kINPUT) {
            if (!context_->setInputShape(tensorName, dims)) {
                throw std::runtime_error("Failed to set input tensor shape");
            }
            const size_t size = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<size_t>());
            auto mem = std::make_unique<HostDeviceMem>(size, dtype);
            bindings_.push_back(mem->device);
            inputs_.push_back(std::move(mem));
        } else {
            const size_t size = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<size_t>());
            auto mem = std::make_unique<HostDeviceMem>(size, dtype);
            bindings_.push_back(mem->device);
            outputs_.push_back(std::move(mem));
        }
    }
}

}
}
