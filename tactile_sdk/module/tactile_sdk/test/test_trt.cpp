#include <catch2/catch_all.hpp>

#include <src/trt.h> 

using namespace sharpa::tactile;

TEST_CASE("trt", "[tactile_sdk]") {
    // int batch_size{3};
    // std::vector<float> to_infer(320 * 240 * batch_size);
    // TensorRTInference trt("config/models/model_epoch_100__HA3_010_finetune.onnx", batch_size);
    // std::vector<std::vector<float>> ret = trt.infer(to_infer.data(), to_infer.size());
    // CHECK(ret[0].size() == batch_size * 3);
    // CHECK(ret[1].size() == batch_size * 240 * 320);
}
