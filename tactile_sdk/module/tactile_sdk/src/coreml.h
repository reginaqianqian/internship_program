#pragma once

#include <def.h>
#include "engine_base.h"

namespace sharpa {
namespace tactile {

class CoremlInference : public EngineBase {
public:
    CoremlInference(const std::string& onnxPath, int defaultBatchSize = 1);
    std::vector<std::vector<float>> infer(const float* input, int size) override;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace tactile
}  // namespace sharpa
