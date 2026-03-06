
#pragma once

#include "engine_base.h"

namespace sharpa {
namespace tactile {

class DummyEngine : public EngineBase {
public:
    std::vector<std::vector<float>> infer(const float *input, int size) override;
};

}
}
