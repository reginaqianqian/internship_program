#pragma once

#include <vector>

namespace sharpa {
namespace tactile {
    
class EngineBase {
public:    
virtual std::vector<std::vector<float>> infer(const float *input, int size) = 0;
};

}
}
