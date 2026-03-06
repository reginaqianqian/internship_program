
#include "version_map.h"

#include <map>

namespace sharpa {
namespace tactile {

/** mapping from sdk version to firmware version */
const static std::map<std::string, std::string> kVersionMap_{
    {"1.1.1", "1.1.0"},
    {"1.1.2", "1.1.0"},
    {"1.1.3", "1.1.0"},
    {"1.1.4", "1.1.0"},
    {"2.0.0", "1.3.4"},
};

std::string version_map(const std::string &version) {
    auto iter = kVersionMap_.find(version);
    if (iter != kVersionMap_.end()) return iter->second;
    return version;
}

}
}
