#pragma once

#include <array>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <def.h>
#include "protocol_def.h"

namespace sharpa {
namespace tactile {

class ContentParser {
public:
    ContentParser(int channel, int num_packs, bool decode_jpeg);
    int num_pack() const;
    int num_recv() const;
    void append_pack(int pack_id, std::vector<uint8_t> pack);
    Frame::Ptr parse() const;
    double get_ts() const { return ts_; }

private:
    std::vector<std::vector<uint8_t>> packs_;
    int num_pack_;
    int num_recv_{0};
    int channel_;
    double ts_;
    bool decode_jpeg_;
};

using RawFrame = std::optional<std::shared_ptr<ContentParser>>;

}  // namespace tactile
}  // namespace sharpa
