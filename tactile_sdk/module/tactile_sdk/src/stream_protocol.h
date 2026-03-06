#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include <touch.h>
#include "content_protocol.h"

namespace sharpa {
namespace tactile {

constexpr int BUFFER_POOL_SIZE = 2;  // 2 倍缓存

class StreamParser {
public:
    StreamParser(const std::vector<int>& channel,
                 int allowed_pack_loss,
                 bool decode_jpeg);
    struct FrameStat {
        std::shared_ptr<ContentParser> parser{nullptr};
        int id{-1}, packet_got{0}, packet_loss{0}, frame_got{0}, frame_loss{0};
        double start_ts{-1.};
    };

    struct ChannelStatistic {
        int frame_got{0}, frame_loss{0}, packet_got{0}, packet_loss{0};
        double start_ts{-1.};
    };
    struct StreamStatistic {
        std::unordered_map<int, ChannelStatistic> channel_statistic;
        unsigned long long invalid_pack{0};
        unsigned long long total_pack{0};
    };

    using FrameStats = std::unordered_map<int, FrameStat>;
    using MultiFrameStats =
        std::unordered_map<int, std::array<FrameStat, BUFFER_POOL_SIZE>>;

    RawFrame receive(const std::vector<uint8_t>& data);
    StreamStatistic summary();
    RawFrame parse_(int channel);
    RawFrame receive_multi(const std::vector<uint8_t>& data);

private:
    void init_frame_(int channel, int frame_id, int num_packs);
    FrameStat& frame_(int channel);
    RawFrame ret_parse_(int channel, bool is_failed);

    int find_idle_buffer(int channel, int frame_id);
    bool find_receiving_buffer(int channel, int frame_id, int pack_id, int& idx);
    void init_multi_frame_(int channel, int idx, int frame_id, int num_packs);
    RawFrame parse_multi_(int channel, int idx);
    RawFrame ret_parse_multi_(int channel, int idx, bool is_failed);
    static bool isBefore(uint16_t a, uint16_t b);
    static bool isBeforeFrame(const FrameStat& a, const FrameStat& b);
    MultiFrameStats multi_buf_;

    int allowed_pack_loss_;
    int num_dropped_start_frames_{0};
    const int num_drop_start_frames_{10};
    FrameStats buf_;
    bool decode_jpeg_;

    unsigned long long invalid_pack_count_{0};
    unsigned long long total_pack_count_{0};
};

}  // namespace tactile
}  // namespace sharpa
