
#include <zlib.h>
#include <fmt/core.h>

#include "protocol_def.h"
#include "stream_protocol.h"
#include "util.h"
#include <iostream>

namespace sharpa {
namespace tactile {

StreamParser::StreamParser(const std::vector<int>& channel,
                           int allowed_pack_loss,
                           bool decode_jpeg)
    : allowed_pack_loss_(allowed_pack_loss), decode_jpeg_(decode_jpeg) {
    for (auto ch : channel)
        buf_[ch] = {};
    for (auto ch : channel)
        multi_buf_[ch] = {};
}

    bool StreamParser::isBefore(uint16_t a, uint16_t b) {
        //32768是frameID循环计数最大值的一半
        return (a < b && b - a < 32768) || (a > b && a - b > 32768);
    }

    bool StreamParser::isBeforeFrame(const FrameStat& a, const FrameStat& b) {
        return isBefore(a.id, b.id);
    }


    // 查找空闲 buffer
    int StreamParser::find_idle_buffer(int channel, int frame_id) {
        for (int i = 0; i < BUFFER_POOL_SIZE; ++i) {
            if (multi_buf_[channel][i].id == -1) {
                return i;
            }
        }
        // 没有空闲？尝试覆盖最老的 RECEIVING（假设严重乱序）
        // 现在所有 buffer 都在 RECEIVING 状态
        int oldest_idx = -1;
        auto it = std::min_element(multi_buf_[channel].begin(), multi_buf_[channel].end(), isBeforeFrame);
        if (isBefore(it->id, frame_id) || (ts_unix() - it->parser->get_ts() > 1.0f)) {
            return std::distance(multi_buf_[channel].begin(), it);
        }
        return -1;
        // 可选：只覆盖超过 100ms 的
    }

    bool StreamParser::find_receiving_buffer(int channel,
                                             int frame_id,
                                             int pack_id,
                                             int& idx) {
        auto now_time = ts_unix();
        for (int i = 0; i < BUFFER_POOL_SIZE; ++i) {
            if (multi_buf_[channel][i].id == frame_id) {
                idx = i;
                //获取以秒为单位的时间now_time, 因为ts也是以秒为单位的
                bool res = ((now_time - multi_buf_[channel][i].parser->get_ts()) > 1.0) ||
                           (pack_id >= multi_buf_[channel][i].parser->num_pack());
                return !res;
            }
        }
        idx=-1;
        return false;
    }

    void StreamParser::init_multi_frame_(int channel, int idx, int frame_id, int num_packs) {
        auto &f = multi_buf_[channel][idx];
        if(f.id != -1) {
            f.packet_loss+=f.parser->num_recv();
        }

        f.id = frame_id;
        f.parser = std::make_shared<ContentParser>(channel, num_packs, decode_jpeg_);
    }

    RawFrame StreamParser::parse_multi_(int channel, int idx) {
        if(idx < 0) return ret_parse_multi_(channel, idx, true);
            
        auto &f = multi_buf_[channel][idx];
        if(f.parser->num_pack() == f.parser->num_recv()) return ret_parse_multi_(channel, idx, false);
        return ret_parse_multi_(channel, idx, true);
    }


    RawFrame StreamParser::ret_parse_multi_(int channel, int idx, bool is_failed) {
        if(is_failed) {
            return std::nullopt;
        }
        auto &f = multi_buf_[channel][idx];
        f.frame_got += 1;
        f.packet_got+=f.parser->num_recv();
        f.id=-1;
        return f.parser;
    }


    RawFrame
    StreamParser::receive_multi(const std::vector<uint8_t> &data) {
        total_pack_count_++;
        RawFrame ret{std::nullopt};
        /* parse data */
        const uint8_t* ptr = data.data();
        auto h4 = (StreamHeader0_4*)ptr;
        ptr += sizeof(StreamHeader0_4);
        auto h12 = (StreamHeader1_12*)ptr;
        ptr += sizeof(StreamHeader1_12);
        int size_payload = data.data() + data.size() - ptr - sizeof(StreamTail_4);
        std::vector<uint8_t> payload(ptr, ptr + size_payload);
        ptr += size_payload;
        auto tail4 = (StreamTail_4*)ptr;

        /* reject invalid pack */
        if ((h4->ini_id0 != PACKET_INI0 || h4->ini_id1 != PACKET_INI1) ||
            (h4->protocal_version != PROTOCAL_VERSION &&
             h4->protocal_version != PROTOCAL_OLD_VERSION) ||
            (h12->device_type != DEVICE_TYPE || h12->pack_type != PACKET_TYPE) ||
            (h12->pack_id < 0 || h12->pack_id >= h12->num_pack)) {
            invalid_pack_count_++;
            return ret;
        }

        if(tail4->crc != 0xffff) {
            uint32_t crc = crc32(0L, payload.data(), payload.size());
            if(tail4->crc != crc) {
                invalid_pack_count_++;
                return ret;
            }
        }
        
        /* parse */
        //这里用buf_而不是multi_buf_是因为最后统计的时候用的是buf_中的start_ts，这里只是用它来记录时间
        auto frame_iter = buf_.find(h12->channel);
        if(frame_iter == buf_.end()) {
            invalid_pack_count_++;
            return ret;  /* channel not exist */
        }
        auto &f = frame_iter->second;
        if(f.start_ts < 0) f.start_ts = ts_unix();
        
        int idx=-1;
        bool if_find =
            find_receiving_buffer(h12->channel, h12->frame_id, h12->pack_id, idx);
        if(!if_find && idx != -1){
            //这里表示极端情况下buffer[ch]中存了一个不完整的frame：
            //但是这个frame是在上一个frameID循环周期中接收并存储的，经过了一整个frmaeID循环，仍然没有接收完整并导致其一直无法发出去(如果发出去了ID会被置为-1)
            //此时又接收到有相同的frameID的pack了
            ++(multi_buf_[h12->channel][idx].frame_loss);  /* frame loss */
            init_multi_frame_(h12->channel, idx, h12->frame_id, h12->num_pack);
        }
        if(idx!=-1) {
            multi_buf_[h12->channel][idx].parser->append_pack(h12->pack_id, payload);
        }
        else {
            idx = find_idle_buffer(h12->channel, h12->frame_id);
            if(idx != -1) {
                if(multi_buf_[h12->channel][idx].id != -1)
                    ++(multi_buf_[h12->channel][idx].frame_loss);  /* frame loss */
                init_multi_frame_(h12->channel, idx, h12->frame_id, h12->num_pack);
                multi_buf_[h12->channel][idx].parser->append_pack(h12->pack_id, payload);
            }
            else{
                //这里因为idx==-1,所以统计包丢失的任务就选择buffer中当前通道的index为0的FrameStat上
                //但是这并不影响，因为最后会统一计算的时候是按照channel来进行统计的
                multi_buf_[h12->channel][0].packet_loss++; 
            }                    
        }
        ret = parse_multi_(h12->channel, idx);
        return ret;
    }

    StreamParser::StreamStatistic StreamParser::summary() {
        StreamStatistic stat;
        for(auto& ch: multi_buf_){
            for(auto& fs: ch.second){
                stat.channel_statistic[ch.first].frame_got += fs.frame_got;
                stat.channel_statistic[ch.first].frame_loss += fs.frame_loss;
                stat.channel_statistic[ch.first].packet_got += fs.packet_got;
                stat.channel_statistic[ch.first].packet_loss += fs.packet_loss;
            }
            stat.channel_statistic[ch.first].start_ts = buf_[ch.first].start_ts;
        }
        stat.invalid_pack = invalid_pack_count_;
        stat.total_pack = total_pack_count_;
        return stat;
    }

    void StreamParser::init_frame_(int channel, int frame_id, int num_packs) {
        auto &f = frame_(channel);
        f.id = frame_id;
        f.parser = std::make_shared<ContentParser>(channel, num_packs, decode_jpeg_);
    }


    StreamParser::FrameStat &StreamParser::frame_(int channel) {
        auto frame_iter = buf_.find(channel);
        assert_throw(frame_iter != buf_.end(), fmt::format("channel {} doesn't exist", channel));
        return frame_iter->second;
    }


    RawFrame
    StreamParser::receive(const std::vector<uint8_t> &data) {
        RawFrame ret{std::nullopt};
        /* parse data */
        const uint8_t *ptr = data.data();
        auto h4 = (StreamHeader0_4 *)ptr;
        ptr += sizeof(StreamHeader0_4);
        auto h12 = (StreamHeader1_12 *)ptr;
        ptr += sizeof(StreamHeader1_12);
        int size_payload = data.data() + data.size() - ptr - sizeof(StreamTail_4);
        std::vector<uint8_t> payload(ptr, ptr + size_payload);
        ptr += size_payload;
        auto tail4 = (StreamTail_4 *)ptr;

        /* reject invalid pack */
        if((h4->ini_id0 != PACKET_INI0 || h4->ini_id1 != PACKET_INI1)
            ||(h4->protocal_version != PROTOCAL_VERSION && h4->protocal_version != PROTOCAL_OLD_VERSION)
            ||(h12->device_type != DEVICE_TYPE || h12->pack_type != PACKET_TYPE)
            ||(h12->pack_id<0 || h12->pack_id>=h12->num_pack)) {
            ++invalid_pack_count_;  
            return ret;
        }

        if(tail4->crc != 0xffff) {
            uint32_t crc = crc32(0L, payload.data(), payload.size());
            if(tail4->crc != crc) {
                ++invalid_pack_count_;  
                return ret;
            }
        }
        /* parse */
        auto frame_iter = buf_.find(h12->channel);
        if(frame_iter == buf_.end()) {
            ++invalid_pack_count_;    
            return ret;  /* channel not exist */
        }
        auto &f = frame_iter->second;
        if(f.start_ts < 0) f.start_ts = ts_unix();

        /* insure first packet is received */
        if(f.id < 0) {
            LOG_DEBUG("{} got pack 0", ts_unix());

            init_frame_(h12->channel, h12->frame_id, h12->num_pack);
        }
        /* update frame */
        if(h12->frame_id != f.id) {
            if (!isBefore(h12->frame_id, f.id) ||
                (ts_unix() - f.parser->get_ts() > 1.0f)) {
                f.packet_loss += f.parser->num_recv();
                ++f.frame_loss;
                init_frame_(h12->channel, h12->frame_id, h12->num_pack);
                f.parser->append_pack(h12->pack_id, payload);
            } else {
                ++f.packet_loss;
            }
        } else {
            assert_throw(f.parser.get(), "parser is None");
            f.parser->append_pack(h12->pack_id, payload);
        }
        ret = parse_(h12->channel);
        return ret;
    }


    RawFrame StreamParser::parse_(int channel) {
        auto &f = frame_(channel);
        if(f.parser->num_pack() == f.parser->num_recv()) return ret_parse_(channel, false);
        return ret_parse_(channel, true);
    }


    RawFrame StreamParser::ret_parse_(int channel, bool is_failed) {
        if(is_failed) {
            return std::nullopt;
        }
        auto &f = frame_(channel);
        f.packet_got += f.parser->num_recv();
        f.frame_got += 1;
        f.id = -1;
        return f.parser;
    }

}
}
