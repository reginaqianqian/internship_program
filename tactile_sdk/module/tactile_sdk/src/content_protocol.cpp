
#include "content_protocol.h"
#include "util.h"

#include <turbojpeg.h>
#include <cstring>
#include <iostream>

namespace sharpa {
namespace tactile {

// static std::vector<unsigned char>
std::shared_ptr<DataBlock> decode_jpeg(const unsigned char* jpeg_data,
                                       size_t jpeg_size,
                                       size_t out_height,
                                       size_t out_width) {
    tjhandle handle = tjInitDecompress();
    if (!handle) {
        LOG_ERROR("failed to initialize TurboJPEG decompressor");
        return {};
    }
    try {
        /* get image info */
        int width, height, jpegSubsamp, jpegColorspace;
        if (tjDecompressHeader3(handle, jpeg_data, jpeg_size,
                               &width, &height, &jpegSubsamp, &jpegColorspace) != 0) {
            throw std::runtime_error(tjGetErrorStr());
        }
        /* allocate output buffer (GRAY format = 1 byte per pixel) */
        auto block = std::make_shared<DataBlock>(
            Shape{{1, out_height, out_width}},
            sizeof(uint8_t)
        );
        /* decompress */
        if (tjDecompress2(handle, jpeg_data, jpeg_size,
                         (uint8_t *)block->data(), out_width,
                         0 /* pitch */, out_height,
                         TJPF_GRAY,
                         TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT) != 0) {
            throw std::runtime_error(tjGetErrorStr());
        }
        tjDestroy(handle);
        return block;
    } catch (const std::exception& e) {
        tjDestroy(handle);
        LOG_ERROR("Invalid JPEG: {}", e.what());
        return nullptr;
    }
}

bool Frame::is_to_infer() const {
    bool has_raw = (content.find("RAW") != content.end());
    bool has_deform = content.find("DEFORM_JPG") != content.end();
    bool has_f6 = (content.find("F6") != content.end());
    /* frames without 'RAW' cannot be inferred, frames with 'DEFORM' and 'F6' don't need infer */
    return has_raw && (!has_deform || !has_f6);
}

Frame::Ptr Frame::copy() const {
    auto ret = std::make_shared<Frame>(*this);
    for(auto [key, val] : content) {
        ret->content[key] = std::make_shared<DataBlock>(*val);
    }
    return ret;
}

ContentParser::ContentParser(int channel, int num_pack, bool decode_jpeg)
    : channel_(channel), num_pack_(num_pack), decode_jpeg_(decode_jpeg) {
    packs_.resize(num_pack);
    ts_ = ts_unix();
}

void ContentParser::append_pack(int pack_id, std::vector<uint8_t> pack) {
    if(packs_[pack_id].empty()) {
        num_recv_+=1;
    }
    packs_[pack_id] = std::move(pack);
}

int ContentParser::num_pack() const { return num_pack_; }
int ContentParser::num_recv() const { return num_recv_; }

Frame::Ptr ContentParser::parse() const {
    /* join bytes together */
    std::vector<uint8_t> joined;
    for(const auto& byte_array : packs_) {
        joined.insert(joined.end(), byte_array.begin(), byte_array.end());
    }

    auto ret = std::make_shared<Frame>();
    ret->channel = channel_;
    uint8_t *ptr_header = joined.data();
    /* parse header0 */
    auto header0 = (ContentHeader0_24 *)ptr_header;
    ptr_header += sizeof(ContentHeader0_24);
    ret->frame_id = header0->frame_id;
    // ret->ts = header0->time_stamp;
    ret->ts = ts_; /* use host-side timestamp instead */
    /* get number of 1 flags */
    std::vector<int> flags;
    for(int i = 0; i < CONTENT_FLAG_LENGTH; ++i)
        if(header0->content_flags & (1 << i)) flags.push_back(i);
    uint8_t *ptr_data = ptr_header + sizeof(ContentHeaderDynamicBlock8) * flags.size();
    /* get indices of all 1-bit */
    for(auto i : flags) {
        std::string key(get_content_flag(i));
        /* parse block */
        auto h = (ContentHeaderDynamicBlock8 *)ptr_header;
        ptr_header += sizeof(ContentHeaderDynamicBlock8);
        auto block = std::make_shared<DataBlock>(
            Shape{{(size_t)h->size_0, (size_t)h->size_1, (size_t)h->size_2}},
            get_unit_size(h->data_type)
        );
        memcpy(block->data(), ptr_data, block->nbytes());
        ptr_data += block->nbytes();

        if(key == "RAW_JPG" && decode_jpeg_==true) {
            auto block_raw =
                decode_jpeg((uint8_t*)block->data(), block->nbytes(), 240, 320);
            if(block_raw) ret->content.try_emplace("RAW", block_raw);
        }

        if(key == "DEFORM_JPG" && decode_jpeg_==true) {
            auto block_deform =
                decode_jpeg((uint8_t*)block->data(), block->nbytes(), 240, 240);
            if(block_deform) ret->content.try_emplace("DEFORM", block_deform);
        }

        ret->content.emplace(key, block);
    }
    return ret;
}

}
}
