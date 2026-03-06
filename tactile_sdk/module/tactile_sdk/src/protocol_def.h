#pragma once

#include <stdint.h>

namespace sharpa {
namespace tactile {

const inline uint8_t PACKET_INI0 = 0xbb;
const inline uint8_t PACKET_INI1 = 0xee;
const inline uint8_t DEVICE_TYPE = 0x00;  /** 0x00 for sharpa wave */
const inline uint8_t PACKET_TYPE = 0x02;  /** 0x02 for tactile data */

/**
 * bit [2,7) protocol version
 * bit [0,2) protocol type: 0 packet; 1 stream;
 */
const inline uint8_t PROTOCAL_VERSION = 0b01000000;  /** version 0 type stream */
const inline uint8_t PROTOCAL_OLD_VERSION = 0b00000001;  /** version 0 type stream */

/* stream related */

typedef struct {
    uint8_t ini_id0;
    uint8_t ini_id1;
    uint8_t protocal_version;
    uint8_t reserved;
} StreamHeader0_4;

typedef struct {
    uint8_t device_type;
    uint8_t pack_type;
    uint16_t channel;
    uint16_t frame_id;
    uint16_t num_pack;
    uint16_t pack_id;
    uint16_t reserved;
} StreamHeader1_12;

typedef struct {
    uint32_t crc;
} StreamTail_4;

typedef struct {
    uint8_t protocal_version;
    uint8_t reserved0;
    uint16_t frame_id;
    uint32_t content_flags;
    double time_stamp;
    uint64_t reserved1;
} ContentHeader0_24;

const inline uint8_t CONTENT_FLAG_LENGTH = 32;  /** 32-bit content flag */

typedef struct {
    uint16_t size_0;
    uint16_t size_2;
    uint32_t size_1 : 24;
    uint8_t data_type;  /* one-byte */
} ContentHeader1_8;

typedef enum {
    RESERVED_0 = 1 << 0,
    F6 = 1 << 1,
    RESERVED_1 = 1 << 2,
    DEFORM_JPG = 1 << 3,
    RAW_JPG = 1 << 4,
    TEMPERATURE = 1 << 5,
    AUDIO = 1 << 6,
} ContentFlag;

static inline const char *get_content_flag(int idx) {
    switch (idx) {
    case 0: return "RESERVED_0";
    case 1: return "F6";
    case 2: return "RESERVED_1";
    case 3: return "DEFORM_JPG";
    case 4: return "RAW_JPG";
    case 5: return "TEMPERATURE";
    case 6: return "AUDIO";
    case 7: return "CONTACT_POINT";
    default: return "";
    }
}

typedef struct {
    uint16_t size_0;
    uint16_t size_2;
    uint32_t size_1 : 24;
    uint8_t data_type;  /* one-byte */
} ContentHeaderDynamicBlock8;

static inline int get_unit_size(uint8_t data_type) {
    switch(data_type) {
    case 0: return 1;  // bool  TO_DO need to be optimized
    case 1: return 1;  // int8
    case 2: return 2;  // int16
    case 3: return 4;  // int32
    case 4: return 8;  // int64
    case 5: return 4;  // float32
    case 6: return 8;  // float64
    default: return -1;
    }
}

typedef struct {
    ContentHeader1_8 header;
    ContentFlag flag;
    uint8_t *data;
} ContentDynamicBlock;

typedef struct {
    int num_block;
    ContentDynamicBlock *block;
} ContentMsg;

}
}
