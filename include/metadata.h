#pragma once
#include <stdint.h>
#include <string>
#include <vector>

struct SliceMetadata {
    int label;
    std::string filename;
    int x, y, width, height;
};

#pragma pack(push, 1)
struct SliceRecordHeader {
    uint32_t magic;  // "SLIC" = 0x534C4943
    uint32_t count;
    uint32_t original_width;
    uint32_t original_height;
};

struct SliceRecordFixed {
    int32_t label;
    int32_t x;
    int32_t y;
    int32_t width;
    int32_t height;
    uint16_t filename_len;  // length of following filename bytes
};
#pragma pack(pop)


void write_metadata_binary(const std::vector<SliceMetadata>& metadata, const std::string& path, uint32_t image_width, uint32_t image_height);
std::vector<SliceMetadata> read_metadata_binary(const std::string& path, uint32_t& image_width, uint32_t& image_height);
