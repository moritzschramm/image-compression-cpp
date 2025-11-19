#pragma once
#include <thread>
#include <future>
#include <fstream>
#include <mutex>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

struct SliceMetadata {
    int label;
    std::string filename;
    int x, y, width, height;
};

#pragma pack(push, 1)
struct SliceRecordHeader {
    uint32_t magic;  // "SLIC" = 0x534C4943
    uint32_t count;
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

void write_metadata_binary(const std::vector<SliceMetadata>& metadata, const std::string& path);
std::vector<SliceMetadata> read_metadata_binary(const std::string& path);

cv::Mat slice_image(const cv::Mat& input, const torch::Tensor& mask, int label, cv::Rect& out_box);
bool write_slices(const cv::Mat& input, const torch::Tensor& mask,
    const std::filesystem::path& output_path, const std::filesystem::path& file_directory_path, const std::string& extension);
