#pragma once
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

cv::Mat slice_image(const cv::Mat& input, const torch::Tensor& mask, int label, cv::Rect& out_box);
bool write_slices(const cv::Mat& input, const torch::Tensor& mask,
    const std::filesystem::path& output_path, const std::filesystem::path& file_directory_path, const std::string& extension);
