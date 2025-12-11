#pragma once
#include <opencv2/opencv.hpp>
#include <filesystem>

bool write_image(const std::filesystem::path& filename, const cv::Mat& image);
