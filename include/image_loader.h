#pragma once
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

std::vector<std::filesystem::path> find_image_files_recursively(const std::filesystem::path&, const std::string&);
cv::Mat load_image(const std::filesystem::path& path);
