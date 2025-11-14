#include <algorithm>
#include <iostream>
#include <string>
#include "image_loader.hpp"

namespace fs = std::filesystem;

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c) {
            return std::tolower(c);
        });
    return s;
}

std::vector<fs::path> find_image_files_recursively(const fs::path& dir, const std::string& image_format) {
    std::vector<fs::path> image_files;

    if(!fs::exists(dir) || !fs::is_directory(dir)) {
        std::cerr << "Invalid directory: " << dir << "\n";
        return image_files;
    }

    for(const auto& entry : fs::recursive_directory_iterator(dir)) {
        if(entry.is_regular_file()) {
            const auto ext = to_lower(entry.path().extension().string());
            if(ext == "." + image_format) {
                image_files.push_back(entry.path());
            }
        }
    }
    return image_files;
}

cv::Mat load_image(const fs::path& path) {

    cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "Error: Could not load image" << std::endl;
        return cv::Mat();
    }

    return img;
}
