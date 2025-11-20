#include "image_loader.h"
#include <algorithm>
#include <iostream>
#include <string>


std::string to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
        [](unsigned char c) { return std::tolower(c); });
    return s;
}

std::vector<std::filesystem::path> find_image_files_recursively(const std::filesystem::path& dir, const std::string& image_format)
{
    std::vector<std::filesystem::path> image_files;

    if(!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        std::cerr << "Invalid directory: " << dir << "\n";
        return image_files;
    }

    for(const auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
        if(entry.is_regular_file()) {
            const auto ext = to_lower(entry.path().extension().string());
            if(ext == "." + image_format) {
                image_files.push_back(entry.path());
            }
        }
    }
    return image_files;
}

cv::Mat load_image(const std::filesystem::path& path)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "Error: Could not load image" << std::endl;
        return cv::Mat();
    }

    return img;
}
