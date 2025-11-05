#pragma once
#include <cstdint>
#include <vector>
#include <filesystem>

struct Image {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels; // RGBA, 4 bytes/pixel
};

std::vector<std::filesystem::path> find_image_files(const std::filesystem::path&, const std::string&);
Image load_png(const std::filesystem::path& path);
