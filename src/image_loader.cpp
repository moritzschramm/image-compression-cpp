#include <algorithm>
#include <cstdio>
#include <iostream>
#include <string>
#include <png.h>
#include "image_loader.hpp"

namespace fs = std::filesystem;

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return s;
}

std::vector<fs::path> find_image_files(const fs::path& dir, const std::string& image_format) {
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

Image load_png(const fs::path& path) {
    Image img;

    FILE* fp = std::fopen(path.string().c_str(), "rb");
    if(!fp) {
        std::cerr << "Failed to open file: " << path << "\n";
        return img;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if(!png) {
        fclose(fp);
        return img;
    }

    png_infop info = png_create_info_struct(png);
    if(!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        fclose(fp);
        return img;
    }

    if(setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        fclose(fp);
        return img;
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    int width  = png_get_image_width(png, info);
    int height = png_get_image_height(png, info);
    int color  = png_get_color_type(png, info);
    int depth  = png_get_bit_depth(png, info);

    // Convert everything to 8-bit RGBA
    if(depth == 16)
        png_set_strip_16(png);

    if(color == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);

    if(color == PNG_COLOR_TYPE_GRAY && depth < 8)
        png_set_expand_gray_1_2_4_to_8(png);

    if(png_get_valid(png, info, PNG_INFO_tRNS))
        png_set_tRNS_to_alpha(png);

    if(color == PNG_COLOR_TYPE_RGB || color == PNG_COLOR_TYPE_GRAY)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

    if(color & PNG_COLOR_MASK_ALPHA)
        ; // already has alpha

    png_read_update_info(png, info);

    img.width = width;
    img.height = height;
    img.pixels.resize(width * height * 4);

    std::vector<png_bytep> rows(height);
    for(int y = 0; y < height; y++)
        rows[y] = img.pixels.data() + y * width * 4;

    png_read_image(png, rows.data());

    png_destroy_read_struct(&png, &info, nullptr);
    fclose(fp);
    return img;
}
