#include <iostream>
#include <string>
#include "huffman.hpp"
#include "image_loader.hpp"


// change this so that the given directory is relative to the directory you are in while executing the program
const std::string DATASET_DIR = "../dataset";
const std::string IMAGE_FORMAT = "png";

int main() {

    auto paths = find_image_files(DATASET_DIR, IMAGE_FORMAT);

    for(const auto& path : paths) {
        Image img = load_png(path);
        std::cout << "Size: " << img.width << "x" << img.height << " Pixel count: " << img.pixels.size() << std::endl;

        /* access pixels like this:
        int x = 100, y = 100;
        if(x < img.width && y < img.height) {
            int i = (y * img.width + x) * 4;
            unsigned char r = img.pixels[i + 0];
            unsigned char g = img.pixels[i + 1];
            unsigned char b = img.pixels[i + 2];
            unsigned char a = img.pixels[i + 3];
            std::cout << "RGBA: " << (int)r << " " << (int)g << " " << (int)b << " " << (int)a << "\n";
        }
        */
    }

    std::cout << "Enter text (binary-safe input, but reads until newline):\n";
    std::string input;
    getline(std::cin, input);

    std::vector<uint8_t> data(input.begin(), input.end());

    std::string encoded;
    auto codes = encodeHuffman(data, encoded);

    std::cout << "Encoded: " << encoded << "\n";

    return 0;
}
