#include <iostream>
#include <string>
#include <torch/torch.h>
#include "image_loader.hpp"


// change this so that the given directory is relative to the directory you are in while executing the program
const std::string DATASET_DIR = "../dataset";
const std::string IMAGE_FORMAT = "png";

int main() {

    torch::Tensor t = torch::rand({3, 3});
    std::cout << "Tensor:\n" << t << "\n";

    auto t2 = torch::relu(t);
    std::cout << "ReLU:\n" << t2 << "\n";

    auto paths = find_image_files_recursively(DATASET_DIR, IMAGE_FORMAT);

    for(const auto& path : paths) {
        std::cout << path << std::endl;
        cv::Mat img = load_image(path);
        std::cout << "Size: " << img.size().width << "x" << img.size().height << " channels: " << img.channels() << std::endl;
    }

    return 0;
}
