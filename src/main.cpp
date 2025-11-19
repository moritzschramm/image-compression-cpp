#include <iostream>
#include <string>
#include <torch/torch.h>
#include "image_loader.h"
#include "image_slicer.h"


// change this so that the given directory is relative to the directory you are in while executing the program
const std::filesystem::path DATASET_DIR = "./dataset";
const std::filesystem::path RESULTS_DIR = "./results";
const std::string IMAGE_FORMAT = "png";

torch::Tensor make_quadrant_mask(int H, int W) {
    auto mask = torch::empty({H, W}, torch::kUInt8);

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int label = (y < H/2 ? 0 : 2) + (x < W/2 ? 0 : 1);
            mask[y][x] = label;
        }
    }
    return mask;
}

int main()
{
    auto paths = find_image_files_recursively(DATASET_DIR, IMAGE_FORMAT);

    for(const auto& path : paths)
    {
        std::cout << path << std::endl;

        cv::Mat input = load_image(path);

        torch::Tensor mask = make_quadrant_mask(input.rows, input.cols);

        write_slices(input, mask, RESULTS_DIR, path.stem(), IMAGE_FORMAT);
    }

    return 0;
}
