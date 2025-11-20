#include <iostream>
#include <torch/torch.h>
#include "configuration.h"
#include "image_loader.h"
#include "image_slicer.h"



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
        break;
    }

    return 0;
}
