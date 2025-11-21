#include <iostream>
#include <torch/torch.h>
#include "configuration.h"
#include "image_loader.h"
#include "image_slicer.h"

torch::Tensor make_triangle_mask(int H, int W)
{
    auto mask = torch::zeros({H, W}, torch::kInt64);

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {

            bool top    = y < H / 2;
            bool left   = x < W / 2;
            bool diag1  = x < y;                 // below main diagonal
            bool diag2  = x < (H - y - 1);       // below anti-diagonal

            int label;

            if (top && left) {
                // Top-left triangle
                label = diag1 ? 0 : 1;
            }
            else if (top && !left) {
                // Top-right triangle
                label = diag2 ? 1 : 0;
            }
            else if (!top && left) {
                // Bottom-left triangle
                label = diag2 ? 2 : 3;
            }
            else {
                // Bottom-right triangle
                label = diag1 ? 3 : 2;
            }

            mask[y][x] = label;
        }
    }

    return mask.to(torch::kUInt8);   // compact format for slicing
}

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

    std::cout << "Found " << std::to_string(paths.size()) << " images" << std::endl;

    for(const auto& path : paths)
    {
        std::cout << path << std::endl;

        cv::Mat input = load_image(path);

        torch::Tensor mask = make_triangle_mask(input.rows, input.cols);

        write_slices(input, mask, RESULTS_DIR, path.stem(), IMAGE_FORMAT);
        break;
    }

    return 0;
}
