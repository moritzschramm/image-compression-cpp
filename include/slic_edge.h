#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

// Output: edges [2, H, W] (float32, CPU)
//   edges[0, y, x] = horizontal edge between (y,x) and (y,x+1) for x in [0, W-2]
//   edges[1, y, x] = vertical   edge between (y,x) and (y+1,x) for y in [0, H-2]
// Values: 1.0 for connect (same superpixel), 0.0 for cut (superpixel boundary)
torch::Tensor slic_edge_costs(
    const cv::Mat& input,
    int region_size = 20,
    float ruler = 0.0f,
    int iters = 10,
    int slic_algorithm = cv::ximgproc::SLIC
);
