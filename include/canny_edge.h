#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

// Output: edges [2, H, W] (float32, CPU)
//   edges[0, y, x] = horizontal edge between (y,x) and (y,x+1) for x in [0, W-2]
//   edges[1, y, x] = vertical   edge between (y,x) and (y+1,x) for y in [0, H-2]
// Values: 1.0 for connect (no edge), 0.0 for cut (edge present)
torch::Tensor canny_edge_costs(
    const cv::Mat& img,
    double canny_low = 50.0,
    double canny_high = 150.0,
    int aperture_size = 3,
    bool L2gradient = true,
    int blur_ksize = 3,
    double blur_sigma = 1.0
);
