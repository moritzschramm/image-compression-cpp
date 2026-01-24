#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

// Output: edges [2, H, W] (float32, CPU)
//   edges[0, y, x] = horizontal edge between (y,x) and (y,x+1) for x in [0, W-2]
//   edges[1, y, x] = vertical   edge between (y,x) and (y+1,x) for y in [0, H-2]
// Values: 1.0 for connect (same segment), 0.0 for cut (segment boundary)
torch::Tensor watershed_edge_costs(
    const cv::Mat& input,
    int seed_stride = 16,
    int blur_ksize = 3,
    double blur_sigma = 1.0
);
