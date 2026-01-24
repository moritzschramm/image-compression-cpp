#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/segmentation.hpp>

// Output: edges [2, H, W] (float32, CPU)
//   edges[0, y, x] = horizontal edge between (y,x) and (y,x+1) for x in [0, W-2]
//   edges[1, y, x] = vertical   edge between (y,x) and (y+1,x) for y in [0, H-2]
// Values: 1.0 for connect (same segment), 0.0 for cut (segment boundary)
torch::Tensor graph_based_edge_costs(
    const cv::Mat& input,
    float sigma = 1.0f,
    float k = 100.0f,
    int min_size = 250
);
