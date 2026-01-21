#pragma once
#include <torch/torch.h>
//#include <torch/extension.h>

torch::Tensor compute_rewards_batched(
    const torch::Tensor& images_bchw_f32,
    const torch::Tensor& labels,
    const torch::Tensor& image_sizes,
    int32_t min_pixels_per_segment = 1,
    int L_min = 4,
    float beta = 0.012167f,
    float b_match_token = 18.0f,
    float gamma = 0.1f,
    double overhead_base = 9.308622,
    bool adaptive_filter = true,
    double lambda = 0.5
);
