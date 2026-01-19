#pragma once
#include <torch/torch.h>
//#include <torch/extension.h>

// bboxes: int32 [K,4] = (x0,y0,x1,y1), counts: int32 [K]
void compute_counts_bboxes_from_compact_labels_cuda(
    const torch::Tensor& labels_compact_hw_i64, // [H,W] int64, values 0..K-1
    torch::Tensor& counts_k_i32,                // [K] int32
    torch::Tensor& bboxes_k4_i32                // [K,4] int32
);
