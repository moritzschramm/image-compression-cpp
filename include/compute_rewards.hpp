#pragma once
#include <segment_extract.hpp>
#include <torch/torch.h>

/*
 * Compute rewards based on segmentation and image
 * - image: [4,H,W]
 * - node_labels: Labelled pixels given by RAMA
 * returns scalar tensor
 */
inline torch::Tensor compute_rewards(const torch::Tensor& image, const torch::Tensor& node_labels)
{
    auto segments = extract_segments_bgra_cuda(image, node_labels);

    // TODO estimate size of whole image
    // TODO estimate size of each segment
    // TODO compute ratio between sizes as reward signal

    torch::Tensor t;

    return t;
}
