#pragma once
#include <segment_extract.hpp>
#include <torch/torch.h>

/*
 * Compute rewards based on segmentation and image
 * - image: [4,H,W]
 * - node_labels: Labelled pixels given by RAMA
 * returns scalar tensor
 */
inline double compute_rewards(const torch::Tensor& image, const torch::Tensor& node_labels, const int size_image)
{
    auto segments = extract_segments_bgra_cuda(image, node_labels);

    double size_segments = 0.0;
    double sumsq = 0.0;
    int num_pixels = image.size(1) * image.size(2);

    for (auto segment : segments)
    {
        const uint8_t* img_dev = segment.rgba_tensor.data_ptr<uint8_t>();
        int H = (int)segment.rgba_tensor.size(0);
        int W = (int)segment.rgba_tensor.size(1);
        int C = 4;

        size_segments += estimate_png_size_from_device_image(img_dev, W, H, C,
            4, 0.012167f, 18.000000f, 0.100000f, 9.308622, true);

        double p = static_cast<double>(H * W) / num_pixels;
        sumsq += p * p;
    }

    const double lambda = 0.1;    

    const int k_min = 2;
    const int k = segments.size();

    double G = (size_image - size_segments) / size_image;

    double P = 1.0 - sumsq;
    double R = G - lambda * P;

    std::cout << "G: " << G << " P: " << P << " R: " << R << " k: " << k << " s_base: " << size_image << " s_seg: " << size_segments << std::endl;

    return R;
}
