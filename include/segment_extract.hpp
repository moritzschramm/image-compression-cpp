#pragma once

#include <torch/torch.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <vector>
#include <cstdint>

struct GpuSegmentRGBA {
    int64_t label = -1;
    cv::Rect bbox_xywh;          // bbox in original image coords (x,y,w,h)
    torch::Tensor rgba_tensor;   // CUDA, uint8, [h, w, 4], contiguous
    cv::cuda::GpuMat rgba_mat;   // CV_8UC4 view into rgba_tensor (no copy)
};

inline torch::Tensor ensure_hwc_u8_cuda(const torch::Tensor& img_in, bool assume_input_rgb = true, bool output_bgra = true) {
    TORCH_CHECK(img_in.defined(), "image tensor is undefined");
    TORCH_CHECK(img_in.is_cuda(), "image must be a CUDA tensor");
    TORCH_CHECK(img_in.dim() == 2 || img_in.dim() == 3, "image must be [H,W] or [H,W,C] or [C,H,W]");

    torch::Tensor img = img_in;

    // If CHW convert to HWC
    if (img.dim() == 3) {
        const auto s0 = img.size(0), s1 = img.size(1), s2 = img.size(2);
        // heuristic: if first dim is small (1/3/4) and last two are larger, treat as CHW
        if ((s0 == 1 || s0 == 3 || s0 == 4) && s1 > 4 && s2 > 4) {
            img = img.permute({1, 2, 0});
        }
    }

    // ensure 3D HWC
    if (img.dim() == 2) {
        img = img.unsqueeze(-1);
    }

    TORCH_CHECK(img.dim() == 3, "internal: expected HWC after normalization");
    TORCH_CHECK(img.size(2) == 1 || img.size(2) == 3 || img.size(2) == 4,
                "image channel count must be 1, 3, or 4 after normalization");

    // convert dtype to uint8
    if (img.scalar_type() != torch::kUInt8) {
        // assume float in [0,1] or [0,255], clamp to [0,255]
        img = img.to(torch::kFloat32);
        img = img.clamp(0.0f, 255.0f);
        // if it looks like [0,1], scale
        // cheap heuristic: max <= 1.0 => scale by 255
        const auto mx = img.max().item<float>();
        if (mx <= 1.0f) img = img * 255.0f;
        img = img.to(torch::kUInt8);
    }

    // if 4 channels, drop alpha
    /*if (img.size(2) == 4) {
        img = img.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
    }*/

    // If 1 channel, expand to 3.
    if (img.size(2) == 1) {
        img = img.repeat({1, 1, 3});
    }

    // Optional RGB->BGR swap for OpenCV conventions.
    if (output_bgra) {
        // output is BGRA later; so make base 3ch BGR now.
        // If input is already BGR, set assume_input_rgb=false.
        if (assume_input_rgb) {
            auto idx = torch::tensor({2, 1, 0}, torch::TensorOptions().dtype(torch::kLong).device(img.device()));
            img = img.index({torch::indexing::Slice(), torch::indexing::Slice(), idx});
        }
    }

    return img.contiguous();
}

inline torch::Tensor ensure_labels_hw_i64_cuda(const torch::Tensor& labels_in, int64_t H, int64_t W) {
    TORCH_CHECK(labels_in.defined(), "labels tensor is undefined");
    TORCH_CHECK(labels_in.is_cuda(), "labels must be a CUDA tensor");
    TORCH_CHECK(labels_in.numel() == H * W,
                "labels must have H*W elements (node labels per pixel). Got numel=",
                labels_in.numel(), " expected=", H * W);

    torch::Tensor lab = labels_in;
    if (lab.scalar_type() != torch::kInt64) lab = lab.to(torch::kInt64);
    lab = lab.view({H, W}).contiguous();
    return lab;
}

/*
 * Extract connected-component segments from a node label map
 * - image: CUDA tensor [H,W,C] (or [C,H,W]) with C=1/3/4; uint8 or float
 * - node_labels: CUDA tensor with H*W elements (or [H,W]) of component ids (int)
 * returns one BGRA cv::cuda::GpuMat per label, cropped to bbox, with alpha=0 outside the segment.
 */
inline std::vector<GpuSegmentRGBA> extract_segments_bgra_cuda(
    const torch::Tensor& image,
    const torch::Tensor& node_labels,
    bool assume_input_rgb = true,
    int64_t min_pixels_per_segment = 1,
    cv::cuda::Stream stream = cv::cuda::Stream::Null() // pass a stream for explicit stream control
) {
    (void)stream;

    torch::NoGradGuard ng;

    torch::Tensor img3 = ensure_hwc_u8_cuda(image, assume_input_rgb, /*output_bgra=*/true);
    const int64_t H = img3.size(0);
    const int64_t W = img3.size(1);

    torch::Tensor labels_hw;
    if (node_labels.dim() == 2) {
        TORCH_CHECK(node_labels.size(0) == H && node_labels.size(1) == W,
                    "labels [H,W] must match image H,W");
        labels_hw = node_labels;
        if (labels_hw.scalar_type() != torch::kInt64) labels_hw = labels_hw.to(torch::kInt64);
        labels_hw = labels_hw.contiguous();
    } else {
        labels_hw = ensure_labels_hw_i64_cuda(node_labels, H, W);
    }

    // unique labels (CPU list for looping). This is small compared to image tensors
    torch::Tensor unique_labels_cpu = std::get<0>(labels_hw.flatten().sort()).to(torch::kCPU);

    std::vector<GpuSegmentRGBA> out;
    out.reserve(static_cast<size_t>(unique_labels_cpu.numel()));

    for (int64_t i = 0; i < unique_labels_cpu.numel(); ++i) {
        const int64_t lbl = unique_labels_cpu[i].item<int64_t>();

        // mask: [H,W] bool on GPU
        torch::Tensor mask = (labels_hw == lbl);

        // skip tiny/empty segments
        const int64_t pix = mask.sum().item<int64_t>();
        if (pix < min_pixels_per_segment) continue;

        // find bbox via nonzero coords on GPU then reduce
        torch::Tensor coords = mask.nonzero(); // [K,2] (y,x)
        TORCH_CHECK(coords.numel() > 0, "internal: nonzero returned empty for a non-empty mask");

        torch::Tensor ys = coords.select(1, 0);
        torch::Tensor xs = coords.select(1, 1);

        const int64_t y0 = ys.min().item<int64_t>();
        const int64_t y1 = ys.max().item<int64_t>();
        const int64_t x0 = xs.min().item<int64_t>();
        const int64_t x1 = xs.max().item<int64_t>();

        const int64_t h = y1 - y0 + 1;
        const int64_t w = x1 - x0 + 1;

        // crop image and mask to bbox
        auto ysl = torch::indexing::Slice(y0, y1 + 1);
        auto xsl = torch::indexing::Slice(x0, x1 + 1);

        torch::Tensor img_crop = img3.index({ysl, xsl, torch::indexing::Slice()}).contiguous();       // [h,w,3] u8
        torch::Tensor m_crop   = mask.index({ysl, xsl}).to(torch::kUInt8).contiguous();               // [h,w] u8 {0,1}

        // zero RGB outside mask, set alpha = 255 inside, 0 outside
        torch::Tensor rgb = img_crop * m_crop.unsqueeze(-1);          // [h,w,3] u8
        torch::Tensor a   = (m_crop * 255).unsqueeze(-1);             // [h,w,1] u8
        torch::Tensor rgba = torch::cat({rgb, a}, /*dim=*/2).contiguous(); // [h,w,4] u8 (BGRA)

        // wrap tensor memory as cv::cuda::GpuMat (no copy)
        // IMPORTANT: rgba_mat is only valid as long as rgba_tensor is alive
        const size_t step_bytes = static_cast<size_t>(rgba.stride(0)) * rgba.element_size();
        auto* ptr = reinterpret_cast<void*>(rgba.data_ptr<uint8_t>());
        cv::cuda::GpuMat mat(static_cast<int>(h), static_cast<int>(w), CV_8UC4, ptr, step_bytes);

        GpuSegmentRGBA seg;
        seg.label = lbl;
        seg.bbox_xywh = cv::Rect(static_cast<int>(x0), static_cast<int>(y0), static_cast<int>(w), static_cast<int>(h));
        seg.rgba_tensor = rgba;
        seg.rgba_mat = mat;

        out.emplace_back(std::move(seg));
    }

    return out;
}
