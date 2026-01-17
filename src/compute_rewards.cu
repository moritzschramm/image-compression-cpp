#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "segment_stats.cuh"
#include "png_size_estimator_masked.cuh"

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); } } while(0)
#endif

// Convert float32 [B,3,H,W] -> uint8 [B,H,W,4] with alpha=255
__global__ void chw3_f32_to_hwc4_u8_kernel(
    const float* __restrict__ in_chw3,
    uint8_t* __restrict__ out_hwc4,
    int B, int H, int W)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t N = (int64_t)B * H * W;
    if (idx >= N) return;

    const int b = (int)(idx / (H * W));
    const int rem = (int)(idx - (int64_t)b * H * W);
    const int y = rem / W;
    const int x = rem - y * W;

    // input index for CHW: ((b*C + c)*H + y)*W + x
    const int64_t base_c = ((int64_t)b * 3) * H * W + (int64_t)y * W + x;
    const float v0 = in_chw3[base_c + 0LL * H * W];
    const float v1 = in_chw3[base_c + 1LL * H * W];
    const float v2 = in_chw3[base_c + 2LL * H * W];

    auto to_u8 = [](float v) __device__ -> uint8_t {
        // clamp + round to nearest int
        v *= 255.f;
        v = v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
        int iv = (int)lrintf(v);
        iv = iv < 0 ? 0 : (iv > 255 ? 255 : iv);
        return (uint8_t)iv;
    };

    const uint8_t c0 = to_u8(v0);
    const uint8_t c1 = to_u8(v1);
    const uint8_t c2 = to_u8(v2);

    // output HWC4: (((b*H + y)*W + x)*4 + c)
    const int64_t out = (((int64_t)b * H + y) * W + x) * 4;
    out_hwc4[out + 0] = c0;
    out_hwc4[out + 1] = c1;
    out_hwc4[out + 2] = c2;
    out_hwc4[out + 3] = 255;
}

torch::Tensor compute_rewards_batched(
    const torch::Tensor& images_bchw_f32,  // [B,3,H,W] float32 (from CV_32FC3 blob)
    const torch::Tensor& labels_bhw,       // [B,H,W] integer (any), CUDA
    const torch::Tensor& image_sizes_b,    // [B] int64/float64, CUDA
    int32_t min_pixels_per_segment,
    int L_min,
    float beta,
    float b_match_token,
    float gamma,
    double overhead_base,
    bool adaptive_filter,
    double lambda)
{
    torch::NoGradGuard ng;

    TORCH_CHECK(images_bchw_f32.dim() == 4 && images_bchw_f32.size(1) == 3, "images must be [B,3,H,W]");
    TORCH_CHECK(images_bchw_f32.scalar_type() == torch::kFloat32, "images must be float32 (CV_32FC3)");
    TORCH_CHECK(labels_bhw.is_cuda(), "labels must be CUDA");
    TORCH_CHECK(image_sizes_b.is_cuda(), "image_sizes must be CUDA");
    TORCH_CHECK(labels_bhw.dim() == 3, "labels must be [B,H,W]");
    TORCH_CHECK(image_sizes_b.numel() == images_bchw_f32.size(0), "image_sizes must have B elements");

    // Ensure everything on the same CUDA device
    const auto dev = labels_bhw.device();
    c10::cuda::CUDAGuard dg(dev);

    // Move images to same device if needed
    torch::Tensor images = images_bchw_f32;
    if (!images.is_cuda() || images.device() != dev) {
        images = images.to(dev, /*non_blocking=*/true);
    }
    images = images.contiguous();

    const int64_t B = images.size(0);
    const int64_t H = images.size(2);
    const int64_t W = images.size(3);

    TORCH_CHECK(labels_bhw.size(0) == B && labels_bhw.size(1) == H && labels_bhw.size(2) == W,
                "labels must match images: [B,H,W]");

    // labels -> int64 contiguous
    torch::Tensor labs = labels_bhw;
    if (labs.scalar_type() != torch::kInt64) labs = labs.to(torch::kInt64);
    labs = labs.contiguous();

    // image_sizes -> [B] float64 contiguous
    torch::Tensor sizes = image_sizes_b.view({-1});
    if (sizes.scalar_type() != torch::kFloat64) sizes = sizes.to(torch::kFloat64);
    sizes = sizes.contiguous();

    // Convert images to uint8 RGBA HWC on GPU once for the whole batch: [B,H,W,4]
    auto imgs_rgba = torch::empty({B, H, W, 4},
                                  torch::TensorOptions().device(dev).dtype(torch::kUInt8));

    const auto stream = c10::cuda::getCurrentCUDAStream().stream();
    const int threads = 256;
    const int64_t Npix = B * H * W;
    const int blocks = (int)((Npix + threads - 1) / threads);

    chw3_f32_to_hwc4_u8_kernel<<<blocks, threads, 0, stream>>>(
        images.data_ptr<float>(),
        imgs_rgba.data_ptr<uint8_t>(),
        (int)B, (int)H, (int)W);
    CUDA_CHECK(cudaGetLastError());

    // Output rewards [B] float64 on CUDA
    auto rewards = torch::empty({B}, torch::TensorOptions().device(dev).dtype(torch::kFloat64));
    auto opts_f64 = torch::TensorOptions().device(dev).dtype(torch::kFloat64);
    const double num_pixels = (double)(H * W);

    for (int64_t b = 0; b < B; ++b) {
        // Views (no copies)
        auto img_hwc4 = imgs_rgba[b]; // [H,W,4] uint8 contiguous view
        auto lab_hw   = labs[b];      // [H,W] int64 contiguous view

        // unique + inverse (compact ids 0..K-1)
        auto flat = lab_hw.reshape({-1});
        auto uniq_tup = at::_unique(flat, /*sorted=*/true, /*return_inverse=*/true);
        torch::Tensor inverse = std::get<1>(uniq_tup); // [H*W]
        const int64_t K = std::get<0>(uniq_tup).numel();

        if (K <= 0) {
            rewards[b].fill_(0.0);
            continue;
        }

        torch::Tensor lab_compact = inverse.view({H, W}).contiguous(); // [H,W] int64, 0..K-1

        // counts + bboxes on GPU
        auto counts = torch::empty({K}, torch::TensorOptions().device(dev).dtype(torch::kInt32));
        auto bboxes = torch::empty({K, 4}, torch::TensorOptions().device(dev).dtype(torch::kInt32));
        compute_counts_bboxes_from_compact_labels_cuda(lab_compact, counts, bboxes);

        // segment sizes on GPU
        auto seg_sizes = torch::zeros({K}, opts_f64);

        // Need bbox extents on host for variable-size estimator launches
        auto bboxes_cpu = bboxes.to(torch::kCPU);

        const uint8_t* img_dev = img_hwc4.data_ptr<uint8_t>();     // packed HWC4
        const int64_t* lab_dev = lab_compact.data_ptr<int64_t>();  // compact HW
        const int32_t* cnt_dev = counts.data_ptr<int32_t>();
        double* seg_sizes_dev  = seg_sizes.data_ptr<double>();

        for (int64_t k = 0; k < K; ++k) {
            const int32_t x0 = bboxes_cpu[k][0].item<int32_t>();
            const int32_t y0 = bboxes_cpu[k][1].item<int32_t>();
            const int32_t x1 = bboxes_cpu[k][2].item<int32_t>();
            const int32_t y1 = bboxes_cpu[k][3].item<int32_t>();
            if (x1 < x0 || y1 < y0) continue;

            const int w = (int)(x1 - x0 + 1);
            const int h = (int)(y1 - y0 + 1);

            // Writes seg_sizes[k] on GPU (float64) using current PyTorch stream
            estimate_png_size_masked_segment_to_output(
                img_dev, (int)W, (int)H, /*channels=*/4,
                lab_dev,
                cnt_dev,
                k,
                (int)x0, (int)y0, w, h,
                min_pixels_per_segment,
                L_min, beta, b_match_token, gamma, overhead_base,
                adaptive_filter,
                &seg_sizes_dev[k]);
        }

        // Reward on GPU
        auto valid = (counts >= min_pixels_per_segment);                 // [K] bool
        auto counts_f = counts.to(torch::kFloat64);
        auto p = (counts_f / num_pixels) * valid.to(torch::kFloat64);    // invalid -> 0
        auto sumsq = (p * p).sum();
        auto P = 1.0 - sumsq;

        auto size_segments = seg_sizes.sum();
        auto size_image_t  = sizes[b];                                   // CUDA scalar float64
        auto G = (size_image_t - size_segments) / size_image_t;

        auto R = G - lambda * P;                                         // CUDA scalar float64
        rewards[b].copy_(R);
    }

    return rewards;
}
