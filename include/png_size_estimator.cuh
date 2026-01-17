#pragma once
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); } } while(0)
#endif

struct PngEstimatorWorkspace {
    // device buffers (persistent)
    at::Tensor residuals_u8;   // [cap_N]
    at::Tensor hist_u32;       // [channels*256]
    at::Tensor costs_u32;      // [cap_h*5]
    at::Tensor filter_u8;      // [cap_h]
    at::Tensor match_symbols_u64; // [1]
    at::Tensor match_count_u64;   // [1]
    at::Tensor match_len_sum_u64; // [1]
    at::Tensor Hc_f64;         // [channels]
    at::Tensor Hbar_f64;       // [1]

    int64_t cap_N = 0;
    int     cap_h = 0;
    int     cap_channels = 0;

    void ensure(int64_t N, int h, int channels, c10::Device dev) {
        auto dopt_u8  = at::TensorOptions().device(dev).dtype(at::kByte);
        auto dopt_u32 = at::TensorOptions().device(dev).dtype(at::kInt);
        auto dopt_u64 = at::TensorOptions().device(dev).dtype(at::kLong);   // uint64 via reinterpret ok if you prefer kUInt64
        auto dopt_f64 = at::TensorOptions().device(dev).dtype(at::kDouble);

        if (channels != cap_channels) {
            hist_u32 = at::empty({channels * 256}, dopt_u32);
            Hc_f64   = at::empty({channels}, dopt_f64);
            cap_channels = channels;
        }
        if (N > cap_N) {
            residuals_u8 = at::empty({N}, dopt_u8);
            cap_N = N;
        }
        if (h > cap_h) {
            costs_u32  = at::empty({(int64_t)h * 5}, dopt_u32);
            filter_u8  = at::empty({h}, dopt_u8);
            cap_h = h;
        }
        if (!match_symbols_u64.defined()) {
            match_symbols_u64 = at::empty({1}, dopt_u64);
            match_count_u64   = at::empty({1}, dopt_u64);
            match_len_sum_u64 = at::empty({1}, dopt_u64);
            Hbar_f64          = at::empty({1}, dopt_f64);
        }
    }
};

// Thread-local workspace per device (simple, fast)
PngEstimatorWorkspace& get_png_ws(int device_index);

// Writes one size estimate into out_dev[0] (device pointer) on the *current PyTorch CUDA stream*.
void estimate_png_size_masked_segment_to_output(
    const uint8_t* img_hwc_u8, int full_W, int full_H, int channels,
    const int64_t* labels_compact_hw, // [full_H*full_W], values 0..K-1
    const int32_t* counts_k,          // [K]
    int64_t seg_id_k,                 // compact id k
    int x0, int y0, int w, int h,
    int32_t min_pixels,
    int L_min, float beta, float b_match_token, float gamma, double overhead_base,
    bool adaptive_filter,
    double* out_dev // device pointer to one double (or sizes[k])
);
