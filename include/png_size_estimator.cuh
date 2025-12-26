#pragma once
#include <cstdint>
#include <opencv2/core/cuda.hpp>

#define CUDA_CHECK(expr)                                     \
    do {                                                     \
        cudaError_t err = (expr);                            \
        if (err != cudaSuccess) {                            \
            std::fprintf(stderr,                             \
                         "CUDA error %s at %s:%d\n",         \
                         cudaGetErrorString(err),            \
                         __FILE__, __LINE__);                \
            std::exit(EXIT_FAILURE);                         \
        }                                                    \
    } while (0)


// estimates PNG size (bytes) from an interleaved uint8 image already on the GPU.
// image layout: packed, interleaved: [y][x][c], size = width*height*channels bytes.
double estimate_png_size_from_device_image(
    const uint8_t* img_dev,
    int width,
    int height,
    int channels,
    int L_min = 4,
    float beta = 0.3f,
    float b_match_token = 18.0f,
    float gamma = 0.1f,
    double overhead_base = 300.0,
    bool adaptive_filter = true);

// estimator, input is a pitched image buffer on GPU (e.g. cv::cuda::GpuMat)
// gpu_mat_data: pointer to first byte of row 0
// step_bytes: bytes per row (pitch)
// width/height in pixels, channels in bytes per pixel channel count
double estimate_png_size_from_pitched_device_image(
    const void* gpu_mat_data,
    int step_bytes,
    int width,
    int height,
    int channels,
    int L_min = 4,
    float beta = 0.3f,
    float b_match_token = 18.0f,
    float gamma = 0.1f,
    double overhead_base = 300.0,
    bool adaptive_filter = true);

// estimator using CUDA GpuMat directly
double estimate_png_size_from_GpuMat(
    const cv::cuda::GpuMat& gpu,
    int L_min = 4,
    float beta = 0.3f,
    float b_match_token = 18.0f,
    float gamma = 0.1f,
    double overhead_base = 300.0,
    bool adaptive_filter = true);
