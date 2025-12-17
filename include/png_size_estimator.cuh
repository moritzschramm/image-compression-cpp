#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

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

uint8_t* move_img_to_gpu(const cv::Mat& image);
void free_img_from_gpu(uint8_t* img_dev);

double estimate_png_size_from_device_image(
    const uint8_t* img_dev,
    int width,
    int height,
    int channels,
    int L_min = 4,
    float beta = 0.3f,
    float b_match_token = 18.0f,
    float gamma = 0.1f,
    double overhead_base = 300.0);

double estimate_png_size_from_GpuMat(
    const void* gpu_mat_data,
    int step_bytes,
    int width,
    int height,
    int channels);
