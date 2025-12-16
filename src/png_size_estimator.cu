#include "png_size_estimator.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>

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

uint8_t* move_img_to_gpu(const cv::Mat& image)
{
    auto img = image;
    int width    = img.cols;
    int height   = img.rows;
    int channels = img.channels();

    size_t N = static_cast<size_t>(width) * height * channels;

    if (!img.isContinuous())
        img = img.clone();

    const uint8_t* img_host_ptr = img.data;

    uint8_t* img_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&img_dev, N));
    CUDA_CHECK(cudaMemcpy(img_dev, img_host_ptr, N, cudaMemcpyHostToDevice));

    return img_dev;
}

void free_img_from_gpu(uint8_t* img_dev)
{
    CUDA_CHECK(cudaFree(img_dev));
}

// simple paeth predictor as used by PNG
__device__ inline int paeth_predictor(int a, int b, int c)
{
    int p  = a + b - c;
    int pa = abs(p - a);
    int pb = abs(p - b);
    int pc = abs(p - c);

    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

// kernel: compute Paeth residuals and per-channel histograms.
// image is interleaved uint8_t: [y][x][c] with c in [0, channels)
__global__ void compute_residuals_and_hist_kernel(
    const uint8_t* __restrict__ img,
    uint8_t* __restrict__ residuals,
    unsigned int* __restrict__ hist,  // size = channels * 256
    int width, int height, int channels)
{
    extern __shared__ unsigned int sh_hist[];

    const int bins_per_channel = 256;
    const int total_bins = bins_per_channel * channels;

    for (int i = threadIdx.x; i < total_bins; i += blockDim.x) {
        sh_hist[i] = 0;
    }
    __syncthreads();

    const int N = width * height * channels;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < N) {
        int pixel_idx = idx / channels;
        int c         = idx % channels;
        int x         = pixel_idx % width;
        int y         = pixel_idx / width;

        int cur = img[idx];

        // neighbor values for Paeth. Use 0 outside image
        int left      = 0;
        int up        = 0;
        int up_left   = 0;

        if (x > 0) {
            int left_idx = idx - channels; // previous pixel, same channel
            left = img[left_idx];
        }
        if (y > 0) {
            int up_pixel_idx = (y - 1) * width + x;
            int up_idx       = up_pixel_idx * channels + c;
            up = img[up_idx];
        }
        if (x > 0 && y > 0) {
            int ul_pixel_idx = (y - 1) * width + (x - 1);
            int ul_idx       = ul_pixel_idx * channels + c;
            up_left = img[ul_idx];
        }

        int pred = paeth_predictor(left, up, up_left);
        int r    = cur - pred;              // signed residual
        uint8_t r8 = static_cast<uint8_t>(r & 0xFF); // wrap to [0,255]

        residuals[idx] = r8;

        int bin_idx = c * bins_per_channel + static_cast<int>(r8);
        atomicAdd(&sh_hist[bin_idx], 1U);

        idx += blockDim.x * gridDim.x;
    }

    __syncthreads();

    // accumulate shared hist into global hist
    for (int i = threadIdx.x; i < total_bins; i += blockDim.x) {
        if (sh_hist[i] > 0) {
            atomicAdd(&hist[i], sh_hist[i]);
        }
    }
}

// kernel: approximate match statistics via run-lengths in the residual stream.
//
// treat residuals[] as a 1D sequence (length N). Each thread processes
// a contiguous chunk independently. Runs that cross chunk boundaries are broken
__global__ void run_length_stats_kernel(
    const uint8_t* __restrict__ residuals,
    size_t N,
    int L_min,
    unsigned long long* match_symbols_out,
    unsigned long long* match_count_out,
    unsigned long long* match_length_sum_out)
{
    // Global accumulators
    __shared__ unsigned long long sh_match_symbols;
    __shared__ unsigned long long sh_match_count;
    __shared__ unsigned long long sh_match_length_sum;

    if (threadIdx.x == 0) {
        sh_match_symbols     = 0;
        sh_match_count       = 0;
        sh_match_length_sum  = 0;
    }
    __syncthreads();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x * gridDim.x;

    size_t chunk_size = (N + num_threads - 1) / num_threads; // ceil
    size_t start = static_cast<size_t>(tid) * chunk_size;
    if (start >= N) {
        return;
    }
    size_t end   = start + chunk_size;
    if (end > N) end = N;

    if (end - start == 0) {
        return;
    }

    uint8_t cur = residuals[start];
    int run_len = 1;

    unsigned long long local_match_symbols    = 0;
    unsigned long long local_match_count      = 0;
    unsigned long long local_match_length_sum = 0;

    for (size_t i = start + 1; i < end; ++i) {
        uint8_t v = residuals[i];
        if (v == cur) {
            ++run_len;
        } else {
            if (run_len >= L_min) {
                local_match_symbols    += run_len;
                local_match_count      += 1;
                local_match_length_sum += run_len;
            }
            cur = v;
            run_len = 1;
        }
    }

    // last run in this chunk
    if (run_len >= L_min) {
        local_match_symbols    += run_len;
        local_match_count      += 1;
        local_match_length_sum += run_len;
    }

    atomicAdd(&sh_match_symbols,     local_match_symbols);
    atomicAdd(&sh_match_count,       local_match_count);
    atomicAdd(&sh_match_length_sum,  local_match_length_sum);

    __syncthreads();

    if (threadIdx.x == 0) {
        if (sh_match_symbols > 0) {
            atomicAdd(match_symbols_out,    sh_match_symbols);
        }
        if (sh_match_count > 0) {
            atomicAdd(match_count_out,      sh_match_count);
        }
        if (sh_match_length_sum > 0) {
            atomicAdd(match_length_sum_out, sh_match_length_sum);
        }
    }
}

// host helper: estimate PNG size (bytes) from device image pointer.
// img_dev:  interleaved uint8, size = width * height * channels
// channels: typically 1, 3, or 4
double estimate_png_size_from_device_image(
    const uint8_t* img_dev,
    int width,
    int height,
    int channels,
    int L_min = 4,
    float beta = 0.3f,
    float b_match_token = 18.0f,
    float gamma = 0.1f,
    double overhead_base = 300.0)
{
    if (width <= 0 || height <= 0 || channels <= 0) {
        throw std::runtime_error("Invalid image dimensions/channels");
    }

    const size_t N = static_cast<size_t>(width) * height * channels;

    uint8_t* residuals_dev = nullptr;
    unsigned int* hist_dev = nullptr;

    CUDA_CHECK(cudaMalloc(&residuals_dev, N * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&hist_dev, channels * 256 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(hist_dev, 0, channels * 256 * sizeof(unsigned int)));

    // launch residual + histogram kernel
    const int block_size = 256;
    const int grid_size  = (static_cast<int>(N) + block_size - 1) / block_size;
    const int max_grid   = 65535;
    const int launch_grid = grid_size > max_grid ? max_grid : grid_size;

    size_t shmem_size = channels * 256 * sizeof(unsigned int);
    compute_residuals_and_hist_kernel<<<launch_grid, block_size, shmem_size>>>(
        img_dev, residuals_dev, hist_dev, width, height, channels);
    CUDA_CHECK(cudaGetLastError());

    // copy histogram to host
    std::vector<unsigned int> hist_host(channels * 256);
    CUDA_CHECK(cudaMemcpy(hist_host.data(),
                          hist_dev,
                          hist_host.size() * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    // compute per-channel entropy on host
    std::vector<double> H_c(channels, 0.0);

    for (int c = 0; c < channels; ++c) {
        const unsigned int* h = &hist_host[c * 256];
        unsigned long long count_c = 0;
        for (int v = 0; v < 256; ++v) {
            count_c += h[v];
        }
        if (count_c == 0) {
            H_c[c] = 0.0;
            continue;
        }
        double inv_total = 1.0 / static_cast<double>(count_c);
        double H = 0.0;
        for (int v = 0; v < 256; ++v) {
            if (h[v] == 0) continue;
            double p = h[v] * inv_total;
            H -= p * std::log2(p);
        }
        H_c[c] = H; // bits per symbol in this channel
    }

    double H_bar = 0.0;
    for (int c = 0; c < channels; ++c) {
        H_bar += H_c[c];
    }
    H_bar /= static_cast<double>(channels); // mean over channels

    // run-length stats kernel for match approximation
    unsigned long long *d_match_symbols, *d_match_count, *d_match_length_sum;
    CUDA_CHECK(cudaMalloc(&d_match_symbols,     sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_match_count,       sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_match_length_sum,  sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemset(d_match_symbols,     0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_match_count,       0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_match_length_sum,  0, sizeof(unsigned long long)));

    const int rl_block_size = 256;
    const int rl_grid_size  = 256; // heuristic; can tune
    run_length_stats_kernel<<<rl_grid_size, rl_block_size>>>(
        residuals_dev, N, L_min,
        d_match_symbols, d_match_count, d_match_length_sum);
    CUDA_CHECK(cudaGetLastError());

    unsigned long long h_match_symbols    = 0;
    unsigned long long h_match_count      = 0;
    unsigned long long h_match_length_sum = 0;

    CUDA_CHECK(cudaMemcpy(&h_match_symbols,
                          d_match_symbols,
                          sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_match_count,
                          d_match_count,
                          sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_match_length_sum,
                          d_match_length_sum,
                          sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));

    // free temporary device buffers
    CUDA_CHECK(cudaFree(residuals_dev));
    CUDA_CHECK(cudaFree(hist_dev));
    CUDA_CHECK(cudaFree(d_match_symbols));
    CUDA_CHECK(cudaFree(d_match_count));
    CUDA_CHECK(cudaFree(d_match_length_sum));

    // compute f_match and L_bar
    double f_match = 0.0;
    double L_bar   = static_cast<double>(L_min);

    if (N > 0 && h_match_symbols > 0) {
        f_match = static_cast<double>(h_match_symbols) /
                  static_cast<double>(N);
    }
    if (h_match_count > 0) {
        L_bar = static_cast<double>(h_match_length_sum) /
                static_cast<double>(h_match_count);
    }

    // bit-cost model
    double b_lit   = static_cast<double>(H_bar) + static_cast<double>(beta);
    double b_match = (static_cast<double>(b_match_token) / L_bar) +
                     static_cast<double>(gamma);

    double b_data = (1.0 - f_match) * b_lit + f_match * b_match; // bits / symbol

    // overhead in bytes: PNG/zlib headers + one filter byte per scanline
    double S_overhead = overhead_base + static_cast<double>(height);

    // total estimated size (bytes)
    double S_est = S_overhead +
                   (static_cast<double>(N) * b_data) / 8.0;

    return S_est;
}

// example host-side usage stub (expects img_dev already filled).
void run_example()
{
    // dummy 512x512 RGBA filled with zeros
    int width = 512;
    int height = 512;
    int channels = 4;
    size_t N = static_cast<size_t>(width) * height * channels;

    uint8_t* img_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&img_dev, N));
    CUDA_CHECK(cudaMemset(img_dev, 0, N)); // black image

    double est_size = estimate_png_size_from_device_image(img_dev, width, height, channels);

    std::cout << "Estimated PNG size: " << est_size << " bytes" << std::endl;

    CUDA_CHECK(cudaFree(img_dev));
    return 0;
}

// copy pitched to packed device buffer
__global__ void copy_pitched_to_packed(
    const uint8_t* src,
    int src_step,
    uint8_t* dst,
    int width,
    int height,
    int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // pixel index
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (x >= width || y >= height) return;

    // src is pitched: row start is src + y*src_step
    const uint8_t* src_row = src + static_cast<size_t>(y) * src_step;
    uint8_t* dst_row = dst + (static_cast<size_t>(y) * width * channels);

    // copy pixel (all channels)
    for (int c = 0; c < channels; ++c)
        dst_row[x * channels + c] = src_row[x * channels + c];
}

// overload: accept a GpuMats pitched memory
double estimate_png_size_from_GpuMat(
    const void* gpu_mat_data,
    int step_bytes,
    int width,
    int height,
    int channels)
{
    size_t packed_size = static_cast<size_t>(width) * height * channels;

    uint8_t* packed_dev = nullptr;
    cudaMalloc(&packed_dev, packed_size);

    dim3 block(32, 8);
    dim3 grid(
        (width  + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    copy_pitched_to_packed<<<grid, block>>>(
        static_cast<const uint8_t*>(gpu_mat_data),
        step_bytes,
        packed_dev,
        width,
        height,
        channels
    );
    cudaDeviceSynchronize();

    double result = estimate_png_size_from_device_image(
        packed_dev,
        width,
        height,
        channels
    );

    cudaFree(packed_dev);
    return result;
}
