#include "png_size_estimator.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <cmath>
#include <stdexcept>


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

// ---------------------------
// Utility: pitched->packed copy (device-to-device)
// ---------------------------
__global__ void copy_pitched_to_packed_kernel(
    const uint8_t* __restrict__ src,
    int src_step,
    uint8_t* __restrict__ dst,
    int width,
    int height,
    int channels)
{
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) return;

    const uint8_t* src_row = src + (size_t)y * (size_t)src_step;
    uint8_t* dst_row = dst + (size_t)y * (size_t)width * (size_t)channels;

    int base = x * channels;
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        if (c < channels) dst_row[base + c] = src_row[base + c];
    }
}

// ---------------------------
// Kernel 1: per-row filter costs (heuristic SAD on signed residuals)
// filters: 0 None, 1 Sub, 2 Up, 3 Avg, 4 Paeth
// ---------------------------
__global__ void compute_filter_costs_per_row_kernel(
    const uint8_t* __restrict__ img,
    uint32_t* __restrict__ costs_out, // [height*5]
    int width, int height, int channels)
{
    int y = (int)blockIdx.x;
    if (y >= height) return;

    extern __shared__ uint32_t sh[];
    uint32_t* sh_none  = sh + 0 * blockDim.x;
    uint32_t* sh_sub   = sh + 1 * blockDim.x;
    uint32_t* sh_up    = sh + 2 * blockDim.x;
    uint32_t* sh_avg   = sh + 3 * blockDim.x;
    uint32_t* sh_paeth = sh + 4 * blockDim.x;

    uint32_t local_none = 0, local_sub = 0, local_up = 0, local_avg = 0, local_paeth = 0;

    for (int x = (int)threadIdx.x; x < width; x += (int)blockDim.x) {
        for (int c = 0; c < channels; ++c) {
            int idx = (y * width + x) * channels + c;
            int cur = (int)img[idx];

            int left = 0, up = 0, up_left = 0;
            if (x > 0) left = (int)img[idx - channels];
            if (y > 0) {
                int up_idx = ((y - 1) * width + x) * channels + c;
                up = (int)img[up_idx];
            }
            if (x > 0 && y > 0) {
                int ul_idx = ((y - 1) * width + (x - 1)) * channels + c;
                up_left = (int)img[ul_idx];
            }

            // None
            {
                uint8_t r8 = (uint8_t)cur;
                int s = (int)((int8_t)r8);
                local_none += (uint32_t)abs(s);
            }
            // Sub
            {
                int pred = left;
                uint8_t r8 = (uint8_t)((cur - pred) & 0xFF);
                int s = (int)((int8_t)r8);
                local_sub += (uint32_t)abs(s);
            }
            // Up
            {
                int pred = up;
                uint8_t r8 = (uint8_t)((cur - pred) & 0xFF);
                int s = (int)((int8_t)r8);
                local_up += (uint32_t)abs(s);
            }
            // Avg
            {
                int pred = (left + up) >> 1;
                uint8_t r8 = (uint8_t)((cur - pred) & 0xFF);
                int s = (int)((int8_t)r8);
                local_avg += (uint32_t)abs(s);
            }
            // Paeth
            {
                int pred = paeth_predictor(left, up, up_left);
                uint8_t r8 = (uint8_t)((cur - pred) & 0xFF);
                int s = (int)((int8_t)r8);
                local_paeth += (uint32_t)abs(s);
            }
        }
    }

    sh_none[threadIdx.x]  = local_none;
    sh_sub[threadIdx.x]   = local_sub;
    sh_up[threadIdx.x]    = local_up;
    sh_avg[threadIdx.x]   = local_avg;
    sh_paeth[threadIdx.x] = local_paeth;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sh_none[threadIdx.x]  += sh_none[threadIdx.x + stride];
            sh_sub[threadIdx.x]   += sh_sub[threadIdx.x + stride];
            sh_up[threadIdx.x]    += sh_up[threadIdx.x + stride];
            sh_avg[threadIdx.x]   += sh_avg[threadIdx.x + stride];
            sh_paeth[threadIdx.x] += sh_paeth[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        uint32_t* row = costs_out + y * 5;
        row[0] = sh_none[0];
        row[1] = sh_sub[0];
        row[2] = sh_up[0];
        row[3] = sh_avg[0];
        row[4] = sh_paeth[0];
    }
}

// ---------------------------
// Kernel 2: select filter per row (min-cost)
// ---------------------------
__global__ void select_filter_per_row_kernel(
    const uint32_t* __restrict__ costs, // [height*5]
    uint8_t* __restrict__ filter_id,    // [height]
    int height)
{
    int y = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (y >= height) return;

    const uint32_t* row = costs + y * 5;
    uint32_t best = row[0];
    uint8_t best_id = 0;

    #pragma unroll
    for (int f = 1; f < 5; ++f) {
        uint32_t v = row[f];
        if (v < best) { best = v; best_id = (uint8_t)f; }
    }
    filter_id[y] = best_id;
}

// ---------------------------
// Kernel 3: compute residual stream given selected per-row filter
// residuals: packed interleaved like img
// ---------------------------
__global__ void compute_residuals_with_selected_filter_kernel(
    const uint8_t* __restrict__ img,
    const uint8_t* __restrict__ filter_id, // [height]
    uint8_t* __restrict__ residuals,
    int width, int height, int channels)
{
    size_t N = (size_t)width * (size_t)height * (size_t)channels;
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    while (idx < N) {
        int pixel_idx = (int)(idx / (size_t)channels);
        int c         = (int)(idx % (size_t)channels);
        int x         = pixel_idx % width;
        int y         = pixel_idx / width;

        int cur = (int)img[idx];

        int left = 0, up = 0, up_left = 0;
        if (x > 0) left = (int)img[idx - (size_t)channels];
        if (y > 0) {
            int up_idx = ((y - 1) * width + x) * channels + c;
            up = (int)img[(size_t)up_idx];
        }
        if (x > 0 && y > 0) {
            int ul_idx = ((y - 1) * width + (x - 1)) * channels + c;
            up_left = (int)img[(size_t)ul_idx];
        }

        uint8_t f = filter_id[y];
        int pred = 0;
        if (f == 0) pred = 0;
        else if (f == 1) pred = left;
        else if (f == 2) pred = up;
        else if (f == 3) pred = (left + up) >> 1;
        else pred = paeth_predictor(left, up, up_left);

        uint8_t r8 = (f == 0) ? (uint8_t)cur : (uint8_t)((cur - pred) & 0xFF);
        residuals[idx] = r8;

        idx += (size_t)blockDim.x * (size_t)gridDim.x;
    }
}

// ---------------------------
// Histogram residuals per channel into hist[channels*256]
// ---------------------------
__global__ void histogram_residuals_kernel(
    const uint8_t* __restrict__ residuals,
    uint32_t* __restrict__ hist,
    int width, int height, int channels)
{
    extern __shared__ uint32_t sh_hist[];
    int total_bins = channels * 256;

    for (int i = threadIdx.x; i < total_bins; i += (int)blockDim.x)
        sh_hist[i] = 0;
    __syncthreads();

    size_t N = (size_t)width * (size_t)height * (size_t)channels;
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    while (idx < N) {
        int c = (int)(idx % (size_t)channels);
        uint8_t r8 = residuals[idx];
        atomicAdd(&sh_hist[c * 256 + (int)r8], 1U);
        idx += (size_t)blockDim.x * (size_t)gridDim.x;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < total_bins; i += (int)blockDim.x) {
        uint32_t v = sh_hist[i];
        if (v) atomicAdd(&hist[i], v);
    }
}

// ---------------------------
// Match proxy: run-length stats on residual stream
// ---------------------------
__global__ void run_length_stats_kernel(
    const uint8_t* __restrict__ residuals,
    size_t N,
    int L_min,
    unsigned long long* match_symbols_out,
    unsigned long long* match_count_out,
    unsigned long long* match_length_sum_out)
{
    __shared__ unsigned long long sh_match_symbols;
    __shared__ unsigned long long sh_match_count;
    __shared__ unsigned long long sh_match_length_sum;

    if (threadIdx.x == 0) {
        sh_match_symbols    = 0;
        sh_match_count      = 0;
        sh_match_length_sum = 0;
    }
    __syncthreads();

    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int num_threads = (int)(blockDim.x * gridDim.x);

    size_t chunk_size = (N + (size_t)num_threads - 1) / (size_t)num_threads;
    size_t start = (size_t)tid * chunk_size;
    if (start >= N) return;
    size_t end = start + chunk_size;
    if (end > N) end = N;
    if (end - start == 0) return;

    uint8_t cur = residuals[start];
    int run_len = 1;

    unsigned long long local_match_symbols = 0;
    unsigned long long local_match_count = 0;
    unsigned long long local_match_length_sum = 0;

    for (size_t i = start + 1; i < end; ++i) {
        uint8_t v = residuals[i];
        if (v == cur) {
            ++run_len;
        } else {
            if (run_len >= L_min) {
                local_match_symbols    += (unsigned long long)run_len;
                local_match_count      += 1ULL;
                local_match_length_sum += (unsigned long long)run_len;
            }
            cur = v;
            run_len = 1;
        }
    }
    if (run_len >= L_min) {
        local_match_symbols    += (unsigned long long)run_len;
        local_match_count      += 1ULL;
        local_match_length_sum += (unsigned long long)run_len;
    }

    atomicAdd(&sh_match_symbols,    local_match_symbols);
    atomicAdd(&sh_match_count,      local_match_count);
    atomicAdd(&sh_match_length_sum, local_match_length_sum);
    __syncthreads();

    if (threadIdx.x == 0) {
        if (sh_match_symbols)    atomicAdd(match_symbols_out,    sh_match_symbols);
        if (sh_match_count)      atomicAdd(match_count_out,      sh_match_count);
        if (sh_match_length_sum) atomicAdd(match_length_sum_out, sh_match_length_sum);
    }
}

// ---------------------------
// Internal helper: compute residuals+hist from packed img_dev
// ---------------------------
static void compute_residuals_and_hist(
    const uint8_t* img_dev,
    uint8_t* residuals_dev,
    uint32_t* hist_dev,
    int width,
    int height,
    int channels,
    bool adaptive_filter)
{
    // Histogram init
    CUDA_CHECK(cudaMemset(hist_dev, 0, (size_t)channels * 256 * sizeof(uint32_t)));

    const int block_size = 256;
    size_t N = (size_t)width * (size_t)height * (size_t)channels;
    int grid_size = (int)((N + (size_t)block_size - 1) / (size_t)block_size);
    int launch_grid = grid_size > 65535 ? 65535 : grid_size;

    if (adaptive_filter) {
        // allocate per-row costs + filter ids
        uint32_t* costs_dev = nullptr;
        uint8_t* filter_dev = nullptr;
        CUDA_CHECK(cudaMalloc(&costs_dev, (size_t)height * 5 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&filter_dev, (size_t)height * sizeof(uint8_t)));

        // Kernel 1
        dim3 k1_grid(height);
        dim3 k1_block(256);
        size_t k1_shmem = 5ULL * (size_t)k1_block.x * sizeof(uint32_t);
        compute_filter_costs_per_row_kernel<<<k1_grid, k1_block, k1_shmem>>>(
            img_dev, costs_dev, width, height, channels);
        CUDA_CHECK(cudaGetLastError());

        // Kernel 2
        dim3 k2_block(256);
        dim3 k2_grid((height + (int)k2_block.x - 1) / (int)k2_block.x);
        select_filter_per_row_kernel<<<k2_grid, k2_block>>>(costs_dev, filter_dev, height);
        CUDA_CHECK(cudaGetLastError());

        // Kernel 3
        compute_residuals_with_selected_filter_kernel<<<launch_grid, block_size>>>(
            img_dev, filter_dev, residuals_dev, width, height, channels);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(costs_dev));
        CUDA_CHECK(cudaFree(filter_dev));
    } else {
        // if disabled adaptive filter -> still need residuals
        // use filter_id==Paeth for all rows by faking filter_id in a temporary buffer
        uint8_t* filter_dev = nullptr;
        CUDA_CHECK(cudaMalloc(&filter_dev, (size_t)height * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemset(filter_dev, 4, (size_t)height * sizeof(uint8_t))); // 4 = Paeth

        compute_residuals_with_selected_filter_kernel<<<launch_grid, block_size>>>(
            img_dev, filter_dev, residuals_dev, width, height, channels);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(filter_dev));
    }

    // histogram residuals
    size_t shmem_hist = (size_t)channels * 256 * sizeof(uint32_t);
    histogram_residuals_kernel<<<launch_grid, block_size, shmem_hist>>>(
        residuals_dev, hist_dev, width, height, channels);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------
// Public API: packed device image
// ---------------------------
double estimate_png_size_from_device_image(
    const uint8_t* img_dev,
    int width,
    int height,
    int channels,
    int L_min,
    float beta,
    float b_match_token,
    float gamma,
    double overhead_base,
    bool adaptive_filter)
{
    if (!img_dev) throw std::runtime_error("img_dev is null");
    if (width <= 0 || height <= 0 || channels <= 0 || channels > 4)
        throw std::runtime_error("Invalid width/height/channels (channels must be 1..4)");

    const size_t N = (size_t)width * (size_t)height * (size_t)channels;

    uint8_t* residuals_dev = nullptr;
    uint32_t* hist_dev = nullptr;

    CUDA_CHECK(cudaMalloc(&residuals_dev, N * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&hist_dev, (size_t)channels * 256 * sizeof(uint32_t)));

    compute_residuals_and_hist(img_dev, residuals_dev, hist_dev, width, height, channels, adaptive_filter);

    // copy histogram to host
    std::vector<uint32_t> hist_host((size_t)channels * 256);
    CUDA_CHECK(cudaMemcpy(hist_host.data(),
                          hist_dev,
                          hist_host.size() * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    // compute per channel entropy on host
    std::vector<double> H_c((size_t)channels, 0.0);
    for (int c = 0; c < channels; ++c) {
        const uint32_t* h = &hist_host[(size_t)c * 256];

        unsigned long long count_c = 0;
        for (int v = 0; v < 256; ++v) count_c += (unsigned long long)h[v];

        if (count_c == 0) { H_c[(size_t)c] = 0.0; continue; }

        double inv_total = 1.0 / (double)count_c;
        double H = 0.0;
        for (int v = 0; v < 256; ++v) {
            uint32_t hv = h[v];
            if (!hv) continue;
            double p = (double)hv * inv_total;
            H -= p * std::log2(p);
        }
        H_c[(size_t)c] = H;
    }

    double H_bar = 0.0;
    for (int c = 0; c < channels; ++c) H_bar += H_c[(size_t)c];
    H_bar /= (double)channels;

    // match proxy on GPU (run-length)
    unsigned long long *d_match_symbols = nullptr, *d_match_count = nullptr, *d_match_len_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&d_match_symbols, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_match_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_match_len_sum, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_match_symbols, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_match_count, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_match_len_sum, 0, sizeof(unsigned long long)));

    const int rl_block = 256;
    const int rl_grid = 256;
    run_length_stats_kernel<<<rl_grid, rl_block>>>(residuals_dev, N, L_min, d_match_symbols, d_match_count, d_match_len_sum);
    CUDA_CHECK(cudaGetLastError());

    unsigned long long h_match_symbols = 0, h_match_count = 0, h_match_len_sum = 0;
    CUDA_CHECK(cudaMemcpy(&h_match_symbols, d_match_symbols, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_match_count,   d_match_count,   sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_match_len_sum, d_match_len_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_match_symbols));
    CUDA_CHECK(cudaFree(d_match_count));
    CUDA_CHECK(cudaFree(d_match_len_sum));
    CUDA_CHECK(cudaFree(hist_dev));
    CUDA_CHECK(cudaFree(residuals_dev));

    // compute f_match and L_bar
    double f_match = 0.0;
    double L_bar = (double)L_min;
    if (N > 0 && h_match_symbols > 0) f_match = (double)h_match_symbols / (double)N;
    if (h_match_count > 0) L_bar = (double)h_match_len_sum / (double)h_match_count;

    // bit cost model
    double b_lit   = H_bar + (double)beta;
    double b_match = ((double)b_match_token / L_bar) + (double)gamma;
    double b_data  = (1.0 - f_match) * b_lit + f_match * b_match;

    // overhead: PNG/zlib headers + one filter byte per scanline
    double S_overhead = overhead_base + (double)height;

    // total bytes
    double S_est = S_overhead + ((double)N * b_data) / 8.0;
    return S_est;
}

// ---------------------------
// Public API: pitched device image (e.g., cv::cuda::GpuMat)
// ---------------------------
double estimate_png_size_from_pitched_device_image(
    const void* gpu_mat_data,
    int step_bytes,
    int width,
    int height,
    int channels,
    int L_min,
    float beta,
    float b_match_token,
    float gamma,
    double overhead_base,
    bool adaptive_filter)
{
    if (!gpu_mat_data) throw std::runtime_error("gpu_mat_data is null");
    if (step_bytes <= 0) throw std::runtime_error("step_bytes must be > 0");

    const size_t packed_size = (size_t)width * (size_t)height * (size_t)channels;

    uint8_t* packed_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&packed_dev, packed_size));

    dim3 block(32, 8);
    dim3 grid((width + (int)block.x - 1) / (int)block.x,
              (height + (int)block.y - 1) / (int)block.y);

    copy_pitched_to_packed_kernel<<<grid, block>>>(
        (const uint8_t*)gpu_mat_data, step_bytes, packed_dev, width, height, channels);
    CUDA_CHECK(cudaGetLastError());

    double out = estimate_png_size_from_device_image(
        packed_dev, width, height, channels, L_min, beta, b_match_token, gamma, overhead_base, adaptive_filter);

    CUDA_CHECK(cudaFree(packed_dev));
    return out;
}

double estimate_png_size_from_GpuMat(
    const cv::cuda::GpuMat& gpu,
    int L_min,
    float beta,
    float b_match_token,
    float gamma,
    double overhead_base,
    bool adaptive_filter)
{
    if (gpu.empty()) throw std::runtime_error("GpuMat is empty");
    if (gpu.depth() != CV_8U) throw std::runtime_error("GpuMat must be CV_8U");
    int channels = gpu.channels();
    if (channels < 1 || channels > 4) throw std::runtime_error("GpuMat channels must be 1..4");

    return estimate_png_size_from_pitched_device_image(
        gpu.ptr<uint8_t>(), (int)gpu.step, gpu.cols, gpu.rows, channels,
        L_min, beta, b_match_token, gamma, overhead_base, adaptive_filter);
}
