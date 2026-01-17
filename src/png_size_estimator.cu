#include "png_size_estimator_masked.cuh"
#include <cmath>

// Paeth predictor (device)
__device__ inline int paeth_predictor(int a, int b, int c) {
    int p  = a + b - c;
    int pa = abs(p - a);
    int pb = abs(p - b);
    int pc = abs(p - c);
    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

// ---------------------------
// Kernel 1 (masked): per-row filter costs in bbox
// ---------------------------
__global__ void compute_filter_costs_per_row_masked_kernel(
    const uint8_t* __restrict__ img,    // full image HWC
    const int64_t* __restrict__ labels, // full labels HW
    int full_W, int full_H,
    int x0, int y0, int width, int height,
    int64_t k,
    uint32_t* __restrict__ costs_out,   // [height*5]
    int channels)
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

    const int gy = y0 + y;
    if (gy < 0 || gy >= full_H) return;

    for (int x = (int)threadIdx.x; x < width; x += (int)blockDim.x) {
        const int gx = x0 + x;
        if (gx < 0 || gx >= full_W) continue;

        const int64_t idx_lab = (int64_t)gy * full_W + gx;
        const bool in_seg = (labels[idx_lab] == k);

        for (int c = 0; c < channels; ++c) {
            const int64_t idx_img = ((int64_t)gy * full_W + gx) * channels + c;

            int cur = in_seg ? (int)img[idx_img] : 0;

            int left = 0, up = 0, up_left = 0;

            if (x > 0) {
                const int gx_l = gx - 1;
                const int64_t il = ((int64_t)gy * full_W + gx_l);
                const bool in_l = (labels[il] == k);
                left = in_l ? (int)img[idx_img - channels] : 0;
            }
            if (y > 0) {
                const int gy_u = gy - 1;
                const int64_t iu = ((int64_t)gy_u * full_W + gx);
                const bool in_u = (labels[iu] == k);
                const int64_t idx_up = ((int64_t)gy_u * full_W + gx) * channels + c;
                up = in_u ? (int)img[idx_up] : 0;
            }
            if (x > 0 && y > 0) {
                const int gx_ul = gx - 1, gy_ul = gy - 1;
                const int64_t iul = ((int64_t)gy_ul * full_W + gx_ul);
                const bool in_ul = (labels[iul] == k);
                const int64_t idx_ul = ((int64_t)gy_ul * full_W + gx_ul) * channels + c;
                up_left = in_ul ? (int)img[idx_ul] : 0;
            }

            // None (heuristic uses signed residual of raw byte)
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
// Kernel 3 (masked): residuals in bbox
// ---------------------------
__global__ void compute_residuals_with_selected_filter_masked_kernel(
    const uint8_t* __restrict__ img,
    const int64_t* __restrict__ labels,
    int full_W, int full_H,
    int x0, int y0, int width, int height,
    int64_t k,
    const uint8_t* __restrict__ filter_id, // [height]
    uint8_t* __restrict__ residuals,       // [width*height*channels]
    int channels)
{
    const size_t N = (size_t)width * (size_t)height * (size_t)channels;
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    while (idx < N) {
        int pixel_idx = (int)(idx / (size_t)channels);
        int c         = (int)(idx % (size_t)channels);
        int x         = pixel_idx % width;
        int y         = pixel_idx / width;

        const int gx = x0 + x;
        const int gy = y0 + y;

        int cur = 0, left = 0, up = 0, up_left = 0;

        if ((unsigned)gx < (unsigned)full_W && (unsigned)gy < (unsigned)full_H) {
            const int64_t il = (int64_t)gy * full_W + gx;
            const bool in_seg = (labels[il] == k);
            const int64_t ii = ((int64_t)gy * full_W + gx) * channels + c;
            cur = in_seg ? (int)img[ii] : 0;

            if (x > 0) {
                const int gx_l = gx - 1;
                const int64_t il2 = (int64_t)gy * full_W + gx_l;
                const bool in_l = (labels[il2] == k);
                left = in_l ? (int)img[ii - channels] : 0;
            }
            if (y > 0) {
                const int gy_u = gy - 1;
                const int64_t iu = (int64_t)gy_u * full_W + gx;
                const bool in_u = (labels[iu] == k);
                const int64_t iu_img = ((int64_t)gy_u * full_W + gx) * channels + c;
                up = in_u ? (int)img[iu_img] : 0;
            }
            if (x > 0 && y > 0) {
                const int gx_ul = gx - 1, gy_ul = gy - 1;
                const int64_t iul = (int64_t)gy_ul * full_W + gx_ul;
                const bool in_ul = (labels[iul] == k);
                const int64_t iul_img = ((int64_t)gy_ul * full_W + gx_ul) * channels + c;
                up_left = in_ul ? (int)img[iul_img] : 0;
            }
        }

        const uint8_t f = filter_id[y];
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
// Entropy from hist on GPU
// hist: [channels*256], count_per_channel = width*height
// ---------------------------
__global__ void entropy_from_hist_kernel(
    const uint32_t* __restrict__ hist,
    double* __restrict__ Hc,   // [channels]
    int channels,
    uint32_t count_per_channel)
{
    int c = (int)blockIdx.x;
    if (c >= channels) return;

    int v = (int)threadIdx.x; // 0..255 assumed
    __shared__ double sh[256];

    double acc = 0.0;
    if (v < 256) {
        uint32_t hv = hist[c * 256 + v];
        if (hv) {
        double p = (double)hv / (double)count_per_channel;
        acc = -p * ::log2(p);
        }
    }
    sh[v] = acc;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (v < stride) sh[v] += sh[v + stride];
        __syncthreads();
    }
    if (v == 0) Hc[c] = sh[0];
}

// ---------------------------
// Mean of channels
// ---------------------------
__global__ void mean_channels_kernel(const double* __restrict__ Hc, double* __restrict__ Hbar, int channels) {
    // one block, 256 threads
    __shared__ double sh[256];
    int t = (int)threadIdx.x;
    double v = 0.0;
    if (t < channels) v = Hc[t];
    sh[t] = v;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) sh[t] += sh[t + stride];
        __syncthreads();
    }
    if (t == 0) *Hbar = sh[0] / (double)channels;
}

// ---------------------------
// Compute size formula
// ---------------------------
__global__ void compute_size_kernel(
    double* __restrict__ out,
    double Hbar,
    unsigned long long match_symbols,
    unsigned long long match_count,
    unsigned long long match_len_sum,
    size_t N,
    int L_min,
    double beta,
    double b_match_token,
    double gamma,
    double overhead_base,
    int height)
{
    double f_match = 0.0;
    if (N > 0 && match_symbols > 0) f_match = (double)match_symbols / (double)N;

    double L_bar = (double)L_min;
    if (match_count > 0) L_bar = (double)match_len_sum / (double)match_count;

    double b_lit   = Hbar + beta;
    double b_match = (b_match_token / L_bar) + gamma;
    double b_data  = (1.0 - f_match) * b_lit + f_match * b_match;

    double S_overhead = overhead_base + (double)height;
    double S_est = S_overhead + ((double)N * b_data) / 8.0;
    *out = S_est;
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

// workspace store
PngEstimatorWorkspace& get_png_ws(int device_index) {
    static thread_local std::vector<PngEstimatorWorkspace> ws(16); // adjust if >16 GPUs
    return ws.at((size_t)device_index);
}

// main entrypoint
void estimate_png_size_masked_segment_to_output(
    const uint8_t* img_hwc_u8, int full_W, int full_H, int channels,
    const int64_t* labels_compact_hw,
    const int32_t* counts_k,
    int64_t seg_id_k,
    int x0, int y0, int w, int h,
    int32_t min_pixels,
    int L_min, float beta, float b_match_token, float gamma, double overhead_base,
    bool adaptive_filter,
    double* out_dev)
{
    if (!img_hwc_u8 || !labels_compact_hw || !counts_k || !out_dev) throw std::runtime_error("null ptr");
    if (channels < 1 || channels > 4) throw std::runtime_error("channels must be 1..4");
    if (w <= 0 || h <= 0) {
        // write 0
        CUDA_CHECK(cudaMemsetAsync(out_dev, 0, sizeof(double), c10::cuda::getCurrentCUDAStream().stream()));
        return;
    }

    const auto stream = c10::cuda::getCurrentCUDAStream().stream();
    const int dev = c10::cuda::current_device();
    auto& ws = get_png_ws(dev);
    ws.ensure((int64_t)w * h * channels, h, channels, c10::Device(c10::kCUDA, dev));

    // If too small, skip (device-side check)
    // We avoid a sync by reading counts on device in a tiny kernel:
    // (simpler: just run and let bbox be small, but min_pixels was requested)
    // Do it with a 1-thread kernel:
    auto counts_ptr = counts_k;
    auto out_ptr = out_dev;
    const int64_t k = seg_id_k;
    const int32_t mp = min_pixels;
    auto skip_check = [] __global__ (const int32_t* counts, int64_t k, int32_t mp, double* out) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            if (counts[(int)k] < mp) *out = 0.0;
        }
    };
    skip_check<<<1,1,0,stream>>>(counts_ptr, k, mp, out_ptr);
    CUDA_CHECK(cudaGetLastError());

    const size_t N = (size_t)w * (size_t)h * (size_t)channels;

    // hist = 0, match = 0
    CUDA_CHECK(cudaMemsetAsync(ws.hist_u32.data_ptr(), 0, (size_t)channels * 256 * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(ws.match_symbols_u64.data_ptr(), 0, sizeof(unsigned long long), stream));
    CUDA_CHECK(cudaMemsetAsync(ws.match_count_u64.data_ptr(),   0, sizeof(unsigned long long), stream));
    CUDA_CHECK(cudaMemsetAsync(ws.match_len_sum_u64.data_ptr(), 0, sizeof(unsigned long long), stream));

    // adaptive filter: costs + filter + residuals
    const int block_size = 256;
    int grid_size = (int)((N + (size_t)block_size - 1) / (size_t)block_size);
    int launch_grid = grid_size > 65535 ? 65535 : grid_size;

    uint8_t* residuals = (uint8_t*)ws.residuals_u8.data_ptr();
    uint32_t* costs    = (uint32_t*)ws.costs_u32.data_ptr();
    uint8_t*  filter   = (uint8_t*)ws.filter_u8.data_ptr();

    if (adaptive_filter) {
        dim3 k1_grid(h);
        dim3 k1_block(256);
        size_t k1_shmem = 5ULL * (size_t)k1_block.x * sizeof(uint32_t);

        compute_filter_costs_per_row_masked_kernel<<<k1_grid, k1_block, k1_shmem, stream>>>(
            img_hwc_u8, labels_compact_hw, full_W, full_H, x0, y0, w, h, seg_id_k, costs, channels);
        CUDA_CHECK(cudaGetLastError());

        dim3 k2_block(256);
        dim3 k2_grid((h + (int)k2_block.x - 1) / (int)k2_block.x);

        select_filter_per_row_kernel<<<k2_grid, k2_block, 0, stream>>>(costs, filter, h);
        CUDA_CHECK(cudaGetLastError());

        compute_residuals_with_selected_filter_masked_kernel<<<launch_grid, block_size, 0, stream>>>(
            img_hwc_u8, labels_compact_hw, full_W, full_H, x0, y0, w, h, seg_id_k, filter, residuals, channels);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // all rows Paeth (4)
        CUDA_CHECK(cudaMemsetAsync(filter, 4, (size_t)h * sizeof(uint8_t), stream));
        compute_residuals_with_selected_filter_masked_kernel<<<launch_grid, block_size, 0, stream>>>(
            img_hwc_u8, labels_compact_hw, full_W, full_H, x0, y0, w, h, seg_id_k, filter, residuals, channels);
        CUDA_CHECK(cudaGetLastError());
    }

    // histogram
    size_t shmem_hist = (size_t)channels * 256 * sizeof(uint32_t);
    histogram_residuals_kernel<<<launch_grid, block_size, shmem_hist, stream>>>(
        residuals, (uint32_t*)ws.hist_u32.data_ptr(), w, h, channels);
    CUDA_CHECK(cudaGetLastError());

    // entropy on GPU
    const uint32_t count_per_channel = (uint32_t)((uint64_t)w * (uint64_t)h);
    entropy_from_hist_kernel<<<channels, 256, 0, stream>>>(
        (const uint32_t*)ws.hist_u32.data_ptr(),
        (double*)ws.Hc_f64.data_ptr(),
        channels, count_per_channel);
    CUDA_CHECK(cudaGetLastError());

    mean_channels_kernel<<<1, 256, 0, stream>>>(
        (const double*)ws.Hc_f64.data_ptr(),
        (double*)ws.Hbar_f64.data_ptr(),
        channels);
    CUDA_CHECK(cudaGetLastError());

    // run-length proxy
    run_length_stats_kernel<<<256, 256, 0, stream>>>(
        residuals,
        N,
        L_min,
        (unsigned long long*)ws.match_symbols_u64.data_ptr(),
        (unsigned long long*)ws.match_count_u64.data_ptr(),
        (unsigned long long*)ws.match_len_sum_u64.data_ptr());
    CUDA_CHECK(cudaGetLastError());

    // compute size (GPU) -> out_dev
    auto launch_size = [] __global__ (
        double* out,
        const double* Hbar,
        const unsigned long long* ms,
        const unsigned long long* mc,
        const unsigned long long* mls,
        size_t N,
        int L_min,
        double beta,
        double b_match_token,
        double gamma,
        double overhead_base,
        int height)
    {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            compute_size_kernel(out, *Hbar, *ms, *mc, *mls, N, L_min, beta, b_match_token, gamma, overhead_base, height);
        }
    };

    launch_size<<<1, 1, 0, stream>>>(
        out_dev,
        (const double*)ws.Hbar_f64.data_ptr(),
        (const unsigned long long*)ws.match_symbols_u64.data_ptr(),
        (const unsigned long long*)ws.match_count_u64.data_ptr(),
        (const unsigned long long*)ws.match_len_sum_u64.data_ptr(),
        N, L_min,
        (double)beta, (double)b_match_token, (double)gamma, overhead_base, h);
    CUDA_CHECK(cudaGetLastError());
}
