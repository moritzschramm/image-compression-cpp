#include "segment_stats.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); } } while(0)
#endif

__global__ void init_counts_bboxes_kernel(int32_t* counts, int32_t* bboxes, int K, int W, int H) {
    int k = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (k >= K) return;
    counts[k] = 0;
    // x0,y0,x1,y1
    bboxes[k*4 + 0] = W;
    bboxes[k*4 + 1] = H;
    bboxes[k*4 + 2] = -1;
    bboxes[k*4 + 3] = -1;
}

__global__ void stats_kernel(
    const int64_t* __restrict__ lab, // [H*W]
    int H, int W,
    int32_t* __restrict__ counts,    // [K]
    int32_t* __restrict__ bboxes)    // [K*4]
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t N = (int64_t)H * W;
    if (idx >= N) return;

    const int64_t k = lab[idx];
    // labels are 0..K-1
    const int x = (int)(idx % W);
    const int y = (int)(idx / W);

    atomicAdd(&counts[(int)k], 1);
    atomicMin(&bboxes[(int)k*4 + 0], x);
    atomicMin(&bboxes[(int)k*4 + 1], y);
    atomicMax(&bboxes[(int)k*4 + 2], x);
    atomicMax(&bboxes[(int)k*4 + 3], y);
}

void compute_counts_bboxes_from_compact_labels_cuda(
    const torch::Tensor& labels_compact_hw_i64,
    torch::Tensor& counts_k_i32,
    torch::Tensor& bboxes_k4_i32)
{
    TORCH_CHECK(labels_compact_hw_i64.is_cuda(), "labels must be CUDA");
    TORCH_CHECK(labels_compact_hw_i64.scalar_type() == torch::kInt64, "labels must be int64");
    TORCH_CHECK(labels_compact_hw_i64.dim() == 2, "labels must be [H,W]");
    TORCH_CHECK(counts_k_i32.is_cuda() && bboxes_k4_i32.is_cuda(), "outputs must be CUDA");
    TORCH_CHECK(counts_k_i32.scalar_type() == torch::kInt32, "counts must be int32");
    TORCH_CHECK(bboxes_k4_i32.scalar_type() == torch::kInt32, "bboxes must be int32");
    TORCH_CHECK(bboxes_k4_i32.size(1) == 4, "bboxes must be [K,4]");

    const int H = (int)labels_compact_hw_i64.size(0);
    const int W = (int)labels_compact_hw_i64.size(1);
    const int K = (int)counts_k_i32.numel();

    const auto stream = c10::cuda::getCurrentCUDAStream().stream();

    const int threads = 256;
    init_counts_bboxes_kernel<<<(K + threads - 1)/threads, threads, 0, stream>>>(
        counts_k_i32.data_ptr<int32_t>(),
        bboxes_k4_i32.data_ptr<int32_t>(),
        K, W, H);
    CUDA_CHECK(cudaGetLastError());

    const int64_t N = (int64_t)H * W;
    stats_kernel<<<(N + threads - 1)/threads, threads, 0, stream>>>(
        labels_compact_hw_i64.data_ptr<int64_t>(),
        H, W,
        counts_k_i32.data_ptr<int32_t>(),
        bboxes_k4_i32.data_ptr<int32_t>());
    CUDA_CHECK(cudaGetLastError());
}
