#include <thrust/device_vector.h>
#include "rama_wrapper.cuh"
#include <vector>
#include <algorithm>
#include "rama_cuda.h"
#include "multicut_solver_options.h"


torch::Tensor rama_torch(
    const torch::Tensor& _i,
    const torch::Tensor& _j,
    const torch::Tensor& _costs)
{
	CHECK_INPUT(_i);
	CHECK_INPUT(_j);
	CHECK_INPUT(_costs);
	if (_i.size(0) != _j.size(0) || _i.size(0) != _costs.size(0))
		throw std::runtime_error("Input shapes must match");
    if (_i.scalar_type() != _j.scalar_type())
		throw std::runtime_error("Node indices i, j should be of same type");

    TORCH_CHECK(_i.dim() == 1, "i should be one-dimensional");
    TORCH_CHECK(_j.dim() == 1, "j should be one-dimensional");
    TORCH_CHECK(_costs.dim() == 1, "costs should be one-dimensional");

	multicut_solver_options opts;
	opts.verbose = false;

	thrust::device_vector<int> i(_i.data_ptr<int32_t>(), _i.data_ptr<int32_t>() + _i.size(0));
	thrust::device_vector<int> j(_j.data_ptr<int32_t>(), _j.data_ptr<int32_t>() + _j.size(0));
	thrust::device_vector<float> costs(_costs.data_ptr<float>(), _costs.data_ptr<float>() + _costs.size(0));
	thrust::device_vector<int> node_mapping;
    std::vector<std::vector<int>> timeline;
	double lb;
	const int device = _costs.device().index();
	if (device < 0)
		throw std::runtime_error("Invalid device ID");
    std::tie(node_mapping, lb, timeline) = rama_cuda(std::move(i), std::move(j), std::move(costs), opts, device);

	torch::Tensor node_mapping_torch = at::empty({long(node_mapping.size())}, _i.options());
    thrust::copy(node_mapping.begin(), node_mapping.end(), node_mapping_torch.data_ptr<int32_t>());

	return node_mapping_torch;
}

// Batched version:
//  _i:      [E]   int32 CUDA
//  _j:      [E]   int32 CUDA
//  _costs:  [B,E] float32 CUDA (contiguous strongly recommended)
// Returns:
//  labels:  [B,N] int32 CUDA, where N = max(i,j)+1 (num nodes)
torch::Tensor rama_torch_batched(
    const torch::Tensor& _i,
    const torch::Tensor& _j,
    const torch::Tensor& _costs_be)
{
    CHECK_INPUT(_i);
    CHECK_INPUT(_j);
    CHECK_INPUT(_costs_be);

    TORCH_CHECK(_i.dim() == 1, "i should be one-dimensional [E]");
    TORCH_CHECK(_j.dim() == 1, "j should be one-dimensional [E]");
    TORCH_CHECK(_costs_be.dim() == 2, "costs should be two-dimensional [B,E]");

    TORCH_CHECK(_i.scalar_type() == torch::kInt32, "i must be int32");
    TORCH_CHECK(_j.scalar_type() == torch::kInt32, "j must be int32");
    TORCH_CHECK(_costs_be.scalar_type() == torch::kFloat32, "costs must be float32");

    const int64_t E = _i.size(0);
    TORCH_CHECK(_j.size(0) == E, "i/j size mismatch");
    TORCH_CHECK(_costs_be.size(1) == E, "costs second dim must equal E");

    const int64_t B = _costs_be.size(0);

    const int device = _costs_be.device().index();
    TORCH_CHECK(device >= 0, "Invalid device ID");
    TORCH_CHECK(_i.device().index() == device && _j.device().index() == device,
                "i/j/costs must be on the same CUDA device");

    // Ensure contiguous so per-row pointer arithmetic is valid and fast.
    auto costs_be = _costs_be.contiguous();

    // Build solver options (disable heavy features).
    multicut_solver_options opts;
    opts.verbose = false;
    opts.dump_timeline = false;

    // Copy i/j once into thrust vectors.
    thrust::device_vector<int> i_base(_i.data_ptr<int32_t>(), _i.data_ptr<int32_t>() + E);
    thrust::device_vector<int> j_base(_j.data_ptr<int32_t>(), _j.data_ptr<int32_t>() + E);

    // Determine number of nodes once (sync once).
    const int64_t max_i = _i.max().item<int64_t>();
    const int64_t max_j = _j.max().item<int64_t>();
    const int64_t N = std::max(max_i, max_j) + 1;
    TORCH_CHECK(N > 0, "Invalid number of nodes inferred from i/j");

    // Output tensor [B,N] on CUDA, int32.
    torch::Tensor out = at::empty({B, N}, _i.options());

    // Solve each sample (sequential). This removes the per-sample torch->C++ boundary.
    for (int64_t b = 0; b < B; ++b) {
        // Copy costs row into a thrust vector (still required with current rama_cuda API).
        auto costs_b = costs_be[b]; // [E] view, contiguous because costs_be is contiguous
        thrust::device_vector<float> costs(costs_b.data_ptr<float>(), costs_b.data_ptr<float>() + E);

        // rama_cuda currently consumes i/j by move, so we need per-sample copies.
        thrust::device_vector<int> i = i_base;
        thrust::device_vector<int> j = j_base;

        thrust::device_vector<int> node_mapping;
        std::vector<std::vector<int>> timeline;
        double lb = 0.0;

        std::tie(node_mapping, lb, timeline) =
            rama_cuda(std::move(i), std::move(j), std::move(costs), opts, device);

        TORCH_CHECK((int64_t)node_mapping.size() == N,
                    "rama_cuda returned node_mapping of unexpected size");

        // Write into out[b]
        auto out_b = out[b];
        thrust::copy(node_mapping.begin(), node_mapping.end(), out_b.data_ptr<int32_t>());
    }

    return out;
}
