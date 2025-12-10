#include <thrust/device_vector.h>
#include "rama_wrapper.h"
#include <vector>
#include "rama_cuda.h"


std::vector<torch::Tensor> rama_torch(
    const torch::Tensor& _i,
    const torch::Tensor& _j,
    const torch::Tensor& _costs,
	const multicut_solver_options& opts)
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

	torch::Tensor lb_torch = at::empty({1}, _i.options());
	lb_torch.toType(torch::kFloat64);
	lb_torch.fill_(lb);
	return {node_mapping_torch, lb_torch};
}
