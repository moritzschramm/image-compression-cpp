// Overload to reuse cached i and j
std::tuple<thrust::device_vector<int>, double, std::vector<std::vector<int>>>
rama_cuda(
    const thrust::device_vector<int>& i,
    const thrust::device_vector<int>& j,
    thrust::device_vector<float>&& costs,
    const multicut_solver_options& opts,
    int device
);

// Batched variant
std::tuple<thrust::device_vector<int>, std::vector<double>>
rama_cuda_batched(
    const thrust::device_vector<int>& i,
    const thrust::device_vector<int>& j,
    const thrust::device_vector<float>& costs_be, // [B*E]
    int B, int E,
    int num_nodes,
    const multicut_solver_options& opts,
    int device
);
