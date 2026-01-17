// This overload allows callers to keep i/j cached and avoids rebuilding them from torch tensors.
// NOTE: It still copies i/j internally because the existing move-taking rama_cuda consumes them.
std::tuple<thrust::device_vector<int>, double, std::vector<std::vector<int>>>
rama_cuda(
    const thrust::device_vector<int>& i,
    const thrust::device_vector<int>& j,
    thrust::device_vector<float>&& costs,
    const multicut_solver_options& opts,
    int device
) {
    cudaSetDevice(device);

    // Basic sanity checks (optional but helpful)
    if (i.size() != j.size())
        throw std::runtime_error("rama_cuda(i,j,costs): i/j size mismatch");
    if (i.size() != costs.size())
        throw std::runtime_error("rama_cuda(i,j,costs): costs size mismatch");

    // Copy i/j because the existing implementation takes rvalue refs and moves into dCOO.
    // This still avoids having to reconstruct i/j from torch tensors every call.
    thrust::device_vector<int> i_copy = i;
    thrust::device_vector<int> j_copy = j;

    // Forward to existing implementation
    return rama_cuda(std::move(i_copy), std::move(j_copy), std::move(costs), opts, device);
}

// Batched variant
std::tuple<thrust::device_vector<int>, std::vector<double>>
rama_cuda_batched(
    const thrust::device_vector<int>& i_base,
    const thrust::device_vector<int>& j_base,
    const thrust::device_vector<float>& costs_be,
    int B, int E,
    int num_nodes,
    const multicut_solver_options& opts,
    int device
) {
    cudaSetDevice(device);

    if ((int)i_base.size() != E || (int)j_base.size() != E)
        throw std::runtime_error("rama_cuda_batched: i/j size must equal E");
    if ((int)costs_be.size() != B * E)
        throw std::runtime_error("rama_cuda_batched: costs_be must be B*E");

    // Batched + timeline is not feasible.
    if (opts.dump_timeline)
        throw std::runtime_error("rama_cuda_batched: disable dump_timeline");

    thrust::device_vector<int> labels_be((size_t)B * (size_t)num_nodes);
    std::vector<double> lbs(B);

    for (int b = 0; b < B; ++b) {
        // costs copy for this sample
        auto cb = costs_be.begin() + (size_t)b * (size_t)E;
        thrust::device_vector<float> costs(cb, cb + E);

        // i/j must be mutable because rama_cuda currently moves them into dCOO
        thrust::device_vector<int> i = i_base;
        thrust::device_vector<int> j = j_base;

        thrust::device_vector<int> node_mapping;
        double lb;
        std::vector<std::vector<int>> timeline;
        std::tie(node_mapping, lb, timeline) =
            rama_cuda(std::move(i), std::move(j), std::move(costs), opts, device);

        if ((int)node_mapping.size() != num_nodes)
            throw std::runtime_error("rama_cuda_batched: node_mapping size != num_nodes");

        thrust::copy(node_mapping.begin(), node_mapping.end(),
                     labels_be.begin() + (size_t)b * (size_t)num_nodes);

        lbs[b] = lb;
    }

    return {std::move(labels_be), std::move(lbs)};
}
