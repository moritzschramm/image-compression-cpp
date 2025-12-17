#pragma once
#include <torch/torch.h>
#include <cmath>

struct SampleOut {
    torch::Tensor w;       // [B,E]
    torch::Tensor logp;    // [B]
    torch::Tensor entropy; // [B]
};

inline SampleOut sample_gaussian_policy(
    const torch::Tensor& mu,        // [B,E]
    const torch::Tensor& sigma,     // [B,E]
) {
    // sample: w = mu + sigma * noise
    auto noise = torch::randn_like(mu);
    auto w = mu + sigma * noise;

    // log_prob per element
    // logp_e = -0.5 * ((w-mu)/sigma)^2 - log(sigma) - 0.5*log(2*pi)
    const double log2pi = std::log(2.0 * M_PI);
    auto z = (w - mu) / sigma;
    auto logp_elem = -0.5 * z.pow(2) - torch::log(sigma) - 0.5 * log2pi;

    // sum over edges => [B]
    auto logp = logp_elem.sum(/*dim=*/1);

    // entropy per element: 0.5*(1+log(2*pi)) + log(sigma)
    auto ent_elem = 0.5 * (1.0 + log2pi) + torch::log(sigma);
    auto entropy = ent_elem.sum(/*dim=*/1);

    return {w, logp, entropy};
}
