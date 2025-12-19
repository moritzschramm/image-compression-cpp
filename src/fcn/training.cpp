#include <c10/core/DeviceType.h>
#include <torch/torch.h>
#include "fcn/EdgeUNet.h"
#include "gaussian_policy.hpp"
#include "ema_baseline.hpp"
#include "rama_wrapper.cuh"
#include "png_size_estimator.cuh"

torch::Tensor flatten_grid_edges(const torch::Tensor& x)
{
    // x: [B, 4, H, W]
    TORCH_CHECK(x.dim() == 4, "Expected [B, 4, H, W]");

    const int64_t B = x.size(0);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    // horizontal edges: channels 0,1 — drop last column
    auto h = x.slice(1, 0, 2).slice(3, 0, W - 1);  // [B, 2, H, W-1]

    // vertical edges: channels 2,3 — drop last row
    auto v = x.slice(1, 2, 4).slice(2, 0, H - 1);  // [B, 2, H-1, W]

    // flatten spatial dims
    auto h_flat = h.reshape({B, 2, -1});  // [B, 2, H*(W-1)]
    auto v_flat = v.reshape({B, 2, -1});  // [B, 2, (H-1)*W]

    // concatenate edge lists
    return torch::cat({h_flat, v_flat}, /*dim=*/2);  // [B, 2, E]
}

void build_rama_indices(
    int32_t H,
    int32_t W,
    std::vector<int32_t>& i_idx,
    std::vector<int32_t>& j_idx)
{
    i_idx.clear();
    j_idx.clear();

    // horizontal edges
    for (int32_t r = 0; r < H; ++r) {
        for (int32_t c = 0; c < W - 1; ++c) {
            int32_t u = r * W + c;
            int32_t v = r * W + (c + 1);

            i_idx.push_back(u);
            j_idx.push_back(v);
        }
    }

    // vertical edges
    for (int32_t r = 0; r < H - 1; ++r) {
        for (int32_t c = 0; c < W; ++c) {
            int32_t u = r * W + c;
            int32_t v = (r + 1) * W + c;

            i_idx.push_back(u);
            j_idx.push_back(v);
        }
    }
}

int main()
{
    const auto device = torch::kCUDA;

    EdgeUNet model;
    torch::load(model, "fcn_pretrained.pt");

    model->to(device);

    torch::manual_seed(0);

    torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(1e-4));

    EMABaseline baseline(0.99);

    const double entropy_coef = 1e-4;

    std::vector<int32_t> i_idx;
    std::vector<int32_t> j_idx;

    build_rama_indices(512, 512); // assuming height and width of 512x512

    for (int step = 0; step < 1e5; ++step) {
        torch::Tensor image = /* TODO load batch */ torch::randn({4, 4, 512, 512}, device);

        // forward: raw_mu, raw_sigma [B,E]
        // first and second output channel: horizontal edges
        // third and fourth output channel: vertical edges
        auto out = model->forward(image);
        auto flat = flatten_grid_edges(out);

        torch::Tensor raw_mu = flat.select(1, 0);
        torch::Tensor raw_sigma = flat.select(1, 1);

        const double mu_scale = 2.0;
        const double sigma_min = 0.02;
        const double sigma_scale = 0.3;
        mu = mu_scale * torch::tanh(raw_mu);
        sigma = sigma_min + sigma_scale * torch::softplus(raw_sigma);
        sigma = torch::clamp(sigma, 0.02, 0.3);

        // sample weights + compute logp/entropy on GPU
        auto samp = sample_gaussian_policy(mu, sigma);

        // multicut + reward outside autograd
        torch::Tensor rewards = torch::ones({4}, device);
        {
            torch::NoGradGuard ng;

            torch::Tensor node_labels = rama_torch(i_idx, j_idx, samp.w.detach().contiguous()); // TODO maybe convert to float32

            //rewards = compute_rewards(node_labels);
        }

        // baseline update
        auto b = baseline.update(rewards);

        // advantage [B] on device
        auto adv = (rewards - b).detach();

        // Optional: advantage normalization (often stabilizes)
        adv = (adv - adv.mean()) / (adv.std(/*unbiased=*/false) + 1e-6);

        // loss: -(adv * logp).mean() - entropy_coef * ent.mean()
        auto loss = -(adv * samp.logp).mean() - entropy_coef * samp.ent.mean();

        opt.zero_grad();
        loss.backward();
        opt.step();

        if (step % 100 == 0) {
            auto loss_v = loss.detach().to(torch::kCPU).item<double>();
            auto rmean = rewards.mean().item<double>();
            auto bval  = b.item<double>();
            std::cout << "step=" << step
                        << " loss=" << loss_v
                        << " Rmean=" << rmean
                        << " baseline=" << bval
                        << std::endl;
        }
    }


    torch::save(model, "fcn_" + std::to_string(std::time(0)) + ".pt");

    return 0;
}
