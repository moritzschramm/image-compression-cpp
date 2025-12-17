#include <c10/core/DeviceType.h>
#include <torch/torch.h>
#include "fcn/EdgeUNet.h"
#include "gaussian_policy.hpp"
#include "ema_baseline.hpp"

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

    for (int step = 0; step < 1e5; ++step) {
        torch::Tensor image = /* TODO load batch */ torch::randn({4, 3, 256, 256}, device);

        // forward: mu, raw_sigma [B,E]
        auto out = model->forward(image);
        torch::Tensor raw_mu = out.first; // TODO
        torch::Tensor raw_sigma = out.second;

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
        /*{
            torch::NoGradGuard ng;

            auto segs = multicut_solve_batch(samp.w.detach());
            rewards = compute_rewards(segs);
        }*/

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
