#pragma once
#include <torch/torch.h>

struct EMABaseline {
    double momentum = 0.99;
    bool initialized = false;
    torch::Tensor value;

    explicit EMABaseline(double m = 0.99) : momentum(m) {}

    torch::Tensor update(const torch::Tensor& rewards) {
        torch::NoGradGuard ng;
        auto mean_r = rewards.mean().detach();
        if (!initialized) {
            value = mean_r.clone();
            initialized = true;
        } else {
            value = value * momentum + mean_r * (1.0 - momentum);
        }
        return value;
    }
};
