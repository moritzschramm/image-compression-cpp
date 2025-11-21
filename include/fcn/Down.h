#pragma once
#include <torch/torch.h>
#include "DoubleConv.h"

struct DownImpl : torch::nn::Module {
    torch::nn::Sequential down;

    DownImpl(int in_ch, int out_ch) {
        down = torch::nn::Sequential(
            torch::nn::MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        );
        register_module("down", down);
    }

    torch::Tensor forward(const torch::Tensor& x) {
        return down->forward(x);
    }
};

TORCH_MODULE(Down);
