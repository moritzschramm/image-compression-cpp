#pragma once
#include <torch/torch.h>

struct DoubleConvImpl : torch::nn::Module {
    torch::nn::Sequential conv;

    DoubleConvImpl(int in_ch, int out_ch) {
        conv = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_ch, out_ch, 3).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))
        );
        register_module("conv", conv);
    }

    torch::Tensor forward(const torch::Tensor& x) {
        return conv->forward(x);
    }
};

TORCH_MODULE(DoubleConv);
