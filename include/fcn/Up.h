#pragma once
#include <torch/torch.h>
#include "DoubleConv.h"

struct UpImpl : torch::nn::Module {
    torch::nn::ConvTranspose2d up{nullptr};
    DoubleConv conv{nullptr};

    UpImpl(int in_ch, int out_ch)
        : up(torch::nn::ConvTranspose2dOptions(in_ch, out_ch, 2).stride(2)),
          conv(in_ch, out_ch)
    {
        register_module("up", up);
        register_module("conv", conv);
    }

    torch::Tensor forward(torch::Tensor x1, torch::Tensor x2) {
        x1 = up->forward(x1);

        // Padding correction
        auto diffY = x2.size(2) - x1.size(2);
        auto diffX = x2.size(3) - x1.size(3);

        x1 = torch::constant_pad_nd(x1, {diffX/2, diffX - diffX/2, diffY/2, diffY - diffY/2});

        auto x = torch::cat({x2, x1}, 1);
        return conv->forward(x);
    }
};

TORCH_MODULE(Up);
