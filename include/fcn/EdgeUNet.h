#pragma once
#include <torch/torch.h>
#include "DoubleConv.h"
#include "Down.h"
#include "Up.h"

struct EdgeUNetImpl : torch::nn::Module {
    DoubleConv inc{nullptr};
    Down down1{nullptr}, down2{nullptr};
    Up up1{nullptr}, up2{nullptr};
    torch::nn::Conv2d outc{nullptr};

    EdgeUNetImpl(int in_channels = 4, int edge_channels = 2)
        : inc(in_channels, 64),
          down1(64, 128),
          down2(128, 256),
          up1(256, 128),
          up2(128, 64),
          outc(torch::nn::Conv2dOptions(64, edge_channels, 1))
    {
        register_module("inc", inc);
        register_module("down1", down1);
        register_module("down2", down2);
        register_module("up1", up1);
        register_module("up2", up2);
        register_module("outc", outc);
    }

    torch::Tensor forward(const torch::Tensor& x) {
        auto x1 = inc->forward(x);
        auto x2 = down1->forward(x1);
        auto x3 = down2->forward(x2);

        auto u1 = up1->forward(x3, x2);
        auto u2 = up2->forward(u1, x1);

        return outc->forward(u2);
    }
};

TORCH_MODULE(EdgeUNet);
