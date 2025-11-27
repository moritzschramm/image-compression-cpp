#include <c10/core/DeviceType.h>
#include <torch/torch.h>
#include "fcn/EdgeUNet.h"

int main()
{
    const auto device = torch::kCUDA;

    EdgeUNet model;
    torch::load(model, "fcn_pretrained.pt");

    model->to(device);

    // TODO training loop where segment-level scores are used to create edge supervision signals

    torch::save(model, "fcn_" + std::to_string(std::time(0)) + ".pt");

    return 0;
}
