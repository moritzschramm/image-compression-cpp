#include <torch/torch.h>
#include "fcn/EdgeUNet.h"

int main() {
    //torch::Device device(torch::kCUDA);

    EdgeUNet model(3, 4);
    //model->to(device);

    auto input = torch::rand({1, 3, 256, 256});//.to(device);
    auto output = model->forward(input);

    std::cout << "Output shape: " << output.sizes() << std::endl;
}
