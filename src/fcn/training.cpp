#include <c10/core/DeviceType.h>
#include <torch/torch.h>
#include "configuration.h"
#include "fcn/EdgeUNet.h"
#include "fcn/EdgeDataset.h"
#include "gaussian_policy.hpp"
#include "ema_baseline.hpp"
#include "image_loader.h"
#include "rama_wrapper.cuh"
#include "png_size_estimator.cuh"
#include "compute_rewards.hpp"

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
    torch::manual_seed(0);

    const auto device = torch::kCUDA;

    EdgeUNet model;
    torch::load(model, "fcn_pretrained_1767799034_epoch_1.pt");

    model->to(device);

    torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(1e-4));

    EMABaseline baseline(0.99);

    const double entropy_coef = 1e-4;

    std::vector<int32_t> i_idx;
    std::vector<int32_t> j_idx;

    build_rama_indices(256, 256, i_idx, j_idx); // assuming height and width of 256x256

    auto i_device = torch::tensor(i_idx, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto j_device = torch::tensor(j_idx, torch::TensorOptions().dtype(torch::kInt32).device(device));

    auto image_paths = find_image_files_recursively(DATASET_DIR, IMAGE_FORMAT);

    auto train_dataset = EdgeDataset(image_paths, /*create_targets=*/false)
        .map(torch::data::transforms::Stack<>());

    const size_t BATCH_SIZE = 8;

    auto train_loader = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions()
            .batch_size(BATCH_SIZE)
            .workers(4)
            .drop_last(true)
    );

    for (int epoch = 0; epoch < 50; ++epoch) {
        model->train();
        int batch_count = 0;

        for (auto& batch : *train_loader) {
            // batch.data: [B,4,H,W], float32
            auto images = batch.data.to(device, /*non_blocking=*/true);
            auto image_sizes = batch.target;

            // forward: raw_mu, raw_sigma [B,E]
            // first and second output channel: horizontal edges
            // third and fourth output channel: vertical edges
            // flatten edges to go from [B,4,H,W] -> [B,2,E] -> 2x [B,E]
            auto out = model->forward(images);
            auto flat = flatten_grid_edges(out);

            torch::Tensor raw_mu = flat.select(1, 0);
            torch::Tensor raw_sigma = flat.select(1, 1);

            const double mu_scale = 2.0;
            const double sigma_min = 0.02;
            const double sigma_scale = 0.3;
            auto mu = mu_scale * torch::tanh(raw_mu);
            auto sigma = sigma_min + sigma_scale * torch::softplus(raw_sigma);
            sigma = torch::clamp(sigma, 0.02, 0.3);

            // sample weights + compute logp/entropy on GPU
            auto samp = sample_gaussian_policy(mu, sigma);

            // multicut + reward outside autograd
            torch::Tensor rewards = torch::empty({BATCH_SIZE}, images.options().dtype(torch::kFloat32));
            {
                torch::NoGradGuard ng;

                for (int i = 0; i < BATCH_SIZE; ++i) {
                    auto edge_costs = samp.w[i].detach().to(torch::kFloat32).contiguous();
                    torch::Tensor node_labels = rama_torch(i_device, j_device, edge_costs);
                    rewards[i] = compute_rewards(images[i], node_labels, image_sizes[i][0].item<int>());
                }
            }

            // baseline update
            auto b = baseline.update(rewards);

            // advantage [B] on device
            auto adv = (rewards - b).detach();

            adv = (adv - adv.mean()) / (adv.std(/*unbiased=*/false) + 1e-6);

            // loss: -(adv * logp).mean() - entropy_coef * ent.mean()
            auto loss = -(adv * samp.logp).mean() - entropy_coef * samp.entropy.mean();

            opt.zero_grad();
            loss.backward();
            opt.step();

            batch_count++;

            //if (batch_count % 100 == 0) {
                auto loss_v = loss.detach().to(torch::kCPU).item<double>();
                auto rmean = rewards.mean().item<double>();
                auto bval  = b.item<double>();
                std::cout << "step=" << batch_count
                            << " loss=" << loss_v
                            << " Rmean=" << rmean
                            << " baseline=" << bval
                            << std::endl;
            //}
            if (batch_count % 1000)
                torch::save(model, "fcn_training_" + std::to_string(std::time(0)) + ".pt");
        }
    }


    torch::save(model, "fcn_" + std::to_string(std::time(0)) + ".pt");

    return 0;
}
