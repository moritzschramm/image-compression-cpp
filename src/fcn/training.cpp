#include <c10/core/DeviceType.h>
#include <torch/torch.h>
#include "configuration.h"
#include "fcn/EdgeUNet.h"
#include "fcn/EdgeDataset.h"
#include "gaussian_policy.hpp"
#include "ema_baseline.hpp"
#include "image_loader.h"
#include "rama_wrapper.cuh"
#include "compute_rewards.cuh"


torch::Tensor flatten_grid_edges(const torch::Tensor& x)
{
    // x: [B, 4, H, W]
    TORCH_CHECK(x.dim() == 4, "Expected [B, 4, H, W]");

    const int64_t B = x.size(0);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    // horizontal edges: channels 0,1 — drop last column
    auto h = x.slice(1, 0, 2).slice(3, 0, W - 1).contiguous();  // [B, 2, H, W-1]

    // vertical edges: channels 2,3 — drop last row
    auto v = x.slice(1, 2, 4).slice(2, 0, H - 1).contiguous();  // [B, 2, H-1, W]

    // flatten spatial dims
    auto h_flat = h.flatten(2);  // [B, 2, H*(W-1)]
    auto v_flat = v.flatten(2);  // [B, 2, (H-1)*W]

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
    const auto TRAIN_DATASET_SIZE = 1e6;
    const auto VAL_DATASET_SIZE = 64;

    EdgeUNet model;
    torch::load(model, "fcn_pretrained_1768820146_best.pt");

    model->to(device);

    torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(1e-4));

    EMABaseline baseline(0.99);

    const double entropy_coef = 1e-4;

    const int H = 256;
    const int W = 256;

    std::vector<int32_t> i_idx;
    std::vector<int32_t> j_idx;

    build_rama_indices(H, W, i_idx, j_idx); // assuming height and width of 256x256

    auto i_device = torch::tensor(i_idx, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto j_device = torch::tensor(j_idx, torch::TensorOptions().dtype(torch::kInt32).device(device));

    // -------------------------
    // Train dataset loader
    // -------------------------
    auto train_image_paths = find_image_files_recursively(DATASET_DIR, IMAGE_FORMAT);

    if (train_image_paths.size() > TRAIN_DATASET_SIZE) train_image_paths.resize(TRAIN_DATASET_SIZE);

    auto train_dataset = EdgeDataset(train_image_paths, /*create_targets=*/false)
        .map(torch::data::transforms::Stack<>());

    const size_t BATCH_SIZE = 8;

    auto train_loader = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions()
            .batch_size(BATCH_SIZE)
            .workers(4)
            .drop_last(true)
    );

    // -------------------------
    // Val dataset loader
    // -------------------------
    auto val_image_paths = find_image_files_recursively(VAL_DATASET_DIR, IMAGE_FORMAT);

    if (val_image_paths.size() > VAL_DATASET_SIZE) val_image_paths.resize(VAL_DATASET_SIZE);

    auto val_dataset = EdgeDataset(val_image_paths, /*create_targets=*/false)
        .map(torch::data::transforms::Stack<>());

    auto val_loader = torch::data::make_data_loader(
        std::move(val_dataset),
        torch::data::DataLoaderOptions()
            .batch_size(BATCH_SIZE)
            .workers(2)
            .drop_last(false)
    );

    const auto run_id = std::to_string(std::time(nullptr));

    for (int epoch = 0; epoch < 50; ++epoch) {
        model->train();
        int batch_count = 0;

        for (auto& batch : *train_loader) {
            auto images = batch.data.to(device, /*non_blocking=*/true); // batch.data: [B,3,H,W], float32
            auto image_sizes = batch.target.to(device, /*non_blocking=*/true);

            // forward: raw_mu, raw_sigma [B,E]
            // first and second output channel: horizontal edges
            // third and fourth output channel: vertical edges
            // flatten edges to go from [B,4,H,W] -> [B,2,E] -> 2x [B,E]
            auto out = model->forward(images);
            auto flat = flatten_grid_edges(out);

            torch::Tensor raw_mu = flat.select(1, 0);
            torch::Tensor raw_sigma = flat.select(1, 1);

            const double mu_scale = 2.0;
            const double sigma_min = 0.1;
            const double sigma_max = 0.9;
            auto mu = mu_scale * torch::tanh(0.5 * raw_mu);
            auto sigma = sigma_min + (sigma_max - sigma_min) * torch::sigmoid(raw_sigma);

            // sample weights + compute logp/entropy on GPU
            auto samp = sample_gaussian_policy(mu, sigma);

            // multicut + reward outside autograd
            torch::Tensor rewards = torch::empty({BATCH_SIZE}, images.options().dtype(torch::kFloat32));
            {
                torch::NoGradGuard ng;

                auto edge_costs = samp.w.detach().to(torch::kFloat32).contiguous();
                torch::Tensor node_labels = rama_torch_batched(i_device, j_device, edge_costs);

                const int64_t B = node_labels.size(0);
                node_labels = node_labels.view({B, H, W}).contiguous();

                rewards = compute_rewards_batched(images, node_labels, image_sizes);
            }

            // baseline update
            auto b = baseline.update(rewards).to(rewards.device());
            auto adv = (rewards - b).detach();

            adv = (adv - adv.mean()) / (adv.std(false).clamp_min(1e-6));

            opt.zero_grad();

            const double E = static_cast<double>(mu.size(1));
            auto loss = -(adv * (samp.logp / E)).mean() - entropy_coef * (samp.entropy / E).mean();

            loss.backward();

            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);

            opt.step();

            batch_count++;

            if (batch_count % 100 == 0) {
                auto loss_v = loss.detach().to(torch::kCPU).item<double>();
                auto rmean = rewards.mean().item<double>();
                auto bval  = b.item<double>();
                std::cout << "step=" << batch_count
                            << " loss=" << loss_v
                            << " Rmean=" << rmean
                            << " baseline=" << bval
                            << std::endl;
                model->eval();
                torch::NoGradGuard ng;

                double rsum = 0.0;
                int n = 0;

                for (auto& batch : *val_loader) {
                    auto images = batch.data.to(device, /*non_blocking=*/true);
                    auto image_sizes = batch.target.to(device, /*non_blocking=*/true);

                    auto out = model->forward(images);
                    auto flat = flatten_grid_edges(out);
                    auto mu = mu_scale * torch::tanh(0.5 * flat.select(1,0));

                    auto edge_costs = mu.to(torch::kFloat32).contiguous();
                    torch::Tensor node_labels = rama_torch_batched(i_device, j_device, edge_costs);
                    const int64_t B = node_labels.size(0);
                    node_labels = node_labels.view({B, H, W}).contiguous();
                    rewards = compute_rewards_batched(images, node_labels, image_sizes);

                    rsum += rewards.sum().item<double>();
                    n += images.size(0);
                }

                std::cout << "Eval reward mean=" << (rsum / std::max(1,n)) << "\n";
                torch::save(model, "fcn_training_" + run_id + ".pt");
                model->train();
            }
        }
    }


    torch::save(model, "fcn_training_" + run_id + "_final.pt");

    return 0;
}
