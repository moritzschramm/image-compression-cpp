#include <filesystem>
#include <torch/torch.h>
#include <vector>
#include "configuration.h"
#include "image_loader.h"
#include "image_writer.h"
#include "fcn/EdgeDataset.h"
#include "fcn/EdgeUNet.h"


int main()
{
    torch::manual_seed(0);

    const auto device = torch::kCUDA;

    auto image_paths = find_image_files_recursively(DATASET_DIR, IMAGE_FORMAT);

    auto train_dataset = EdgeDataset(image_paths, /*create_targets=*/true)
        .map(torch::data::transforms::Stack<>());

    const size_t BATCH_SIZE = 8;

    auto train_loader = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions()
            .batch_size(BATCH_SIZE)  // make sure images have the same dimensions; otherwise set batch size to 1
            .workers(4)
            .drop_last(true)
    );

    std::cout << "Loaded pretraining data" << std::endl;

    EdgeUNet model;
    model->to(device);

    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(/*learning_rate=*/1e-3)
    );

    //torch::nn::SmoothL1Loss criterion(torch::nn::SmoothL1LossOptions().beta(0.1));

    int epochs = 1;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        model->train();
        float total_loss = 0.0f;
        int batch_count = 0;

        for (auto& batch : *train_loader) {
            auto imgs = batch.data.to(device, /*non_blocking=*/true);
            auto targets = batch.target.to(device, /*non_blocking=*/true);

            //int64_t sample_id = batch.target.index({0, 0, 0, 0}).to(torch::kCPU).item<int64_t>();
            //std::cout << "id: " << sample_id << std::endl;

            optimizer.zero_grad();

            auto outputs = model->forward(imgs);

            // masks are target channels 4 and 5 -> [B,2,H,W]
            auto m0 = targets.index({torch::indexing::Slice(), 4, torch::indexing::Slice(), torch::indexing::Slice()}).unsqueeze(1); // [B,1,H,W]
            auto m1 = targets.index({torch::indexing::Slice(), 5, torch::indexing::Slice(), torch::indexing::Slice()}).unsqueeze(1); // [B,1,H,W]
            auto mask = torch::cat({m0, m0, m1, m1}, 1);                                  // [B,4,H,W]

            auto t0 = targets.index({torch::indexing::Slice(), 0, torch::indexing::Slice(), torch::indexing::Slice()}).unsqueeze(1);
            auto t1 = targets.index({torch::indexing::Slice(), 1, torch::indexing::Slice(), torch::indexing::Slice()}).unsqueeze(1);
            auto t2 = targets.index({torch::indexing::Slice(), 2, torch::indexing::Slice(), torch::indexing::Slice()}).unsqueeze(1);
            auto t3 = targets.index({torch::indexing::Slice(), 3, torch::indexing::Slice(), torch::indexing::Slice()}).unsqueeze(1);
            auto target = torch::cat({t0, t1, t2, t3}, 1);

            // masked L1 (normalize by valid count)
            auto abs_err = (outputs - target).abs();
            auto denom = mask.sum().clamp_min(1.0);
            auto loss = (abs_err * mask).sum() / denom;

            loss.backward();
            optimizer.step();

            total_loss += loss.item<float>();
            batch_count++;

            if (batch_count == 10000)
                torch::save(model, "fcn_pretrained_canny_" + std::to_string(std::time(0)) + "_epoch_" + std::to_string(epoch) + ".pt");

            if (batch_count % 500 == 0 || batch_count == 1) {
                auto pred_sel = torch::cat({
                    outputs.index({torch::indexing::Slice(), 0, torch::indexing::Slice(), torch::indexing::Slice()}).unsqueeze(1),
                    outputs.index({torch::indexing::Slice(), 2, torch::indexing::Slice(), torch::indexing::Slice()}).unsqueeze(1)
                }, 1); // [B,2,H,W]

                auto target_sel = torch::cat({
                    targets.index({torch::indexing::Slice(), 0, torch::indexing::Slice(), torch::indexing::Slice()}).unsqueeze(1),
                    targets.index({torch::indexing::Slice(), 2, torch::indexing::Slice(), torch::indexing::Slice()}).unsqueeze(1)
                }, 1); // [B,2,H,W]

                double tau = 0.05;  // choose based on target statistics

                auto sign_pred   = torch::sign(pred_sel);
                auto sign_target = torch::sign(target_sel);

                // ignore near-zero targets
                auto valid = target_sel.abs() > tau;

                auto correct =
                    (sign_pred == sign_target).logical_and(valid);

                double sign_accuracy =
                    correct.sum().item<double>() /
                    valid.sum().item<double>();

                auto valid_count = valid.sum().item<int64_t>();
                auto total_count = valid.numel();

                auto t = target_sel.detach();
                auto s = torch::sign(target_sel);
                auto flat = s.flatten();
                auto n_total = flat.numel();

                auto n_neg  = (flat == -1).sum().item<int64_t>();
                auto n_zero = (flat ==  0).sum().item<int64_t>();
                auto n_pos  = (flat ==  1).sum().item<int64_t>();

                std::cout << "sign counts: "
                        << "(-1)=" << n_neg
                        << ", (0)=" << n_zero
                        << ", (+1)=" << n_pos
                        << " / total=" << n_total << "\n";

                std::cout << "Epoch [" << epoch << "/" << epochs
                            << "] Batch [" << batch_count
                            << "] Loss: " << loss.item<float>()
                            << " Sign accuracy: " << sign_accuracy
                            << " valid: " << (double)valid_count / (double)total_count
                            << " target min=" << t.min().item<double>()
                            << " max=" << t.max().item<double>()
                            << " mean=" << t.mean().item<double>()
                            << " std=" << t.std().item<double>() << std::endl;
            }
        }
        std::cout << "Epoch [" << epoch << "/" << epochs
                              << "] Average Loss: " << (total_loss/batch_count) << std::endl;
        torch::save(model, "fcn_pretrained_" + std::to_string(std::time(0)) + "_epoch_" + std::to_string(epoch) + ".pt");
    }

    torch::save(model, "fcn_pretrained_" + std::to_string(std::time(0)) + ".pt");
}
