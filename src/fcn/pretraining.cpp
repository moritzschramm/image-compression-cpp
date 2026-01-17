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

    torch::optim::AdamW optimizer(
        model->parameters(),
        torch::optim::AdamWOptions(1e-3).weight_decay(1e-4)
    );

    int epochs = 10;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        model->train();
        float total_loss = 0.0f;
        int batch_count = 0;

        for (auto& batch : *train_loader) {
            auto imgs = batch.data.to(device, /*non_blocking=*/true);
            auto targets = batch.target.to(device, /*non_blocking=*/true);

            optimizer.zero_grad();

            auto outputs = model->forward(imgs);

            using torch::indexing::Slice;

            auto target = targets.index({Slice(), Slice(0, 4), Slice(), Slice()});   // [B,4,H,W]
            auto m = targets.index({Slice(), Slice(4, 6), Slice(), Slice()});        // [B,2,H,W]
            auto mask = m.repeat_interleave(2, /*dim=*/1);                           // [B,4,H,W]
            mask = mask.to(outputs.dtype());

            auto loss_map = torch::smooth_l1_loss(outputs, target, torch::nn::functional::SmoothL1LossFuncOptions().reduction(torch::kNone));
            auto loss = (loss_map * mask).sum() / mask.sum().clamp_min(1.0);

            loss.backward();
            optimizer.step();

            total_loss += loss.item<float>();
            batch_count++;

            if (batch_count % 500 == 0 || batch_count == 1) {
                torch::NoGradGuard ng;

                auto pred_sel = outputs.index({Slice(), Slice(0,4), Slice(), Slice()})
                                           .index({Slice(), torch::tensor({0,2}, torch::kLong).to(outputs.device()), Slice(), Slice()});

                auto target_sel = targets.index({Slice(), Slice(0,4), Slice(), Slice()})
                                            .index({Slice(), torch::tensor({0,2}, torch::kLong).to(targets.device()), Slice(), Slice()});

                const double tau = 0.05;
                auto valid = target_sel.abs() > tau;
                auto valid_count = valid.sum().item<double>();
                double sign_accuracy = 0.0;
                if (valid_count > 0) {
                    auto correct = (torch::sign(pred_sel) == torch::sign(target_sel)).logical_and(valid);
                    sign_accuracy = correct.sum().item<double>() / valid_count;
                }

                //auto t = target_sel.detach();
                //auto s = torch::sign(target_sel);

                std::cout << "Epoch [" << epoch << "/" << epochs
                            << "] Batch [" << batch_count
                            << "] Loss: " << loss.item<float>()
                            << " Sign accuracy: " << sign_accuracy << std::endl;
                            //<< " target min=" << t.min().item<double>()
                            //<< " max=" << t.max().item<double>()
                            //<< " mean=" << t.mean().item<double>()
                            //<< " std=" << t.std().item<double>() << std::endl;
            }
        }
        std::cout << "Epoch [" << epoch << "/" << epochs
                              << "] Average Loss: " << (total_loss/batch_count) << std::endl;
        torch::save(model, "fcn_pretrained_" + std::to_string(std::time(0)) + "_epoch_" + std::to_string(epoch) + ".pt");
    }

    torch::save(model, "fcn_pretrained_" + std::to_string(std::time(0)) + ".pt");
}
