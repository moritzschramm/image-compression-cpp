#include <filesystem>
#include <torch/torch.h>
#include <vector>
#include "configuration.h"
#include "image_loader.h"
#include "image_writer.h"
#include "fcn/EdgeDataset.h"
#include "fcn/EdgeUNet.h"
#include "png_size_estimator.cuh"


int main()
{
    const auto device = torch::kCUDA;

    auto image_paths = find_image_files_recursively(DATASET_DIR, IMAGE_FORMAT);

    auto train_dataset = EdgeDataset(image_paths, /*create_targets=*/true)
        .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions()
            .batch_size(8)  // make sure images have the same dimensions; otherwise set batch size to 1
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

    torch::nn::SmoothL1Loss criterion(torch::nn::SmoothL1LossOptions().beta(0.1));

    int epochs = 50;

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

            // TODO discard loss at borders?

            /*auto pred_sel = torch::stack(
                { outputs.index({torch::indexing::Slice(), 0, torch::indexing::Slice(), torch::indexing::Slice()}),
                outputs.index({torch::indexing::Slice(), 2, torch::indexing::Slice(), torch::indexing::Slice()}) });

            auto target_sel = torch::stack(
                { targets.index({torch::indexing::Slice(), 0, torch::indexing::Slice(), torch::indexing::Slice()}),
                targets.index({torch::indexing::Slice(), 2, torch::indexing::Slice(), torch::indexing::Slice()}) });*/

            auto loss = criterion(outputs, targets);
            loss.backward();
            optimizer.step();

            total_loss += loss.item<float>();
            batch_count++;

            if (batch_count % 500 == 0)
                std::cout << "Epoch [" << epoch << "/" << epochs
                                  << "] Batch [" << batch_count
                                  << "] Loss: " << loss.item<float>() << std::endl;
        }
        std::cout << "Epoch [" << epoch << "/" << epochs
                              << "] Average Loss: " << (total_loss/batch_count) << std::endl;
    }

    torch::save(model, "fcn_pretrained_" + std::to_string(std::time(0)) + ".pt");
}
