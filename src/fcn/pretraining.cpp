#include <torch/torch.h>
#include "configuration.h"
#include "image_loader.h"
#include "fcn/EdgeDatset.h"
#include "fcn/EdgeUNet.h"

int main()
{
    auto image_paths = find_image_files_recursively(DATASET_DIR, IMAGE_FORMAT);
    std::vector<std::string> train_targets; // TODO

    auto train_dataset = EdgeDataset(image_paths, train_targets)
        .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions()
            .batch_size(2)
            .workers(4)
    );

    EdgeUNet model(/*in_channels=*/3, /*edge_channels=*/4);
    model->to(torch::kCUDA);

    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(/*learning_rate=*/1e-3)
    );

    torch::nn::L1Loss criterion;

    int epochs = 50;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        model->train();
        float total_loss = 0.0f;
        int batch_count = 0;

        for (auto& batch : *train_loader) {
            auto imgs = batch.data.to(torch::kCUDA);
            auto tgts = batch.target.to(torch::kCUDA);

            optimizer.zero_grad();

            auto outputs = model->forward(imgs);

            auto loss = criterion(outputs, tgts);
            loss.backward();
            optimizer.step();

            total_loss += loss.item<float>();
            batch_count++;
        }

        std::cout << "Epoch " << epoch
                  << " | Loss: " << (total_loss / batch_count)
                  << std::endl;
    }
}
