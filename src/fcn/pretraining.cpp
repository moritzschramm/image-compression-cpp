#include <filesystem>
#include <torch/torch.h>
#include <vector>
#include "configuration.h"
#include "image_loader.h"
#include "pattern_generator.h"
#include "fcn/EdgeDatset.h"
#include "fcn/EdgeUNet.h"

void create_target(const std::filesystem::path& target_path, const std::filesystem::path& image_path)
{
    cv::Mat image = load_image(image_path);

    std::cout << "Loading image for target creation: " << image_path.relative_path() << " with image type: " << image.type() << std::endl;

    CV_Assert(!image.empty());
    CV_Assert(image.channels() == 1 || image.channels() == 3 || image.channels() == 4);

    const int H = image.rows;
    const int W = image.cols;

    cv::Mat work;
    if (image.depth() != CV_8U)
        image.convertTo(work, CV_8U, 1.0 / 256.0);  // scaling for 16-bit → 8-bit
    else
        work = image;

    if (work.channels() == 1)
        cv::cvtColor(work, work, cv::COLOR_GRAY2RGBA);
    else if (work.channels() == 3)
        cv::cvtColor(work, work, cv::COLOR_BGR2RGBA);

    // max diff = sum over channels of max abs difference (255)
    const float max_raw_diff = 255.0f * image.channels();

    torch::Tensor out = torch::zeros({2, H, W}, torch::TensorOptions().dtype(torch::kFloat32));
    auto A = out.accessor<float, 3>();

    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            // Read current pixel
            float I0[4] = {0,0,0,0};
            const cv::Vec4b& v = work.at<cv::Vec4b>(y,x);
            I0[0] = v[0]; I0[1] = v[1]; I0[2] = v[2]; I0[3] = v[3];

            auto compute_cost = [&](int nx, int ny) -> float {
                float I1[4] = {0,0,0,0};
                const cv::Vec4b& v = work.at<cv::Vec4b>(ny,nx);
                I1[0] = v[0]; I1[1] = v[1]; I1[2] = v[2]; I1[3] = v[3];

                float diff = 0.f;
                for (int c = 0; c < work.channels(); ++c)
                    diff += std::abs(I0[c] - I1[c]);

                float aff  = 1.0f - (diff / max_raw_diff);   // [0,1]
                float cost = aff * 2.0f - 1.0f;              // [-1,1]
                return cost;
            };

            // right
            if (x + 1 < W) A[0][y][x] = compute_cost(x+1, y);

            // down
            if (y + 1 < H) A[1][y][x] = compute_cost(x, y+1);

            // for now, only use 2 output channels, so undirected edges
            // left
            //if (x - 1 >= 0) A[1][y][x] = compute_cost(x-1, y);
            // up
            //if (y - 1 >= 0) A[3][y][x] = compute_cost(x, y-1);
        }
    }

    torch::save(out, target_path);
}

void create_random_patterns()
{
    if (!std::filesystem::exists(CACHE_DIR / "random_patterns")) {
        std::filesystem::create_directories(CACHE_DIR / "random_patterns");
    }

    auto write_random_image = [&](size_t idx, std::function<cv::Mat(int,int,bool)> random_pattern_generator, int w, int h, bool alpha) -> void {

        auto target_path = CACHE_DIR / "random_patterns" / (std::to_string(idx) + ".png");

        if (!std::filesystem::exists(target_path)) {
            cv::imwrite(target_path, random_pattern_generator(w, h, alpha), {cv::IMWRITE_PNG_COMPRESSION, COMPRESSION_LEVEL});
            std::cout << "created random pattern " << target_path << std::endl;
        }
    };

    int w = 1024, h = 1024;
    size_t idx = 0;
    size_t batch_size = 100;
    for (; idx < batch_size; idx++) write_random_image(idx, generate_repetition_pattern, w, h, true);
    for (; idx < batch_size*2; idx++) write_random_image(idx, generate_repetition_pattern, w, h, false);
    for (; idx < batch_size*3; idx++) write_random_image(idx, generate_monochrome_region, w, h, true);
    for (; idx < batch_size*4; idx++) write_random_image(idx, generate_monochrome_region, w, h, false);
    for (; idx < batch_size*5; idx++) write_random_image(idx, generate_low_variance_noise, w, h, true);
    for (; idx < batch_size*6; idx++) write_random_image(idx, generate_low_variance_noise, w, h, false);
    for (; idx < batch_size*7; idx++) write_random_image(idx, generate_low_frequency_noise, w, h, true);
    for (; idx < batch_size*8; idx++) write_random_image(idx, generate_low_frequency_noise, w, h, false);
    for (; idx < batch_size*9; idx++) write_random_image(idx, generate_random_row_copies, w, h, true);
    for (; idx < batch_size*10; idx++) write_random_image(idx, generate_random_row_copies, w, h, false);
}

std::vector<std::string> find_or_create_targets(const std::vector<std::filesystem::path>& image_paths)
{
    std::vector<std::string> target_paths;
    for (const auto& image_path : image_paths)
    {
        auto relative_path = image_path.lexically_relative(DATASET_DIR);
        if (relative_path.empty())
            throw std::runtime_error("Path does not start with DATASET_DIR");

        if (!std::filesystem::exists(CACHE_DIR)) {
            std::filesystem::create_directories(CACHE_DIR);
        }

        auto target_path = CACHE_DIR / relative_path.replace_extension(".pt");

        if (!std::filesystem::exists(target_path)) {
            std::filesystem::create_directories(target_path.parent_path());
            create_target(target_path, image_path);
        }

        target_paths.push_back(target_path);
    }
    return target_paths;
}

int main()
{
    const auto device = torch::kCPU;//torch::kCUDA;

    create_random_patterns();

    auto image_paths = find_image_files_recursively(DATASET_DIR, IMAGE_FORMAT);
    std::vector<std::string> train_targets = find_or_create_targets(image_paths);

    auto train_dataset = EdgeDataset(image_paths, train_targets)
        .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions()
            .batch_size(1)
            .workers(4)
    );

    std::cout << "Loaded pretraining data" << std::endl;

    EdgeUNet model;
    model->to(device);

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
            auto imgs = batch.data.to(device);
            auto tgts = batch.target.to(device);

            optimizer.zero_grad();

            auto outputs = model->forward(imgs);

            auto loss = criterion(outputs, tgts);
            loss.backward();
            optimizer.step();

            total_loss += loss.item<float>();
            batch_count++;

            std::cout << "Epoch [" << epoch << "/" << epochs
                                  << "] Batch [" << batch_count
                                  << "] Loss: " << loss.item<float>() << std::endl;
        }
        std::cout << "Epoch [" << epoch << "/" << epochs
                              << "] Average Loss: " << (total_loss/batch_count) << std::endl;
    }

    torch::save(model, "fcn_pretrained_" + std::to_string(std::time(0)) + ".pt");
}
