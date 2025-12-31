#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <filesystem>
#include <image_loader.h>

torch::Tensor create_target(const cv::Mat& image)
{
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

    torch::Tensor out = torch::zeros({4, H, W}, torch::TensorOptions().dtype(torch::kFloat32));
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

            // hard code sigma value for pretraining
            // right
            if (x + 1 < W) {
                A[0][y][x] = compute_cost(x+1, y);
                A[1][y][x] = 0.1;
            }

            // down
            if (y + 1 < H) {
                A[2][y][x] = compute_cost(x, y+1);
                A[3][y][x] = 0.1;
            }
        }
    }

    return out;
}

struct EdgeDataset : torch::data::Dataset<EdgeDataset> {
    std::vector<std::filesystem::path> image_paths;
    const bool create_targets;

    EdgeDataset(const std::vector<std::filesystem::path>& imgs, bool create_targets)
        : image_paths(imgs), create_targets(create_targets) {}

    torch::data::Example<> get(size_t idx) override {

        cv::Mat img = load_image(image_paths[idx]);

        if (img.channels() == 1)
            cv::cvtColor(img, img, cv::COLOR_GRAY2BGRA);
        else if (img.channels() == 3)
            cv::cvtColor(img, img, cv::COLOR_BGR2BGRA);

        img.convertTo(img, CV_32FC4, 1.0 / 255.0);

        auto input = torch::from_blob(
            img.data, {img.rows, img.cols, 4}, torch::kFloat32
        ).permute({2, 0, 1}).clone();

        torch::Tensor target;

        if (create_targets) {
            target = create_target(img);
        }

        return {input, target};
    }

    torch::optional<size_t> size() const override {
        return image_paths.size();
    }
};
