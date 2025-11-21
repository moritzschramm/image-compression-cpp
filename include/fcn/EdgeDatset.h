#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <filesystem>

struct EdgeDataset : torch::data::Dataset<EdgeDataset> {
    std::vector<std::filesystem::path> images;
    std::vector<std::string> targets;

    EdgeDataset(const std::vector<std::filesystem::path>& imgs,
                const std::vector<std::string>& tgts)
        : images(imgs), targets(tgts) {}

    torch::data::Example<> get(size_t idx) override {
        // load image using OpenCV (RGB normalized)
        cv::Mat img = cv::imread(images[idx], cv::IMREAD_COLOR);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        img.convertTo(img, CV_32F, 1.0 / 255.0);

        torch::Tensor input = torch::from_blob(
            img.data, {img.rows, img.cols, 3}, torch::kFloat
        ).permute({2, 0, 1}).clone(); // (3,H,W)

        // load target tensor
        torch::Tensor target;
        torch::load(target, targets[idx]);

        return {input, target};
    }

    torch::optional<size_t> size() const override {
        return images.size();
    }
};
