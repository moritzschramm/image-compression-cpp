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

        cv::Mat img = cv::imread(images[idx], cv::IMREAD_UNCHANGED);

        if (img.channels() == 1)
            cv::cvtColor(img, img, cv::COLOR_GRAY2BGRA);
        else if (img.channels() == 3)
            cv::cvtColor(img, img, cv::COLOR_BGR2BGRA);

        img.convertTo(img, CV_32FC4, 1.0 / 255.0);

        auto input = torch::from_blob(
            img.data, {img.rows, img.cols, 4}, torch::kFloat32
        ).permute({2, 0, 1}).clone();

        // load target tensor
        torch::Tensor target;
        torch::load(target, targets[idx]);

        return {input, target};
    }

    torch::optional<size_t> size() const override {
        return images.size();
    }
};
