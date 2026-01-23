#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <cstdint>
#include <system_error>
#include <stdexcept>
#include <vector>
#include "image_loader.h"
#include "configuration.h"
#include "slic_edge.hpp"
#include "canny_edge.hpp"
#include "graph_based_edge.hpp"
#include "watershed_edge.hpp"

// Target layout (C,H,W):
// 0: cost_right   (learned)  {0,1} where 1 = connect
// 1: cost_down    (learned)  {0,1} where 1 = connect
// 2: mask_right   (1 if x+1<W else 0)
// 3: mask_down    (1 if y+1<H else 0)

inline torch::Tensor compute_edge_costs(const cv::Mat& img) {
    switch (EDGE_TARGET) {
        case EdgeTargetType::SLIC:
            return slic_edge_costs(img);
        case EdgeTargetType::CANNY:
            return canny_edge_costs(img);
        case EdgeTargetType::GRAPH:
            return graph_based_edge_costs(img);
        case EdgeTargetType::WATERSHED:
            return watershed_edge_costs(img);
        default:
            return slic_edge_costs(img);
    }
}

torch::Tensor create_target_with_mask(const cv::Mat& img) {

    const int H = img.rows;
    const int W = img.cols;

    torch::Tensor edges = compute_edge_costs(img);

    torch::Tensor out = torch::zeros({4, H, W}, torch::TensorOptions().dtype(torch::kFloat32));

    // cost
    out.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice(0, W-1)},
                    edges.index({0, torch::indexing::Slice(), torch::indexing::Slice(0, W-1)}));
    out.index_put_({1, torch::indexing::Slice(0, H-1), torch::indexing::Slice()},
                    edges.index({1, torch::indexing::Slice(0, H-1), torch::indexing::Slice()}));

    // masks
    out.index_put_({2, torch::indexing::Slice(), torch::indexing::Slice(0, W-1)}, 1.0f);
    out.index_put_({3, torch::indexing::Slice(0, H-1), torch::indexing::Slice()}, 1.0f);

    return out;
}

static cv::Mat to_f32c3_01_or_throw(const cv::Mat& img_any) {
    CV_Assert(!img_any.empty());
    CV_Assert(img_any.channels() == 1 || img_any.channels() == 3 || img_any.channels() == 4);

    cv::Mat img3;
    if (img_any.channels() == 1) {
        cv::cvtColor(img_any, img3, cv::COLOR_GRAY2BGR);
    } else if (img_any.channels() == 4) {
        cv::cvtColor(img_any, img3, cv::COLOR_BGRA2BGR);
    } else { // 3 channels
        img3 = img_any;
    }

    cv::Mat img_f;
    if (img3.depth() == CV_8U) {
        img3.convertTo(img_f, CV_32FC3, 1.0 / 255.0);
    } else if (img3.depth() == CV_16U) {
        img3.convertTo(img_f, CV_32FC3, 1.0 / 65535.0);
    } else if (img3.depth() == CV_32F) {
        CV_Assert(img3.type() == CV_32FC3);
        img_f = img3;
    } else {
        CV_Assert(false && "Unsupported image depth. Use 8U, 16U, or 32F.");
    }
    return img_f;
}

std::uintmax_t file_size_bytes(const std::filesystem::path& p) {
    std::error_code ec;
    const auto sz = std::filesystem::file_size(p, ec);
    if (ec) {
        throw std::runtime_error("file_size failed for '" + p.string() + "': " + ec.message());
    }
    return sz; // bytes
}

struct EdgeDataset : torch::data::Dataset<EdgeDataset> {
    std::vector<std::filesystem::path> image_paths;
    const bool create_targets;

    EdgeDataset(const std::vector<std::filesystem::path>& imgs, bool create_targets)
        : image_paths(imgs), create_targets(create_targets) {}

    torch::data::Example<> get(size_t idx) override {
        cv::Mat img = load_image(image_paths[idx]);

        cv::Mat img_f = to_f32c3_01_or_throw(img);

        // Input tensor: [3,H,W]
        auto input = torch::from_blob(
            img_f.data, {img_f.rows, img_f.cols, 3}, torch::kFloat32
        ).permute({2, 0, 1}).contiguous().clone();

        torch::Tensor target;
        if (create_targets) {
            target = create_target_with_mask(img); // [4,H,W]
        } else {
            // file size of image
            const double sz = static_cast<double>(file_size_bytes(image_paths[idx]));
            target = torch::scalar_tensor(sz, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
        }

        input = input.pin_memory();
        target = target.pin_memory();

        return {input, target};
    }

    torch::optional<size_t> size() const override {
        return image_paths.size();
    }
};
