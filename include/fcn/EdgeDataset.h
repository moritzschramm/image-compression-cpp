#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <cstdint>
#include <system_error>
#include <stdexcept>
#include <vector>
#include "image_loader.h"
#include "canny_edge.hpp"

// Target layout (C,H,W):
// 0: cost_right   (learned)  [-1,1]
// 1: sigma_right  (fixed)    0.1
// 2: cost_down    (learned)  [-1,1]
// 3: sigma_down   (fixed)    0.1
// 4: mask_right   (1 if x+1<W else 0)
// 5: mask_down    (1 if y+1<H else 0)

static inline float luma_bgr01(const cv::Vec3f& bgr01) {
    // OpenCV BGR order
    return 0.114f * bgr01[0] + 0.587f * bgr01[1] + 0.299f * bgr01[2]; // in [0,1]
}

torch::Tensor create_target_with_mask(const cv::Mat& img) {

    const int H = img.rows;
    const int W = img.cols;

    torch::Tensor edges = canny_edge_costs(img);
    auto E = edges.accessor<int8_t, 3>();

    torch::Tensor out = torch::zeros({6, H, W}, torch::TensorOptions().dtype(torch::kFloat32));
    auto A = out.accessor<float, 3>();

    // sigma fixed everywhere (mask will exclude borders)
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            if (x + 1 < W) {
                A[0][y][x] = static_cast<float>(E[0][y][x]);    // mu horizontal
                A[1][y][x] = 0.1f;                              // sigma horizontal
                A[4][y][x] = 1.0f;                              // mask
            }
            if (y + 1 < H) {
                A[2][y][x] = static_cast<float>(E[1][y][x]);    // mu vertical
                A[3][y][x] = 0.1f;                              // sigma vertical
                A[5][y][x] = 1.0f;                              // mask
            }
        }
    }

    /*for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const cv::Vec3f& p0 = img.at<cv::Vec3f>(y, x);
            const float L0 = luma_bgr01(p0);

            // right neighbor
            if (x + 1 < W) {
                const cv::Vec3f& p1 = img.at<cv::Vec3f>(y, x + 1);
                const float L1 = luma_bgr01(p1);

                float d = L1 - L0;                 // signed, in [-1,1]
                d = std::max(-1.0f, std::min(1.0f, d));

                A[0][y][x] = d;
                A[4][y][x] = 1.0f;                 // mask_right
            }

            // down neighbor
            if (y + 1 < H) {
                const cv::Vec3f& p1 = img.at<cv::Vec3f>(y + 1, x);
                const float L1 = luma_bgr01(p1);

                float d = L1 - L0;
                d = std::max(-1.0f, std::min(1.0f, d));

                A[2][y][x] = d;
                A[5][y][x] = 1.0f;                 // mask_down
            }
        }
    }*/

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
        ).permute({2, 0, 1}).clone();

        torch::Tensor target;
        if (create_targets) {
            target = create_target_with_mask(img); // [6,H,W]
        } else {
            // file size of image
            target = torch::tensor({(int)file_size_bytes(image_paths[idx])}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
        }

        return {input, target};
    }

    torch::optional<size_t> size() const override {
        return image_paths.size();
    }
};
