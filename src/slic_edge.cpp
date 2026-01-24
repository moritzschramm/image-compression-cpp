#include "slic_edge.h"
#include <stdexcept>

namespace {
cv::Mat to_bgr_f32_or_throw(const cv::Mat& input) {
    if (input.empty()) throw std::runtime_error("slic_edge_costs: input image is empty");

    cv::Mat img_bgr;
    if (input.channels() == 1) {
        cv::cvtColor(input, img_bgr, cv::COLOR_GRAY2BGR);
    } else if (input.channels() == 3) {
        img_bgr = input;
    } else if (input.channels() == 4) {
        cv::cvtColor(input, img_bgr, cv::COLOR_BGRA2BGR);
    } else {
        throw std::runtime_error("slic_edge_costs: unsupported channel count");
    }

    cv::Mat img;
    if (img_bgr.depth() == CV_8U) {
        img_bgr.convertTo(img, CV_32FC3, 1.0 / 255.0);
    } else if (img_bgr.depth() == CV_16U) {
        img_bgr.convertTo(img, CV_32FC3, 1.0 / 65535.0);
    } else if (img_bgr.depth() == CV_32F) {
        if (img_bgr.type() == CV_32FC3) {
            img = img_bgr;
        } else {
            img_bgr.convertTo(img, CV_32FC3);
        }
    } else {
        throw std::runtime_error("slic_edge_costs: expected 8-bit, 16-bit, or 32-bit float image");
    }

    if (!img.isContinuous()) img = img.clone();
    return img;
}
} // namespace

torch::Tensor slic_edge_costs(
    const cv::Mat& input,
    int region_size,
    float ruler,
    int iters,
    int slic_algorithm
) {
    cv::Mat img = to_bgr_f32_or_throw(input);

    const int H = img.rows;
    const int W = img.cols;

    auto sp = cv::ximgproc::createSuperpixelSLIC(img, slic_algorithm, region_size, ruler);
    sp->iterate(iters);
    sp->enforceLabelConnectivity();

    cv::Mat labels; // CV_32S, HxW
    sp->getLabels(labels);

    auto edges = torch::zeros({2, H, W}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    float* e = edges.data_ptr<float>();
    const int64_t plane = static_cast<int64_t>(H) * static_cast<int64_t>(W);

    // Horizontal edges (channel 0): compare (y,x) vs (y,x+1)
    for (int y = 0; y < H; ++y) {
        const int* row = labels.ptr<int>(y);
        int64_t base = static_cast<int64_t>(y) * W;
        for (int x = 0; x < W - 1; ++x) {
            const float v = (row[x] == row[x + 1]) ? 1.0f : 0.0f;
            e[0 * plane + base + x] = v;
        }
        // last column remains 0 (ignored)
    }

    // Vertical edges (channel 1): compare (y,x) vs (y+1,x)
    for (int y = 0; y < H - 1; ++y) {
        const int* row0 = labels.ptr<int>(y);
        const int* row1 = labels.ptr<int>(y + 1);
        int64_t base = static_cast<int64_t>(y) * W;
        for (int x = 0; x < W; ++x) {
            const float v = (row0[x] == row1[x]) ? 1.0f : 0.0f;
            e[1 * plane + base + x] = v;
        }
    }
    // last row in channel 1 remains 0 (ignored)

    return edges.contiguous();
}
