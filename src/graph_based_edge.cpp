#include "graph_based_edge.h"
#include <stdexcept>

namespace {
cv::Mat to_bgr_u8_or_throw(const cv::Mat& img_any) {
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

    cv::Mat img_u8;
    if (img3.depth() == CV_8U) {
        img_u8 = img3;
    } else if (img3.depth() == CV_16U) {
        img3.convertTo(img_u8, CV_8UC3, 1.0 / 257.0); // 65535/255 ≈ 257
    } else if (img3.depth() == CV_32F) {
        double minv = 0.0, maxv = 0.0;
        cv::minMaxLoc(img3, &minv, &maxv);
        if (maxv <= 1.0 + 1e-6 && minv >= 0.0 - 1e-6) {
            img3.convertTo(img_u8, CV_8UC3, 255.0);
        } else if (maxv > minv) {
            const double scale = 255.0 / (maxv - minv);
            const double shift = -minv * scale;
            img3.convertTo(img_u8, CV_8UC3, scale, shift);
        } else {
            img_u8 = cv::Mat(img3.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        }
    } else {
        CV_Assert(false && "Unsupported image depth. Use 8U, 16U, or 32F.");
    }

    if (!img_u8.isContinuous()) img_u8 = img_u8.clone();
    return img_u8;
}
} // namespace

torch::Tensor graph_based_edge_costs(
    const cv::Mat& input,
    float sigma,
    float k,
    int min_size
) {
    cv::Mat img_u8 = to_bgr_u8_or_throw(input);

    const int H = img_u8.rows;
    const int W = img_u8.cols;

    auto segmenter = cv::ximgproc::segmentation::createGraphSegmentation(sigma, k, min_size);
    cv::Mat labels;
    segmenter->processImage(img_u8, labels);

    cv::Mat labels32;
    if (labels.depth() == CV_32S) {
        labels32 = labels;
    } else {
        labels.convertTo(labels32, CV_32S);
    }

    auto edges = torch::zeros({2, H, W}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    float* e = edges.data_ptr<float>();
    const int64_t plane = static_cast<int64_t>(H) * static_cast<int64_t>(W);

    // Horizontal edges (channel 0): compare (y,x) vs (y,x+1)
    for (int y = 0; y < H; ++y) {
        const int* row = labels32.ptr<int>(y);
        int64_t base = static_cast<int64_t>(y) * W;
        for (int x = 0; x < W - 1; ++x) {
            const float v = (row[x] == row[x + 1]) ? 1.0f : 0.0f;
            e[0 * plane + base + x] = v;
        }
        // last column remains 0 (ignored)
    }

    // Vertical edges (channel 1): compare (y,x) vs (y+1,x)
    for (int y = 0; y < H - 1; ++y) {
        const int* row0 = labels32.ptr<int>(y);
        const int* row1 = labels32.ptr<int>(y + 1);
        int64_t base = static_cast<int64_t>(y) * W;
        for (int x = 0; x < W; ++x) {
            const float v = (row0[x] == row1[x]) ? 1.0f : 0.0f;
            e[1 * plane + base + x] = v;
        }
    }
    // last row in channel 1 remains 0 (ignored)

    return edges.contiguous();
}
