#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

static inline cv::Mat to_bgr_u8_or_throw_ws(const cv::Mat& img_any) {
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

// Output: edges [2, H, W] (float32, CPU)
//   edges[0, y, x] = horizontal edge between (y,x) and (y,x+1) for x in [0, W-2]
//   edges[1, y, x] = vertical   edge between (y,x) and (y+1,x) for y in [0, H-2]
// Values: 1.0 for connect (same segment), 0.0 for cut (segment boundary)
inline torch::Tensor watershed_edge_costs(
    const cv::Mat& input,
    int seed_stride = 16,
    int blur_ksize = 3,
    double blur_sigma = 1.0
) {
    CV_Assert(seed_stride >= 2);

    cv::Mat img_u8 = to_bgr_u8_or_throw_ws(input);
    if (blur_ksize >= 3 && (blur_ksize % 2 == 1)) {
        cv::GaussianBlur(img_u8, img_u8, cv::Size(blur_ksize, blur_ksize), blur_sigma);
    }

    const int H = img_u8.rows;
    const int W = img_u8.cols;

    // Seed markers on a regular grid to get superpixel-like regions.
    cv::Mat markers = cv::Mat::zeros(H, W, CV_32S);
    int label = 0;
    const int y0 = seed_stride / 2;
    const int x0 = seed_stride / 2;
    for (int y = y0; y < H; y += seed_stride) {
        int* row = markers.ptr<int>(y);
        for (int x = x0; x < W; x += seed_stride) {
            row[x] = ++label;
        }
    }

    // If the image is tiny, ensure at least one marker.
    if (label == 0) {
        markers.at<int>(H / 2, W / 2) = 1;
    }

    cv::watershed(img_u8, markers);

    auto edges = torch::zeros({2, H, W}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    float* e = edges.data_ptr<float>();
    const int64_t plane = static_cast<int64_t>(H) * static_cast<int64_t>(W);

    // Horizontal edges (channel 0)
    for (int y = 0; y < H; ++y) {
        const int* row = markers.ptr<int>(y);
        int64_t base = static_cast<int64_t>(y) * W;
        for (int x = 0; x < W - 1; ++x) {
            const int a = row[x];
            const int b = row[x + 1];
            const float v = (a > 0 && b > 0 && a == b) ? 1.0f : 0.0f;
            e[0 * plane + base + x] = v;
        }
        // last column remains 0 (ignored)
    }

    // Vertical edges (channel 1)
    for (int y = 0; y < H - 1; ++y) {
        const int* row0 = markers.ptr<int>(y);
        const int* row1 = markers.ptr<int>(y + 1);
        int64_t base = static_cast<int64_t>(y) * W;
        for (int x = 0; x < W; ++x) {
            const int a = row0[x];
            const int b = row1[x];
            const float v = (a > 0 && b > 0 && a == b) ? 1.0f : 0.0f;
            e[1 * plane + base + x] = v;
        }
    }
    // last row remains 0 (ignored)

    return edges.contiguous();
}
