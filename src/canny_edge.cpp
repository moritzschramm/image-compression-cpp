#include "canny_edge.h"
#include <stdexcept>

namespace {
cv::Mat to_gray_u8_any(const cv::Mat& img_any) {
    CV_Assert(!img_any.empty());

    cv::Mat gray;
    const int ch = img_any.channels();
    if (ch == 1) {
        gray = img_any;
    } else if (ch == 3) {
        cv::cvtColor(img_any, gray, cv::COLOR_BGR2GRAY);
    } else if (ch == 4) {
        cv::cvtColor(img_any, gray, cv::COLOR_BGRA2GRAY);
    } else {
        throw std::runtime_error("Unsupported channel count: " + std::to_string(ch));
    }

    cv::Mat gray_u8;
    switch (gray.depth()) {
        case CV_8U:
            gray_u8 = gray;
            break;
        case CV_16U:
            gray.convertTo(gray_u8, CV_8U, 1.0 / 257.0); // 65535/255 ≈ 257
            break;
        case CV_32F: {
            double minv = 0.0, maxv = 0.0;
            cv::minMaxLoc(gray, &minv, &maxv);

            if (maxv <= 1.0 + 1e-6 && minv >= 0.0 - 1e-6) {
                gray.convertTo(gray_u8, CV_8U, 255.0);
            } else if (maxv > minv) {
                const double scale = 255.0 / (maxv - minv);
                const double shift = -minv * scale;
                gray.convertTo(gray_u8, CV_8U, scale, shift);
            } else {
                gray_u8 = cv::Mat(gray.size(), CV_8U, cv::Scalar(0));
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported depth: " + std::to_string(gray.depth()));
    }

    if (!gray_u8.isContinuous()) gray_u8 = gray_u8.clone();
    return gray_u8;
}
} // namespace

torch::Tensor canny_edge_costs(
    const cv::Mat& img,
    double canny_low,
    double canny_high,
    int aperture_size,
    bool L2gradient,
    int blur_ksize,
    double blur_sigma
) {
    cv::Mat gray = to_gray_u8_any(img);
    const int H = gray.rows;
    const int W = gray.cols;

    if (blur_ksize >= 3 && (blur_ksize % 2 == 1)) {
        cv::GaussianBlur(gray, gray, cv::Size(blur_ksize, blur_ksize), blur_sigma);
    }

    cv::Mat edges; // CV_8U, values {0,255}
    cv::Canny(gray, edges, canny_low, canny_high, aperture_size, L2gradient);

    auto out = torch::zeros({2, H, W},
                            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).requires_grad(false));
    auto acc = out.accessor<float, 3>();

    // Horizontal edges: between (y,x) and (y,x+1)
    for (int y = 0; y < H; ++y) {
        const uint8_t* e = edges.ptr<uint8_t>(y);
        for (int x = 0; x < W - 1; ++x) {
            const bool is_edge = (e[x] != 0) || (e[x + 1] != 0);
            acc[0][y][x] = is_edge ? 0.0f : 1.0f;
        }
        // last column remains 0 (ignored)
    }

    // Vertical edges: between (y,x) and (y+1,x)
    for (int y = 0; y < H - 1; ++y) {
        const uint8_t* e0 = edges.ptr<uint8_t>(y);
        const uint8_t* e1 = edges.ptr<uint8_t>(y + 1);
        for (int x = 0; x < W; ++x) {
            const bool is_edge = (e0[x] != 0) || (e1[x] != 0);
            acc[1][y][x] = is_edge ? 0.0f : 1.0f;
        }
    }
    // last row remains 0 (ignored)

    return out;
}
