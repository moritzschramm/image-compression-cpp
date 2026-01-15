#include <torch/torch.h>
#include <opencv2/opencv.hpp>

// Returns a CPU tensor of shape [2, H, W] with values in {-1, +1} (dtype int8)
// Channel 0 (horizontal): edge between (y,x) and (y,x+1) stored at (y,x)
// Channel 1 (vertical):   edge between (y,x) and (y+1,x) stored at (y,x)
// Last column (ch0) and last row (ch1) are set to +1 (no neighbor to form an edge with)
inline torch::Tensor canny_edge_costs(
    const cv::Mat& rgb_u8,
    double canny_low = 50.0,
    double canny_high = 150.0,
    int aperture_size = 3,
    bool L2gradient = true,
    int blur_ksize = 3,          // set 0 or 1 to disable blur
    double blur_sigma = 1.0
) {
    CV_Assert(!rgb_u8.empty());
    CV_Assert(rgb_u8.type() == CV_8UC3);

    const int H = rgb_u8.rows;
    const int W = rgb_u8.cols;

    cv::Mat gray;
    cv::cvtColor(rgb_u8, gray, cv::COLOR_BGR2GRAY);

    if (blur_ksize >= 3 && (blur_ksize % 2 == 1)) {
        cv::GaussianBlur(gray, gray, cv::Size(blur_ksize, blur_ksize), blur_sigma);
    }

    cv::Mat edges; // CV_8U, values {0,255}; Canny edges are typically 1-pixel wide
    cv::Canny(gray, edges, canny_low, canny_high, aperture_size, L2gradient);

    // Output tensor: int8 values {-1, +1}
    auto out = torch::full({2, H, W}, /*value=*/int8_t{+1},
                           torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU).requires_grad(false));
    auto acc = out.accessor<int8_t, 3>();

    // Horizontal edges: between (y,x) and (y,x+1) -> store at (y,x) in channel 0
    for (int y = 0; y < H; ++y) {
        const uint8_t* e = edges.ptr<uint8_t>(y);
        for (int x = 0; x < W - 1; ++x) {
            const bool is_edge = (e[x] != 0) || (e[x + 1] != 0);
            if (is_edge) acc[0][y][x] = int8_t{-1};
        }
        // acc[0][y][W-1] stays +1
    }

    // Vertical edges: between (y,x) and (y+1,x) -> store at (y,x) in channel 1
    for (int y = 0; y < H - 1; ++y) {
        const uint8_t* e0 = edges.ptr<uint8_t>(y);
        const uint8_t* e1 = edges.ptr<uint8_t>(y + 1);
        for (int x = 0; x < W; ++x) {
            const bool is_edge = (e0[x] != 0) || (e1[x] != 0);
            if (is_edge) acc[1][y][x] = int8_t{-1};
        }
    }
    // acc[1][H-1][x] stays +1

    return out;
}
