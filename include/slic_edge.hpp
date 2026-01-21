#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

// Output: edges [2, H, W] (float32, CPU)
//   edges[0, y, x] = horizontal edge between (y,x) and (y,x+1) for x in [0, W-2]
//   edges[1, y, x] = vertical   edge between (y,x) and (y+1,x) for y in [0, H-2]
// Values: +1 inside a superpixel, -1 on superpixel borders
torch::Tensor slic_edge_costs(
    const cv::Mat& input,
    int region_size = 20,
    float ruler = 10.0f,
    int iters = 10,
    int slic_algorithm = cv::ximgproc::SLICO
) {
    cv::Mat image;
    if (input.empty()) throw std::runtime_error("slic_edge_costs: input image is empty");
    if (input.depth() != CV_8U) {
        if (input.depth() == CV_16U) {
            input.convertTo(image, CV_8U, 1.0 / 257.0); // 65535/255 ≈ 257
        } else {
            throw std::runtime_error("slic_edge_costs: expected 8-bit or 16-bit image (CV_8U, CV_16U)");
        }
    } else {
        image = input;
    }

    const int H = image.rows;
    const int W = image.cols;

    // SLIC in OpenCV ximgproc expects 3 or 4 channels; convert grayscale to BGR
    cv::Mat img;
    if (image.channels() == 1) {
        cv::cvtColor(image, img, cv::COLOR_GRAY2BGR);
    } else if (image.channels() == 3 || image.channels() == 4) {
        img = image;
    } else {
        throw std::runtime_error("slic_edge_costs: unsupported channel count");
    }

    if (!img.isContinuous()) img = img.clone();

    // Run SLIC superpixels
    auto sp = cv::ximgproc::createSuperpixelSLIC(img, slic_algorithm, region_size, ruler);
    sp->iterate(iters);
    sp->enforceLabelConnectivity();

    cv::Mat labels; // CV_32S, HxW
    sp->getLabels(labels);

    // Create edge tensor: [2, H, W]
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
