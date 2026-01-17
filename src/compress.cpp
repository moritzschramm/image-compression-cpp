#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "configuration.h"
#include "image_loader.h"
#include "image_slicer.h"
#include "fcn/EdgeUNet.h"
#include "rama_wrapper.cuh"

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


torch::Tensor flatten_grid_edges(const torch::Tensor& x)
{
    // x: [B, 4, H, W]
    TORCH_CHECK(x.dim() == 4, "Expected [B, 4, H, W]");

    const int64_t B = x.size(0);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    // horizontal edges: channels 0,1 — drop last column
    auto h = x.slice(1, 0, 2).slice(3, 0, W - 1);  // [B, 2, H, W-1]

    // vertical edges: channels 2,3 — drop last row
    auto v = x.slice(1, 2, 4).slice(2, 0, H - 1);  // [B, 2, H-1, W]

    // flatten spatial dims
    auto h_flat = h.reshape({B, 2, -1});  // [B, 2, H*(W-1)]
    auto v_flat = v.reshape({B, 2, -1});  // [B, 2, (H-1)*W]

    // concatenate edge lists
    return torch::cat({h_flat, v_flat}, /*dim=*/2);  // [B, 2, E]
}

void build_rama_indices(
    int32_t H,
    int32_t W,
    std::vector<int32_t>& i_idx,
    std::vector<int32_t>& j_idx)
{
    i_idx.clear();
    j_idx.clear();

    // horizontal edges
    for (int32_t r = 0; r < H; ++r) {
        for (int32_t c = 0; c < W - 1; ++c) {
            int32_t u = r * W + c;
            int32_t v = r * W + (c + 1);

            i_idx.push_back(u);
            j_idx.push_back(v);
        }
    }

    // vertical edges
    for (int32_t r = 0; r < H - 1; ++r) {
        for (int32_t c = 0; c < W; ++c) {
            int32_t u = r * W + c;
            int32_t v = (r + 1) * W + c;

            i_idx.push_back(u);
            j_idx.push_back(v);
        }
    }
}

int main()
{
    const auto device = torch::kCUDA;

    auto paths = find_image_files_recursively(DATASET_DIR, IMAGE_FORMAT);

    std::cout << "Found " << std::to_string(paths.size()) << " images" << std::endl;

    std::vector<int32_t> i_idx;
    std::vector<int32_t> j_idx;

    build_rama_indices(256, 256, i_idx, j_idx); // assuming height and width of 256x256

    EdgeUNet model;
    torch::load(model, "fcn_trained.pt");
    model->to(device);
    model->eval();

    for(const auto& path : paths)
    {
        torch::InferenceMode guard;

        std::cout << path << std::endl;

        cv::Mat image = load_image(path);

        auto img_f = to_f32c3_01_or_throw(image);

        // Input tensor: [3,H,W]
        auto input = torch::from_blob(
            img_f.data, {img_f.rows, img_f.cols, 3}, torch::kFloat32
        ).permute({2, 0, 1}).clone().to(device);

        torch::Tensor output = model->forward(input);

        auto flat = flatten_grid_edges(output);
        torch::Tensor edge_costs = flat.select(1, 0);
        // torch::Tensor raw_sigma = flat.select(1, 1); // not needed for inference

        std::vector<int32_t> i_idx;
        std::vector<int32_t> j_idx;

        build_rama_indices(image.rows, image.cols, i_idx, j_idx);

        auto i_device = torch::tensor(i_idx, torch::TensorOptions().dtype(torch::kInt32).device(device));
        auto j_device = torch::tensor(j_idx, torch::TensorOptions().dtype(torch::kInt32).device(device));

        torch::Tensor node_labels = rama_torch(i_device, j_device, edge_costs);

        node_labels = node_labels.to(torch::kCPU);

        write_slices(input, node_labels, RESULTS_DIR, path.stem());

        break;
    }

    return 0;
}
