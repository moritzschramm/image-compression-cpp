#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

std::vector<cv::Mat> slice_image(const cv::Mat& input, const torch::Tensor& mask, int num_labels);
