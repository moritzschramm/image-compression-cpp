#include "image_slicer.h"
#include <vector>

std::vector<cv::Mat> slice_image(const cv::Mat& input, const torch::Tensor& mask, int num_labels)
{
    std::vector<cv::Mat> slices;

    for (int label = 0; label < num_labels; ++label)
    {
        torch::Tensor binary = (mask == label).to(torch::kUInt8);

        cv::Mat mask_mat(input.rows, input.cols, CV_8UC1, binary.data_ptr<uint8_t>());

        cv::Mat slice = cv::Mat::zeros(input.rows, input.cols, input.channels() == 3 ? CV_8UC3 : CV_8UC4);

        input.copyTo(slice, mask_mat);

        slices.push_back(slice);
    }

    return slices;
}
