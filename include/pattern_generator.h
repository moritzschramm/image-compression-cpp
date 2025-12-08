#pragma once
#include <opencv2/opencv.hpp>
#include <random>

struct ChannelParams {
    int mean;
    int sigma;
    std::normal_distribution<float> dist;
};

cv::Mat generate_repetition_pattern(int w, int h);
cv::Mat generate_monochrome_region(int w, int h);
cv::Mat generate_low_variance_noise(int w, int h);
cv::Mat generate_low_frequency_noise(int w, int h);
cv::Mat generate_random_row_copies(int w, int h);
