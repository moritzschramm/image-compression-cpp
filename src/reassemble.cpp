#include <opencv2/opencv.hpp>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "metadata.h"

// ensure a mat has 4 channels (RGBA). If input has 3 channels => add opaque alpha
// if 1 channel => duplicate to RGB and add alpha
cv::Mat ensure_rgba(const cv::Mat& src)
{
    cv::Mat dst;
    if (src.channels() == 4) {
        dst = src.clone();
    } else if (src.channels() == 3) {
        cv::cvtColor(src, dst, cv::COLOR_BGR2BGRA);
    } else if (src.channels() == 1) {
        cv::cvtColor(src, dst, cv::COLOR_GRAY2BGRA);
    } else {
        throw std::runtime_error("Unsupported channel count: " + std::to_string(src.channels()));
    }
    return dst;
}

int main()
{

    std::string meta_path = "metadata.bin";
    std::string out_path = "reconstructed.png";

    uint32_t w, h;
    std::vector<SliceMetadata> meta;
    try {
        meta = read_metadata_binary(meta_path, w, h);
    } catch (const std::exception& e) {
        std::cerr << "Error reading metadata: " << e.what() << std::endl;
        return 1;
    }

    int width = w;
    int height = h;

    if (meta.empty()) {
        std::cerr << "No slices in metadata" << std::endl;
        return 1;
    }

    cv::Mat canvas(height, width, CV_8UC4, cv::Scalar(0,0,0,0));

    for (const auto& m : meta) {
        if (m.filename.empty()) {
            std::cerr << "Warning: empty filename for label " << m.label << ", skipping\n";
            continue;
        }

        cv::Mat slice = cv::imread(m.filename, cv::IMREAD_UNCHANGED);
        if (slice.empty()) {
            std::cerr << "Warning: failed to load slice '" << m.filename << "', skipping\n";
            continue;
        }

        cv::Mat slice_rgba = ensure_rgba(slice);

        // validate slice size matches recorded bbox if possible
        if (slice_rgba.cols != m.width || slice_rgba.rows != m.height) {
            std::cerr << "Warning: slice size (" << slice_rgba.cols << "x" << slice_rgba.rows
                      << ") does not match metadata bbox (" << m.width << "x" << m.height
                      << ") for file " << m.filename << ". Using actual slice size.\n";
        }

        // compute destination ROI
        int dst_x = m.x;
        int dst_y = m.y;
        int copy_w = std::min(slice_rgba.cols, width - dst_x);
        int copy_h = std::min(slice_rgba.rows, height - dst_y);

        if (copy_w <= 0 || copy_h <= 0) {
            std::cerr << "Warning: slice '" << m.filename << "' lies outside canvas, skipping\n";
            continue;
        }

        cv::Rect dst_roi(dst_x, dst_y, copy_w, copy_h);
        cv::Rect src_roi(0, 0, copy_w, copy_h);

        cv::Mat canvas_roi = canvas(dst_roi);
        cv::Mat slice_roi = slice_rgba(src_roi);

        // use alpha mask to copy only non-transparent pixels
        std::vector<cv::Mat> channels;
        cv::split(slice_roi, channels); // channels[3] is alpha (BGRA order)

        cv::Mat alpha = channels[3];

        // alpha is 0..255. create mask where alpha > 0
        cv::Mat alpha_mask;
        cv::threshold(alpha, alpha_mask, 0, 255, cv::THRESH_BINARY);

        // copy RGBA channels individually using mask
        cv::Mat dst_channels[4];
        cv::split(canvas_roi, dst_channels);

        // for each color channel, copy where mask is set
        for (int c = 0; c < 4; ++c) {
            slice_roi.forEach<cv::Vec4b>([&](cv::Vec4b &pixel, const int* pos) {
                // no-op; used to ensure slice_roi is continuous in some builds
            });
        }

        slice_roi.copyTo(canvas_roi, alpha_mask);
    }

    if (!cv::imwrite(out_path, canvas)) {
        std::cerr << "Failed to write reconstructed image to " << out_path << "\n";
        return 1;
    }

    std::cout << "Reconstructed image written to " << out_path << "\n";
    return 0;
}
