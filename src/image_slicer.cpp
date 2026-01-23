#include "image_slicer.h"
#include <thread>
#include <future>
#include <mutex>
#include <opencv2/imgcodecs.hpp>

#include "configuration.h"
#include "image_writer.h"
#include "metadata.h"


/*
 * compute the bounding box of a given binary mask
 */
cv::Rect compute_bounding_box(const torch::Tensor& mask)
{
    int height = mask.size(0);
    int width = mask.size(1);

    const uint8_t* ptr = mask.data_ptr<uint8_t>();

    int min_x = width, min_y = height, max_x = -1, max_y = -1;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (ptr[y * width + x] != 0) {
                min_x = std::min(min_x, x);
                max_x = std::max(max_x, x);
                min_y = std::min(min_y, y);
                max_y = std::max(max_y, y);
            }
        }
    }

    if (max_x < 0) {
        // no pixels found
        return cv::Rect();
    }

    return cv::Rect(min_x, min_y, (max_x - min_x + 1), (max_y - min_y + 1));
}


/*
 * slice given input image based on binary mask defined by label
 * out_box contains dimensions of output slice
 */
cv::Mat slice_image(const cv::Mat& input, const torch::Tensor& mask, int label, cv::Rect& out_box)
{
    // only slice for given label
    torch::Tensor binary_mask = (mask == label).to(torch::kUInt8).contiguous();

    // crop to bounding box
    cv::Rect box = compute_bounding_box(binary_mask);
    if (box.width == 0 || box.height == 0) {
        return cv::Mat();
    }

    cv::Mat cropped = input(box).clone();

    cv::Mat mask_mat(input.rows, input.cols, CV_8UC1, binary_mask.data_ptr());
    cv::Mat cropped_mask = mask_mat(box).clone();

    cv::Mat output(cropped.rows, cropped.cols, input.type(), cv::Scalar(0,0,0,0));

    cropped.copyTo(output, cropped_mask);

    out_box.x = box.x;
    out_box.y = box.y;
    out_box.width = box.width;
    out_box.height = box.height;

    return output;
}

/*
 * write slices based on given mask
 * saves one file per slice in parallel
 * files will be written in output_path / file_directory_name / slice_X.<extension>
 */
bool write_slices(const cv::Mat& input, const torch::Tensor& mask,
    const std::filesystem::path& output_path, const std::filesystem::path& file_directory_name)
{
    bool success = true;
    auto dir = (output_path / file_directory_name);

    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir);
    }

    int num_labels = mask.max().item<int>() + 1;
    unsigned num_threads = std::max(1u, std::thread::hardware_concurrency());

    std::vector<std::future<void>> futures;
    std::mutex meta_mutex;
    std::vector<SliceMetadata> metadata;

    for (int label = 0; label < num_labels; ++label) {

        futures.push_back(std::async(std::launch::async, [&, label]() {

            cv::Rect box;
            cv::Mat slice = slice_image(input, mask, label, box);

            if (slice.empty()) return;

            std::string filename = "slice_" + std::to_string(label) + "." + IMAGE_FORMAT;
            success = write_image(dir / filename, slice) && success;

            // store metadata
            SliceMetadata m;
            m.label = label;
            m.filename = filename;
            m.x = box.x;
            m.y = box.y;
            m.width = box.width;
            m.height = box.height;

            std::lock_guard<std::mutex> lock(meta_mutex);
            metadata.push_back(m);
        }));
    }

    for (auto& f : futures) f.get();

    // write metadata
    write_metadata_binary(metadata, dir / "metadata.bin", input.cols, input.rows);

    return success;
}
