#include <thread>
#include <future>
#include <opencv2/opencv.hpp>
#include <filesystem>

#include "configuration.h"
#include "image_loader.h"
#include "image_writer.h"

const std::string SOURCE_FORMAT = "jpeg";
const int WIDTH = 512;
const int HEIGHT = 512;

/**
 * resized and convert images from dataset to target format with correct compression level
 */
int main()
{
    auto image_paths = find_image_files_recursively(DATASET_DIR, SOURCE_FORMAT);

    unsigned num_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::future<void>> futures;

    for(auto image_path : image_paths)
    {
        futures.push_back(std::async(std::launch::async, [&, image_path]() {

            auto image = load_image(image_path);

            cv::resize(image, image, cv::Size(WIDTH, HEIGHT));

            write_image(image_path, image);
        }));
    }

    for (auto& f : futures) f.get();

    return 0;
}