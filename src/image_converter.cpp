#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>

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

    std::atomic<size_t> index{0};
    const size_t N = image_paths.size();
    const unsigned num_threads = std::min(8u, std::thread::hardware_concurrency());

    std::vector<std::thread> workers;

    for (unsigned t = 0; t < num_threads; ++t) {
        workers.emplace_back([&]() {
            while (true) {
                size_t i = index.fetch_add(1);
                if (i >= N) break;

                const auto& image_path = image_paths[i];

                auto image = load_image(image_path);

                cv::resize(image, image, cv::Size(WIDTH, HEIGHT));
                write_image(image_path, image);
            }
        });
    }

    for (auto& t : workers) t.join();

    return 0;
}
