#include "image_writer.h"
#include "configuration.h"

bool write_image(const std::filesystem::path& filename, const cv::Mat& image)
{
    auto f = filename;
    return cv::imwrite(f.replace_extension(IMAGE_FORMAT), image, {cv::IMWRITE_PNG_COMPRESSION, COMPRESSION_LEVEL});
}
