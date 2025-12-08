#include "pattern_generator.h"
#include <algorithm>


cv::Mat generate_repetition_pattern(int W, int H)
{
    int tile = 8;

    cv::Mat small(tile, tile, CV_8UC4);
    cv::randu(small, 0, 256);

    cv::Mat img(H, W, CV_8UC4);

    for (int y = 0; y < H; y += tile)
        for (int x = 0; x < W; x += tile)
            small.copyTo(img(cv::Rect(x, y, tile, tile)));

    return img;
}

cv::Mat generate_monochrome_region(int W, int H)
{
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 255);

    cv::Vec4b color(dist(rng), dist(rng), dist(rng), dist(rng));

    cv::Mat img(H, W, CV_8UC4, color);

    return img;
}

cv::Mat generate_low_variance_noise(int W, int H)
{
    std::mt19937 rng(std::random_device{}());

    std::uniform_int_distribution<int> mean_dist(50, 200);
    std::uniform_int_distribution<int> sigma_dist(2, 15);

    ChannelParams ch[4]; // B, G, R, A
    for (int i = 0; i < 4; i++) {
        int mean  = mean_dist(rng);
        int sigma = sigma_dist(rng);
        ch[i] = { mean, sigma, std::normal_distribution<float>(mean, sigma) };
    }

    cv::Mat img(height, width, CV_8UC4);

    for (int y = 0; y < H; y++) {
        cv::Vec4b* row = img.ptr<cv::Vec4b>(y);
        for (int x = 0; x < W; x++) {

            uint8_t b = std::clamp((int)ch[0].dist(rng), 0, 255);
            uint8_t g = std::clamp((int)ch[1].dist(rng), 0, 255);
            uint8_t r = std::clamp((int)ch[2].dist(rng), 0, 255);
            uint8_t a = std::clamp((int)ch[3].dist(rng), 0, 255);

            row[x] = cv::Vec4b(b, g, r, a);
        }
    }

    return img;
}

cv::Mat generate_low_frequency_noise(int W, int H)
{
    int seedW = 32;
    int seedH = 32;

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> mean_dist(50, 200);
    std::uniform_int_distribution<int> sigma_dist(2, 20);

    ChannelParams ch[4];
    for (int i = 0; i < 4; i++) {
        int mean  = mean_dist(rng);
        int sigma = sigma_dist(rng);
        ch[i] = { mean, sigma, std::normal_distribution<float>(mean, sigma) };
    }

    // small seed noise (low frequency happens when scaled up)
    cv::Mat seed(seedH, seedW, CV_8UC4);

    for (int y = 0; y < seedH; y++) {
        cv::Vec4b* row = seed.ptr<cv::Vec4b>(y);
        for (int x = 0; x < seedW; x++) {

            uint8_t b = std::clamp((int)ch[0].dist(rng), 0, 255);
            uint8_t g = std::clamp((int)ch[1].dist(rng), 0, 255);
            uint8_t r = std::clamp((int)ch[2].dist(rng), 0, 255);
            uint8_t a = std::clamp((int)ch[3].dist(rng), 0, 255);

            row[x] = cv::Vec4b(b, g, r, a);
        }
    }

    cv::Mat img;
    cv::resize(seed, img, cv::Size(W, H), 0, 0, cv::INTER_CUBIC);

    return img;
}

cv::Mat generate_random_row_copies(int W, int H)
{
    cv::Mat row(1, W, CV_8UC4);
    cv::randu(row, 0, 256);

    cv::Mat img(H, W, CV_8UC4);
    for (int y=0; y < H; y++) row.copyTo(img.row(y));

    return img;
}
