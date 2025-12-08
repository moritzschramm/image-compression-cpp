#include "pattern_generator.h"
#include <algorithm>
#include <opencv2/core/hal/interface.h>


cv::Mat generate_repetition_pattern(int W, int H, bool alpha)
{
    int tile = 8;

    cv::Mat small(tile, tile, alpha ? CV_8UC4 : CV_8UC3);
    cv::randu(small, 0, 256);

    cv::Mat img(H, W, alpha ? CV_8UC4 : CV_8UC3);

    for (int y = 0; y < H; y += tile)
        for (int x = 0; x < W; x += tile)
            small.copyTo(img(cv::Rect(x, y, tile, tile)));

    return img;
}

cv::Mat generate_monochrome_region(int W, int H, bool alpha)
{
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 255);

    if (alpha) {
        cv::Vec4b color(dist(rng), dist(rng), dist(rng), dist(rng));

        cv::Mat img(H, W, CV_8UC4, color);

        return img;
    }
    else {
        cv::Vec3b color(dist(rng), dist(rng), dist(rng));

        cv::Mat img(H, W, CV_8UC3, color);

        return img;
    }
}

cv::Mat generate_low_variance_noise(int W, int H, bool alpha)
{
    std::mt19937 rng(std::random_device{}());

    std::uniform_int_distribution<int> mean_dist(50, 200);
    std::uniform_int_distribution<int> sigma_dist(2, 8);

    ChannelParams ch[alpha ? 4 : 3]; // B, G, R, A
    for (int i = 0; i < (alpha ? 4 : 3); i++) {
        int mean  = mean_dist(rng);
        int sigma = sigma_dist(rng);
        ch[i] = { mean, sigma, std::normal_distribution<float>(mean, sigma) };
    }

    cv::Mat img(H, W, alpha ? CV_8UC4 : CV_8UC3);

    for (int y = 0; y < H; y++) {
        if (alpha) {
            cv::Vec4b* row = img.ptr<cv::Vec4b>(y);
            for (int x = 0; x < W; x++) {

                uint8_t b = std::clamp((int)ch[0].dist(rng), 0, 255);
                uint8_t g = std::clamp((int)ch[1].dist(rng), 0, 255);
                uint8_t r = std::clamp((int)ch[2].dist(rng), 0, 255);
                uint8_t a = std::clamp((int)ch[3].dist(rng), 0, 255);

                row[x] = cv::Vec4b(b, g, r, a);
            }
        }
        else {
            cv::Vec3b* row = img.ptr<cv::Vec3b>(y);
            for (int x = 0; x < W; x++) {

                uint8_t b = std::clamp((int)ch[0].dist(rng), 0, 255);
                uint8_t g = std::clamp((int)ch[1].dist(rng), 0, 255);
                uint8_t r = std::clamp((int)ch[2].dist(rng), 0, 255);

                row[x] = cv::Vec3b(b, g, r);
            }
        }
    }

    return img;
}

cv::Mat generate_low_frequency_noise(int W, int H, bool alpha)
{
    int seedW = 32;
    int seedH = 32;

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> mean_dist(50, 200);
    std::uniform_int_distribution<int> sigma_dist(2, 20);

    ChannelParams ch[alpha ? 4 : 3];
    for (int i = 0; i < (alpha ? 4 : 3); i++) {
        int mean  = mean_dist(rng);
        int sigma = sigma_dist(rng);
        ch[i] = { mean, sigma, std::normal_distribution<float>(mean, sigma) };
    }

    // small seed noise (low frequency happens when scaled up)
    cv::Mat seed(seedH, seedW, alpha ? CV_8UC4 : CV_8UC3);

    for (int y = 0; y < seedH; y++) {
        if (alpha) {
            cv::Vec4b* row = seed.ptr<cv::Vec4b>(y);
            for (int x = 0; x < seedW; x++) {

                uint8_t b = std::clamp((int)ch[0].dist(rng), 0, 255);
                uint8_t g = std::clamp((int)ch[1].dist(rng), 0, 255);
                uint8_t r = std::clamp((int)ch[2].dist(rng), 0, 255);
                uint8_t a = std::clamp((int)ch[3].dist(rng), 0, 255);

                row[x] = cv::Vec4b(b, g, r, a);
            }
        }
        else {
            cv::Vec3b* row = seed.ptr<cv::Vec3b>(y);
            for (int x = 0; x < seedW; x++) {

                uint8_t b = std::clamp((int)ch[0].dist(rng), 0, 255);
                uint8_t g = std::clamp((int)ch[1].dist(rng), 0, 255);
                uint8_t r = std::clamp((int)ch[2].dist(rng), 0, 255);

                row[x] = cv::Vec3b(b, g, r);
            }
        }
    }

    cv::Mat img;
    cv::resize(seed, img, cv::Size(W, H), 0, 0, cv::INTER_CUBIC);

    return img;
}

cv::Mat generate_random_row_copies(int W, int H, bool alpha)
{
    cv::Mat row(1, W, alpha ? CV_8UC4 : CV_8UC3);
    cv::randu(row, 0, 256);

    cv::Mat img(H, W, alpha ? CV_8UC4 : CV_8UC3);
    for (int y = 0; y < H; y++) row.copyTo(img.row(y));

    return img;
}

cv::Mat generate_random_noise(int W, int H, bool alpha)
{
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 255);

    cv::Mat img(H, W, alpha ? CV_8UC4 : CV_8UC3);

    for (int y = 0; y < H; y++) {
        if (alpha) {
            cv::Vec4b* row = img.ptr<cv::Vec4b>(y);
            for (int x = 0; x < W; x++) {

                uint8_t b = dist(rng);
                uint8_t g = dist(rng);
                uint8_t r = dist(rng);
                uint8_t a = dist(rng);

                row[x] = cv::Vec4b(b, g, r, a);
            }
        }
        else {
            cv::Vec3b* row = img.ptr<cv::Vec3b>(y);
            for (int x = 0; x < W; x++) {

                uint8_t b = dist(rng);
                uint8_t g = dist(rng);
                uint8_t r = dist(rng);

                row[x] = cv::Vec3b(b, g, r);
            }
        }
    }

    return img;
}
