#include "pattern_generator.h"
#include <algorithm>
#include <queue>
#include <random>
#include <opencv2/opencv.hpp>
#include <opencv2/core/hal/interface.h>

void create_random_patterns()
{
    if (!std::filesystem::exists(CACHE_DIR / "random_patterns")) {
        std::filesystem::create_directories(CACHE_DIR / "random_patterns");
    }

    auto write_random_image = [&](size_t idx, std::function<cv::Mat(int,int,bool)> random_pattern_generator, int w, int h, bool alpha) -> void {

        auto target_path = CACHE_DIR / "random_patterns" / std::to_string(idx);

        if (!std::filesystem::exists(target_path.replace_extension(IMAGE_FORMAT))) {
            write_image(target_path, random_pattern_generator(w, h, alpha));
            std::cout << "created random pattern " << target_path << std::endl;
        }
    };

    int w = 1024, h = 1024;
    size_t idx = 0;
    size_t batch_size = 100;
    for (; idx < batch_size; idx++) write_random_image(idx, generate_repetition_pattern, w, h, true);
    for (; idx < batch_size*2; idx++) write_random_image(idx, generate_repetition_pattern, w, h, false);
    for (; idx < batch_size*3; idx++) write_random_image(idx, generate_monochrome_region, w, h, true);
    for (; idx < batch_size*4; idx++) write_random_image(idx, generate_monochrome_region, w, h, false);
    for (; idx < batch_size*5; idx++) write_random_image(idx, generate_low_variance_noise, w, h, true);
    for (; idx < batch_size*6; idx++) write_random_image(idx, generate_low_variance_noise, w, h, false);
    for (; idx < batch_size*7; idx++) write_random_image(idx, generate_low_frequency_noise, w, h, true);
    for (; idx < batch_size*8; idx++) write_random_image(idx, generate_low_frequency_noise, w, h, false);
    for (; idx < batch_size*9; idx++) write_random_image(idx, generate_random_row_copies, w, h, true);
    for (; idx < batch_size*10; idx++) write_random_image(idx, generate_random_row_copies, w, h, false);
}

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

cv::Mat generate_random_partition(int H, int W, int numSegments)
{
    cv::Mat mask(H, W, CV_32S, cv::Scalar(-1));

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> distW(0, W-1);
    std::uniform_int_distribution<int> distH(0, H-1);

    std::vector<cv::Point> seeds;
    seeds.reserve(numSegments);

    for (int i = 0; i < numSegments; i++)
        seeds.emplace_back(distW(rng), distH(rng));

    std::vector<std::queue<cv::Point>> queues(numSegments);
    for (int i = 0; i < numSegments; i++)
    {
        queues[i].push(seeds[i]);
        mask.at<int>(seeds[i].y, seeds[i].x) = i;
    }

    auto neighbors = [&](int y, int x) {
        std::vector<cv::Point> nb;
        if (y > 0)     nb.emplace_back(x, y-1);
        if (y < H-1)   nb.emplace_back(x, y+1);
        if (x > 0)     nb.emplace_back(x-1, y);
        if (x < W-1)   nb.emplace_back(x+1, y);
        return nb;
    };

    bool somethingFilled = true;

    while (somethingFilled)
    {
        somethingFilled = false;

        // random order of segment expansion
        std::vector<int> order(numSegments);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);

        for (int idx : order)
        {
            if (queues[idx].empty())
                continue;

            auto p = queues[idx].front();
            queues[idx].pop();

            auto nb = neighbors(p.y, p.x);
            std::shuffle(nb.begin(), nb.end(), rng);

            for (auto &q : nb)
            {
                int &cell = mask.at<int>(q.y, q.x);
                if (cell == -1)
                {
                    cell = idx;
                    queues[idx].push(q);
                    somethingFilled = true;
                }
            }
        }
    }

    return mask;
}



cv::Mat colorize_segmentation(const cv::Mat& mask)
{
    int H = mask.rows;
    int W = mask.cols;

    // Determine the maximum label
    int maxLabel = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            maxLabel = std::max(maxLabel, mask.at<int>(y, x));

    int numLabels = maxLabel + 1;

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 255);

    // Assign random colors to each label
    std::vector<cv::Vec3b> colors(numLabels);
    for (int i = 0; i < numLabels; i++)
        colors[i] = cv::Vec3b(dist(rng), dist(rng), dist(rng));

    // Create output image
    cv::Mat out(H, W, CV_8UC3);
    for (int y = 0; y < H; y++)
    {
        for (int x = 0; x < W; x++)
        {
            int label = mask.at<int>(y, x);
            out.at<cv::Vec3b>(y, x) = colors[label];
        }
    }

    return out;
}
