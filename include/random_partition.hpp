
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

struct Rect {
    int64_t x0, y0, w, h; // [x0, x0+w) × [y0, y0+h)
};

static inline bool can_split_v(const Rect& r, int64_t min_w) { return r.w >= 2 * min_w; }
static inline bool can_split_h(const Rect& r, int64_t min_h) { return r.h >= 2 * min_h; }

// Output: int8 tensor of shape [2,H,W]
//   out[0,y,x] = horizontal edge between (y,x) and (y,x+1)  (right-neighbor).  +1 same-rect, -1 boundary
//   out[1,y,x] = vertical   edge between (y,x) and (y+1,x)  (bottom-neighbor). +1 same-rect, -1 boundary
// For the last column/row (no neighbor), value is +1 unless include_image_boundary==true, then -1
torch::Tensor random_rect_partition(
        int64_t H,
        int64_t W,
        int64_t min_h = 8,
        int64_t min_w = 8,
        double split_prob = 0.75,
        int64_t min_rect_count = 1,
        uint64_t seed = 0) {

    if (H <= 0 || W <= 0) throw std::invalid_argument("H and W must be > 0");
    if (min_h <= 0 || min_w <= 0) throw std::invalid_argument("min_h/min_w must be > 0");
    if (split_prob < 0.0 || split_prob > 1.0) throw std::invalid_argument("split_prob must be in [0,1]");
    if (min_rect_count <= 0) throw std::invalid_argument("min_rect_count must be >= 1");

    // Feasibility (coarse but useful): if both dims >= min_* is enforced, max rectangles is bounded by area
    const int64_t min_area = min_h * min_w;
    const int64_t max_by_area = (min_area > 0) ? (H * W) / min_area : 1;
    if (min_rect_count > std::max<int64_t>(1, max_by_area) &&
            (H >= min_h && W >= min_w)) {
        throw std::invalid_argument("min_rect_count not achievable (exceeds area-based upper bound)");
    }
    if ((H < 2 * min_h && W < 2 * min_w) && min_rect_count > 1) {
        // No split possible in either direction at root
        throw std::invalid_argument("min_rect_count > 1 not achievable with given min_h/min_w (no splits possible)");
    }

    std::mt19937_64 rng(seed ? seed : std::random_device{}());
    std::bernoulli_distribution do_split(split_prob);

    // 1) BSP tiling, forcing splits until min_rect_count is reached (if possible)
    std::vector<Rect> pending;
    pending.reserve(1024);
    pending.push_back(Rect{0, 0, W, H});

    std::vector<Rect> rects;
    rects.reserve(1024);

    while (!pending.empty()) {
        const int64_t current_total = static_cast<int64_t>(pending.size() + rects.size());
        const bool need_more = (current_total < min_rect_count);

        Rect r{};

        if (need_more) {
            // Choose a splittable pending rect (largest-area heuristic)
            int best_i = -1;
            int64_t best_area = -1;
            for (int i = 0; i < (int)pending.size(); ++i) {
                const Rect& c = pending[i];
                if (can_split_v(c, min_w) || can_split_h(c, min_h)) {
                    const int64_t area = c.w * c.h;
                    if (area > best_area) { best_area = area; best_i = i; }
                }
            }
            if (best_i < 0) {
                // Cannot split further; finalize remaining
                rects.insert(rects.end(), pending.begin(), pending.end());
                pending.clear();
                break;
            }
            r = pending[best_i];
            pending[best_i] = pending.back();
            pending.pop_back();
        } else {
            r = pending.back();
            pending.pop_back();
        }

        const bool canV = can_split_v(r, min_w);
        const bool canH = can_split_h(r, min_h);

        bool split_now = false;
        if (need_more) {
            split_now = (canV || canH);
        } else {
            split_now = (canV || canH) && do_split(rng);
        }

        if (!split_now) {
            rects.push_back(r);
            continue;
        }

        // Pick split orientation (biased by aspect ratio if both possible)
        bool splitV = false;
        if (canV && canH) {
            const double pV = (double)r.w / (double)(r.w + r.h);
            std::bernoulli_distribution pickV(pV);
            splitV = pickV(rng);
        } else {
            splitV = canV;
        }

        if (splitV) {
            std::uniform_int_distribution<int64_t> dist(r.x0 + min_w, r.x0 + r.w - min_w);
            const int64_t sx = dist(rng);

            pending.push_back(Rect{r.x0, r.y0, sx - r.x0, r.h});
            pending.push_back(Rect{sx,   r.y0, (r.x0 + r.w) - sx, r.h});
        } else {
            std::uniform_int_distribution<int64_t> dist(r.y0 + min_h, r.y0 + r.h - min_h);
            const int64_t sy = dist(rng);

            pending.push_back(Rect{r.x0, r.y0, r.w, sy - r.y0});
            pending.push_back(Rect{r.x0, sy,   r.w, (r.y0 + r.h) - sy});
        }
    }

    if ((int64_t)rects.size() < min_rect_count) {
        throw std::runtime_error("Could not reach min_rect_count with given constraints");
    }

    // 2) ID map [H,W] (CPU int32)
    auto id_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    torch::Tensor ids = torch::empty({H, W}, id_opts);
    int32_t* idp = ids.data_ptr<int32_t>();

    for (int32_t rid = 0; rid < (int32_t)rects.size(); ++rid) {
        const Rect& rr = rects[rid];
        for (int64_t y = rr.y0; y < rr.y0 + rr.h; ++y) {
            int32_t* row = idp + y * W;
            std::fill(row + rr.x0, row + (rr.x0 + rr.w), rid);
        }
    }

    // 3) Edge tensor [2,H,W] (CPU int8), initialized to +1
    auto out_opts = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU);
    torch::Tensor out = torch::full({2, H, W}, int8_t(+1), out_opts);
    int8_t* outp = out.data_ptr<int8_t>();

    // Layout: out is contiguous [2,H,W]
    // idx_out(c,y,x) = (c*H + y)*W + x
    auto idx_out = [H, W](int64_t c, int64_t y, int64_t x) -> int64_t {
        return (c * H + y) * W + x;
    };

    for (int64_t y = 0; y < H; ++y) {
        for (int64_t x = 0; x < W; ++x) {
            const int64_t idx = y * W + x;
            const int32_t v = idp[idx];

            // horizontal edge (to right neighbor)
            if (x + 1 < W) {
                outp[idx_out(0, y, x)] = (idp[idx + 1] == v) ? int8_t(+1) : int8_t(-1);
            }

            // vertical edge (to bottom neighbor)
            if (y + 1 < H) {
                outp[idx_out(1, y, x)] = (idp[idx + W] == v) ? int8_t(+1) : int8_t(-1);
            }
        }
    }

    return out; // [2,H,W], int8, CPU
}
