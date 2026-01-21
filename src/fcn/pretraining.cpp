#include <filesystem>
#include <torch/torch.h>
#include <vector>
#include "configuration.h"
#include "image_loader.h"
#include "image_writer.h"
#include "fcn/EdgeDataset.h"
#include "fcn/EdgeUNet.h"


struct BatchStats {
    torch::Tensor loss;      // scalar
    torch::Tensor valid_w;   // scalar weight for averaging (number of valid pixels)
    int64_t correct = 0;
    int64_t valid = 0;
};

BatchStats compute_loss_and_signacc(
    const torch::Tensor& outputs,   // [B,6,H,W]
    const torch::Tensor& targets,   // [B,6,H,W]
    const torch::Tensor& pos_weight, // scalar tensor on same device/dtype float
    double w_sign = 1.0,
    double w_reg  = 0.3,
    double w_sig  = 0.1
) {
    // Predictions
    auto p_cost_r = outputs.select(1, 0); // [B,H,W]
    auto p_sig_r  = outputs.select(1, 1);
    auto p_cost_d = outputs.select(1, 2);
    auto p_sig_d  = outputs.select(1, 3);

    // Targets + masks
    auto t_cost_r = targets.select(1, 0).to(outputs.dtype());
    auto t_sig_r  = targets.select(1, 1).to(outputs.dtype());
    auto t_cost_d = targets.select(1, 2).to(outputs.dtype());
    auto t_sig_d  = targets.select(1, 3).to(outputs.dtype());
    auto m_r      = targets.select(1, 4).to(outputs.dtype()); // {0,1}
    auto m_d      = targets.select(1, 5).to(outputs.dtype()); // {0,1}

    auto sign_loss_weighted_bce = [&](const torch::Tensor& pred_cost,
                                      const torch::Tensor& tgt_cost,
                                      const torch::Tensor& mask01) {
        auto valid = mask01 > 0;
        auto y = ((tgt_cost < 0) & valid).to(pred_cost.dtype()); // 1 = negative target
        auto logits = -pred_cost;

        auto bce = torch::binary_cross_entropy_with_logits(
            logits, y, /*weight=*/{}, /*pos_weight=*/pos_weight, torch::Reduction::None
        );

        auto denom = mask01.sum().clamp_min(1.0);
        return (bce * mask01).sum() / denom;
    };

    auto cost_reg = [&](const torch::Tensor& pred_cost,
                        const torch::Tensor& tgt_cost,
                        const torch::Tensor& mask01) {
        auto reg = torch::smooth_l1_loss(pred_cost, tgt_cost, torch::Reduction::None);
        auto denom = mask01.sum().clamp_min(1.0);
        return (reg * mask01).sum() / denom;
    };

    auto sigma_reg = [&](const torch::Tensor& pred_sig,
                         const torch::Tensor& tgt_sig,
                         const torch::Tensor& mask01) {
        auto reg = torch::smooth_l1_loss(pred_sig, tgt_sig, torch::Reduction::None);
        auto denom = mask01.sum().clamp_min(1.0);
        return (reg * mask01).sum() / denom;
    };

    // Loss components
    auto loss_sign = 0.5 * (sign_loss_weighted_bce(p_cost_r, t_cost_r, m_r) +
                            sign_loss_weighted_bce(p_cost_d, t_cost_d, m_d));

    auto loss_regv = 0.5 * (cost_reg(p_cost_r, t_cost_r, m_r) +
                            cost_reg(p_cost_d, t_cost_d, m_d));

    auto loss_sigv = 0.5 * (sigma_reg(p_sig_r, t_sig_r, m_r) +
                            sigma_reg(p_sig_d, t_sig_d, m_d));

    auto loss = w_sign * loss_sign + w_reg * loss_regv + w_sig * loss_sigv;

    // Sign accuracy on cost channels only, masked
    auto acc_one = [&](const torch::Tensor& pred_cost,
                       const torch::Tensor& tgt_cost,
                       const torch::Tensor& mask01,
                       int64_t& correct,
                       int64_t& valid_cnt) {
        auto valid = mask01 > 0;
        auto pred_pos = pred_cost > 0;
        auto pred_neg = pred_cost < 0;
        auto tgt_pos  = tgt_cost  > 0;
        auto tgt_neg  = tgt_cost  < 0;

        auto corr = ((pred_pos & tgt_pos) | (pred_neg & tgt_neg)) & valid;
        correct   += corr.sum().item<int64_t>();
        valid_cnt += valid.sum().item<int64_t>();
    };

    int64_t correct = 0, valid_cnt = 0;
    acc_one(p_cost_r, t_cost_r, m_r, correct, valid_cnt);
    acc_one(p_cost_d, t_cost_d, m_d, correct, valid_cnt);

    // Weight for averaging across batches (valid pixels in both directions)
    auto valid_w = 0.5 * (m_r.sum() + m_d.sum()); // scalar tensor

    return BatchStats{loss, valid_w, correct, valid_cnt};
}


double compute_global_pos_weight(torch::data::DataLoader<EdgeDataset>& loader, torch::Device device) {
    torch::NoGradGuard ng;
    double n_neg = 0.0, n_pos = 0.0;

    for (auto& batch : loader) {
        auto tgt = batch.target.to(device);

        auto t_cost_r = tgt.select(1,0);
        auto t_cost_d = tgt.select(1,2);
        auto m_r = tgt.select(1,4) > 0;
        auto m_d = tgt.select(1,5) > 0;

        n_neg += ((t_cost_r < 0) & m_r).sum().item<double>();
        n_pos += ((t_cost_r > 0) & m_r).sum().item<double>();
        n_neg += ((t_cost_d < 0) & m_d).sum().item<double>();
        n_pos += ((t_cost_d > 0) & m_d).sum().item<double>();
    }
    return (n_pos + 1e-6) / (n_neg + 1e-6);
}


int main()
{
    const auto device = torch::kCUDA;

    // -------------------------
    // Train dataset loader
    // -------------------------
    auto train_image_paths = find_image_files_recursively(DATASET_DIR, IMAGE_FORMAT);

    auto train_dataset = EdgeDataset(train_image_paths, /*create_targets=*/true)
        .map(torch::data::transforms::Stack<>());

    const size_t BATCH_SIZE = 8;

    auto train_loader = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions()
            .batch_size(BATCH_SIZE)
            .workers(4)
            .drop_last(true)
    );

    // -------------------------
    // Val dataset loader
    // -------------------------
    auto val_image_paths = find_image_files_recursively(VAL_DATASET_DIR, IMAGE_FORMAT);

    if (val_image_paths.size() > 200) val_image_paths.resize(200);

    auto val_dataset = EdgeDataset(val_image_paths, /*create_targets=*/true)
        .map(torch::data::transforms::Stack<>());

    auto val_loader = torch::data::make_data_loader(
        std::move(val_dataset),
        torch::data::DataLoaderOptions()
            .batch_size(BATCH_SIZE)
            .workers(2)
            .drop_last(false)
    );


    std::cout << "Loaded pretraining data" << std::endl;

    EdgeUNet model;
    model->to(device);

    torch::optim::AdamW optimizer(
        model->parameters(),
        torch::optim::AdamWOptions(1e-3).weight_decay(1e-4)
    );

    using torch::indexing::Slice;

    // convert double to scalar tensor
    auto pos_weight = torch::tensor(
        {static_cast<float>(compute_global_pos_weight(val_loader, device))},
        torch::TensorOptions().dtype(torch::kFloat32).device(device)
    );

    int epochs = 10;
    double best_val_loss = std::numeric_limits<double>::infinity();
    const auto run_id = std::to_string(std::time(nullptr));

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        // =========================
        // Train
        // =========================
        model->train();
        double train_loss_sum = 0.0;
        int train_batches = 0;

        int batch_count = 0;
        for (auto& batch : *train_loader) {
            batch_count++;

            auto imgs = batch.data.to(device, /*non_blocking=*/true);
            auto targets = batch.target.to(device, /*non_blocking=*/true);

            optimizer.zero_grad();

            auto outputs = model->forward(imgs); // [B,6,H,W]

            auto stats = compute_loss_and_signacc(outputs, targets, pos_weight, 1.0, 0.3, 0.1);
            auto loss = stats.loss;

            loss.backward();
            optimizer.step();

            train_loss_sum += loss.item<double>();
            train_batches++;

            if (batch_count % 100 == 0 || batch_count == 1) {
                torch::NoGradGuard ng;

                const double tau = 0.05;
                auto [c_cnt, v_cnt] = sign_counts(outputs, targets, tau);
                double sign_accuracy = (v_cnt > 0.0) ? (c_cnt / v_cnt) : 0.0;

                auto t = targets.index({Slice(), Slice(0, 4), Slice(), Slice()}).detach();

                std::cout << "Epoch [" << epoch << "/" << epochs
                          << "] Batch [" << batch_count
                          << "] Loss: " << loss.item<float>()
                          << " Sign accuracy: " << sign_accuracy
                          << " target min=" << t.min().item<double>()
                          << " max=" << t.max().item<double>()
                          << " mean=" << t.mean().item<double>()
                          << " std=" << t.std().item<double>()
                          << std::endl;

                // =========================
                // Validation
                // =========================
                model->eval();
                torch::NoGradGuard ng;

                double loss_num = 0.0, loss_den = 0.0;
                int64_t correct = 0, valid = 0;

                for (auto& batch : *val_loader) {
                    auto imgs = batch.data.to(device, true);
                    auto targets = batch.target.to(device, true);

                    auto outputs = model->forward(imgs);

                    auto stats = compute_loss_and_signacc(outputs, targets, pos_weight, 1.0, 0.3, 0.1);

                    double w = stats.valid_w.item<double>();
                    loss_num += stats.loss.item<double>() * w;
                    loss_den += w;

                    correct += stats.correct;
                    valid   += stats.valid;
                }

                model->train();

                double val_loss = loss_num / std::max(1e-12, loss_den);
                double val_sign_acc = (valid > 0) ? (double(correct) / double(valid)) : 0.0;

                std::cout << "Epoch [" << epoch << "/" << epochs
                        << "] Val Loss: " << val_loss
                        << " Val Sign accuracy: " << val_sign_acc
                        << std::endl;

                if (val_loss < best_val_loss) {
                    best_val_loss = val_loss;
                    torch::save(model, "fcn_pretrained_" + run_id + "_best.pt");
                    std::cout << "New best val loss: " << best_val_loss << std::endl;
                }
            }
        }

        const double train_avg_loss = train_loss_sum / std::max(1, train_batches);
        std::cout << "Epoch [" << epoch << "/" << epochs
                  << "] Train Average Loss: " << train_avg_loss
                  << std::endl;

        // =========================
        // Checkpointing
        // =========================
        torch::save(model, "fcn_pretrained_" + run_id + "_epoch_" + std::to_string(epoch) + ".pt");
    }

    torch::save(model, "fcn_pretrained_" + run_id + "_final.pt");
    return 0;
}
