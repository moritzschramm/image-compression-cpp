#include <filesystem>
#include <torch/torch.h>
#include <vector>
#include "configuration.h"
#include "image_loader.h"
#include "image_writer.h"
#include "fcn/EdgeDataset.h"
#include "fcn/EdgeUNet.h"


int main()
{
    torch::manual_seed(0);

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

    // Build [B,4,H,W] target and [B,4,H,W] mask from targets [B,6,H,W]
    auto make_target_and_mask = [&](const torch::Tensor& targets_b6hw) {
        auto target = targets_b6hw.index({Slice(), Slice(0, 4), Slice(), Slice()});  // [B,4,H,W]
        auto m = targets_b6hw.index({Slice(), Slice(4, 6), Slice(), Slice()});       // [B,2,H,W]
        auto mask = m.repeat_interleave(2, /*dim=*/1);                               // [B,4,H,W]
        return std::make_pair(target, mask);
    };

    // Accumulate sign-accuracy counts (correct, valid) for channels 0 and 2
    auto sign_counts = [&](const torch::Tensor& outputs_b4hw,
                             const torch::Tensor& targets_b6hw,
                             double tau) {
        auto pred_sel = torch::cat({
            outputs_b4hw.index({Slice(), 0, Slice(), Slice()}).unsqueeze(1),
            outputs_b4hw.index({Slice(), 2, Slice(), Slice()}).unsqueeze(1)
        }, 1); // [B,2,H,W]

        auto target_sel = torch::cat({
            targets_b6hw.index({Slice(), 0, Slice(), Slice()}).unsqueeze(1),
            targets_b6hw.index({Slice(), 2, Slice(), Slice()}).unsqueeze(1)
        }, 1); // [B,2,H,W]

        auto valid = target_sel.abs() > tau;
        auto valid_cnt = valid.sum().item<double>();

        if (valid_cnt <= 0.0) return std::pair<double,double>{0.0, 0.0};

        auto correct = (torch::sign(pred_sel) == torch::sign(target_sel)).logical_and(valid);
        auto correct_cnt = correct.sum().item<double>();
        return std::pair<double,double>{correct_cnt, valid_cnt};
    };

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

            auto outputs = model->forward(imgs);                 // [B,6,H,W]

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

            auto m_r = targets.select(1, 4).to(outputs.dtype());   // [B,H,W] in {0,1}
            auto m_d = targets.select(1, 5).to(outputs.dtype());   // [B,H,W] in {0,1}

            // weighted BCE sign loss for a single cost channel
            auto sign_loss_weighted_bce = [&](const torch::Tensor& pred_cost,
                                            const torch::Tensor& tgt_cost,
                                            const torch::Tensor& mask01) {
                auto valid = mask01 > 0; // bool

                // y=1 means "negative target"
                auto y = ((tgt_cost < 0) & valid).to(pred_cost.dtype());

                // logits = -pred so pred<0 => predicts y=1
                auto logits = -pred_cost;

                auto n_pos = y.sum().to(torch::kFloat32);           // count of negatives
                auto n_all = valid.sum().to(torch::kFloat32);
                auto n_neg = (n_all - n_pos);

                auto pos_weight = ((n_neg + 1e-6f) / (n_pos + 1e-6f))
                                    .clamp_max(20.0f)
                                    .to(pred_cost.device());

                auto bce = torch::binary_cross_entropy_with_logits(
                    logits, y, /*weight=*/{}, /*pos_weight=*/pos_weight, torch::Reduction::None
                );

                auto denom = mask01.sum().clamp_min(1.0);
                return (bce * mask01).sum() / denom;
            };

            // magnitude regression for costs
            auto cost_reg = [&](const torch::Tensor& pred_cost,
                                const torch::Tensor& tgt_cost,
                                const torch::Tensor& mask01) {
                auto reg = torch::smooth_l1_loss(pred_cost, tgt_cost, torch::Reduction::None);
                auto denom = mask01.sum().clamp_min(1.0);
                return (reg * mask01).sum() / denom;
            };

            // regression for sigma (to 0.1)
            auto sigma_reg = [&](const torch::Tensor& pred_sig,
                                const torch::Tensor& tgt_sig,
                                const torch::Tensor& mask01) {
                auto reg = torch::smooth_l1_loss(pred_sig, tgt_sig, torch::Reduction::None);
                auto denom = mask01.sum().clamp_min(1.0);
                return (reg * mask01).sum() / denom;
            };

            // Compute losses
            auto loss_sign_r = sign_loss_weighted_bce(p_cost_r, t_cost_r, m_r);
            auto loss_sign_d = sign_loss_weighted_bce(p_cost_d, t_cost_d, m_d);

            auto loss_reg_r  = cost_reg(p_cost_r, t_cost_r, m_r);
            auto loss_reg_d  = cost_reg(p_cost_d, t_cost_d, m_d);

            auto loss_sig_r  = sigma_reg(p_sig_r,  t_sig_r,  m_r);
            auto loss_sig_d  = sigma_reg(p_sig_d,  t_sig_d,  m_d);

            // Combine
            double w_sign = 1.0;
            double w_reg  = 0.2;
            double w_sig  = 0.2;

            auto loss = w_sign * 0.5 * (loss_sign_r + loss_sign_d)
                    + w_reg  * 0.5 * (loss_reg_r  + loss_reg_d)
                    + w_sig  * 0.5 * (loss_sig_r  + loss_sig_d);


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

                double val_num_sum = 0.0;
                double val_den_sum = 0.0;
                double val_correct_sum = 0.0;
                double val_valid_sum = 0.0;

                for (auto& batch : *val_loader) {
                    auto imgs = batch.data.to(device, /*non_blocking=*/true);
                    auto targets = batch.target.to(device, /*non_blocking=*/true);

                    auto outputs = model->forward(imgs);

                    auto [target, mask] = make_target_and_mask(targets);
                    mask = mask.to(outputs.dtype());

                    auto abs_err = (outputs - target).abs();
                    auto num = (abs_err * mask).sum();
                    auto den = mask.sum().clamp_min(1.0);

                    val_num_sum += num.item<double>();
                    val_den_sum += den.item<double>();

                    auto [c_cnt, v_cnt] = sign_counts(outputs, targets, tau);
                    val_correct_sum += c_cnt;
                    val_valid_sum += v_cnt;
                }
                model->train();

                const double val_loss = val_num_sum / std::max(1.0, val_den_sum);
                const double val_sign_acc = (val_valid_sum > 0.0) ? (val_correct_sum / val_valid_sum) : 0.0;

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
