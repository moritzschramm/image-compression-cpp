#include <filesystem>
#include <torch/torch.h>
#include <vector>
#include "configuration.h"
#include "image_loader.h"
#include "image_writer.h"
#include "fcn/EdgeDataset.h"
#include "fcn/EdgeUNet.h"
#include <cstdint>
#include <algorithm>

struct EdgeMetrics {
    int64_t TP_conn = 0; // predict connect and is connect
    int64_t FP_conn = 0; // predict connect but is cut
    int64_t TN_conn = 0; // predict cut and is cut
    int64_t FN_conn = 0; // predict cut but is connect

    int64_t TP_cut = 0;  // predict cut and is cut
    int64_t FP_cut = 0;  // predict cut but is connect
    int64_t TN_cut = 0;  // predict connect and is connect
    int64_t FN_cut = 0;  // predict connect but is cut

    double precision_conn = 0.0; // TP / (TP+FP)
    double recall_conn    = 0.0; // TP / (TP+FN)
    double f1_conn        = 0.0; // 2PR/(P+R)

    double precision_cut = 0.0;
    double recall_cut    = 0.0;
    double f1_cut        = 0.0;
};

inline EdgeMetrics compute_edge_metrics(
    const torch::Tensor& logits_r,   // [B,H,W] float
    const torch::Tensor& logits_d,   // [B,H,W] float
    const torch::Tensor& y_r,        // [B,H,W] float in {0,1} (1 = connect)
    const torch::Tensor& y_d,        // [B,H,W] float in {0,1} (1 = connect)
    const torch::Tensor& mask_r,     // [B,H,W] float/bool in {0,1}
    const torch::Tensor& mask_d,     // [B,H,W] float/bool in {0,1}
    double thresh = 0.5              // threshold on probability p=sigmoid(logit)
) {
    TORCH_CHECK(logits_r.sizes() == y_r.sizes(), "logits_r and y_r shape mismatch");
    TORCH_CHECK(logits_d.sizes() == y_d.sizes(), "logits_d and y_d shape mismatch");
    TORCH_CHECK(mask_r.sizes()  == y_r.sizes(), "mask_r and y_r shape mismatch");
    TORCH_CHECK(mask_d.sizes()  == y_d.sizes(), "mask_d and y_d shape mismatch");

    // predict connect if p >= thresh  <=>  logit >= log(thresh/(1-thresh))
    double logit_thresh = std::log(thresh / (1.0 - thresh));

    auto pred_conn_r = (logits_r >= logit_thresh);
    auto pred_conn_d = (logits_d >= logit_thresh);

    auto gt_conn_r = (y_r >= 0.5);
    auto gt_conn_d = (y_d >= 0.5);

    auto m_r = (mask_r > 0.5);
    auto m_d = (mask_d > 0.5);

    const double eps = 1e-12;

    // Confusion counts for "connect" as positive class
    auto tp_conn = ((pred_conn_r & gt_conn_r & m_r).sum() + (pred_conn_d & gt_conn_d & m_d).sum()).item<int64_t>();
    auto fp_conn = ((pred_conn_r & (~gt_conn_r) & m_r).sum() + (pred_conn_d & (~gt_conn_d) & m_d).sum()).item<int64_t>();
    auto fn_conn = (((~pred_conn_r) & gt_conn_r & m_r).sum() + ((~pred_conn_d) & gt_conn_d & m_d).sum()).item<int64_t>();
    auto tn_conn = (((~pred_conn_r) & (~gt_conn_r) & m_r).sum() + ((~pred_conn_d) & (~gt_conn_d) & m_d).sum()).item<int64_t>();

    auto pred_cut_r = ~pred_conn_r;
    auto pred_cut_d = ~pred_conn_d;
    auto gt_cut_r = ~gt_conn_r;
    auto gt_cut_d = ~gt_conn_d;

    // Confusion counts for "cut" as positive class
    auto tp_cut = ((pred_cut_r & gt_cut_r & m_r).sum() + (pred_cut_d & gt_cut_d & m_d).sum()).item<int64_t>();
    auto fp_cut = ((pred_cut_r & (~gt_cut_r) & m_r).sum() + (pred_cut_d & (~gt_cut_d) & m_d).sum()).item<int64_t>();
    auto fn_cut = (((~pred_cut_r) & gt_cut_r & m_r).sum() + ((~pred_cut_d) & gt_cut_d & m_d).sum()).item<int64_t>();
    auto tn_cut = (((~pred_cut_r) & (~gt_cut_r) & m_r).sum() + ((~pred_cut_d) & (~gt_cut_d) & m_d).sum()).item<int64_t>();

    EdgeMetrics out;
    out.TP_conn = tp_conn; out.FP_conn = fp_conn; out.FN_conn = fn_conn; out.TN_conn = tn_conn;
    out.TP_cut  = tp_cut;  out.FP_cut  = fp_cut;  out.FN_cut  = fn_cut;  out.TN_cut  = tn_cut;

    out.precision_conn = double(tp_conn) / (double(tp_conn + fp_conn) + eps);
    out.recall_conn    = double(tp_conn) / (double(tp_conn + fn_conn) + eps);
    out.f1_conn        = (2.0 * out.precision_conn * out.recall_conn) / (out.precision_conn + out.recall_conn + eps);

    out.precision_cut = double(tp_cut) / (double(tp_cut + fp_cut) + eps);
    out.recall_cut    = double(tp_cut) / (double(tp_cut + fn_cut) + eps);
    out.f1_cut        = (2.0 * out.precision_cut * out.recall_cut) / (out.precision_cut + out.recall_cut + eps);
    return out;
}


struct BatchStats {
    torch::Tensor loss;      // scalar
    torch::Tensor valid_w;   // scalar weight for averaging (number of valid pixels)
    int64_t correct = 0;
    int64_t valid = 0;
};

torch::Tensor masked_mean(const torch::Tensor& x, const torch::Tensor& mask) {
    // x, m01: same shape, m01 in {0,1}
    auto m = mask.to(x.dtype());
    auto denom = m.sum().clamp_min(1.0);
    return (x * m).sum() / denom;
}

BatchStats compute_loss_and_signacc(
    const torch::Tensor& outputs,   // [B,4,H,W]
    const torch::Tensor& targets,   // [B,4,H,W]
    const torch::Tensor& pos_weight, // scalar tensor
    double w_sign = 1.0,
    double w_sig  = 0.01
) {
    // Targets
    auto y_cost_r = targets.select(1, 0); // [B,H,W] in {0,1}
    auto y_cost_d = targets.select(1, 1); // [B,H,W] in {0,1}
    auto mask_r   = targets.select(1, 2); // [B,H,W] in {0,1}
    auto mask_d   = targets.select(1, 3); // [B,H,W] in {0,1}

    // Predictions
    auto logit_r   = outputs.select(1, 0); // [B,H,W] logits
    auto sigma_r_z = outputs.select(1, 1); // [B,H,W] unconstrained
    auto logit_d   = outputs.select(1, 2);
    auto sigma_d_z = outputs.select(1, 3);

    // Masked BCE with logits for costs
    auto bce_opts = torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions()
                    .reduction(torch::kNone);

    auto bce_r = torch::nn::functional::binary_cross_entropy_with_logits(logit_r, y_cost_r, bce_opts);
    auto bce_d = torch::nn::functional::binary_cross_entropy_with_logits(logit_d, y_cost_d, bce_opts);

    auto pos_w = pos_weight.to(outputs.dtype());

    // y==1 (connect) weight pos_w, y==0 (cut) weight 1
    // With pos_w < 1, this downweights the overrepresented connect class.
    auto w_r = (1.0 - y_cost_r) + y_cost_r * pos_w;
    auto w_d = (1.0 - y_cost_d) + y_cost_d * pos_w;

    auto num = (bce_r * w_r * mask_r).sum() + (bce_d * w_d * mask_d).sum();
    auto den = (w_r * mask_r).sum() + (w_d * mask_d).sum();
    auto loss_sign = num / den.clamp_min(1.0);

    auto denom_r = mask_r.sum().clamp_min(1.0);
    auto denom_d = mask_d.sum().clamp_min(1.0);
    auto valid_w = (denom_r + denom_d); // scalar tensor

    auto p_r = torch::sigmoid(logit_r);
    auto p_d = torch::sigmoid(logit_d);

    // Sigma head: map to [0.1, 0.9]
    // sigma is interpreted as std dev in RL training
    const double sigma_min = 0.1;
    const double sigma_max = 0.9;
    auto sigma_r = sigma_min + (sigma_max - sigma_min) * torch::sigmoid(sigma_r_z);
    auto sigma_d = sigma_min + (sigma_max - sigma_min) * torch::sigmoid(sigma_d_z);
    sigma_r = sigma_r.clamp_min(1e-4);
    sigma_d = sigma_d.clamp_min(1e-4);

    // Self-supervised calibration loss for sigma
    // Detach p so sigma learns to explain current errors
    auto err2_r = (p_r.detach() - y_cost_r).pow(2);
    auto err2_d = (p_d.detach() - y_cost_d).pow(2);

    auto nll_r = 0.5 * (err2_r / sigma_r.pow(2) + torch::log(sigma_r.pow(2)));
    auto nll_d = 0.5 * (err2_d / sigma_d.pow(2) + torch::log(sigma_d.pow(2)));

    auto loss_sig = ( (nll_r * mask_r).sum() + (nll_d * mask_d).sum() ) / valid_w;

    // Total loss
    auto loss = (w_sign * loss_sign) + (w_sig * loss_sig);

    // Sign accuracy on valid edges (threshold 0.5 on sigmoid(logit))
    auto pred_r = (p_r >= 0.5).to(torch::kInt64);
    auto pred_d = (p_d >= 0.5).to(torch::kInt64);
    auto gt_r   = (y_cost_r >= 0.5).to(torch::kInt64);
    auto gt_d   = (y_cost_d >= 0.5).to(torch::kInt64);

    auto m_r_i64 = mask_r.to(torch::kInt64);
    auto m_d_i64 = mask_d.to(torch::kInt64);

    int64_t correct_r = ((pred_r == gt_r).to(torch::kInt64) * m_r_i64).sum().item<int64_t>();
    int64_t correct_d = ((pred_d == gt_d).to(torch::kInt64) * m_d_i64).sum().item<int64_t>();
    int64_t valid_cnt = (mask_r.sum() + mask_d.sum()).item<int64_t>();

    return BatchStats {loss, valid_w.detach(), correct_r + correct_d, valid_cnt};
}

template <typename DataLoader>
double compute_global_pos_weight(DataLoader& loader) {
    torch::NoGradGuard ng;
    double n_neg = 0.0, n_pos = 0.0;

    for (auto& batch : loader) {
        auto tgt = batch.target;

        auto t_cost_r = tgt.select(1, 0);
        auto t_cost_d = tgt.select(1, 1);
        auto m_r = tgt.select(1, 2) > 0;
        auto m_d = tgt.select(1, 3) > 0;

        n_neg += ((t_cost_r < 0.5) & m_r).sum().template item<double>();
        n_pos += ((t_cost_r > 0.5) & m_r).sum().template item<double>();
        n_neg += ((t_cost_d < 0.5) & m_d).sum().template item<double>();
        n_pos += ((t_cost_d > 0.5) & m_d).sum().template item<double>();
    }
    return (n_neg + 1e-6) / (n_pos + 1e-6);
}

// -------------------------
// Pretraining
// -------------------------
int main()
{
    const auto device = torch::kCUDA;
    const auto TRAIN_DATASET_SIZE = 1e5;
    const auto VAL_DATASET_SIZE = 128;

    EdgeUNet model;
    model->to(device);

    torch::optim::AdamW optimizer(
        model->parameters(),
        torch::optim::AdamWOptions(1e-3).weight_decay(1e-4)
    );

    // -------------------------
    // Train dataset loader
    // -------------------------
    auto train_image_paths = find_image_files_recursively(DATASET_DIR, IMAGE_FORMAT);

    if (train_image_paths.size() > TRAIN_DATASET_SIZE) train_image_paths.resize(TRAIN_DATASET_SIZE);

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

    if (val_image_paths.size() > VAL_DATASET_SIZE) val_image_paths.resize(VAL_DATASET_SIZE);

    auto val_dataset = EdgeDataset(val_image_paths, /*create_targets=*/true)
        .map(torch::data::transforms::Stack<>());

    auto val_loader = torch::data::make_data_loader(
        std::move(val_dataset),
        torch::data::DataLoaderOptions()
            .batch_size(BATCH_SIZE)
            .workers(2)
            .drop_last(false)
    );

    auto pos_weight = torch::tensor(
        {static_cast<float>(0.1)}, // connect-class weight; <1 downweights connect to emphasize cuts
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

            auto outputs = model->forward(imgs); // [B,4,H,W]

            auto stats = compute_loss_and_signacc(outputs, targets, pos_weight);
            auto loss = stats.loss;

            loss.backward();
            optimizer.step();

            train_loss_sum += loss.item<double>();
            train_batches++;

            if (batch_count % 100 == 0 || batch_count == 1) {
                // =========================
                // Validation
                // =========================
                model->eval();
                torch::NoGradGuard ng;

                auto logit_r = outputs.select(1,0);
                auto logit_d = outputs.select(1,2);

                auto y_r    = targets.select(1,0);
                auto y_d    = targets.select(1,1);
                auto mask_r = targets.select(1,2);
                auto mask_d = targets.select(1,3);

                EdgeMetrics train_m = compute_edge_metrics(logit_r, logit_d, y_r, y_d, mask_r, mask_d, 0.5);

                double loss_num = 0.0, loss_den = 0.0;
                int64_t correct = 0, valid = 0;
                int64_t val_tp_conn = 0, val_fp_conn = 0, val_fn_conn = 0, val_tn_conn = 0;
                int64_t val_tp_cut = 0, val_fp_cut = 0, val_fn_cut = 0, val_tn_cut = 0;

                for (auto& batch : *val_loader) {
                    auto imgs = batch.data.to(device, true);
                    auto targets = batch.target.to(device, true);

                    auto outputs = model->forward(imgs);

                    auto stats = compute_loss_and_signacc(outputs, targets, pos_weight);

                    double w = stats.valid_w.item<double>();
                    loss_num += stats.loss.item<double>() * w;
                    loss_den += w;

                    correct += stats.correct;
                    valid   += stats.valid;

                    auto v_logit_r = outputs.select(1, 0);
                    auto v_logit_d = outputs.select(1, 2);
                    auto v_y_r     = targets.select(1, 0);
                    auto v_y_d     = targets.select(1, 1);
                    auto v_mask_r  = targets.select(1, 2);
                    auto v_mask_d  = targets.select(1, 3);

                    EdgeMetrics vm = compute_edge_metrics(v_logit_r, v_logit_d, v_y_r, v_y_d, v_mask_r, v_mask_d, 0.5);
                    val_tp_conn += vm.TP_conn; val_fp_conn += vm.FP_conn; val_fn_conn += vm.FN_conn; val_tn_conn += vm.TN_conn;
                    val_tp_cut  += vm.TP_cut;  val_fp_cut  += vm.FP_cut;  val_fn_cut  += vm.FN_cut;  val_tn_cut  += vm.TN_cut;
                }

                model->train();

                double val_loss = loss_num / std::max(1e-12, loss_den);
                double val_sign_acc = (valid > 0) ? (double(correct) / double(valid)) : 0.0;
                const double eps = 1e-12;
                double val_prec_conn = double(val_tp_conn) / (double(val_tp_conn + val_fp_conn) + eps);
                double val_rec_conn  = double(val_tp_conn) / (double(val_tp_conn + val_fn_conn) + eps);
                double val_f1_conn   = (2.0 * val_prec_conn * val_rec_conn) / (val_prec_conn + val_rec_conn + eps);
                double val_prec_cut = double(val_tp_cut) / (double(val_tp_cut + val_fp_cut) + eps);
                double val_rec_cut  = double(val_tp_cut) / (double(val_tp_cut + val_fn_cut) + eps);
                double val_f1_cut   = (2.0 * val_prec_cut * val_rec_cut) / (val_prec_cut + val_rec_cut + eps);

                double train_sign_accuracy = (stats.valid > 0) ? (double(stats.correct) / double(stats.valid)) : 0.0;

                std::cout << "Epoch [" << epoch << "/" << epochs << "] Batch [" << batch_count << "]\n"
                          << "  Train: loss=" << loss.item<float>()
                          << " sign_acc=" << train_sign_accuracy
                          << " conn P/R/F1=" << train_m.precision_conn << "/" << train_m.recall_conn << "/" << train_m.f1_conn
                          << " cut P/R/F1=" << train_m.precision_cut << "/" << train_m.recall_cut << "/" << train_m.f1_cut
                          << "\n"
                          << "  Val:   loss=" << val_loss
                          << " sign_acc=" << val_sign_acc
                          << " conn P/R/F1=" << val_prec_conn << "/" << val_rec_conn << "/" << val_f1_conn
                          << " cut P/R/F1=" << val_prec_cut << "/" << val_rec_cut << "/" << val_f1_cut
                          << std::endl;

                if (val_loss < best_val_loss) {
                    best_val_loss = val_loss;
                    torch::save(model, "fcn_pretrained_" + run_id + "_best.pt");
                    // std::cout << "New best val loss: " << best_val_loss << std::endl;
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
