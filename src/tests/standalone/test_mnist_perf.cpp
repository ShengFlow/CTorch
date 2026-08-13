#include "mnist/mnist_loader.h"
#include "AutoGrad.h"
#include "CtorchError.h"
#include "CtorchScheduler.h"
#include "C3/C3Cleanup.h"
#include "ctQALS/Random.h"
#include "src/kernels/kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using hires = std::chrono::high_resolution_clock;
using ms = std::chrono::duration<double, std::milli>;

static double to_ms(hires::time_point t0, hires::time_point t1) {
    return std::chrono::duration_cast<ms>(t1 - t0).count();
}

struct PerfResult {
    std::string device_name = "CPU";
    int batch_size = 0;
    int epochs = 0;
    double total_ms = 0.0;
    double fwd_ms = 0.0;
    double bwd_ms = 0.0;
    double upd_ms = 0.0;
    double other_ms = 0.0;
    double train_acc = 0.0;
    double test_acc = 0.0;
    double final_loss = 0.0;
};

static DeviceType parse_device(const std::string& s) {
    if (s == "mps" || s == "MPS") return DeviceType::kMPS;
    if (s == "cpu" || s == "CPU") return DeviceType::kCPU;
    std::cerr << "Unknown device: " << s << " (use cpu|mps)\n";
    std::exit(1);
}

static void xavier_init(Tensor& W, ctQALS::rng::Xoshiro256PlusPlus& rng,
                        size_t fan_in, size_t fan_out) {
    float std = std::sqrt(2.0f / static_cast<float>(fan_in + fan_out));
    float* data = W.data_write<float>();
    for (size_t i = 0; i < W.numel(); ++i) {
        float r = 2.0f * rng.uniform_f32() - 1.0f;
        data[i] = r * std;
    }
}

class PerfNet {
public:
    PerfNet(size_t in, size_t h1, size_t h2, size_t out, float lr, DeviceType dev,
            ctQALS::rng::Xoshiro256PlusPlus& rng)
        : learning_rate(lr), device_(dev) {
        W1 = Tensor(ShapeTag{}, {in, h1}, DType::kFloat, dev);
        b1 = Tensor(ShapeTag{}, {h1}, DType::kFloat, dev);
        W2 = Tensor(ShapeTag{}, {h1, h2}, DType::kFloat, dev);
        b2 = Tensor(ShapeTag{}, {h2}, DType::kFloat, dev);
        W3 = Tensor(ShapeTag{}, {h2, out}, DType::kFloat, dev);
        b3 = Tensor(ShapeTag{}, {out}, DType::kFloat, dev);

        W1.requires_grad(true); b1.requires_grad(true);
        W2.requires_grad(true); b2.requires_grad(true);
        W3.requires_grad(true); b3.requires_grad(true);

        xavier_init(W1, rng, in, h1);
        xavier_init(W2, rng, h1, h2);
        xavier_init(W3, rng, h2, out);
        b1.zero(); b2.zero(); b3.zero();
    }

    Tensor forward(const Tensor& x) {
        Tensor z1 = x.matmul(W1) + b1;
        Tensor h1 = z1.relu();
        Tensor z2 = h1.matmul(W2) + b2;
        Tensor h2 = z2.relu();
        Tensor logits = h2.matmul(W3) + b3;
        return logits;
    }

    float train_step(const Tensor& x, const Tensor& y_onehot,
                     double* out_fwd_ms, double* out_bwd_ms, double* out_upd_ms) {
        auto t0 = hires::now();
        Tensor logits = forward(x);
        Tensor loss = logits.cross_entropy(y_onehot);
        float loss_value = loss.item<float>();
        auto t1 = hires::now();

        AutoGrad::backward(loss.getRelatedNode(), false);
        auto t2 = hires::now();

        update_parameters();
        auto t3 = hires::now();

        *out_fwd_ms = to_ms(t0, t1);
        *out_bwd_ms = to_ms(t1, t2);
        *out_upd_ms = to_ms(t2, t3);
        return loss_value;
    }

    void update_parameters() {
        if (device_ == DeviceType::kMPS) {
            MPS_flush_wait(true);
            MPS_update_begin();
            SGD_Step_Zero_MPS_kernel(W1, W1.grad(), learning_rate);
            SGD_Step_Zero_MPS_kernel(b1, b1.grad(), learning_rate);
            SGD_Step_Zero_MPS_kernel(W2, W2.grad(), learning_rate);
            SGD_Step_Zero_MPS_kernel(b2, b2.grad(), learning_rate);
            SGD_Step_Zero_MPS_kernel(W3, W3.grad(), learning_rate);
            SGD_Step_Zero_MPS_kernel(b3, b3.grad(), learning_rate);
            MPS_update_end();
        } else {
            auto sgd = [this](Tensor& param) {
                float* p = param.data_write<float>();
                float* g = param.grad_ptr();
                for (size_t i = 0; i < param.numel(); ++i) {
                    p[i] -= g[i] * learning_rate;
                }
            };
            sgd(W1); sgd(b1); sgd(W2); sgd(b2); sgd(W3); sgd(b3);

            W1.zero_grad(); b1.zero_grad();
            W2.zero_grad(); b2.zero_grad();
            W3.zero_grad(); b3.zero_grad();
        }
    }

    Tensor predict(const Tensor& x) {
        Tensor logits = forward(x);
        if (device_ == DeviceType::kMPS) {
            MPS_flush_wait(true);
        }
        std::vector<size_t> shape = logits.sizes();
        Tensor probs(ShapeTag{}, shape, logits.dtype(), logits.device());
        const float* in = logits.data_read<float>();
        float* out = probs.data_write<float>();
        size_t batch = shape[0];
        size_t classes = shape[1];
        for (size_t i = 0; i < batch; ++i) {
            float max_val = in[i * classes];
            for (size_t j = 1; j < classes; ++j) {
                max_val = std::max(max_val, in[i * classes + j]);
            }
            float sum = 0.0f;
            for (size_t j = 0; j < classes; ++j) {
                float e = std::exp(in[i * classes + j] - max_val);
                out[i * classes + j] = e;
                sum += e;
            }
            for (size_t j = 0; j < classes; ++j) {
                out[i * classes + j] /= sum;
            }
        }
        return probs;
    }

    float learning_rate;
private:
    DeviceType device_;
    Tensor W1, b1, W2, b2, W3, b3;
};

static Tensor make_one_hot(const Tensor& labels, DeviceType dev) {
    size_t batch = labels.numel();
    Tensor one_hot(ShapeTag{}, {batch, 10}, DType::kFloat, dev);
    std::memset(one_hot.data_write<float>(), 0, batch * 10 * sizeof(float));
    const float* lp = labels.data_read<float>();
    float* op = one_hot.data_write<float>();
    for (size_t i = 0; i < batch; ++i) {
        int lab = static_cast<int>(lp[i]);
        op[i * 10 + lab] = 1.0f;
    }
    return one_hot;
}

static float calculate_accuracy(const Tensor& y_pred, const Tensor& y_true) {
    size_t total = y_true.shape()[0];
    int correct = 0;
    const float* pred = y_pred.data_read<float>();
    const float* true_labels = y_true.data_read<float>();
    for (size_t i = 0; i < total; ++i) {
        int pred_label = 0;
        float max_prob = -1.0f;
        for (int j = 0; j < 10; ++j) {
            float prob = pred[i * 10 + j];
            if (prob > max_prob) {
                max_prob = prob;
                pred_label = j;
            }
        }
        if (pred_label == static_cast<int>(true_labels[i])) {
            ++correct;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(total);
}

static PerfResult run_benchmark(DeviceType dev, int batch_size, int epochs,
                                int seed, bool quiet) {
    CtorchScheduler::getInstance();

    MNISTLoader loader(".", dev);
    Tensor train_images, train_labels;
    Tensor test_images, test_labels;
    loader.load_training_data(train_images, train_labels);
    loader.load_test_data(test_images, test_labels);

    size_t train_n = train_images.shape()[0];
    size_t test_n = test_images.shape()[0];
    int num_batches = static_cast<int>(train_n) / batch_size;

    ctQALS::rng::Xoshiro256PlusPlus rng(seed);
    PerfNet net(784, 256, 128, 10, 0.001f, dev, rng);

    PerfResult res;
    res.device_name = (dev == DeviceType::kMPS ? "MPS" : "CPU");
    res.batch_size = batch_size;
    res.epochs = epochs;

    double total_fwd = 0.0, total_bwd = 0.0, total_upd = 0.0;
    double epoch_loss_sum = 0.0;

    auto t_start = hires::now();
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_fwd = 0.0, epoch_bwd = 0.0, epoch_upd = 0.0;
        float epoch_loss = 0.0f;
        for (int b = 0; b < num_batches; ++b) {
            int start = b * batch_size;
            Tensor batch_x(ShapeTag{}, {static_cast<size_t>(batch_size), 784},
                           DType::kFloat, dev);
            Tensor batch_y(ShapeTag{}, {static_cast<size_t>(batch_size)},
                           DType::kFloat, dev);
            std::memcpy(batch_x.data_write<float>(),
                        train_images.data_read<float>() + start * 784,
                        batch_size * 784 * sizeof(float));
            std::memcpy(batch_y.data_write<float>(),
                        train_labels.data_read<float>() + start,
                        batch_size * sizeof(float));
            batch_x.requires_grad(false);
            batch_y.requires_grad(false);

            Tensor y_onehot = make_one_hot(batch_y, dev);
            double fwd_ms = 0.0, bwd_ms = 0.0, upd_ms = 0.0;
            float loss = net.train_step(batch_x, y_onehot, &fwd_ms, &bwd_ms, &upd_ms);
            epoch_loss += loss;
            epoch_fwd += fwd_ms;
            epoch_bwd += bwd_ms;
            epoch_upd += upd_ms;
        }
        epoch_loss_sum += epoch_loss / static_cast<float>(num_batches);
        total_fwd += epoch_fwd;
        total_bwd += epoch_bwd;
        total_upd += epoch_upd;
        if (!quiet) {
            std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                      << " loss=" << std::fixed << std::setprecision(4)
                      << (epoch_loss / num_batches) << std::endl;
        }
    }
    auto t_end = hires::now();

    res.total_ms = to_ms(t_start, t_end);
    res.fwd_ms = total_fwd;
    res.bwd_ms = total_bwd;
    res.upd_ms = total_upd;
    res.other_ms = res.total_ms - (total_fwd + total_bwd + total_upd);
    res.final_loss = epoch_loss_sum / epochs;

    Tensor train_pred = net.predict(train_images);
    res.train_acc = calculate_accuracy(train_pred, train_labels);
    Tensor test_pred = net.predict(test_images);
    res.test_acc = calculate_accuracy(test_pred, test_labels);

    return res;
}

static void print_result(const PerfResult& r) {
    double samples = static_cast<double>(r.batch_size) *
                     static_cast<double>(60000 / r.batch_size) * r.epochs;
    double throughput = samples / (r.total_ms / 1000.0);
    double fwd_pct = 100.0 * r.fwd_ms / r.total_ms;
    double bwd_pct = 100.0 * r.bwd_ms / r.total_ms;
    double upd_pct = 100.0 * r.upd_ms / r.total_ms;
    double other_pct = 100.0 * r.other_ms / r.total_ms;

    std::cout << "=== Benchmark Result ===\n"
              << "Device:        " << r.device_name << "\n"
              << "Batch size:    " << r.batch_size << "\n"
              << "Epochs:        " << r.epochs << "\n"
              << "Total time:    " << std::fixed << std::setprecision(1)
              << r.total_ms << " ms\n"
              << "Throughput:    " << std::setprecision(1) << throughput
              << " samples/s\n"
              << "Forward:       " << std::setprecision(1) << r.fwd_ms
              << " ms (" << std::setprecision(2) << fwd_pct << "%)\n"
              << "Backward:      " << std::setprecision(1) << r.bwd_ms
              << " ms (" << std::setprecision(2) << bwd_pct << "%)\n"
              << "Update:        " << std::setprecision(1) << r.upd_ms
              << " ms (" << std::setprecision(2) << upd_pct << "%)\n"
              << "Other/data:    " << std::setprecision(1) << r.other_ms
              << " ms (" << std::setprecision(2) << other_pct << "%)\n"
              << "Train acc:     " << std::setprecision(2) << (r.train_acc * 100.0f)
              << "%\n"
              << "Test acc:      " << std::setprecision(2) << (r.test_acc * 100.0f)
              << "%\n"
              << "Final loss:    " << std::setprecision(4) << r.final_loss << "\n";
}

static void run_sweep(DeviceType dev, const std::vector<int>& batch_sizes,
                      int epochs, int seed, bool quiet) {
    std::cout << "=== Batch size sweep ("
              << (dev == DeviceType::kMPS ? "MPS" : "CPU") << ", "
              << epochs << " epoch) ===\n"
              << std::setw(6) << "BS"
              << std::setw(12) << "time(ms)"
              << std::setw(14) << "samples/s"
              << std::setw(12) << "fwd%"
              << std::setw(12) << "bwd%"
              << std::setw(12) << "upd%"
              << std::setw(12) << "other%"
              << std::setw(12) << "test_acc"
              << "\n";
    for (int bs : batch_sizes) {
        PerfResult r = run_benchmark(dev, bs, epochs, seed, quiet);
        double samples = static_cast<double>(bs) *
                         static_cast<double>(60000 / bs) * epochs;
        double throughput = samples / (r.total_ms / 1000.0);
        std::cout << std::setw(6) << bs
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.total_ms
                  << std::setw(14) << std::setprecision(1) << throughput
                  << std::setw(12) << std::setprecision(1)
                  << (100.0 * r.fwd_ms / r.total_ms)
                  << std::setw(12) << std::setprecision(1)
                  << (100.0 * r.bwd_ms / r.total_ms)
                  << std::setw(12) << std::setprecision(1)
                  << (100.0 * r.upd_ms / r.total_ms)
                  << std::setw(12) << std::setprecision(1)
                  << (100.0 * r.other_ms / r.total_ms)
                  << std::setw(12) << std::setprecision(2) << (r.test_acc * 100.0f)
                  << "\n";
    }
}

static void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0
              << " --device cpu|mps [--batch N] [--epochs N] [--seed N]\n"
              << "       " << argv0
              << " --sweep --device cpu|mps [--epochs N] [--seed N] [--quiet]\n";
}

int main(int argc, char** argv) {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    std::string device_str = "mps";
    int batch_size = 128;
    int epochs = 1;
    int seed = 42;
    bool sweep = false;
    bool quiet = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device_str = argv[++i];
        } else if (arg == "--batch" && i + 1 < argc) {
            batch_size = std::atoi(argv[++i]);
        } else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::atoi(argv[++i]);
        } else if (arg == "--sweep") {
            sweep = true;
        } else if (arg == "--quiet") {
            quiet = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    DeviceType dev = parse_device(device_str);

    if (sweep) {
        std::vector<int> batch_sizes = {32, 64, 128, 256, 512};
        if (dev == DeviceType::kCPU) {
            // CPU sweep uses smaller set to keep total time reasonable
            batch_sizes = {64, 128, 256};
        }
        run_sweep(dev, batch_sizes, epochs, seed, quiet);
    } else {
        PerfResult r = run_benchmark(dev, batch_size, epochs, seed, quiet);
        print_result(r);
    }

    ct::c3::shutdownAll();
    return 0;
}
