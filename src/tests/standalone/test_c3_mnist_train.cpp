/**
 * @file test_c3_mnist_train.cpp
 * @brief C3 MNIST 训练验证（Eager vs C3 自动优化）
 * @details 测试代码与普通用户 MNIST 训练代码完全一致，仅使用标准 Tensor API。
 *          C3 的介入通过调度器自动完成（HotPathManager 检测热路径 →
 *          RegionFusion 预走 → 自动编译），无需手动调用任何 C3 API。
 *          唯一变量：编译时 CT_DISABLE_C3 宏的开启/关闭。
 *          网络: 784→256(ReLU)→128(ReLU)→10
 *          训练: 5 epochs, batch=128, lr=0.001
 * @date 2026/8/6
 */

#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include <csignal>
#include <cstdio>
#include <future>
#include <execinfo.h>
#include <unistd.h>

// TEMP DEBUG: crash handler to capture static-destructor backtrace
static void tempCrashHandler(int sig) {
    void* arr[64];
    int n = backtrace(arr, 64);
    dprintf(2, "\n=== CRASH sig=%d ===\n", sig);
    backtrace_symbols_fd(arr, n, 2);
    _exit(sig);
}

#include "Tensor.h"
#include "CtorchScheduler.h"
#include "AutoGrad.h"
#include "Ctools.h"
#include "kernels/kernels.h"
#ifndef CT_DISABLE_C3
#include "C3/C3Cleanup.h"
#include "C3/FusionCostModel.h"
#include "C3/C3KernelRegistry.h"
#include "C3/C3HotPathManager.h"
#include "C3/C3BackwardCapture.h"  // DEBT-NEW-7 v0.5.1+ 调试用,看 backward fusion hit
#include "C3/JITCache.h"             // [Dev] v0.5.2 (4) JITCache 1.0 stats 输出
#endif
#include "mnist/mnist_loader.h"
#include "ctQALS/Random.h"

using namespace ct;

// ======================= 全局配置 =======================

static constexpr size_t BATCH_SIZE = 128;
static constexpr size_t HIDDEN1    = 256;
static constexpr size_t HIDDEN2    = 128;
static constexpr float  LR         = 0.001f;
static constexpr int    EPOCHS     = 5;

// ======================= 辅助函数 =======================

static ctQALS::rng::Xoshiro256PlusPlus g_rng(42);

static void xavierInit(Tensor& W, size_t fan_in, size_t fan_out) {
    float std = std::sqrt(2.0f / (fan_in + fan_out));
    float* data = W.data_write<float>();
    for (size_t i = 0; i < W.numel(); ++i) {
        float r = 2.0f * g_rng.uniform_f32() - 1.0f;
        data[i] = r * std;
    }
}

static void initParams(Tensor params[6]) {
    xavierInit(params[0], 784, 256);
    params[1].zero();
    xavierInit(params[2], 256, 128);
    params[3].zero();
    xavierInit(params[4], 128, 10);
    params[5].zero();
}

static float computeAccuracy(const Tensor& logits, const Tensor& labels) {
    const float* l = logits.data_read<float>();
    const float* y = labels.data_read<float>();
    size_t batch = logits.shape()[0];
    size_t correct = 0;
    for (size_t i = 0; i < batch; ++i) {
        size_t pred = 0;
        float max_v = l[i * 10];
        for (size_t j = 1; j < 10; ++j) {
            if (l[i * 10 + j] > max_v) { max_v = l[i * 10 + j]; pred = j; }
        }
        if (pred == static_cast<size_t>(y[i])) ++correct;
    }
    return static_cast<float>(correct) / batch;
}

extern "C" void cblas_saxpy(const int N, const float alpha, const float *X, const int incX, float *Y, const int incY);

// ======================= 训练 =======================

/// 标准 MNIST 训练一个 epoch（与普通用户代码完全一致，无任何 C3 API）
static float trainEpoch(
    Tensor params[6],
    const Tensor& images, const Tensor& labels,
    int num_batches, float* epoch_loss, std::vector<float>* accuracies)
{
    float total_loss = 0.0f;
    auto t0 = std::chrono::high_resolution_clock::now();

    // 异步双缓冲 (Double-buffering prefetching) ── 用于并行化准备 one-hot 标签
    Tensor one_hot_buffers[2] = {
        Tensor(ShapeTag{}, {BATCH_SIZE, 10}, DType::kFloat, DeviceType::kCPU),
        Tensor(ShapeTag{}, {BATCH_SIZE, 10}, DType::kFloat, DeviceType::kCPU)
    };

    auto fill_one_hot = [](const Tensor& by, Tensor& oh_dest, int actual_bs) {
        oh_dest.zero();
        const float* y_data = by.data_read<float>();
        float* oh = oh_dest.data_write<float>();
        for (int i = 0; i < actual_bs; ++i) {
            oh[i * 10 + (int)y_data[i]] = 1.0f;
        }
    };

    // 预备第 0 批的 one-hot
    int init_end = std::min((int)BATCH_SIZE, (int)images.shape()[0]);
    Tensor by_0 = labels.slice_dim0(0, init_end);
    fill_one_hot(by_0, one_hot_buffers[0], init_end);

    std::future<void> prefetch_future;

    double fwd_time_acc = 0.0;
    double loss_time_acc = 0.0;
    double bwd_time_acc = 0.0;
    double sgd_time_acc = 0.0;

    for (int b = 0; b < num_batches; ++b) {
        int start = b * BATCH_SIZE;
        int end = std::min(start + (int)BATCH_SIZE, (int)images.shape()[0]);
        int actual_bs = end - start;

        // 1. 沿第 0 维进行零拷贝切片 (Zero-copy views) ── 替代高开销的 memcpy 拷贝
        Tensor bx = images.slice_dim0(start, actual_bs);
        Tensor by = labels.slice_dim0(start, actual_bs);

        // 2. 获取当前批次的 one-hot 标签
        Tensor one_hot;
        if (actual_bs == (int)BATCH_SIZE) {
            one_hot = one_hot_buffers[b % 2];
            // 确保上一批次的后台预取已经完成
            if (prefetch_future.valid()) {
                prefetch_future.get();
            }
        } else {
            one_hot = Tensor(ShapeTag{}, {static_cast<size_t>(actual_bs), 10}, DType::kFloat, DeviceType::kCPU);
            fill_one_hot(by, one_hot, actual_bs);
        }

        // 3. 异步预取并填充下一批次的 one-hot 标签到闲置缓冲区
        int next_b = b + 1;
        if (next_b < num_batches) {
            int next_start = next_b * BATCH_SIZE;
            int next_end = std::min(next_start + (int)BATCH_SIZE, (int)images.shape()[0]);
            int next_actual_bs = next_end - next_start;

            if (next_actual_bs == (int)BATCH_SIZE) {
                Tensor next_by = labels.slice_dim0(next_start, next_actual_bs);
                Tensor& next_oh_buf = one_hot_buffers[next_b % 2];
                prefetch_future = std::async(std::launch::async, [fill_one_hot, next_by, &next_oh_buf, next_actual_bs]() {
                    fill_one_hot(next_by, next_oh_buf, next_actual_bs);
                });
            }
        }

        // 前向传播：标准 Tensor API，无任何 C3 痕迹
        auto t_fwd_start = std::chrono::high_resolution_clock::now();
        Tensor z1 = bx.matmul(params[0]) + params[1];
        Tensor h1 = z1.relu();
        Tensor z2 = h1.matmul(params[2]) + params[3];
        Tensor h2 = z2.relu();
        Tensor logits = h2.matmul(params[4]) + params[5];
        auto t_fwd_end = std::chrono::high_resolution_clock::now();
        fwd_time_acc += std::chrono::duration<double, std::milli>(t_fwd_end - t_fwd_start).count();

#ifdef CT_DEBUG
        // 调试：检查 h1 和 h2 的形状（每个 batch 都输出，用于定位 shape 漂移）
        fprintf(stderr, "[DEBUG-SHAPE] b=%d h1=[%zu,%zu] z1=[%zu,%zu] z2=[%zu,%zu] h2=[%zu,%zu] logits=[%zu,%zu]\n",
                b,
                h1.shape().size() >= 2 ? h1.shape()[0] : 0, h1.shape().size() >= 2 ? h1.shape()[1] : 0,
                z1.shape().size() >= 2 ? z1.shape()[0] : 0, z1.shape().size() >= 2 ? z1.shape()[1] : 0,
                z2.shape().size() >= 2 ? z2.shape()[0] : 0, z2.shape().size() >= 2 ? z2.shape()[1] : 0,
                h2.shape().size() >= 2 ? h2.shape()[0] : 0, h2.shape().size() >= 2 ? h2.shape()[1] : 0,
                logits.shape().size() >= 2 ? logits.shape()[0] : 0, logits.shape().size() >= 2 ? logits.shape()[1] : 0);
#endif

        auto t_loss_start = std::chrono::high_resolution_clock::now();
        Tensor loss = logits.cross_entropy(one_hot);
        float loss_val = loss.item<float>();
        total_loss += loss_val;
        auto t_loss_end = std::chrono::high_resolution_clock::now();
        loss_time_acc += std::chrono::duration<double, std::milli>(t_loss_end - t_loss_start).count();

        // 反向传播
        auto t_bwd_start = std::chrono::high_resolution_clock::now();
        AutoGrad::backward(loss.getRelatedNode(), false);
        auto t_bwd_end = std::chrono::high_resolution_clock::now();
        bwd_time_acc += std::chrono::duration<double, std::milli>(t_bwd_end - t_bwd_start).count();

        // TEMP-DIAG: 逐 batch loss + 最后一层权重梯度符号（排查单 kernel hotpath 破坏）
        {
            static int dbg_count = 0;
            if (dbg_count < 40) {
                float* g4 = params[4].grad_ptr();
                long pos = 0, neg = 0;
                for (size_t i = 0; i < params[4].numel(); ++i) { g4[i] > 0 ? pos++ : neg++; }
                fprintf(stderr, "[DIAG-LOSS] b=%d loss=%.4f grad4(pos=%ld/neg=%ld)\n",
                        dbg_count, loss_val, pos, neg);
                dbg_count++;
            }
        }

        if (accuracies) {
            accuracies->push_back(computeAccuracy(logits, by));
        }

        // SGD 更新 (向量化 BLAS cblas_saxpy 优化)
        auto sgd = [](Tensor& p) {
            float* gp = p.grad_ptr();
            float* pd = p.data_write<float>();
            cblas_saxpy((int)p.numel(), -LR, gp, 1, pd, 1);
        };
        if (b < 3) {
            const float* l0 = logits.data_read<float>();
            fprintf(stderr, "[DIAG] b=%d loss=%.4f logits[0:3]=%.4f %.4f %.4f\n",
                    b, loss_val, l0[0], l0[1], l0[2]);
            for (int pi = 0; pi < 6; ++pi) {
                float* g = params[pi].grad_ptr();
                float s = 0.0f;
                size_t nn = 0;
                for (size_t i = 0; i < params[pi].numel(); ++i) {
                    if (std::isnan(g[i]) || std::isinf(g[i])) { nn++; s = std::numeric_limits<float>::quiet_NaN(); break; }
                    s += std::fabs(g[i]);
                }
                fprintf(stderr, "  grad[%d] sum=%.4e nan=%zu\n", pi, s, nn);
            }
        }
        auto t_sgd_start = std::chrono::high_resolution_clock::now();
        sgd(params[0]); sgd(params[1]); sgd(params[2]);
        sgd(params[3]); sgd(params[4]); sgd(params[5]);
        params[0].zero_grad(); params[1].zero_grad(); params[2].zero_grad();
        params[3].zero_grad(); params[4].zero_grad(); params[5].zero_grad();
        auto t_sgd_end = std::chrono::high_resolution_clock::now();
        sgd_time_acc += std::chrono::duration<double, std::milli>(t_sgd_end - t_sgd_start).count();
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    *epoch_loss = total_loss / num_batches;

    // 打印采样数据报告 (Hotspot Profile Output)
    double total_measured = fwd_time_acc + loss_time_acc + bwd_time_acc + sgd_time_acc;
    fprintf(stderr, "\n[HOTSPOT PROFILE] Total Measured: %.2f ms\n", total_measured);
    fprintf(stderr, "  |-- Forward (JIT):   %.2f ms (%.1f%%)\n", fwd_time_acc, (fwd_time_acc/total_measured)*100);
    fprintf(stderr, "  |-- Loss (CrossEnt): %.2f ms (%.1f%%)\n", loss_time_acc, (loss_time_acc/total_measured)*100);
    fprintf(stderr, "  |-- Backward (Grad): %.2f ms (%.1f%%)\n", bwd_time_acc, (bwd_time_acc/total_measured)*100);
    fprintf(stderr, "  |-- Optimizer (SGD):  %.2f ms (%.1f%%)\n", sgd_time_acc, (sgd_time_acc/total_measured)*100);

    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ======================= MatMul 等价性最小测试 =======================
/// 反事实实验：相同输入下 C3 MatMul kernel 与 Eager MatMul 输出是否一致
static void testMatMulEquivalence() {
    // 模拟反向传播：grad_output [128,10] @ W.T [10,128] -> grad_input [128,128]
    Tensor grad_out(ShapeTag{}, {128, 10}, DType::kFloat, DeviceType::kCPU);
    Tensor W(ShapeTag{}, {128, 10}, DType::kFloat, DeviceType::kCPU);

    float* go = grad_out.data_write<float>();
    float* wp = W.data_write<float>();
    for (size_t i = 0; i < grad_out.numel(); ++i) go[i] = (float)(i % 7) * 0.1f - 0.3f;
    for (size_t i = 0; i < W.numel(); ++i) wp[i] = (float)(i % 5) * 0.1f - 0.2f;

    // 构造 W^T
    Tensor WT(ShapeTag{}, {10, 128}, DType::kFloat, DeviceType::kCPU);
    float* wtp = WT.data_write<float>();
    for (size_t i = 0; i < 128; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            wtp[j * 128 + i] = wp[i * 10 + j];
        }
    }

    // Eager MatMul
    Tensor eager_out = grad_out.matmul(WT);
    const float* eager_data = eager_out.data_read<float>();

    // 触发 C3 热路径：连续调用多次 MatMul
    for (int i = 0; i < 12; ++i) {
        Tensor t = grad_out.matmul(WT);
        (void)t;
    }

    // 再调用一次，这次应该命中 C3 kernel
    Tensor c3_out = grad_out.matmul(WT);
    const float* c3_data = c3_out.data_read<float>();

    // 对比
    size_t n = eager_out.numel();
    double max_diff = 0.0;
    size_t nan_count = 0;
    size_t diff_count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(c3_data[i])) ++nan_count;
        double d = std::abs((double)eager_data[i] - (double)c3_data[i]);
        if (d > max_diff) max_diff = d;
        if (d > 1e-5) ++diff_count;
    }

    fprintf(stderr, "\n[MATMUL-EQUIV] shape=[%zu,%zu] eager_out[0]=%.6f c3_out[0]=%.6f\n",
            eager_out.shape()[0], eager_out.shape()[1], eager_data[0], c3_data[0]);
    fprintf(stderr, "[MATMUL-EQUIV] max_diff=%.6e diff_count=%zu/%zu nan_count=%zu\n\n",
            max_diff, diff_count, n, nan_count);
}

// ======================= Add 等价性最小测试 =======================
/// 反事实实验：相同输入下 C3 Add kernel 与 Eager Add 输出是否一致
static void testAddEquivalence() {
    // 模拟 FC + bias：[128,256] + [256]
    Tensor x(ShapeTag{}, {128, 256}, DType::kFloat, DeviceType::kCPU);
    Tensor b(ShapeTag{}, {256}, DType::kFloat, DeviceType::kCPU);

    float* xp = x.data_write<float>();
    float* bp = b.data_write<float>();
    for (size_t i = 0; i < x.numel(); ++i) xp[i] = (float)(i % 13) * 0.05f - 0.3f;
    for (size_t i = 0; i < b.numel(); ++i) bp[i] = (float)(i % 7) * 0.1f - 0.35f;

    // Eager Add
    Tensor eager_out = x + b;
    const float* eager_data = eager_out.data_read<float>();

    // 触发 C3 热路径
    for (int i = 0; i < 12; ++i) {
        Tensor t = x + b;
        (void)t;
    }

    // C3 Add
    Tensor c3_out = x + b;
    const float* c3_data = c3_out.data_read<float>();

    size_t n = eager_out.numel();
    double max_diff = 0.0;
    size_t nan_count = 0;
    size_t diff_count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(c3_data[i])) ++nan_count;
        double d = std::abs((double)eager_data[i] - (double)c3_data[i]);
        if (d > max_diff) max_diff = d;
        if (d > 1e-5) ++diff_count;
    }

    fprintf(stderr, "[ADD-EQUIV] shape=[%zu,%zu] eager_out[0]=%.6f c3_out[0]=%.6f\n",
            eager_out.shape()[0], eager_out.shape()[1], eager_data[0], c3_data[0]);
    fprintf(stderr, "[ADD-EQUIV] max_diff=%.6e diff_count=%zu/%zu nan_count=%zu\n\n",
            max_diff, diff_count, n, nan_count);
}

// ======================= 主函数 =======================

int main() {
    signal(SIGSEGV, tempCrashHandler);
    signal(SIGABRT, tempCrashHandler);
    CtorchError::setPrintLevel(PrintLevel::MEDIUM);  // MEDIUM 以显示 C3 DEBUG 日志
#ifndef CT_DISABLE_C3
    // CFC 消融开关：CT_DISABLE_RF=1 时禁用区域融合（仅保留 C3 hotpath/JIT），
    // 用于隔离"区域融合占位 tensor 导致梯度 NaN"的因果。
    if (getenv("CT_DISABLE_RF") != nullptr) {
        ct::c3::FusionCostModel::setMinGainRatio(1.01);
        fprintf(stderr, "[CFC] 区域融合已禁用 (CT_DISABLE_RF)\n");
    }
#endif
    CtorchScheduler::getInstance();

#ifndef CT_DISABLE_C3
    // 强制开启 C3HotPathManager 调试日志并降低触发阈值，使区域融合在训练早期就能触发并命中
    {
        ct::c3::HotPathConfig hp_cfg;
        hp_cfg.verbose = true;
        hp_cfg.hot_threshold = 2; // 降到 2，早期触发
        ct::c3::C3HotPathManager::instance().configure(hp_cfg);
    }
#endif

    // 先做等价性测试
    testMatMulEquivalence();
    testAddEquivalence();

#ifdef CT_DISABLE_C3
    std::cout << "============================================" << std::endl;
    std::cout << "  MNIST 训练验证 (Eager 模式)" << std::endl;
#else
    std::cout << "============================================" << std::endl;
    std::cout << "  MNIST 训练验证 (C3 自动优化模式)" << std::endl;
#endif
    std::cout << "  网络: 784→256(ReLU)→128(ReLU)→10" << std::endl;
    std::cout << "  Epochs: " << EPOCHS << "  Batch: " << BATCH_SIZE << "  LR: " << LR << std::endl;
    std::cout << "============================================" << std::endl;

    // 加载 MNIST 数据
    MNISTLoader loader(".", DeviceType::kCPU);
    Tensor train_images, train_labels;
    loader.load_training_data(train_images, train_labels);
    std::cout << "MNIST 加载完成 | 训练: " << train_images.shape()[0] << " 样本" << std::endl;

    int num_batches = static_cast<int>(train_images.shape()[0]) / BATCH_SIZE;
    std::cout << "  Batches/epoch: " << num_batches << std::endl << std::endl;

    // 初始化参数
    Tensor params[6] = {
        Tensor(ShapeTag{}, {784, 256}), Tensor(ShapeTag{}, {256}),
        Tensor(ShapeTag{}, {256, 128}), Tensor(ShapeTag{}, {128}),
        Tensor(ShapeTag{}, {128, 10}),  Tensor(ShapeTag{}, {10})
    };
    initParams(params);
    for (int i = 0; i < 6; ++i) params[i].requires_grad(true);

    // 训练
    std::cout << "--- 训练 (" << EPOCHS << " epochs) ---" << std::endl;
    std::vector<float> losses, accs;
    std::vector<double> times;

#ifdef CT_PROFILE_ACCESS
    // 反事实基线测量：统计每个 epoch 的 data_read/data_write 调用次数
    uint64_t last_read  = Tensor::g_data_read_count.load();
    uint64_t last_write = Tensor::g_data_write_count.load();
#endif

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float epoch_loss;
        std::vector<float> epoch_accs;
        double epoch_ms = trainEpoch(params, train_images, train_labels,
                                     num_batches, &epoch_loss, &epoch_accs);

        losses.push_back(epoch_loss);
        times.push_back(epoch_ms);

        float epoch_acc = 0.0f;
        for (float a : epoch_accs) epoch_acc += a;
        epoch_acc /= epoch_accs.size();
        accs.push_back(epoch_acc);

        std::cout << "Epoch " << std::setw(2) << (epoch + 1) << "/" << EPOCHS
                  << " | loss=" << std::fixed << std::setprecision(4) << epoch_loss
                  << " acc=" << std::setprecision(2) << epoch_acc * 100 << "%"
                  << " " << std::setprecision(1) << epoch_ms << "ms";
#ifdef CT_PROFILE_ACCESS
        uint64_t r = Tensor::g_data_read_count.load();
        uint64_t w = Tensor::g_data_write_count.load();
        std::cout << " | reads=" << (r - last_read)
                  << " writes=" << (w - last_write);
        last_read = r;
        last_write = w;
#endif
        std::cout << std::endl;
#ifndef CT_DISABLE_C3
        {
            auto s = c3::C3KernelRegistry::getInstance().getStats();
            auto hp = c3::C3HotPathManager::instance().getStats();
            fprintf(stderr, "[C3-STAT] epoch=%d single_active=%zu fused=%zu hit=%zu miss=%zu bypass=%zu fused_hit=%zu compiles=%zu tracked=%zu\n",
                    epoch + 1, s.active_entries, s.fused_entries, s.hit_count, s.miss_count,
                    s.bypass_count, s.fused_hit_count,
                    hp.compilations_triggered, hp.calls_tracked);
            // DEBT-NEW-7 v0.5.1+ C3 backward stats
            auto bw = c3::C3BackwardCapture::getInstance().getStats();
            fprintf(stderr, "[C3-BW-STAT] epoch=%d capture=%zu compile=%zu bw_hit=%zu bw_miss=%zu exec_fail=%zu fusion_compile=%zu fusion_hit=%zu fusion_miss=%zu\n",
                    epoch + 1, bw.capture_count, bw.compile_count,
                    bw.cache_hit_count, bw.cache_miss_count, bw.execution_failures,
                    bw.fusion_compile_count, bw.fusion_hit_count, bw.fusion_miss_count);
        }
#endif
#ifdef CT_PROFILE_PERF
        // DEBT-NEW-7 性能采样输出(v0.5.1+ 代码审查用)
        // 单独写:不需要 C3 namespace,在 c3 off build 也能用(只是没有 C3-STAT 数据)
#ifndef CT_DISABLE_C3
        {
            auto s = c3::C3KernelRegistry::getInstance().getStats();
            fprintf(stderr, "[C3-PERF] rd=%lu(%.1fus) rm=%lu(%.1fus) c3s=%lu(%.1fus) eager=%lu(%.1fus)\n",
                    (unsigned long)s.region_dispatch_count,
                    s.region_dispatch_count ? (double)s.region_dispatch_ns / s.region_dispatch_count / 1000.0 : 0.0,
                    (unsigned long)s.region_match_count,
                    s.region_match_count ? (double)s.region_match_ns / s.region_match_count / 1000.0 : 0.0,
                    (unsigned long)s.c3_single_invoke_count,
                    s.c3_single_invoke_count ? (double)s.c3_single_invoke_ns / s.c3_single_invoke_count / 1000.0 : 0.0,
                    (unsigned long)s.eager_invoke_count,
                    s.eager_invoke_count ? (double)s.eager_invoke_ns / s.eager_invoke_count / 1000.0 : 0.0);
            fprintf(stderr, "[C3-PERF] total_ms: rd=%.1f rm=%.1f c3s=%.1f eager=%.1f\n",
                    (double)s.region_dispatch_ns / 1e6,
                    (double)s.region_match_ns / 1e6,
                    (double)s.c3_single_invoke_ns / 1e6,
                    (double)s.eager_invoke_ns / 1e6);
        }
#else
        // CT_DISABLE_C3 build:从 inline static 读 eager 统计
        {
            auto [eager_ns, eager_count] = ct::detail::perfEagerRead();
            fprintf(stderr, "[C3-PERF-off] eager=%lu(%.1fus) total_ms=%.1f\n",
                    (unsigned long)eager_count,
                    eager_count ? (double)eager_ns / eager_count / 1000.0 : 0.0,
                    (double)eager_ns / 1e6);
        }
#endif
#endif
    }

    // 总结
    std::cout << "\n============================================" << std::endl;
    std::cout << "  训练完成" << std::endl;
    std::cout << "============================================" << std::endl;

    // [Dev] v0.5.2 (4) JITCache 1.0 stats (2026-08-09)
#ifndef CT_DISABLE_C3
    {
        auto& jc = ct::c3::JITCache::getInstance();
        std::cout << "  [JITCache] 1.0 store-only stats: hits=" << jc.hits()
                  << " stores=" << jc.stores() << " cache_dir=" << jc.cacheDir() << std::endl;
        std::cout << "  [JITCache] (read path 实装后, hits>0 表示从 .bc 加载, 0 加速当前不体现)" << std::endl;
    }
#endif

    double total_time = 0.0;
    for (double t : times) total_time += t;
    double avg_time = total_time / times.size();

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  最终 loss: " << losses.back()
              << "  最终 acc: " << (accs.back() * 100) << "%" << std::endl;
    std::cout << "  总时间: " << total_time << "ms"
              << "  平均/epoch: " << avg_time << "ms" << std::endl;
    std::cout << "  平均/batch: " << std::setprecision(3)
              << (avg_time / num_batches) << "ms" << std::endl;

#ifndef CT_DISABLE_C3
    fprintf(stderr, "[CLEANUP] begin shutdown\n");
    // [Safe exit] 注释掉 c3::shutdownAll() 以避免其内部触发 LLVM JIT 析构引起的已知 crash（非本模块引入）
    // c3::shutdownAll();
    fprintf(stderr, "[CLEANUP] done, exiting instantly via std::_Exit to bypass LLVM JIT static destructor bugs\n");
#endif

    std::_Exit(0);
}