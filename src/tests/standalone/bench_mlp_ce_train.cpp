/**
 * @file bench_mlp_ce_train.cpp
 * @brief MLP+CE 端到端训练 step 时延基准（C3 vs Eager）
 * @details 2 层 MLP（IN -> H -> NUM_CLASSES）+ CrossEntropy loss，N 个 training step。
 *          量化每 step wall-clock（C3 ON 相对 Eager 加速比），用于评估 P0.2（CE backward
 *          接入 + P0.2.1 broadcast 修复）后端到端收益。
 *
 * 用法：
 *   # C3 路径（默认）
 *   ./bench_mlp_ce_train                          # 默认规模（B=64, IN=784, H=128, NC=10）
 *   ./bench_mlp_ce_train 64 784 128 10 50         # 显式指定
 *
 *   # Eager 对比
 *   CTORCH_DISABLE_C3_BACKWARD=1 ./bench_mlp_ce_train
 *
 * @date 2026-08-30
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <cstdlib>

#include "Tensor.h"
#include "CtorchScheduler.h"
#include "AutoGrad.h"
#include "C3/C3BackwardCapture.h"
#include "C3/C3Engine.h"
#include "C3/C3Cleanup.h"

using namespace ct;
using namespace ct::c3;

using hires = std::chrono::high_resolution_clock;
using us = std::chrono::duration<double, std::micro>;

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 0) ? 0.5 * (v[n/2 - 1] + v[n/2]) : v[n/2];
}

static void fill(Tensor& t, float seed) {
    float* p = t.data_write<float>();
    for (size_t i = 0; i < t.numel(); ++i) p[i] = seed + (float)(i % 13) * 0.013f;
}

// 一次完整 training step：forward + loss + backward + SGD 更新
static double one_step(Tensor& x, Tensor& y_onehot,
                       Tensor& W1, Tensor& b1,
                       Tensor& W2, Tensor& b2,
                       float lr)
{
    // forward
    Tensor h_pre = x.matmul(W1);     // [B, H]
    h_pre = h_pre + b1;              // broadcast bias
    Tensor h = h_pre.relu();         // [B, H]
    Tensor logits = h.matmul(W2);    // [B, NC]
    logits = logits + b2;
    Tensor loss = logits.cross_entropy(y_onehot);  // [1]

    // backward
    AutoGrad::backward(loss.getRelatedNode(), false);

    // SGD 更新
    {
        Tensor gW1 = W1.grad();
        Tensor gb1 = b1.grad();
        Tensor gW2 = W2.grad();
        Tensor gb2 = b2.grad();
        if (gW1.data_read<float>() != nullptr) {
            float* p = W1.data_write<float>();
            const float* g = gW1.data_read<float>();
            for (size_t i = 0; i < W1.numel(); ++i) p[i] -= lr * g[i];
        }
        if (gb1.data_read<float>() != nullptr) {
            float* p = b1.data_write<float>();
            const float* g = gb1.data_read<float>();
            for (size_t i = 0; i < b1.numel(); ++i) p[i] -= lr * g[i];
        }
        if (gW2.data_read<float>() != nullptr) {
            float* p = W2.data_write<float>();
            const float* g = gW2.data_read<float>();
            for (size_t i = 0; i < W2.numel(); ++i) p[i] -= lr * g[i];
        }
        if (gb2.data_read<float>() != nullptr) {
            float* p = b2.data_write<float>();
            const float* g = gb2.data_read<float>();
            for (size_t i = 0; i < b2.numel(); ++i) p[i] -= lr * g[i];
        }
    }
    return loss.data_read<float>()[0];
}

int main(int argc, char** argv) {
    size_t B = 64;
    size_t IN = 784;
    size_t H = 128;
    size_t NC = 10;
    int STEPS = 50;
    int WARMUP = 5;
    float LR = 0.01f;

    if (argc >= 5) {
        B = (size_t)std::atoll(argv[1]);
        IN = (size_t)std::atoll(argv[2]);
        H = (size_t)std::atoll(argv[3]);
        NC = (size_t)std::atoll(argv[4]);
    }
    if (argc >= 6) STEPS = std::atoi(argv[5]);
    if (argc >= 7) WARMUP = std::atoi(argv[6]);

    const char* mode = "C3 ON";
    if (std::getenv("CTORCH_DISABLE_C3_BACKWARD") != nullptr) {
        mode = "Eager (C3 OFF)";
    }

    std::cout << "=== MLP+CE training step latency ===" << std::endl;
    std::cout << "  B=" << B << " IN=" << IN << " H=" << H << " NC=" << NC
              << "  steps=" << STEPS << "  warmup=" << WARMUP << std::endl;
    std::cout << "  mode: " << mode << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    // 构造权重
    Tensor W1(ShapeTag{}, {IN, H}, DType::kFloat, DeviceType::kCPU);
    Tensor b1(ShapeTag{}, {H}, DType::kFloat, DeviceType::kCPU);
    Tensor W2(ShapeTag{}, {H, NC}, DType::kFloat, DeviceType::kCPU);
    Tensor b2(ShapeTag{}, {NC}, DType::kFloat, DeviceType::kCPU);
    fill(W1, 0.05f); fill(b1, 0.0f); fill(W2, 0.07f); fill(b2, 0.0f);
    W1.requires_grad(true); b1.requires_grad(true);
    W2.requires_grad(true); b2.requires_grad(true);

    // 构造输入 + one-hot target
    Tensor x(ShapeTag{}, {B, IN}, DType::kFloat, DeviceType::kCPU);
    fill(x, 0.3f);
    Tensor y(ShapeTag{}, {B, NC}, DType::kFloat, DeviceType::kCPU);
    {
        float* yp = y.data_write<float>();
        for (size_t i = 0; i < B; ++i) {
            size_t c = i % NC;
            for (size_t j = 0; j < NC; ++j) yp[i * NC + j] = (j == c) ? 1.0f : 0.0f;
        }
    }

    // warmup
    double last_loss = 0;
    for (int i = 0; i < WARMUP; ++i) {
        CtorchScheduler::getInstance().resetRegionFusion();
        last_loss = one_step(x, y, W1, b1, W2, b2, LR);
    }
    // warmup 后再清一次（让 C3 cache 全 miss 状态进入正式测）
    auto stats_after_warmup = C3BackwardCapture::getInstance().getStats();

    // 正式 benchmark
    std::vector<double> step_us;
    for (int i = 0; i < STEPS; ++i) {
        auto t0 = hires::now();
        last_loss = one_step(x, y, W1, b1, W2, b2, LR);
        auto t1 = hires::now();
        step_us.push_back(std::chrono::duration_cast<us>(t1 - t0).count());
    }
    auto stats_final = C3BackwardCapture::getInstance().getStats();

    double med = median(step_us);
    double p10 = step_us[STEPS / 10];
    double p90 = step_us[STEPS * 9 / 10];

    std::cout << "  step latency (us): median=" << med << "  p10=" << p10 << "  p90=" << p90
              << "  min=" << step_us.front() << "  max=" << step_us.back() << std::endl;
    std::cout << "  last_loss = " << last_loss << "  (just a sanity check, should decrease-ish)" << std::endl;

    std::cout << "\n  [C3 Backward Stats] (warmup→final)" << std::endl;
    std::cout << "    cache_hit          " << (stats_final.cache_hit_count - stats_after_warmup.cache_hit_count) << std::endl;
    std::cout << "    cache_miss         " << (stats_final.cache_miss_count - stats_after_warmup.cache_miss_count) << std::endl;
    std::cout << "    fusion_compiles    " << (stats_final.fusion_compile_count - stats_after_warmup.fusion_compile_count) << std::endl;
    std::cout << "    fusion_hits        " << (stats_final.fusion_hit_count - stats_after_warmup.fusion_hit_count) << std::endl;
    std::cout << "    fusion_misses      " << (stats_final.fusion_miss_count - stats_after_warmup.fusion_miss_count) << std::endl;
    std::cout << "    mimo_compiles      " << (stats_final.mimo_compile_count - stats_after_warmup.mimo_compile_count) << std::endl;
    std::cout << "    mimo_hits          " << (stats_final.mimo_hit_count - stats_after_warmup.mimo_hit_count) << std::endl;

    // compile error stats
    auto err_stats = C3Engine::getInstance().getCompileErrorStats();
    std::cout << "    compile_errors     " << (err_stats.total_failures) << std::endl;

    std::cout.flush();
    // 跳过 c3::shutdownAll()：mutex lock failed 崩溃（与 benchmark 无关）
    return 0;
}
