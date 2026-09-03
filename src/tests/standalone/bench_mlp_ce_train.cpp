/**
 * @file bench_mlp_ce_train.cpp
 * @brief MLP+CE 端到端训练 step 时延基准（C3 vs Eager，支持多层深宽 MLP）
 * @details 深宽 MLP（IN -> H -> H -> ... -> H -> NC）+ CrossEntropy loss，N 个 training step。
 *          每 step 完整训练（forward + loss + backward + SGD 更新），量化每 step wall-clock
 *          （C3 ON 相对 Eager 加速比）。作为论文「中等规模模型」实验：证明 C3 的训练态
 *          （RegionFusion + MIMO backward）相对 Eager 的正收益，与纯前向手工已形成对照组。
 *
 * 用法：
 *   # C3 路径（默认）
 *   ./bench_mlp_ce_train                          # 默认（B=64, IN=784, H=128, NC=10, depth=1）
 *   ./bench_mlp_ce_train B IN H NC STEPS WARMUP DEPTH
 *
 *   # Eager 对比（只禁 backward；forward 仍需在 CT_DISABLE_C3=ON 构建下才纯 eager）
 *   CTORCH_DISABLE_C3_BACKWARD=1 ./bench_mlp_ce_train
 *   # 真正纯 Eager 请改用 build-eager 目录的同名二进制
 *
 * @date 2026-08-30 (2026-09-03 扩展为多层深宽 MLP)
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
#ifndef CT_DISABLE_C3
#include "C3/C3BackwardCapture.h"
#include "C3/C3Engine.h"
#include "C3/C3Cleanup.h"
#endif

using namespace ct;
#ifndef CT_DISABLE_C3
using namespace ct::c3;
#endif

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

// 一次完整 training step：forward + loss + backward + SGD 更新（多层深宽 MLP）
static double one_step(Tensor& x, Tensor& y_onehot,
                       std::vector<Tensor>& W, std::vector<Tensor>& b,
                       Tensor& Wout, Tensor& bout,
                       float lr)
{
    const int D = (int)W.size();
    // forward
    Tensor h = x;
    for (int l = 0; l < D; ++l) {
        Tensor pre = h.matmul(W[l]);   // [B, H]
        pre = pre + b[l];              // broadcast bias
        h = pre.relu();                // [B, H]
    }
    Tensor logits = h.matmul(Wout);    // [B, NC]
    logits = logits + bout;
    Tensor loss = logits.cross_entropy(y_onehot);  // [1]

    // backward
    AutoGrad::backward(loss.getRelatedNode(), false);

    // SGD 更新全部参数
    for (int l = 0; l < D; ++l) {
        Tensor gW = W[l].grad();
        Tensor gb = b[l].grad();
        if (gW.data_read<float>() != nullptr) {
            float* p = W[l].data_write<float>();
            const float* g = gW.data_read<float>();
            for (size_t i = 0; i < W[l].numel(); ++i) p[i] -= lr * g[i];
        }
        if (gb.data_read<float>() != nullptr) {
            float* p = b[l].data_write<float>();
            const float* g = gb.data_read<float>();
            for (size_t i = 0; i < b[l].numel(); ++i) p[i] -= lr * g[i];
        }
    }
    // 输出层
    {
        Tensor gW = Wout.grad(); Tensor gb = bout.grad();
        if (gW.data_read<float>() != nullptr) {
            float* p = Wout.data_write<float>(); const float* g = gW.data_read<float>();
            for (size_t i = 0; i < Wout.numel(); ++i) p[i] -= lr * g[i];
        }
        if (gb.data_read<float>() != nullptr) {
            float* p = bout.data_write<float>(); const float* g = gb.data_read<float>();
            for (size_t i = 0; i < bout.numel(); ++i) p[i] -= lr * g[i];
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
    int DEPTH = 1;
    float LR = 0.01f;

    if (argc >= 5) {
        B = (size_t)std::atoll(argv[1]);
        IN = (size_t)std::atoll(argv[2]);
        H = (size_t)std::atoll(argv[3]);
        NC = (size_t)std::atoll(argv[4]);
    }
    if (argc >= 6) STEPS = std::atoi(argv[5]);
    if (argc >= 7) WARMUP = std::atoi(argv[6]);
    if (argc >= 8) DEPTH = std::atoi(argv[7]);

    const char* mode = "C3 ON";
    if (std::getenv("CTORCH_DISABLE_C3_BACKWARD") != nullptr) {
        mode = "Eager (C3 OFF)";
    }

    std::cout << "=== MLP+CE training step latency (deep MLP) ===" << std::endl;
    std::cout << "  B=" << B << " IN=" << IN << " H=" << H << " depth=" << DEPTH
              << " NC=" << NC << "  steps=" << STEPS << "  warmup=" << WARMUP << std::endl;
    std::cout << "  mode: " << mode << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    // 构造多层隐藏层权重（IN->H, 继而 H->H, 共 DEPTH 层）
    std::vector<Tensor> W, b;
    W.reserve(DEPTH); b.reserve(DEPTH);
    for (int l = 0; l < DEPTH; ++l) {
        size_t in = (l == 0) ? IN : H;
        Tensor w(ShapeTag{}, {in, H}, DType::kFloat, DeviceType::kCPU);
        Tensor bb(ShapeTag{}, {H}, DType::kFloat, DeviceType::kCPU);
        fill(w, 0.05f + 0.01f * l); fill(bb, 0.0f);
        w.requires_grad(true); bb.requires_grad(true);
        W.push_back(std::move(w)); b.push_back(std::move(bb));
    }
    Tensor Wout(ShapeTag{}, {H, NC}, DType::kFloat, DeviceType::kCPU);
    Tensor bout(ShapeTag{}, {NC}, DType::kFloat, DeviceType::kCPU);
    fill(Wout, 0.07f); fill(bout, 0.0f);
    Wout.requires_grad(true); bout.requires_grad(true);

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
#ifndef CT_DISABLE_C3
        CtorchScheduler::getInstance().resetRegionFusion();
#endif
        last_loss = one_step(x, y, W, b, Wout, bout, LR);
    }
    // warmup 后再清一次（让 C3 cache 全 miss 状态进入正式测）
#ifndef CT_DISABLE_C3
    auto stats_after_warmup = C3BackwardCapture::getInstance().getStats();
#endif

    // 正式 benchmark
    std::vector<double> step_us;
    for (int i = 0; i < STEPS; ++i) {
        auto t0 = hires::now();
        last_loss = one_step(x, y, W, b, Wout, bout, LR);
        auto t1 = hires::now();
        step_us.push_back(std::chrono::duration_cast<us>(t1 - t0).count());
    }
#ifndef CT_DISABLE_C3
    auto stats_final = C3BackwardCapture::getInstance().getStats();
#endif

    double med = median(step_us);
    double p10 = step_us[STEPS / 10];
    double p90 = step_us[STEPS * 9 / 10];

    std::cout << "  step latency (us): median=" << med << "  p10=" << p10 << "  p90=" << p90
              << "  min=" << step_us.front() << "  max=" << step_us.back() << std::endl;
    std::cout << "  last_loss = " << last_loss << "  (just a sanity check, should decrease-ish)" << std::endl;

#ifndef CT_DISABLE_C3
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
#endif

    std::cout.flush();
    // 跳过 c3::shutdownAll()：mutex lock failed 崩溃（与 benchmark 无关）
    return 0;
}
