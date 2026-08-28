/**
 * @file bench_wide_mlp_e2e.cpp
 * @brief 端到端宽 MLP 前向基准：寻找 C3 端到端收益 > 调度税的规模
 * @details 网络 = 输入 -> (relu(x@W+b)) x L 层 -> 输出。纯前向（经真实调度器，含
 *          区域调度税 + C3 融合），可同时用于 C3 ON（build）与 C3 OFF（build_eager）
 *          两种构建，比较端到端耗时。目标：融合省下的内存往返 > 调度税时 C3 更快。
 * @date 2026-08-28
 */
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cstdlib>
#include "Tensor.h"
#include "CtorchScheduler.h"
#include "C3/C3HotPathManager.h"
#include "C3/C3Config.h"
#include "C3/C3Cleanup.h"

using namespace ct;
using namespace ct::c3;
using hires = std::chrono::high_resolution_clock;
using ms = std::chrono::duration<double, std::milli>;

static size_t B = 64, H = 4096, L = 4, STEPS = 20;

static void fill(Tensor& t, float seed) {
    float* p = t.data_write<float>();
    for (size_t i = 0; i < t.numel(); ++i) p[i] = seed + (float)(i % 17) * 0.01f;
}

int main(int argc, char** argv) {
    if (argc >= 5) { B = (size_t)std::atoll(argv[1]); H = (size_t)std::atoll(argv[2]); L = (size_t)std::atoll(argv[3]); STEPS = (size_t)std::atoll(argv[4]); }
    const size_t IN = 512;

    std::cout << "=== Wide-MLP forward e2e (B=" << B << " H=" << H << " L=" << L << " steps=" << STEPS << ") ===\n";
#ifndef CT_DISABLE_C3
    std::cout << "  mode: C3 ON\n";
    HotPathConfig hp_cfg; hp_cfg.hot_threshold = 2;
    C3HotPathManager::instance().configure(hp_cfg);
#else
    std::cout << "  mode: Eager (C3 OFF)\n";
#endif
    std::cout.flush();

    // 权重（前向不需要 autograd，直接普通张量）
    std::vector<Tensor> W, b;
    W.reserve(L + 1); b.reserve(L + 1);
    {
        W.push_back(Tensor(ShapeTag{}, {IN, H})); fill(W.back(), 0.1f);
        b.push_back(Tensor(ShapeTag{}, {H})); b.back().zero();
        for (size_t i = 1; i < L; ++i) { W.push_back(Tensor(ShapeTag{}, {H, H})); fill(W.back(), 0.1f); b.push_back(Tensor(ShapeTag{}, {H})); b.back().zero(); }
        W.push_back(Tensor(ShapeTag{}, {H, H})); fill(W.back(), 0.1f);
        b.push_back(Tensor(ShapeTag{}, {H})); b.back().zero();
    }
    Tensor x(ShapeTag{}, {B, IN}); fill(x, 0.3f);

    // 预热（让 C3 编译完成；不计时）
    for (size_t s = 0; s < std::min(STEPS, (size_t)3); ++s) {
        Tensor h = x;
        for (size_t i = 0; i <= L; ++i) h = (h.matmul(W[i]) + b[i]).relu();
    }

    auto t0 = hires::now();
    volatile float sink = 0;
    for (size_t s = 0; s < STEPS; ++s) {
        Tensor h = x;
        for (size_t i = 0; i <= L; ++i) h = (h.matmul(W[i]) + b[i]).relu();
        sink += h.data_read<float>()[0];
    }
    auto t1 = hires::now();
    double total = std::chrono::duration_cast<ms>(t1 - t0).count();

    std::cout << std::fixed << std::setprecision(1)
              << "  total=" << total << " ms  (" << total / STEPS << " ms/step)  (sink=" << sink << ")\n";
    std::cout.flush();

#ifndef CT_DISABLE_C3
    auto s = C3KernelRegistry::getInstance().getStats();
    fprintf(stderr, "[E2E-C3-STAT] fused_hit=%zu compiles=%zu rd=%lu rm=%lu\n",
            s.fused_hit_count, C3HotPathManager::instance().getStats().compilations_triggered,
            (unsigned long)s.region_dispatch_count, (unsigned long)s.region_match_count);
    ct::c3::shutdownAll();
#endif
    std::cout.flush();
    std::_Exit(0);
}
