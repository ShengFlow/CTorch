/**
 * @file exp_c3_warmup.cpp
 * @brief C3 预热需求实验：对比 0 预热 vs 预热的前 N 轮耗时
 * @details 逐轮打印每次迭代耗时，观察：
 *   1. 首次迭代是否特别慢（JIT 编译触发门槛）
 *   2. 第 ~40 轮后（fusion_compile 触发）是否有抖动
 *   3. 预热 vs 不预热对稳态耗时的影响
 *
 * 用法：
 *   ./exp_c3_warmup 0 80   # 0 warmup, 80 measure iters
 *   ./exp_c3_warmup 20 80  # 20 warmup, 80 measure iters
 */
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <thread>
#include <cstdlib>

#include "Tensor.h"
#include "CtorchScheduler.h"
#include "AutoGrad.h"
#include "C3/C3BackwardCapture.h"
#include "C3/C3Engine.h"

using namespace ct;
using namespace ct::c3;

static inline double now_ms() {
    using clock = std::chrono::high_resolution_clock;
    auto t = clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(t).count();
}

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n == 0) return NAN;
    if (n % 2 == 0) return 0.5 * (v[n/2 - 1] + v[n/2]);
    return v[n/2];
}

int main(int argc, char** argv) {
    int N_WARMUP = (argc > 1) ? std::atoi(argv[1]) : 0;
    int N_MEASURE = (argc > 2) ? std::atoi(argv[2]) : 80;
    const char* exp_tag = (argc > 3) ? argv[3] : (N_WARMUP == 0 ? "NO-WARMUP" : "WITH-WARMUP");

    std::cout << "=== C3 Warmup 实验 ===" << std::endl;
    std::cout << "  tag      : " << exp_tag << std::endl;
    std::cout << "  warmup   : " << N_WARMUP << std::endl;
    std::cout << "  measure  : " << N_MEASURE << std::endl;

    auto& sched = CtorchScheduler::getInstance();
    (void)sched;

    const int M = 512, K = 512;
    std::cout << "  shape    : [" << M << "x" << K << "]" << std::endl;

    auto x_base = Tensor(ShapeTag{}, {M, K}, DType::kFloat, DeviceType::kCPU);
    float* xp = x_base.data_write<float>();
    uint32_t rng = 0xdeadbeef;
    for (int i = 0; i < M*K; ++i) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        xp[i] = ((double)rng / 4294967296.0) * 4.0f - 2.0f;
    }

    // ========= Warmup =========
    if (N_WARMUP > 0) {
        std::cout << "\n  [Warmup] " << N_WARMUP << " iters... ";
        std::cout.flush();
        for (int iter = 0; iter < N_WARMUP; ++iter) {
            Tensor x = x_base.clone();
            x.requires_grad(true);
            Tensor y = x.tanh().sigmoid().relu();
            AutoGrad::backward(y.getRelatedNode(), false);
        }
        // 等异步编译
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        for (int iter = 0; iter < 15; ++iter) {
            Tensor x = x_base.clone();
            x.requires_grad(true);
            Tensor y = x.tanh().sigmoid().relu();
            AutoGrad::backward(y.getRelatedNode(), false);
        }
        std::cout << "done." << std::endl;
    }

    // ========= Measure (逐轮打印) =========
    std::vector<double> samples_ms;
    samples_ms.reserve(N_MEASURE);

    std::cout << "\n  [逐轮耗时] (iter, ms):" << std::endl;
    for (int iter = 0; iter < N_MEASURE; ++iter) {
        Tensor x = x_base.clone();
        x.requires_grad(true);

        double t0 = now_ms();
        Tensor y = x.tanh().sigmoid().relu();
        AutoGrad::backward(y.getRelatedNode(), false);
        double t1 = now_ms();

        double dt = t1 - t0;
        samples_ms.push_back(dt);
        printf("    iter%3d:  %7.3f ms%s\n", iter, dt,
               (iter < 15 || dt > 5.0) ? "  <--" : "");
    }

    // ========= 统计 =========
    double p50 = median(samples_ms);
    double sum = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0);
    double mean = sum / samples_ms.size();
    double minv = *std::min_element(samples_ms.begin(), samples_ms.end());
    double maxv = *std::max_element(samples_ms.begin(), samples_ms.end());

    // 前 10 轮平均 vs 后 10 轮平均（看收敛速度）
    double first10_avg = 0, last10_avg = 0;
    for (int i = 0; i < std::min(10, N_MEASURE); ++i) first10_avg += samples_ms[i];
    first10_avg /= std::min(10, N_MEASURE);
    int n_l10 = std::min(10, N_MEASURE);
    for (int i = 0; i < n_l10; ++i) last10_avg += samples_ms[N_MEASURE - 1 - i];
    last10_avg /= n_l10;
    double ratio = first10_avg / last10_avg;

    std::cout << "\n  ---------------- 统计 ----------------" << std::endl;
    printf("    mean     = %.3f ms\n", mean);
    printf("    p50      = %.3f ms\n", p50);
    printf("    min/max  = %.3f / %.3f ms\n", minv, maxv);
    printf("    first10 avg  = %.3f ms\n", first10_avg);
    printf("    last10  avg  = %.3f ms\n", last10_avg);
    printf("    ratio(first10/last10) = %.2fx\n", ratio);
    if (ratio > 1.3) printf("    → 首10轮明显慢 (%.2fx)，预热有用！\n", ratio);
    else             printf("    → 首末差异小 (%.2fx)，预热影响不大～\n", ratio);

    auto stats = C3BackwardCapture::getInstance().getStats();
    std::cout << "\n  [C3 Stats]";
    std::cout << "\n    cache_hit        = " << stats.cache_hit_count
              << "\n    cache_miss       = " << stats.cache_miss_count
              << "\n    fusion_compiles  = " << stats.fusion_compile_count
              << "\n    fusion_hits      = " << stats.fusion_hit_count
              << "\n    fusion_misses    = " << stats.fusion_miss_count << std::endl;

    return 0;
}
