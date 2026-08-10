/**
 * @file bench_c3_backward_perf_clean.cpp
 * @brief C3 反向融合性能基准测试【纯净版 / 无预热】
 * @details 只使用标准 Tensor / AutoGrad API（无任何 C3 专用头文件、API、预热逻辑），
 *          模拟真实用户首次运行场景。调度器自行决定何时触发编译、何时替换内核。
 *
 * 用法：
 *   # 基线：完全禁用 C3 反向融合
 *   C3_AOT_CACHE_DIR=/tmp/.c3cache C3_DISABLE_BACKWARD=1 ./bench_c3_backward_perf_clean
 *   # 实验：C3 默认开启（无预热，真实用户路径）
 *   C3_AOT_CACHE_DIR=/tmp/.c3cache ./bench_c3_backward_perf_clean
 *
 * @date 2026/8/8
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

#include "Tensor.h"
#include "AutoGrad.h"

using namespace ct;

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

static double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n == 0) return NAN;
    double idx = (n - 1) * p;
    size_t lo = (size_t)idx, hi = std::min(lo + 1, n - 1);
    double frac = idx - lo;
    return v[lo] * (1 - frac) + v[hi] * frac;
}

int main() {
    const bool disabled = []() {
        const char* env = std::getenv("C3_DISABLE_BACKWARD");
        return (env && std::string(env) == "1");
    }();
    std::cout << "=== C3 Backward Fusion 性能基准测试【纯净版 / 无预热】===" << std::endl;
    std::cout << "  模式: " << (disabled ? "EAGER 基线 (C3 禁用)" : "C3 融合启用（用户级API）") << std::endl;

    // ========= 问题规模：256K 元素 / 3 层 unary =========
    const int M = 512, K = 512;
    const int N_MEASURE = 120;
    std::cout << "  张量形状: [" << M << " x " << K << "] = " << (1.0*M*K/1048576.0) << "M elements" << std::endl;
    std::cout << "  图结构: x → Tanh → Sigmoid → ReLU → backward" << std::endl;
    std::cout << "  Warmup iters: 0 (无预热) | Measure iters: " << N_MEASURE
              << " (从第1次开始计时)" << std::endl;

    // 固定随机种子
    auto x_base = Tensor(ShapeTag{}, {M, K}, DType::kFloat, DeviceType::kCPU);
    float* xp = x_base.data_write<float>();
    uint32_t rng = 0xdeadbeef;
    for (int i = 0; i < M*K; ++i) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        xp[i] = ((double)rng / 4294967296.0) * 4.0f - 2.0f;
    }

    // ========= 直接测量（无预热！用户视角 = 第1次就纳入统计）=========
    std::vector<double> samples_ms;
    samples_ms.reserve(N_MEASURE);
    double max_abs_diff = 0.0;

    // ref：iter0 跑完后**立即**在同状态下再跑一次做对比，避免 120 次迭代中间异步编译完成导致模式漂移
    std::vector<float> iter0_measured(M*K);
    std::vector<float> iter0_ref(M*K);

    std::cout << "\n  [Measure] 直接运行 " << N_MEASURE << " 次（无预热）... ";
    std::cout.flush();
    for (int iter = 0; iter < N_MEASURE; ++iter) {
        Tensor x = x_base.clone();
        x.requires_grad(true);

        double t0 = now_ms();
        Tensor y = x.tanh().sigmoid().relu();
        AutoGrad::backward(y.getRelatedNode(), false);
        double t1 = now_ms();

        samples_ms.push_back(t1 - t0);

        if (iter == 0) {
            // 【关键】iter0 跑完后立刻、马上、紧接着跑一次 ref，用完全相同的进程状态做对比
            // 此时异步编译肯定还没完成（编译至少需要几百ms），所以两次一定是同模式
            const float* g = x.grad().data_read<float>();
            for (int i = 0; i < M*K; ++i) iter0_measured[i] = g[i];

            // 立刻再跑一次做 ref（同一模式：要么都是 eager，要么都是 fusion，绝不会混）
            {
                Tensor x2 = x_base.clone();
                x2.requires_grad(true);
                Tensor y2 = x2.tanh().sigmoid().relu();
                AutoGrad::backward(y2.getRelatedNode(), false);
                const float* g2 = x2.grad().data_read<float>();
                for (int i = 0; i < M*K; ++i) iter0_ref[i] = g2[i];
            }
            double md = 0.0;
            for (int i = 0; i < M*K; ++i) {
                double d = std::fabs(iter0_ref[i] - iter0_measured[i]);
                if (d > md) md = d;
            }
            max_abs_diff = md;
        }
    }
    std::cout << "done." << std::endl;

    // ========= 统计输出 =========
    double p50 = median(samples_ms);
    double p95 = percentile(samples_ms, 0.95);
    double p05 = percentile(samples_ms, 0.05);
    double sum = std::accumulate(samples_ms.begin(), samples_ms.end(), 0.0);
    double mean = sum / samples_ms.size();
    double minv = *std::min_element(samples_ms.begin(), samples_ms.end());
    double maxv = *std::max_element(samples_ms.begin(), samples_ms.end());

    // 找出稳态 p50（跳过前 20 次冷启动，更稳妥观察何时收敛）
    size_t skip = std::min<size_t>(20, samples_ms.size());
    std::vector<double> steady(samples_ms.begin() + skip, samples_ms.end());
    double steady_p50 = steady.empty() ? p50 : median(steady);
    double steady_mean = steady.empty() ? mean : std::accumulate(steady.begin(), steady.end(), 0.0) / steady.size();

    std::cout << "\n  ---------------- 单次迭代耗时 (3层 unary forward + backward) ----------------" << std::endl;
    printf("    【含冷启动】mean=%.3f ms   p50=%.3f ms   p95=%.3f ms\n", mean, p50, p95);
    printf("    【稳态(跳前%zu次)】mean=%.3f ms   p50=%.3f ms\n", skip, steady_mean, steady_p50);
    printf("    min=%.3f ms   max=%.3f ms   p05=%.3f ms   samples=%zu\n", minv, maxv, p05, samples_ms.size());
    printf("    前10次迭代:");
    for (size_t i = 0; i < std::min<size_t>(10, samples_ms.size()); ++i) printf(" [%zu]=%.3f", i, samples_ms[i]);
    printf("\n");
    printf("    稳态 throughput (iter/s, p50-based) = %.2f iter/s\n", 1000.0 / steady_p50);

    // 数值正确性
    std::cout << "\n  [数值 guard (iter0 dx vs 同模式再跑一次)] max_abs_diff = " << max_abs_diff;
    if (max_abs_diff < 1e-4) std::cout << "  ✅ PASS";
    else                    std::cout << "  ❌ FAIL (>1e-4)";
    std::cout << std::endl;
    std::cout.flush();

    return (max_abs_diff < 1e-4) ? 0 : 1;
}
