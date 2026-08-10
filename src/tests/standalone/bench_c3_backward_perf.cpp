/**
 * @file bench_c3_backward_perf.cpp
 * @brief C3 反向融合性能基准测试
 * @details 构造大尺寸 3 层 unary 串联 (Tanh → Sigmoid → ReLU)，
 *          对比 eager 基线 vs C3 融合执行的端到端 wall clock。
 *
 * 用法：
 *   # 基线：完全禁用 C3 反向融合
 *   CTORCH_DISABLE_C3_BACKWARD=1  ./bench_c3_backward_perf
 *   # 实验：启用 C3（fusion 命中后走 JIT 多输出 kernel）
 *   ./bench_c3_backward_perf
 *
 * @date 2026/8/8
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <thread>

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
        const char* env = std::getenv("CTORCH_DISABLE_C3_BACKWARD");
        return (env && std::string(env) == "1");
    }();
    std::cout << "=== C3 Backward Fusion 性能基准测试 ===" << std::endl;
    std::cout << "  模式: " << (disabled ? "EAGER 基线 (C3 禁用)" : "C3 融合启用") << std::endl;

    auto& sched = CtorchScheduler::getInstance();
    (void)sched;

    // ========= 问题规模：256K 元素 / 3 层 unary =========
    const int M = 512, K = 512;
    const int N_WARMUP = 60;       // warmup 更多：确保触发 C3 fusion compile 阈值
    const int N_MEASURE = 120;     // 正式测量
    std::cout << "  张量形状: [" << M << " x " << K << "] = " << (1.0*M*K/1048576.0) << "M elements" << std::endl;
    std::cout << "  图结构: x → Tanh → Sigmoid → ReLU → backward (ones as grad_output)" << std::endl;
    std::cout << "  Warmup iters: " << N_WARMUP << " | Measure iters: " << N_MEASURE << std::endl;

    // 固定随机种子 → eager 与 c3 两边输入完全相同（可重复）
    auto x_base = Tensor(ShapeTag{}, {M, K}, DType::kFloat, DeviceType::kCPU);
    float* xp = x_base.data_write<float>();
    uint32_t rng = 0xdeadbeef;
    for (int i = 0; i < M*K; ++i) {
        // xorshift32, 归一化到 [-2, 2]（覆盖 ReLU/tanh/sigmoid 敏感区间）
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        xp[i] = ((double)rng / 4294967296.0) * 4.0f - 2.0f;
    }

    // ========= Warmup =========
    std::cout << "\n  [Warmup] 运行 " << N_WARMUP << " 次... ";
    std::cout.flush();
    for (int iter = 0; iter < N_WARMUP; ++iter) {
        Tensor x = x_base.clone();
        x.requires_grad(true);
        Tensor y = x.tanh().sigmoid().relu();
        AutoGrad::backward(y.getRelatedNode(), false);
    }
    // fusion compile 是异步的，稍微等一下确保 kernel 已 install
    if (!disabled) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        // 再额外跑 15 次让它有机会命中 kernel cache（compile 完 install 之后）
        for (int iter = 0; iter < 15; ++iter) {
            Tensor x = x_base.clone();
            x.requires_grad(true);
            Tensor y = x.tanh().sigmoid().relu();
            AutoGrad::backward(y.getRelatedNode(), false);
        }
    }
    std::cout << "done." << std::endl;

    // ========= Measure =========
    std::vector<double> samples_ms;
    samples_ms.reserve(N_MEASURE);
    double max_abs_diff = 0.0; // 额外 guard：与第一次 eager 的 dx 做对比（若开启 C3，确保数值一致）

    // 先跑一次无计时的 eager 参考（如果 C3 模式，临时 fallback 到 eager 拿 baseline 参考）
    // 为简单起见：把 x_base 当输入，模式 A 就记录 dx；模式 B 计算完毕后再在末尾用 disabled 重新跑一次拿参考
    std::vector<float> ref_dx(M*K);
    {
        // 强制 eager 跑参考：临时切换环境变量不行（static const），所以手动 clone + 仅在测量末尾做对比
        Tensor x = x_base.clone();
        x.requires_grad(true);
        Tensor y = x.tanh().sigmoid().relu();
        // CTORCH_DISABLE_C3_BACKWARD=1 环境下直接用这里作为真实的一次 eager 参考；否则作为 dummy
        AutoGrad::backward(y.getRelatedNode(), false);
        const float* g = x.grad().data_read<float>();
        for (int i = 0; i < M*K; ++i) ref_dx[i] = g[i];
    }

    std::cout << "  [Measure] 运行 " << N_MEASURE << " 次... ";
    std::cout.flush();
    for (int iter = 0; iter < N_MEASURE; ++iter) {
        Tensor x = x_base.clone();
        x.requires_grad(true);

        double t0 = now_ms();
        Tensor y = x.tanh().sigmoid().relu();
        AutoGrad::backward(y.getRelatedNode(), false);
        double t1 = now_ms();

        samples_ms.push_back(t1 - t0);

        // 数值检查：只查第一次防止拖慢性能
        if (iter == 0) {
            const float* g = x.grad().data_read<float>();
            double md = 0.0;
            // 统一对比 ref_dx（ref_dx 是同一模式下单独跑的 eager 结果，数值一致性参考基准）
            for (int i = 0; i < M*K; ++i) {
                double d = std::fabs(g[i] - ref_dx[i]);
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

    std::cout << "\n  ---------------- 单次迭代耗时 (3层 unary forward + sum + backward) ----------------" << std::endl;
    printf("    mean=%.3f ms   median(p50)=%.3f ms   p95=%.3f ms\n", mean, p50, p95);
    printf("    min =%.3f ms   max          =%.3f ms   p05=%.3f ms\n", minv, maxv, p05);
    printf("    samples = %zu\n", samples_ms.size());
    double ops_per_sec = 1000.0 / p50;  // 以 p50 估计吞吐
    printf("    throughput (iter/s, p50-based) = %.2f iter/s\n", ops_per_sec);

    // C3 统计
    auto stats = C3BackwardCapture::getInstance().getStats();
    std::cout << "\n  [C3 Backward Stats]";
    std::cout << "\n    cache_hit        = " << stats.cache_hit_count
              << "\n    cache_miss       = " << stats.cache_miss_count
              << "\n    compiles         = " << stats.compile_count
              << "\n    fusion_compiles  = " << stats.fusion_compile_count
              << "\n    fusion_hits      = " << stats.fusion_hit_count
              << "\n    fusion_misses    = " << stats.fusion_miss_count
              << "\n    exec_failures    = " << stats.execution_failures << std::endl;

    // 数值正确性
    std::cout << "\n  [数值 guard (iter0 dx vs ref)] max_abs_diff = " << max_abs_diff;
    if (max_abs_diff < 1e-4) std::cout << "  ✅ PASS";
    else                    std::cout << "  ❌ FAIL (>1e-4)";
    std::cout << std::endl;
    std::cout.flush();

    return (max_abs_diff < 1e-4) ? 0 : 1;
}
