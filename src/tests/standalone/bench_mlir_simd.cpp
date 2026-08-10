/**
 * @file bench_mlir_simd.cpp
 * @brief 综合 benchmark：MLIR+SIMD 批处理 vs 直接 SIMD vs 朴素标量
 * @details 对比三种路径在 Sigmoid/Tanh 上的性能，测量不同规模下的吞吐。
 * @date 2026/08/03
 */

#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <numeric>

#include "Tensor.h"
#include "CtorchScheduler.h"
#include "C3/Graph.h"
#include "C3/C3Engine.h"
#include "kernels/SIMDWrapper.h"
#include "kernels/SIMDMath.h"

using namespace ct;
using namespace ct::c3;
using clk = std::chrono::steady_clock;

// ====================== 测试配置 ======================
static constexpr size_t SIZES[] = {1024, 16384, 131072, 1048576, 4194304};
static constexpr int NUM_SIZES = 5;
static constexpr int WARMUP = 10;
static constexpr int ITERS = 100;

// ====================== 基准实现 ======================

/// 朴素标量 Sigmoid（每元素调用 expf）
static void scalar_sigmoid(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i)
        out[i] = 1.0f / (1.0f + std::exp(-in[i]));
}

/// 朴素标量 Tanh（每元素调用 tanhf）
static void scalar_tanh(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i)
        out[i] = std::tanh(in[i]);
}

// ====================== 构建测试图 ======================

static Graph buildSigmoidGraph(const std::vector<size_t>& shape) {
    auto desc = TensorDesc::fromShape(shape);
    Graph g;
    size_t in = g.addInput(desc);
    size_t out = g.addNode(SigmoidNode{desc}, {in}, desc);
    g.markOutput(out);
    return g;
}

static Graph buildTanhGraph(const std::vector<size_t>& shape) {
    auto desc = TensorDesc::fromShape(shape);
    Graph g;
    size_t in = g.addInput(desc);
    size_t out = g.addNode(TanhNode{desc}, {in}, desc);
    g.markOutput(out);
    return g;
}

// ====================== 基准引擎 ======================

struct BenchResult {
    std::string label;
    size_t n;
    std::vector<double> us_per_iter;  // 每次迭代的微秒数
};

/// 运行单次基准迭代
template<typename F>
static double bench_once(F&& func, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) func();
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) func();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t0).count();
    return static_cast<double>(elapsed) / iters;
}

/// 运行基准并收集结果
template<typename F>
static BenchResult run_bench(const std::string& label, size_t n, F&& func, int warmup, int iters) {
    BenchResult r;
    r.label = label;
    r.n = n;
    for (int trial = 0; trial < 3; ++trial) {
        r.us_per_iter.push_back(bench_once(func, warmup, iters));
    }
    return r;
}

// ====================== 报告输出 ======================

static void print_separator(char c = '=', int width = 80) {
    std::cout << std::string(width, c) << "\n";
}

static void print_results(const std::vector<BenchResult>& results, const std::string& title) {
    print_separator();
    std::cout << "  " << title << "\n";
    print_separator('-');

    // 表头
    std::cout << std::left << std::setw(28) << "实现"
              << std::right << std::setw(10) << "N"
              << std::setw(12) << "均时(μs)"
              << std::setw(12) << "M 元素/s"
              << std::setw(10) << "加速比"
              << "\n";
    print_separator('-');

    for (size_t si = 0; si < NUM_SIZES; ++si) {
        size_t n = SIZES[si];
        // 找标量基线时间（以 "标量" 开头且 N 匹配的第一个结果）
        double scalar_us = 0.0;
        for (const auto& r : results) {
            if (r.n == n && r.label.find("标量") == 0) {
                scalar_us = r.us_per_iter[0];
                break;
            }
        }
        if (scalar_us == 0.0) scalar_us = 1.0;  // 防除零

        for (const auto& r : results) {
            if (r.n != n) continue;
            double avg_us = std::accumulate(r.us_per_iter.begin(), r.us_per_iter.end(), 0.0) / r.us_per_iter.size();
            double m_elem_s = (static_cast<double>(n) * 1e6) / (avg_us * 1e6);  // M 元素/s
            double speedup = scalar_us / avg_us;

            std::cout << std::left << std::setw(28) << r.label
                      << std::right << std::setw(10) << n
                      << std::setw(12) << std::fixed << std::setprecision(2) << avg_us
                      << std::setw(12) << std::fixed << std::setprecision(2) << m_elem_s
                      << std::setw(10) << std::fixed << std::setprecision(2) << speedup
                      << "\n";
        }
    }
    print_separator();
}

// ====================== 主函数 ======================

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);
    CtorchScheduler::getInstance();

    // 预编译 MLIR kernel（所有测试共享）
    std::cout << "预编译 MLIR kernels...\n";
    auto& engine = C3Engine::getInstance();
    engine.setCompileTimeoutMs(30000);

    // 编译 Sigmoid MLIR kernel（用最大尺寸，所有尺寸共享）
    engine.clearCache();
    Graph sig_graph = buildSigmoidGraph({SIZES[NUM_SIZES - 1]});
    CompileOptions opts;
    opts.opt_level = 2;
    auto sig_future = engine.compileAsync(sig_graph, opts);
    auto sig_kernel = sig_future.get();
    if (!sig_kernel) {
        std::cerr << "FATAL: MLIR Sigmoid 编译失败\n";
        return 1;
    }

    // 编译 Tanh MLIR kernel
    engine.clearCache();
    Graph tanh_graph = buildTanhGraph({SIZES[NUM_SIZES - 1]});
    auto tanh_future = engine.compileAsync(tanh_graph, opts);
    auto tanh_kernel = tanh_future.get();
    if (!tanh_kernel) {
        std::cerr << "FATAL: MLIR Tanh 编译失败\n";
        return 1;
    }

    std::cout << "编译完成，开始基准测试...\n\n";

    // ======================= Sigmoid 基准 =======================
    std::vector<BenchResult> sigmoid_results;

    for (int si = 0; si < NUM_SIZES; ++si) {
        size_t n = SIZES[si];
        Tensor x(ShapeTag{}, {n});
        float* xd = x.data_write<float>();
        for (size_t i = 0; i < n; ++i)
            xd[i] = (i % 2 == 0) ? (i * 0.001f + 0.5f) : (-i * 0.001f - 0.5f);

        Tensor out_scalar(ShapeTag{}, {n});
        Tensor out_simd(ShapeTag{}, {n});

        // [1] 标量基线
        sigmoid_results.push_back(run_bench(
            "标量 (expf)", n,
            [&]() { scalar_sigmoid(xd, out_scalar.data_write<float>(), n); },
            WARMUP, ITERS));

        // [2] 直接 SIMD 调用
        sigmoid_results.push_back(run_bench(
            "直接 SIMD vsigmoid", n,
            [&]() { ct::kernels::simd::vsigmoid(xd, out_simd.data_write<float>(), n); },
            WARMUP, ITERS));

        // [3] MLIR + SIMD 批处理
        sigmoid_results.push_back(run_bench(
            "MLIR+Sigmoid (SIMD)", n,
            [&]() { sig_kernel->execute({x}); },
            WARMUP, ITERS));
    }

    print_results(sigmoid_results, "Sigmoid 性能对比");

    // ======================= Tanh 基准 =======================
    std::vector<BenchResult> tanh_results;

    for (int si = 0; si < NUM_SIZES; ++si) {
        size_t n = SIZES[si];
        Tensor x(ShapeTag{}, {n});
        float* xd = x.data_write<float>();
        for (size_t i = 0; i < n; ++i)
            xd[i] = (i % 2 == 0) ? (i * 0.001f + 0.5f) : (-i * 0.001f - 0.5f);

        Tensor out_scalar(ShapeTag{}, {n});
        Tensor out_simd(ShapeTag{}, {n});

        // [1] 标量基线
        tanh_results.push_back(run_bench(
            "标量 (tanhf)", n,
            [&]() { scalar_tanh(xd, out_scalar.data_write<float>(), n); },
            WARMUP, ITERS));

        // [2] 直接 SIMD 调用
        tanh_results.push_back(run_bench(
            "直接 SIMD vtanh", n,
            [&]() { ct::kernels::simd::vtanh(xd, out_simd.data_write<float>(), n); },
            WARMUP, ITERS));

        // [3] MLIR + SIMD 批处理
        tanh_results.push_back(run_bench(
            "MLIR+Tanh (SIMD)", n,
            [&]() { tanh_kernel->execute({x}); },
            WARMUP, ITERS));
    }

    print_results(tanh_results, "Tanh 性能对比");

    // ======================= 总结分析 =======================
    print_separator('=');
    std::cout << "  总结\n";
    print_separator('=');

    // 计算平均加速比
    auto avg_speedup = [](const std::vector<BenchResult>& results, const std::string& prefix) {
        double total = 0;
        int count = 0;
        for (size_t si = 0; si < NUM_SIZES; ++si) {
            size_t n = SIZES[si];
            double scalar_us = 0;
            double target_us = 0;
            for (const auto& r : results) {
                if (r.n == n && r.label.find("标量") == 0)
                    scalar_us = r.us_per_iter[0];
                if (r.n == n && r.label.find(prefix) == 0)
                    target_us = r.us_per_iter[0];
            }
            if (scalar_us > 0 && target_us > 0) {
                total += scalar_us / target_us;
                count++;
            }
        }
        return count > 0 ? total / count : 0.0;
    };

    double sig_simd_speedup = avg_speedup(sigmoid_results, "直接 SIMD");
    double sig_mlir_speedup = avg_speedup(sigmoid_results, "MLIR+Sigmoid");
    double tanh_simd_speedup = avg_speedup(tanh_results, "直接 SIMD");
    double tanh_mlir_speedup = avg_speedup(tanh_results, "MLIR+Tanh");

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Sigmoid: 直接 SIMD vsigmoid  vs 标量: " << sig_simd_speedup << "x\n";
    std::cout << "  Sigmoid: MLIR+Sigmoid (SIMD) vs 标量: " << sig_mlir_speedup << "x\n";
    std::cout << "  Tanh:    直接 SIMD vtanh     vs 标量: " << tanh_simd_speedup << "x\n";
    std::cout << "  Tanh:    MLIR+Tanh (SIMD)    vs 标量: " << tanh_mlir_speedup << "x\n";

    double sig_overhead = (sig_mlir_speedup > 0 && sig_simd_speedup > 0)
        ? (sig_simd_speedup / sig_mlir_speedup) : 0.0;
    double tanh_overhead = (tanh_mlir_speedup > 0 && tanh_simd_speedup > 0)
        ? (tanh_simd_speedup / tanh_mlir_speedup) : 0.0;

    std::cout << "  MLIR 包装开销 (Sigmoid): " << sig_overhead << "x\n";
    std::cout << "  MLIR 包装开销 (Tanh):    " << tanh_overhead << "x\n";

    std::cout << "\n  结论:\n";
    if (sig_overhead < 1.05)
        std::cout << "    ✅ MLIR 包装开销可忽略 (< 5%)\n";
    else
        std::cout << "    ⚠️  MLIR 包装开销 " << sig_overhead << "x\n";

    print_separator('=');

    return 0;
}