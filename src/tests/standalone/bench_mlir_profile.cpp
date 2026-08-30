/**
 * @file bench_mlir_profile.cpp
 * @brief C3 MLIR 后端性能画像（2026-08-12）
 * @details 回答三个核心问题：
 *          1. MLIR JIT kernel vs 手写 SIMD vs 标量，到底差多少？
 *          2. 多线程分块能否吃满 CPU 多核带宽？
 *          3. 不同规模（1K ~ 4M）下的吞吐曲线，瓶颈是计算还是带宽？
 *
 *          覆盖两类典型算子：
 *            - Add      : 逐元素算术，memory-bound（MLIR 显式向量化主战场）
 *            - Sigmoid  : 超越函数，ALU-bound（手写 SIMD vs MLIR math 对比）
 *
 *          退出护栏：显式 shutdown() + clearCache()，避免 LLVM 全局析构
 *          recursive_mutex 崩溃（bench_c3_backward_perf 已知坑）。
 * @date 2026/08/12
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

#include "Tensor.h"
#include "CtorchError.h"
#include "CtorchScheduler.h"
#include "C3/C3Engine.h"
#include "C3/Graph.h"
#include "kernels/SIMDMath.h"

using namespace ct;
using namespace ct::c3;
using clk = std::chrono::steady_clock;

// ======================= 配置 =======================
static constexpr size_t SIZES[] = {1024, 16384, 131072, 1048576, 4194304};
static constexpr int NUM_SIZES = 5;
static constexpr int WARMUP = 20;
static constexpr int ITERS = 200;

// ======================= 直写实现 =======================

/// 标量 Add 基线
static void scalar_add(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

/// 编译器自动向量化 Add（-O3 -march=native 下 LLVM 应生成 NEON）
static void auto_vec_add(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

/// 标量 Sigmoid 基线
static void scalar_sigmoid(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = 1.0f / (1.0f + std::exp(-in[i]));
}

/// 手写 SIMD Sigmoid（NEON/AVX，SIMDMath）
static void simd_sigmoid(const float* in, float* out, size_t n) {
    ct::kernels::simd::vsigmoid(in, out, n);
}

// ======================= Graph 构建 =======================

static Graph buildAddGraph(const std::vector<size_t>& shape) {
    auto desc = TensorDesc::fromShape(shape);
    Graph g;
    size_t a = g.addInput(desc);
    size_t b = g.addInput(desc);
    size_t out = g.addNode(AddNode{desc, desc}, {a, b}, desc);
    g.markOutput(out);
    return g;
}

static Graph buildSigmoidGraph(const std::vector<size_t>& shape) {
    auto desc = TensorDesc::fromShape(shape);
    Graph g;
    size_t in = g.addInput(desc);
    size_t out = g.addNode(SigmoidNode{desc}, {in}, desc);
    g.markOutput(out);
    return g;
}

// ======================= 多线程分块 =======================

/// 多线程分块：把 [0,n) 切成 nthreads 段，各段一个 worker
/// 用于探明『多核分块能逼近多少带宽上限』
template <typename F>
static void parallel_for(const F& fn, size_t n, int nthreads) {
    if (nthreads <= 1) { fn(0, n); return; }
    std::vector<std::thread> workers;
    workers.reserve(nthreads);
    size_t chunk = (n + nthreads - 1) / nthreads;
    for (int t = 0; t < nthreads; ++t) {
        size_t begin = t * chunk;
        size_t end = std::min(begin + chunk, n);
        if (begin >= end) break;
        workers.emplace_back([&, begin, end]() { fn(begin, end); });
    }
    for (auto& w : workers) w.join();
}

// ======================= 计时代理 =======================

template <typename F>
static double bench_once(F&& func, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) func();
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) func();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t0).count();
    return static_cast<double>(elapsed) / iters;
}

// ======================= 报告 =======================

static void print_sep(char c = '-', int w = 100) {
    std::cout << std::string(w, c) << "\n";
}

static void print_row(const std::string& label, size_t n, double us) {
    double m_elem = (static_cast<double>(n) * 1e6) / (us * 1e6);  // M elem/s
    double gbps = m_elem * 4.0 * 2.0 / 1000.0;  // 读+写，float 4B → GB/s
    std::cout << std::left << std::setw(38) << label
              << std::right << std::setw(10) << (n >= 1024 ? (n / 1024) : n)
              << std::setw(12) << std::fixed << std::setprecision(2) << us
              << std::setw(12) << std::fixed << std::setprecision(2) << m_elem
              << std::setw(12) << std::fixed << std::setprecision(2) << gbps
              << "\n";
}

// ======================= 主函数 =======================

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);
    CtorchScheduler::getInstance();
    auto& engine = C3Engine::getInstance();
    engine.setCompileTimeoutMs(60000);

    std::cout << "核数: " << std::thread::hardware_concurrency() << "\n\n";

    // ============ Add 画像 ============
    {
        Graph g = buildAddGraph({SIZES[NUM_SIZES - 1]});
        CompileOptions opts;
        opts.opt_level = 3;
        engine.clearCache();
        auto kernel = engine.compileAsync(g, opts).get();
        if (!kernel) { std::cerr << "FATAL: MLIR Add 编译失败\n"; return 1; }

        print_sep('=');
        std::cout << "  Add 逐元素（memory-bound）\n";
        print_sep('=');
        for (int si = 0; si < NUM_SIZES; ++si) {
            size_t n = SIZES[si];
            std::vector<float> a(n), b(n), out(n);
            for (size_t i = 0; i < n; ++i) { a[i] = (float)i; b[i] = (float)(i + 1); }

            Tensor ta(ShapeTag{}, {n}), tb(ShapeTag{}, {n});
            std::copy(a.begin(), a.end(), ta.data_write<float>());
            std::copy(b.begin(), b.end(), tb.data_write<float>());

            double s = bench_once([&]() { scalar_add(a.data(), b.data(), out.data(), n); }, WARMUP, ITERS);
            print_row("  标量", n, s);
            double v = bench_once([&]() { auto_vec_add(a.data(), b.data(), out.data(), n); }, WARMUP, ITERS);
            print_row("  自动向量化", n, v);
            double m = bench_once([&]() { kernel->execute({ta, tb}); }, WARMUP, ITERS);
            print_row("  MLIR JIT", n, m);
            print_sep('-');
            std::cout << "    [加速比 vs 标量] 自动向量化=" << std::fixed << std::setprecision(2) << (s / v)
                      << "x  MLIR=" << (s / m) << "x\n";
            std::cout << "    [MLIR 相对自动向量化] " << std::fixed << std::setprecision(2) << (v / m) << "x\n";
            print_sep('-');
            std::cout << "\n";
        }
        engine.clearCache();
    }

    // ============ Sigmoid 画像 ============
    {
        Graph g = buildSigmoidGraph({SIZES[NUM_SIZES - 1]});
        CompileOptions opts;
        opts.opt_level = 3;
        engine.clearCache();
        auto kernel = engine.compileAsync(g, opts).get();
        if (!kernel) { std::cerr << "FATAL: MLIR Sigmoid 编译失败\n"; return 1; }

        print_sep('=');
        std::cout << "  Sigmoid 逐元素（ALU-bound）\n";
        print_sep('=');
        for (int si = 0; si < NUM_SIZES; ++si) {
            size_t n = SIZES[si];
            std::vector<float> in(n), simd_out(n), scalar_out(n);
            for (size_t i = 0; i < n; ++i) in[i] = (i % 2 == 0) ? 0.5f : -0.5f;

            Tensor tin(ShapeTag{}, {n});
            std::copy(in.begin(), in.end(), tin.data_write<float>());

            double s = bench_once([&]() { scalar_sigmoid(in.data(), scalar_out.data(), n); }, WARMUP, ITERS);
            print_row("  标量 (expf)", n, s);
            double v = bench_once([&]() { simd_sigmoid(in.data(), simd_out.data(), n); }, WARMUP, ITERS);
            print_row("  手写 SIMD", n, v);
            double m = bench_once([&]() { kernel->execute({tin}); }, WARMUP, ITERS);
            print_row("  MLIR JIT", n, m);
            print_sep('-');
            std::cout << "    [加速比 vs 标量] SIMD=" << std::fixed << std::setprecision(2) << (s / v)
                      << "x  MLIR=" << (s / m) << "x\n";
            std::cout << "    [MLIR 相对手写 SIMD] " << std::fixed << std::setprecision(2) << (v / m) << "x\n";
            print_sep('-');
            std::cout << "\n";
        }
        engine.clearCache();
    }

    // ============ 多核带宽上限探针 ============
    print_sep('=');
    std::cout << "  多核 parallel_for 带宽上限（Add 分块，4M 元素）\n";
    print_sep('=');
    {
        size_t n = SIZES[NUM_SIZES - 1];  // 4M
        std::vector<float> a(n), b(n), out(n);
        for (size_t i = 0; i < n; ++i) { a[i] = (float)i; b[i] = (float)(i + 1); }
        int cores = static_cast<int>(std::thread::hardware_concurrency());
        if (cores == 0) cores = 4;

        for (int nt : {1, 2, 4, cores}) {
            double t = bench_once([&]() {
                parallel_for([&](size_t lo, size_t hi) {
                    for (size_t i = lo; i < hi; ++i) out[i] = a[i] + b[i];
                }, n, nt);
            }, WARMUP, ITERS);
            print_row("  parallel_for Add x" + std::to_string(nt), n, t);
        }
    }

    print_sep('=');

    // 退出护栏：避免 LLVM 全局析构 recursive_mutex 崩溃
    C3Engine::getInstance().shutdown();
    C3Engine::getInstance().clearCache();

    std::cout << "EXIT=0 (clean)\n";
    return 0;
}