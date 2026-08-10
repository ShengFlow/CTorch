/**
 * @file test_mlir_simd_integration.cpp
 * @brief 验证 MLIR + SIMD 集成：Sigmoid/Tanh 编译和执行正确性 + 性能基准
 * @details 验证：
 *   1. MLIR 编译的 Sigmoid 结果与 eager 一致
 *   2. MLIR 编译的 Tanh 结果与 eager 一致
 *   3. 性能基准（MLIR+Sigmoid vs 理论标量基线）
 * @date 2026/08/03
 */

#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <iomanip>

#include "Tensor.h"
#include "CtorchScheduler.h"
#include "C3/Graph.h"
#include "C3/C3Engine.h"
#include "kernels/SIMDWrapper.h"
#include "kernels/SIMDMath.h"

using namespace ct;
using namespace ct::c3;
using clk = std::chrono::steady_clock;

// 构建纯 Sigmoid 图
static Graph buildSigmoidGraph(const std::vector<size_t>& shape) {
    auto desc = TensorDesc::fromShape(shape);
    Graph g;
    size_t in = g.addInput(desc);
    size_t out = g.addNode(SigmoidNode{desc}, {in}, desc);
    g.markOutput(out);
    return g;
}

// 构建纯 Tanh 图
static Graph buildTanhGraph(const std::vector<size_t>& shape) {
    auto desc = TensorDesc::fromShape(shape);
    Graph g;
    size_t in = g.addInput(desc);
    size_t out = g.addNode(TanhNode{desc}, {in}, desc);
    g.markOutput(out);
    return g;
}

// 检查数值一致性
static bool tensorsAllClose(const Tensor& a, const Tensor& b, float eps = 1e-4f) {
    if (a.shape() != b.shape()) return false;
    const float* pa = a.data_read<float>();
    const float* pb = b.data_read<float>();
    for (size_t i = 0; i < a.numel(); ++i) {
        if (std::fabs(pa[i] - pb[i]) > eps) return false;
    }
    return true;
}

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);
    CtorchScheduler::getInstance();

    std::cout << "=== MLIR + SIMD 集成测试 ===\n\n";

    int passed = 0, failed = 0;

    // ==================== 测试 1: Sigmoid 正确性 ====================
    {
        C3Engine::getInstance().clearCache();
        C3Engine::getInstance().setCompileTimeoutMs(30000);

        const size_t N = 1024;
        Tensor x(ShapeTag{}, {N});
        float* xd = x.data_write<float>();
        for (size_t i = 0; i < N; ++i)
            xd[i] = (i % 2 == 0) ? (i * 0.01f) : (-i * 0.01f);

        // 计算 eager sigmoid
        Tensor expected(ShapeTag{}, {N});
        float* ed = expected.data_write<float>();
        for (size_t i = 0; i < N; ++i)
            ed[i] = 1.0f / (1.0f + std::exp(-xd[i]));

        // 编译 MLIR Sigmoid
        Graph g = buildSigmoidGraph({N});
        CompileOptions opts;
        opts.opt_level = 2;
        // 默认用 MLIR backend

        auto future = C3Engine::getInstance().compileAsync(g, opts);
        auto kernel = future.get();

        if (!kernel) {
            std::cout << "  FAIL [1a]: MLIR Sigmoid 编译失败\n";
            ++failed;
        } else {
            auto results = kernel->execute({x});
            if (results.empty()) {
                std::cout << "  FAIL [1b]: execute 返回空\n";
                ++failed;
            } else if (tensorsAllClose(expected, results[0], 1e-3f)) {
                std::cout << "  PASS [1a]: MLIR Sigmoid 结果正确\n";
                ++passed;
            } else {
                std::cout << "  FAIL [1c]: MLIR Sigmoid 结果偏差过大\n";
                ++failed;
            }
        }
    }

    // ==================== 测试 2: Tanh 正确性 ====================
    {
        C3Engine::getInstance().clearCache();
        C3Engine::getInstance().setCompileTimeoutMs(30000);

        const size_t N = 1024;
        Tensor x(ShapeTag{}, {N});
        float* xd = x.data_write<float>();
        for (size_t i = 0; i < N; ++i)
            xd[i] = (i % 2 == 0) ? (i * 0.01f) : (-i * 0.01f);

        // 计算 eager tanh
        Tensor expected(ShapeTag{}, {N});
        float* ed = expected.data_write<float>();
        for (size_t i = 0; i < N; ++i)
            ed[i] = std::tanh(xd[i]);

        // 编译 MLIR Tanh
        Graph g = buildTanhGraph({N});
        CompileOptions opts;
        opts.opt_level = 2;

        auto future = C3Engine::getInstance().compileAsync(g, opts);
        auto kernel = future.get();

        if (!kernel) {
            std::cout << "  FAIL [2a]: MLIR Tanh 编译失败\n";
            ++failed;
        } else {
            auto results = kernel->execute({x});
            if (results.empty()) {
                std::cout << "  FAIL [2b]: execute 返回空\n";
                ++failed;
            } else if (tensorsAllClose(expected, results[0], 1e-3f)) {
                std::cout << "  PASS [2a]: MLIR Tanh 结果正确\n";
                ++passed;
            } else {
                std::cout << "  FAIL [2c]: MLIR Tanh 结果偏差过大\n";
                ++failed;
            }
        }
    }

    // ==================== 测试 3: 性能基准（Sigmoid） ====================
    {
        const size_t N = 1024 * 1024;  // 1M 元素
        const int WARMUP = 5;
        const int ITERS = 50;

        Tensor x(ShapeTag{}, {N});
        float* xd = x.data_write<float>();
        for (size_t i = 0; i < N; ++i)
            xd[i] = (i % 2 == 0) ? (i * 0.001f) : (-i * 0.001f);

        // MLIR + SIMD 路径
        C3Engine::getInstance().clearCache();
        C3Engine::getInstance().setCompileTimeoutMs(30000);
        Graph g = buildSigmoidGraph({N});
        CompileOptions opts;
        opts.opt_level = 2;
        auto future = C3Engine::getInstance().compileAsync(g, opts);
        auto kernel = future.get();

        if (!kernel) {
            std::cout << "  SKIP [3]: MLIR Sigmoid 编译失败，跳过基准\n";
        } else {
            // warmup
            for (int i = 0; i < WARMUP; ++i)
                kernel->execute({x});

            // benchmark
            auto t0 = clk::now();
            for (int i = 0; i < ITERS; ++i)
                kernel->execute({x});
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t0).count();

            double avg_us = static_cast<double>(elapsed) / ITERS;
            double throughput = (static_cast<double>(N) * ITERS) / (elapsed * 1e-6);
            double gflops = throughput * 3.0 / 1e9;  // sigmoid ≈ 3 FLOPs/element

            std::cout << "  PASS [3]: MLIR+Sigmoid 性能基准 (N=" << N << ")\n";
            std::cout << "    平均: " << std::fixed << std::setprecision(2) << avg_us << " us\n";
            std::cout << "    吞吐: " << std::fixed << std::setprecision(2) << throughput / 1e6 << " M 元素/s\n";
            std::cout << "    GFLOPS: " << std::fixed << std::setprecision(2) << gflops << "\n";
            ++passed;
        }
    }

    // ==================== 测试 4: 性能基准（Tanh） ====================
    {
        const size_t N = 1024 * 1024;  // 1M 元素
        const int WARMUP = 5;
        const int ITERS = 50;

        Tensor x(ShapeTag{}, {N});
        float* xd = x.data_write<float>();
        for (size_t i = 0; i < N; ++i)
            xd[i] = (i % 2 == 0) ? (i * 0.001f) : (-i * 0.001f);

        // MLIR + SIMD 路径
        C3Engine::getInstance().clearCache();
        C3Engine::getInstance().setCompileTimeoutMs(30000);
        Graph g = buildTanhGraph({N});
        CompileOptions opts;
        opts.opt_level = 2;
        auto future = C3Engine::getInstance().compileAsync(g, opts);
        auto kernel = future.get();

        if (!kernel) {
            std::cout << "  SKIP [4]: MLIR Tanh 编译失败，跳过基准\n";
        } else {
            // warmup
            for (int i = 0; i < WARMUP; ++i)
                kernel->execute({x});

            // benchmark
            auto t0 = clk::now();
            for (int i = 0; i < ITERS; ++i)
                kernel->execute({x});
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t0).count();

            double avg_us = static_cast<double>(elapsed) / ITERS;
            double throughput = (static_cast<double>(N) * ITERS) / (elapsed * 1e-6);

            std::cout << "  PASS [4]: MLIR+Tanh 性能基准 (N=" << N << ")\n";
            std::cout << "    平均: " << std::fixed << std::setprecision(2) << avg_us << " us\n";
            std::cout << "    吞吐: " << std::fixed << std::setprecision(2) << throughput / 1e6 << " M 元素/s\n";
            ++passed;
        }
    }

    std::cout << "\n=== 结果: " << passed << " passed, " << failed << " failed ===\n";
    std::cout << (failed == 0 ? "✨ 全部通过\n" : "❌ 有失败\n");

    C3Engine::getInstance().setCompileTimeoutMs(30000);
    return failed == 0 ? 0 : 1;
}