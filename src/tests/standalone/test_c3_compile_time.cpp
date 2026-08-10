/**
 * @file test_c3_compile_time.cpp
 * @brief C3 JIT 编译耗时统计
 * @details 测量 C3 编译流程各阶段耗时，包括图优化、kernel 生成、底层编译等。
 *          为后续编译延迟优化提供基线数据。
 * @date 2026/8/2
 */

#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <numeric>

#include "C3/C3Engine.h"
#include "C3/Graph.h"

// ======================= 计时工具 =======================

struct TimingResult {
    std::string label;
    double total_ms;      // 总耗时 (ms)
    double avg_ms;        // 平均耗时 (ms)
    size_t runs;          // 有效运行次数
};

static std::vector<TimingResult> g_results;

static void record(const std::string& label, double total_ms, size_t runs) {
    g_results.push_back({label, total_ms, total_ms / runs, runs});
}

static void printResults() {
    std::cout << "\n========== C3 Compile Time Breakdown ==========\n";
    std::cout << std::left << std::setw(50) << "Config"
              << std::right << std::setw(14) << "Avg (ms)"
              << std::setw(14) << "Total (ms)"
              << std::setw(10) << "Runs" << "\n";
    std::cout << std::string(88, '-') << "\n";

    for (const auto& r : g_results) {
        std::cout << std::left << std::setw(50) << r.label
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(14) << r.avg_ms
                  << std::setw(14) << r.total_ms
                  << std::setw(10) << r.runs << "\n";
    }
    std::cout << std::string(88, '=') << "\n";
}

// ======================= 测试辅助 =======================

using namespace ct::c3;

/// 构建单节点二元图
static Graph makeBinaryOp(NodeVariant op, TensorDesc ta, TensorDesc tb, TensorDesc tout) {
    Graph g;
    size_t a = g.addInput(ta);
    size_t b = g.addInput(tb);
    size_t node = g.addNode(op, {a, b}, tout);
    g.markOutput(node);
    return g;
}

/// 构建单节点一元图
static Graph makeUnaryOp(NodeVariant op, TensorDesc tin, TensorDesc tout) {
    Graph g;
    size_t x = g.addInput(tin);
    size_t node = g.addNode(op, {x}, tout);
    g.markOutput(node);
    return g;
}

/// 构建 3 层 MLP 图 (784→256→128→10)
static Graph makeMLP3Layer() {
    Graph g;
    auto desc_x  = TensorDesc::fromShape({1, 784});
    auto desc_w1 = TensorDesc::fromShape({784, 256});
    auto desc_b1 = TensorDesc::fromShape({1, 256});
    auto desc_w2 = TensorDesc::fromShape({256, 128});
    auto desc_b2 = TensorDesc::fromShape({1, 128});
    auto desc_w3 = TensorDesc::fromShape({128, 10});
    auto desc_b3 = TensorDesc::fromShape({1, 10});

    size_t x  = g.addInput(desc_x);
    size_t w1 = g.addInput(desc_w1);
    size_t b1 = g.addInput(desc_b1);
    size_t w2 = g.addInput(desc_w2);
    size_t b2 = g.addInput(desc_b2);
    size_t w3 = g.addInput(desc_w3);
    size_t b3 = g.addInput(desc_b3);

    size_t mm1 = g.addNode(MatMulNode{desc_x, desc_w1}, {x, w1}, TensorDesc::fromShape({1, 256}));
    size_t a1  = g.addNode(AddNode{}, {mm1, b1}, TensorDesc::fromShape({1, 256}));
    size_t r1  = g.addNode(ReLUNode{}, {a1}, TensorDesc::fromShape({1, 256}));
    size_t mm2 = g.addNode(MatMulNode{TensorDesc::fromShape({1, 256}), desc_w2}, {r1, w2}, TensorDesc::fromShape({1, 128}));
    size_t a2  = g.addNode(AddNode{}, {mm2, b2}, TensorDesc::fromShape({1, 128}));
    size_t r2  = g.addNode(ReLUNode{}, {a2}, TensorDesc::fromShape({1, 128}));
    size_t mm3 = g.addNode(MatMulNode{TensorDesc::fromShape({1, 128}), desc_w3}, {r2, w3}, TensorDesc::fromShape({1, 10}));
    size_t a3  = g.addNode(AddNode{}, {mm3, b3}, TensorDesc::fromShape({1, 10}));

    g.markOutput(a3);
    return g;
}

/// 构建融合链：N 个连续的 Add+Mul 操作
static Graph makeFusionChain(size_t chain_len) {
    Graph g;
    auto desc = TensorDesc::fromShape({1024, 1024});
    std::vector<size_t> inputs;
    // 2*chain_len + 1 个输入：x, c1, c2, c3, ...
    size_t x = g.addInput(desc);
    for (size_t i = 0; i < chain_len * 2; ++i) {
        inputs.push_back(g.addInput(desc));
    }

    size_t prev = x;
    for (size_t i = 0; i < chain_len; ++i) {
        size_t m = g.addNode(MulNode{}, {prev, inputs[i * 2]}, desc);
        prev = g.addNode(AddNode{}, {m, inputs[i * 2 + 1]}, desc);
    }
    g.markOutput(prev);
    return g;
}

// ======================= 编译耗时测量 =======================

/// 测量编译时间（清除缓存后编译，确保冷启动）
static double measureCompileTime(const Graph& graph, const CompileOptions& opts) {
    auto& engine = C3Engine::getInstance();
    // 编译前清除缓存，确保冷启动
    engine.clearCache();

    auto t0 = std::chrono::high_resolution_clock::now();
    auto kernel = engine.compile(graph, opts);
    auto t1 = std::chrono::high_resolution_clock::now();

    if (!kernel) {
        std::cerr << "  [WARN] compile returned nullptr" << std::endl;
        return -1.0;
    }

    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

/// 测量热启动编译时间（缓存命中）
static double measureCacheHitTime(const Graph& graph, const CompileOptions& opts) {
    auto& engine = C3Engine::getInstance();
    // 先编译一次写入缓存
    engine.compile(graph, opts);
    // 再编译一次，应命中缓存
    auto t0 = std::chrono::high_resolution_clock::now();
    engine.compile(graph, opts);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ======================= 测试用例 =======================

static void testSingleOp() {
    std::cout << "\n--- Single Op Compile Time ---" << std::endl;

    TensorDesc d1024 = TensorDesc::fromShape({1024, 1024});
    TensorDesc d256  = TensorDesc::fromShape({256, 256});
    TensorDesc d1d   = TensorDesc::fromShape({1024});

    // Add
    {
        auto g = makeBinaryOp(AddNode{}, d1024, d1024, d1024);
        CompileOptions opts;
        opts.enable_cache = false;
        opts.enable_fusion = false;

        // Handwritten
        opts.backend = C3Backend::Handwritten;
        double hw = measureCompileTime(g, opts);
        record("Add (1024x1024) Handwritten", hw, 1);

        // MLIR
        opts.backend = C3Backend::MLIR;
        double mlir = measureCompileTime(g, opts);
        record("Add (1024x1024) MLIR", mlir, 1);
    }

    // MatMul
    {
        auto g = makeBinaryOp(MatMulNode{d256, d256}, d256, d256, d256);
        CompileOptions opts;
        opts.enable_cache = false;
        opts.enable_fusion = false;

        opts.backend = C3Backend::Handwritten;
        double hw = measureCompileTime(g, opts);
        record("MatMul (256x256) Handwritten", hw, 1);

        opts.backend = C3Backend::MLIR;
        double mlir = measureCompileTime(g, opts);
        record("MatMul (256x256) MLIR", mlir, 1);
    }

    // ReLU
    {
        auto g = makeUnaryOp(ReLUNode{}, d1024, d1024);
        CompileOptions opts;
        opts.enable_cache = false;
        opts.enable_fusion = false;

        opts.backend = C3Backend::Handwritten;
        double hw = measureCompileTime(g, opts);
        record("ReLU (1024x1024) Handwritten", hw, 1);

        opts.backend = C3Backend::MLIR;
        double mlir = measureCompileTime(g, opts);
        record("ReLU (1024x1024) MLIR", mlir, 1);
    }
}

static void testMLP() {
    std::cout << "\n--- MLP Graph Compile Time ---" << std::endl;

    auto g = makeMLP3Layer();

    // 3-layer MLP, 多节点图
    CompileOptions opts;
    opts.enable_cache = false;

    // 不融合
    opts.enable_fusion = false;
    opts.backend = C3Backend::Handwritten;
    double hw_nf = measureCompileTime(g, opts);
    record("3-Layer MLP (784→256→128→10) Handwritten Non-Fused", hw_nf, 1);

    opts.backend = C3Backend::MLIR;
    double mlir_nf = measureCompileTime(g, opts);
    record("3-Layer MLP (784→256→128→10) MLIR Non-Fused", mlir_nf, 1);

    // 融合
    opts.enable_fusion = true;
    opts.backend = C3Backend::Handwritten;
    double hw_f = measureCompileTime(g, opts);
    record("3-Layer MLP (784→256→128→10) Handwritten Fused", hw_f, 1);

    opts.backend = C3Backend::MLIR;
    double mlir_f = measureCompileTime(g, opts);
    record("3-Layer MLP (784→256→128→10) MLIR Fused", mlir_f, 1);
}

static void testFusionChain() {
    std::cout << "\n--- Fusion Chain Compile Time ---" << std::endl;

    CompileOptions opts;
    opts.enable_cache = false;
    opts.enable_fusion = true;

    for (size_t len : {2, 4, 6}) {
        auto g = makeFusionChain(len);

        opts.backend = C3Backend::Handwritten;
        double hw = measureCompileTime(g, opts);
        record("Fusion Chain (" + std::to_string(len) + " ops) Handwritten", hw, 1);

        opts.backend = C3Backend::MLIR;
        double mlir = measureCompileTime(g, opts);
        record("Fusion Chain (" + std::to_string(len) + " ops) MLIR", mlir, 1);
    }
}

static void testCacheHit() {
    std::cout << "\n--- Cache Hit Latency ---" << std::endl;

    TensorDesc d1024 = TensorDesc::fromShape({1024, 1024});
    auto g = makeBinaryOp(AddNode{}, d1024, d1024, d1024);

    CompileOptions opts;
    opts.enable_cache = true;
    opts.enable_fusion = false;

    opts.backend = C3Backend::Handwritten;
    double hw = measureCacheHitTime(g, opts);
    record("Cache Hit Add (1024x1024) Handwritten", hw, 1);

    opts.backend = C3Backend::MLIR;
    double mlir = measureCacheHitTime(g, opts);
    record("Cache Hit Add (1024x1024) MLIR", mlir, 1);
}

static void testHotCompile() {
    std::cout << "\n--- Hot Compile (warm cache, different shapes) ---" << std::endl;

    // 编译 3 个不同形状的 MatMul，模拟 MLP 中不同层的编译
    CompileOptions opts;
    opts.enable_cache = true;
    opts.enable_fusion = false;

    std::vector<std::pair<size_t, size_t>> shapes = {
        {784, 256}, {256, 128}, {128, 10}
    };

    for (auto [k, n] : shapes) {
        auto dA = TensorDesc::fromShape({1, k});
        auto dB = TensorDesc::fromShape({k, n});
        auto dC = TensorDesc::fromShape({1, n});
        auto g = makeBinaryOp(MatMulNode{dA, dB}, dA, dB, dC);

        // Handwritten
        opts.backend = C3Backend::Handwritten;
        double hw = measureCompileTime(g, opts);
        record("MatMul (1×" + std::to_string(k) + ")×(" + std::to_string(k) + "×" + std::to_string(n) + ") Handwritten", hw, 1);

        // MLIR
        opts.backend = C3Backend::MLIR;
        double mlir = measureCompileTime(g, opts);
        record("MatMul (1×" + std::to_string(k) + ")×(" + std::to_string(k) + "×" + std::to_string(n) + ") MLIR", mlir, 1);
    }
}

static void testParallelCompile() {
    std::cout << "\n--- Parallel Compilation (3-layer MLP, sequential vs parallel) ---" << std::endl;

    // 构建 3 个独立子图（MLP 各层）
    auto g1 = makeBinaryOp(MatMulNode{TensorDesc::fromShape({1, 784}), TensorDesc::fromShape({784, 256})},
                           TensorDesc::fromShape({1, 784}), TensorDesc::fromShape({784, 256}),
                           TensorDesc::fromShape({1, 256}));
    auto g2 = makeBinaryOp(MatMulNode{TensorDesc::fromShape({1, 256}), TensorDesc::fromShape({256, 128})},
                           TensorDesc::fromShape({1, 256}), TensorDesc::fromShape({256, 128}),
                           TensorDesc::fromShape({1, 128}));
    auto g3 = makeBinaryOp(MatMulNode{TensorDesc::fromShape({1, 128}), TensorDesc::fromShape({128, 10})},
                           TensorDesc::fromShape({1, 128}), TensorDesc::fromShape({128, 10}),
                           TensorDesc::fromShape({1, 10}));

    auto& engine = C3Engine::getInstance();
    CompileOptions opts;
    opts.backend = C3Backend::MLIR;
    opts.enable_cache = false;
    opts.enable_fusion = false;

    // 串行编译
    engine.clearCache();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (const auto& g : {g1, g2, g3}) {
        engine.compile(g, opts);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double seq_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    record("3 MatMuls Sequential (MLIR)", seq_ms, 1);

    // 并行编译
    engine.clearCache();
    t0 = std::chrono::high_resolution_clock::now();
    auto kernels = engine.compileParallel({g1, g2, g3}, opts);
    t1 = std::chrono::high_resolution_clock::now();
    double par_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    record("3 MatMuls Parallel (MLIR)", par_ms, 1);

    // 正确性验证
    bool all_valid = true;
    for (auto& k : kernels) {
        if (!k) { all_valid = false; break; }
    }
    if (all_valid) {
        std::cout << "  Parallel compile: all kernels valid ✓\n";
    } else {
        std::cout << "  Parallel compile: SOME KERNELS INVALID ✗\n";
    }

    double speedup = seq_ms / par_ms;
    std::cout << "  Speedup: " << seq_ms << "ms (seq) → " << par_ms
              << "ms (par) = " << std::fixed << std::setprecision(2) << speedup << "x\n";

    // 测试缓存命中：并行编译后再次编译应命中缓存
    t0 = std::chrono::high_resolution_clock::now();
    auto cached_kernels = engine.compileParallel({g1, g2, g3}, opts);
    t1 = std::chrono::high_resolution_clock::now();
    double cache_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    record("3 MatMuls Cache Hit (MLIR)", cache_ms, 1);
    std::cout << "  Cache hit: " << cache_ms << "ms\n";
}

// ======================= main =======================

int main() {
    std::cout << "============================================" << std::endl;
    std::cout << "  C3 JIT Compile Time Benchmark" << std::endl;
    std::cout << "============================================" << std::endl;

    // 1. 单算子编译时间
    testSingleOp();

    // 2. MLP 图编译时间
    testMLP();

    // 3. 融合链编译时间
    testFusionChain();

    // 4. 缓存命中延迟
    testCacheHit();

    // 5. 热启动编译（不同形状）
    testHotCompile();

    // 6. 并行编译 vs 串行编译
    testParallelCompile();

    // 打印汇总
    printResults();

    std::cout << "\nAll compile time measurements done." << std::endl;
    return 0;
}