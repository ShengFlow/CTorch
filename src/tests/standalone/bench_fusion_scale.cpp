/**
 * @file bench_fusion_scale.cpp
 * @brief 规模扫描：寻找 C3 融合 kernel 在这台机器上开始超越 eager 逐算子的规模拐点
 * @details
 *  - elementwise 链: relu(neg(sigmoid(x)*tanh(x)))  (5 个 elementwise op)
 *      eager 需 5 趟内存往返, C3 融合成 1 个 kernel 只读/写一次
 *  - matmul+act 链: sigmoid(X@W+B)
 * 分别在不同规模下对比 C3 fused kernel 与 eager 逐算子耗时, 报告 speedup。
 * 只测 kernel 本身（不经 autograd/热路径调度），隔离"融合收益"。
 * @date 2026-08-28
 */
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>
#include "Tensor.h"
#include "C3/C3Engine.h"
#include "C3/Graph.h"
#include "C3/C3Cleanup.h"

using namespace ct;
using namespace ct::c3;
using hires = std::chrono::high_resolution_clock;
using us = std::chrono::duration<double, std::micro>;

static void fill(Tensor& t, float seed) {
    float* p = t.data_write<float>();
    for (size_t i = 0; i < t.numel(); ++i) p[i] = seed + (float)(i % 13) * 0.01f;
}

// ---------------- elementwise 链 ----------------
// graph: x -> tanh -> sigmoid -> relu  (严格线性链，无分支)
static std::shared_ptr<CompiledKernel> compileElementwise(size_t N) {
    Graph g;
    auto d = TensorDesc::fromShape({N});
    size_t x = g.addInput(d);
    size_t t = g.addNode(TanhNode{d}, {x}, d);
    size_t s = g.addNode(SigmoidNode{d}, {t}, d);
    size_t r = g.addNode(ReLUNode{d}, {s}, d);
    g.markOutput(r);
    CompileOptions opts; opts.backend = C3Backend::MLIR; opts.opt_level = 3;
    try { return C3Engine::getInstance().compile(g, opts); }
    catch (const std::exception& e) { std::cerr << "[compile-EW-FAIL] " << e.what() << "\n"; return nullptr; }
}
static Tensor eagerElementwise(const Tensor& x) {
    Tensor t = x.tanh();
    Tensor s = t.sigmoid();
    return s.relu();
}

// ---------------- matmul+act 链 ----------------
// graph: X[B,K] @ W[K,N] + Bb[N] -> sigmoid -> out
static std::shared_ptr<CompiledKernel> compileMatmulAct(size_t B, size_t K, size_t N) {
    Graph g;
    auto xd = TensorDesc::fromShape({B, K});
    auto wd = TensorDesc::fromShape({K, N});
    auto bd = TensorDesc::fromShape({N});
    auto od = TensorDesc::fromShape({B, N});
    size_t x = g.addInput(xd);
    size_t w = g.addInput(wd);
    size_t bb = g.addInput(bd);
    size_t mm = g.addNode(MatMulNode{xd, wd}, {x, w}, od);
    size_t add = g.addNode(AddNode{od, bd}, {mm, bb}, od);
    size_t sig = g.addNode(SigmoidNode{od}, {add}, od);
    g.markOutput(sig);
    CompileOptions opts; opts.backend = C3Backend::MLIR; opts.opt_level = 3; opts.enable_fusion = true;
    try { return C3Engine::getInstance().compile(g, opts); }
    catch (const std::exception& e) { std::cerr << "[compile-MM-FAIL] " << e.what() << "\n"; return nullptr; }
}
static Tensor eagerMatmulAct(const Tensor& X, const Tensor& W, const Tensor& Bb) {
    Tensor mm = matMul(X, W);
    Tensor add = mm + Bb;
    return add.sigmoid();
}

template <typename F>
static double timeIt(int iters, F&& f) {
    // warmup
    for (int i = 0; i < 3; ++i) f();
    auto t0 = hires::now();
    volatile float sink = 0;
    for (int i = 0; i < iters; ++i) { auto r = f(); sink += r.template data_read<float>()[0]; }
    auto t1 = hires::now();
    (void)sink;
    return std::chrono::duration_cast<us>(t1 - t0).count() / (double)iters;
}

int main() {
    std::cout << "=== C3 fusion-scale sweep (this machine) ===\n";
    std::cout << "  elementwise: relu(neg(sigmoid(x)*tanh(x)))  — fused 1 pass vs eager 5 passes\n\n";

    // ---- elementwise 链 ----
    const size_t ew_sizes[] = {1u<<20, 4u<<20, 16u<<20, 32u<<20, 64u<<20};
    std::cout << "[elementwise chain]\n";
    for (size_t N : ew_sizes) {
        auto k = compileElementwise(N);
        if (!k) { std::cout << "  N=" << N << " compile FAILED\n"; continue; }
        Tensor x(ShapeTag{}, {N}); fill(x, 0.5f);
        Tensor ref = eagerElementwise(x);
        auto eager_t = timeIt(30, [&]{ return eagerElementwise(x); });
        auto c3_t = timeIt(30, [&]{ auto r = k->execute({x}); return r.empty() ? ref : r[0]; });
        double speedup = eager_t / c3_t;
        std::cout << "  N=" << std::setw(9) << N << " (" << std::setw(5) << (N*4/1024/1024) << " MB)  eager="
                  << std::fixed << std::setprecision(1) << eager_t << "us  c3=" << c3_t
                  << "us  speedup=" << std::setprecision(2) << speedup << "x"
                  << (speedup > 1.0 ? "  ✅ C3更快" : "  ❌ eager更快") << "\n";
    }

    // ---- matmul+act 链 ----
    std::cout << "\n[matmul+activation chain: sigmoid(X@W+B)]\n";
    const size_t mm_sizes[][3] = {
        {128, 512, 512}, {128, 1024, 1024}, {64, 2048, 2048},
        {64, 4096, 4096}, {32, 4096, 4096}, {16, 8192, 8192}
    };
    for (auto& s : mm_sizes) {
        size_t B=s[0], K=s[1], N=s[2];
        auto k = compileMatmulAct(B, K, N);
        if (!k) { std::cout << "  B,K,N=" << B << "," << K << "," << N << " compile FAILED\n"; continue; }
        Tensor X(ShapeTag{}, {B, K}); Tensor W(ShapeTag{}, {K, N}); Tensor Bb(ShapeTag{}, {N});
        fill(X, 0.3f); fill(W, 0.2f); fill(Bb, 0.1f);
        Tensor ref = eagerMatmulAct(X, W, Bb);
        auto eager_t = timeIt(20, [&]{ return eagerMatmulAct(X, W, Bb); });
        auto c3_t = timeIt(20, [&]{ auto r = k->execute({X, W, Bb}); return r.empty() ? ref : r[0]; });
        double speedup = eager_t / c3_t;
        std::cout << "  [" << std::setw(3) << B << "x" << std::setw(4) << K << "x" << std::setw(4) << N << "] eager="
                  << std::fixed << std::setprecision(1) << eager_t << "us  c3=" << c3_t
                  << "us  speedup=" << std::setprecision(2) << speedup << "x"
                  << (speedup > 1.0 ? "  ✅ C3更快" : "  ❌ eager更快") << "\n";
    }

    ct::c3::shutdownAll();
    std::cout.flush();
    std::_Exit(0);
}
