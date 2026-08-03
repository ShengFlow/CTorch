/**
 * @file bench_aot_speedup.cpp
 * @brief 量化 AOT cache 的冷启动加速效果
 * @details 测量同一图在"首次编译" vs "AOT 命中" 两种场景下的 wall-clock 时间
 *
 * 用法：
 *   1. 跑一次（cache miss）：记录首次编译时间
 *   2. 跑第二次（cache hit）：记录 AOT dlopen 时间
 *   3. 计算加速比
 */

#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>

#include "C3/AOTCache.h"
#include "C3/C3Engine.h"
#include "C3/Graph.h"
#include "Tensor.h"
#include "Ctools.h"

using namespace ct;
using namespace ct::c3;
using clk = std::chrono::high_resolution_clock;
using us = std::chrono::microseconds;

static Graph buildMatMulGraph(size_t M, size_t K, size_t N) {
    Graph g;
    auto lhs = TensorDesc::fromShape({M, K});
    auto rhs = TensorDesc::fromShape({K, N});
    auto out = TensorDesc::fromShape({M, N});
    size_t a = g.addInput(lhs);
    size_t b = g.addInput(rhs);
    size_t c = g.addNode(MatMulNode{lhs, rhs}, {a, b}, out);
    g.markOutput(c);
    return g;
}

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);
    auto& aot = AOTCache::getInstance();

    std::cout << "=== AOT Cache 冷启动加速 bench ===" << std::endl;
    std::cout << "AOT cache dir: " << aot.getCacheDir() << std::endl;
    std::cout << std::endl;

    // 测试 4 个图：从小到大
    std::vector<std::tuple<size_t, size_t, size_t, std::string>> cases = {
        {16, 16, 16, "tiny (16x16x16)"},
        {64, 64, 64, "small (64x64x64)"},
        {128, 128, 128, "medium (128x128x128)"},
        {256, 256, 256, "large (256x256x256)"},
    };

    std::cout << std::left
              << std::setw(25) << "Graph"
              << std::setw(15) << "miss (us)"
              << std::setw(15) << "hit (us)"
              << std::setw(15) << "speedup"
              << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    for (auto& [M, K, N, name] : cases) {
        C3Engine::getInstance().clearCache();
        aot.evict();  // 清空 AOT 确保首次 miss

        Graph g = buildMatMulGraph(M, K, N);
        CompileOptions opts;
        opts.backend = C3Backend::Handwritten;
        opts.opt_level = 3;

        // 测首次（miss）
        auto t0 = clk::now();
        try {
            auto k = C3Engine::getInstance().compile(g, opts);
            (void)k;
        } catch (const std::exception& e) {
            std::cout << "compile error: " << e.what() << std::endl;
            continue;
        }
        auto t1 = clk::now();
        auto miss_us = std::chrono::duration_cast<us>(t1 - t0).count();

        // 测二次（hit, in-memory cache 应该会命中）
        auto t2 = clk::now();
        try {
            auto k2 = C3Engine::getInstance().compile(g, opts);
            (void)k2;
        } catch (...) {}
        auto t3 = clk::now();
        auto hit_us_inmem = std::chrono::duration_cast<us>(t3 - t2).count();

        // 清空 in-memory，强制走 AOT path
        C3Engine::getInstance().clearCache();
        auto t4 = clk::now();
        try {
            auto k3 = C3Engine::getInstance().compile(g, opts);
            (void)k3;
        } catch (...) {}
        auto t5 = clk::now();
        auto hit_us_aot = std::chrono::duration_cast<us>(t5 - t4).count();

        double speedup_inmem = (double)miss_us / std::max<long long>(hit_us_inmem, 1);
        double speedup_aot = (double)miss_us / std::max<long long>(hit_us_aot, 1);

        std::cout << std::left
                  << std::setw(25) << name
                  << std::setw(15) << miss_us
                  << std::setw(15) << hit_us_aot  // 显示 AOT hit 时间
                  << std::setw(15) << std::to_string(speedup_aot).substr(0, 8) + "x"
                  << std::endl;
    }

    // 总结 stats
    auto stats = C3Engine::getInstance().getAOTCacheStats();
    std::cout << std::endl;
    std::cout << "AOT stats: "
              << "writes=" << stats.writes
              << " hits=" << stats.hits
              << " misses=" << stats.misses
              << " load_failures=" << stats.load_failures
              << std::endl;

    aot.evict();
    return 0;
}
