/**
 * @file bench_subgraph_roi.cpp
 * @brief 子图合并 ROI 验证 benchmark
 * @details 对比 4 种 MLP 执行方案的延迟：
 *          1. Baseline：逐层 Eager dispatch
 *          2. Per-Layer Compile：每层独立 compile + installIntoRegistry
 *          3. Merged：compileMergedSequential（全图融合）
 *          4. MergedPGO：compileMergedPGOSequential（全图融合 + PGO 异步升级）
 *
 *          测试模型：[1024, 512] -> [512, 256] -> [256, 10] 3 层 MLP
 *          每层：MatMul + Add + ReLU
 *
 *          测量：
 *          - 编译时间（一次性）
 *          - 执行延迟：P50 / P95 / P99 / mean / stddev
 *          - 100 次迭代统计（warm-up 10 次后）
 *
 *          用法：
 *          ```
 *          ./build/bench_subgraph_roi [--iterations N] [--warmup N] [--shape M,K1,K2,K3,N]
 *          ```
 *
 * @date 2026/8/3
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>

#include "Tensor.h"
#include "CtorchScheduler.h"
#include "C3/Graph.h"
#include "C3/GraphMerger.h"
#include "C3/C3Engine.h"
#include "C3/PGOManager.h"

using namespace ct;
using namespace ct::c3;
using hires = std::chrono::high_resolution_clock;
using us = std::chrono::duration<double, std::micro>;

// ======================= 工具：构建子图 =======================

static Graph buildFCReLU(const std::vector<size_t>& in_shape,
                          const std::vector<size_t>& w_shape,
                          const std::vector<size_t>& b_shape,
                          const std::vector<size_t>& out_shape,
                          bool add_relu = true) {
    auto in_desc = TensorDesc::fromShape(in_shape);
    auto w_desc = TensorDesc::fromShape(w_shape);
    auto b_desc = TensorDesc::fromShape(b_shape);
    auto out_desc = TensorDesc::fromShape(out_shape);

    Graph g;
    size_t in = g.addInput(in_desc);
    size_t w = g.addInput(w_desc);
    size_t b = g.addInput(b_desc);
    size_t mm = g.addNode(MatMulNode{in_desc, w_desc}, {in, w}, out_desc);
    size_t add = g.addNode(AddNode{out_desc, b_desc}, {mm, b}, out_desc);
    size_t out_id = add;
    if (add_relu) {
        out_id = g.addNode(ReLUNode{out_desc}, {add}, out_desc);
    }
    g.markOutput(out_id);
    return g;
}

static void fillRandom(Tensor& t, float scale = 0.1f) {
    float* data = t.data_write<float>();
    for (size_t i = 0; i < t.numel(); ++i) {
        // 简单伪随机（确定性，便于复现）
        data[i] = scale * std::sin(static_cast<float>(i) * 0.1f);
    }
}

static void fillConst(Tensor& t, float v) {
    float* data = t.data_write<float>();
    std::memset(data, 0, t.numel() * sizeof(float));
    for (size_t i = 0; i < t.numel(); ++i) data[i] = v;
}

// ======================= 工具：延迟统计 =======================

struct LatencyStats {
    double mean_us = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    double p99_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
    double stddev_us = 0.0;
    size_t n = 0;
};

static LatencyStats computeStats(std::vector<double>& samples_us) {
    std::sort(samples_us.begin(), samples_us.end());
    LatencyStats s;
    s.n = samples_us.size();
    if (s.n == 0) return s;
    s.min_us = samples_us.front();
    s.max_us = samples_us.back();
    s.p50_us = samples_us[s.n * 50 / 100];
    s.p95_us = samples_us[s.n * 95 / 100];
    s.p99_us = samples_us[s.n * 99 / 100];
    double sum = std::accumulate(samples_us.begin(), samples_us.end(), 0.0);
    s.mean_us = sum / static_cast<double>(s.n);
    double sq_sum = 0.0;
    for (double v : samples_us) {
        double d = v - s.mean_us;
        sq_sum += d * d;
    }
    s.stddev_us = std::sqrt(sq_sum / static_cast<double>(s.n));
    return s;
}

static void printStats(const std::string& name, const LatencyStats& s,
                        double compile_ms = -1.0) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  " << std::left << std::setw(20) << name
              << "  P50=" << std::setw(9) << s.p50_us
              << "  P95=" << std::setw(9) << s.p95_us
              << "  P99=" << std::setw(9) << s.p99_us
              << "  mean=" << std::setw(9) << s.mean_us
              << "  std=" << std::setw(7) << s.stddev_us
              << "  us";
    if (compile_ms >= 0.0) {
        std::cout << "  (compile=" << std::setprecision(1) << compile_ms << "ms)";
    }
    std::cout << std::endl;
}

// ======================= Benchmark 方案 =======================

struct BenchResult {
    std::string name;
    double compile_ms = 0.0;
    LatencyStats stats;
};

// 方案 0：逐层 Eager dispatch（baseline）
static BenchResult benchEagerPerLayer(
    const std::vector<Graph>& subs,
    const std::vector<Tensor>& inputs,
    int warmup, int iterations) {

    auto& sched = CtorchScheduler::getInstance();
    BenchResult r;
    r.name = "Eager-PerLayer";

    // warm-up
    for (int i = 0; i < warmup; ++i) {
        Tensor prev = inputs[0];
        for (size_t li = 0; li < subs.size(); ++li) {
            Tensor w = inputs[1 + 2 * li];
            Tensor b = inputs[2 + 2 * li];
            Tensor mm = sched.dispatch<op::MatMul>(prev, w);
            Tensor sum = sched.dispatch<op::Add>(mm, b);
            bool has_relu = false;
            for (const auto& node : subs[li].nodes()) {
                if (std::holds_alternative<ReLUNode>(node.op)) { has_relu = true; break; }
            }
            prev = has_relu ? sched.dispatch<op::ReLU>(sum) : sum;
        }
    }

    std::vector<double> samples;
    samples.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        auto t0 = hires::now();
        Tensor prev = inputs[0];
        for (size_t li = 0; li < subs.size(); ++li) {
            Tensor w = inputs[1 + 2 * li];
            Tensor b = inputs[2 + 2 * li];
            Tensor mm = sched.dispatch<op::MatMul>(prev, w);
            Tensor sum = sched.dispatch<op::Add>(mm, b);
            bool has_relu = false;
            for (const auto& node : subs[li].nodes()) {
                if (std::holds_alternative<ReLUNode>(node.op)) { has_relu = true; break; }
            }
            prev = has_relu ? sched.dispatch<op::ReLU>(sum) : sum;
        }
        auto t1 = hires::now();
        samples.push_back(std::chrono::duration_cast<us>(t1 - t0).count());
    }
    r.stats = computeStats(samples);
    return r;
}

// 方案 1：每层独立 compile + installIntoRegistry
static BenchResult benchPerLayerCompile(
    const std::vector<Graph>& subs,
    const std::vector<Tensor>& inputs,
    int warmup, int iterations) {

    auto& engine = C3Engine::getInstance();
    BenchResult r;
    r.name = "PerLayer-Compile";

    // 编译
    auto t_c0 = hires::now();
    CompileOptions opts;
    opts.pgo_mode = false;
    std::vector<std::shared_ptr<CompiledKernel>> kernels;
    for (const auto& g : subs) {
        kernels.push_back(engine.compile(g, opts));
    }
    auto t_c1 = hires::now();
    r.compile_ms = std::chrono::duration_cast<us>(t_c1 - t_c0).count() / 1000.0;

    // warm-up
    for (int i = 0; i < warmup; ++i) {
        Tensor prev = inputs[0];
        for (size_t li = 0; li < subs.size(); ++li) {
            std::vector<Tensor> layer_inputs = {prev, inputs[1 + 2 * li], inputs[2 + 2 * li]};
            prev = kernels[li]->execute(layer_inputs)[0];
        }
    }

    std::vector<double> samples;
    samples.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        auto t0 = hires::now();
        Tensor prev = inputs[0];
        for (size_t li = 0; li < subs.size(); ++li) {
            std::vector<Tensor> layer_inputs = {prev, inputs[1 + 2 * li], inputs[2 + 2 * li]};
            prev = kernels[li]->execute(layer_inputs)[0];
        }
        auto t1 = hires::now();
        samples.push_back(std::chrono::duration_cast<us>(t1 - t0).count());
    }
    r.stats = computeStats(samples);
    return r;
}

// 方案 2：compileMergedSequential
static BenchResult benchMergedSequential(
    const std::vector<Graph>& subs,
    const std::vector<Tensor>& inputs,
    int warmup, int iterations) {

    auto& engine = C3Engine::getInstance();
    BenchResult r;
    r.name = "Merged-Seq";

    // 编译
    auto t_c0 = hires::now();
    auto kernel = engine.compileMergedSequential(subs, {});
    auto t_c1 = hires::now();
    r.compile_ms = std::chrono::duration_cast<us>(t_c1 - t_c0).count() / 1000.0;

    if (!kernel) {
        std::cerr << "  ERR: compileMergedSequential 返回 nullptr\n";
        return r;
    }

    // warm-up
    for (int i = 0; i < warmup; ++i) {
        (void)kernel->execute(inputs);
    }

    std::vector<double> samples;
    samples.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        auto t0 = hires::now();
        (void)kernel->execute(inputs);
        auto t1 = hires::now();
        samples.push_back(std::chrono::duration_cast<us>(t1 - t0).count());
    }
    r.stats = computeStats(samples);
    return r;
}

// 方案 3：compileMergedPGOSequential（PGO 包装）
static BenchResult benchMergedPGO(
    const std::vector<Graph>& subs,
    const std::vector<Tensor>& inputs,
    int warmup, int iterations) {

    auto& engine = C3Engine::getInstance();
    BenchResult r;
    r.name = "Merged-PGO";

    // 编译
    auto t_c0 = hires::now();
    auto kernel = engine.compileMergedPGOSequential(subs, {});
    auto t_c1 = hires::now();
    r.compile_ms = std::chrono::duration_cast<us>(t_c1 - t_c0).count() / 1000.0;

    if (!kernel) {
        std::cerr << "  ERR: compileMergedPGOSequential 返回 nullptr\n";
        return r;
    }

    // PGO 第一次执行是 Eager 解释，需要 warm-up 让 PGO 升级
    // warm-up 阶段多跑一些，确保 O2/Ofast 升级完成
    int pgo_warmup = std::max(warmup, 50);
    for (int i = 0; i < pgo_warmup; ++i) {
        (void)kernel->execute(inputs);
    }
    // 强制 promote + 等待
    if (auto* pgo = dynamic_cast<PGOCompiledKernel*>(kernel.get())) {
        pgo->promote();
    }

    std::vector<double> samples;
    samples.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        auto t0 = hires::now();
        (void)kernel->execute(inputs);
        auto t1 = hires::now();
        samples.push_back(std::chrono::duration_cast<us>(t1 - t0).count());
    }
    r.stats = computeStats(samples);
    return r;
}

// ======================= 主流程 =======================

static void print_usage(const char* argv0) {
    std::cout << "Usage: " << argv0
              << " [--iterations N] [--warmup N] [--shape M,K1,K2,K3,N]\n"
              << "  --iterations N  测量次数（默认 100）\n"
              << "  --warmup N      warm-up 次数（默认 10）\n"
              << "  --shape M,K1,K2,K3,N  MLP shape（默认 1024,512,256,128,10）\n";
}

int main(int argc, char** argv) {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);
    CtorchScheduler::getInstance();
    C3Engine::getInstance().clearCache();

    int iterations = 100;
    int warmup = 10;
    std::vector<size_t> shape = {1024, 512, 256, 128, 10};  // M, K1, K2, K3, N

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::atoi(argv[++i]);
        } else if (arg == "--shape" && i + 1 < argc) {
            shape.clear();
            std::string s = argv[++i];
            size_t pos = 0;
            while ((pos = s.find(',')) != std::string::npos) {
                shape.push_back(std::stoul(s.substr(0, pos)));
                s.erase(0, pos + 1);
            }
            shape.push_back(std::stoul(s));
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (shape.size() != 5) {
        std::cerr << "ERR: --shape 必须 5 个值（M,K1,K2,K3,N）\n";
        return 1;
    }
    size_t M = shape[0], K1 = shape[1], K2 = shape[2], K3 = shape[3], N = shape[4];

    std::cout << "=== C3 子图合并 ROI Benchmark ===\n"
              << "Model: MLP [" << M << ", " << K1 << ", " << K2
              << ", " << K3 << ", " << N << "]\n"
              << "Layers: 3 × (MatMul + Add + ReLU)\n"
              << "Iterations: " << iterations << " (warm-up: " << warmup << ")\n"
              << std::endl;

    // 构造输入（每个 layer 的输入 shape 不同，逐层构造）
    // 实际上 inputs 顺序：[x, w1, b1, w2, b2, w3, b3]
    std::vector<Tensor> inputs;
    Tensor x(ShapeTag{}, {M, K1});
    fillRandom(x, 0.5f);
    inputs.push_back(x);

    // Layer 1: in [M,K1] @ w [K1,K2] + b [K2] -> [M,K2]
    Tensor w1(ShapeTag{}, {K1, K2});
    Tensor b1(ShapeTag{}, {K2});
    fillRandom(w1, 0.05f);
    fillConst(b1, 0.0f);
    inputs.push_back(w1);
    inputs.push_back(b1);

    // Layer 2: in [M,K2] @ w [K2,K3] + b [K3] -> [M,K3]
    Tensor w2(ShapeTag{}, {K2, K3});
    Tensor b2(ShapeTag{}, {K3});
    fillRandom(w2, 0.05f);
    fillConst(b2, 0.0f);
    inputs.push_back(w2);
    inputs.push_back(b2);

    // Layer 3: in [M,K3] @ w [K3,N] + b [N] -> [M,N]
    Tensor w3(ShapeTag{}, {K3, N});
    Tensor b3(ShapeTag{}, {N});
    fillRandom(w3, 0.05f);
    fillConst(b3, 0.0f);
    inputs.push_back(w3);
    inputs.push_back(b3);

    // 构造 3 层子图
    std::vector<Graph> subs = {
        buildFCReLU({M, K1}, {K1, K2}, {K2}, {M, K2}, true),
        buildFCReLU({M, K2}, {K2, K3}, {K3}, {M, K3}, true),
        buildFCReLU({M, K3}, {K3, N}, {N}, {M, N}, true),
    };

    std::vector<BenchResult> results;

    // 方案 0：Eager
    std::cout << "[1/4] 跑 Eager-PerLayer（baseline）..." << std::endl;
    results.push_back(benchEagerPerLayer(subs, inputs, warmup, iterations));
    printStats(results.back().name, results.back().stats);

    // 方案 1：每层独立 compile
    std::cout << "[2/4] 跑 PerLayer-Compile..." << std::endl;
    results.push_back(benchPerLayerCompile(subs, inputs, warmup, iterations));
    printStats(results.back().name, results.back().stats, results.back().compile_ms);

    // 方案 2：compileMergedSequential
    std::cout << "[3/4] 跑 Merged-Seq（compileMergedSequential）..." << std::endl;
    results.push_back(benchMergedSequential(subs, inputs, warmup, iterations));
    printStats(results.back().name, results.back().stats, results.back().compile_ms);

    // 方案 3：compileMergedPGOSequential
    std::cout << "[4/4] 跑 Merged-PGO（compileMergedPGOSequential）..." << std::endl;
    results.push_back(benchMergedPGO(subs, inputs, warmup, iterations));
    printStats(results.back().name, results.back().stats, results.back().compile_ms);

    // ======================= 汇总 =======================
    std::cout << "\n=== 汇总 ===" << std::endl;
    std::cout << std::left << std::setw(20) << "方案"
              << std::right << std::setw(12) << "P50(us)"
              << std::setw(12) << "P95(us)"
              << std::setw(12) << "P99(us)"
              << std::setw(12) << "mean(us)"
              << std::setw(12) << "speedup"
              << std::setw(12) << "compile(ms)"
              << std::endl;
    std::cout << std::string(92, '-') << std::endl;

    double base_p50 = results[0].stats.p50_us;
    for (const auto& r : results) {
        double speedup = base_p50 / r.stats.p50_us;
        std::cout << std::left << std::setw(20) << r.name
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(12) << r.stats.p50_us
                  << std::setw(12) << r.stats.p95_us
                  << std::setw(12) << r.stats.p99_us
                  << std::setw(12) << r.stats.mean_us
                  << std::setw(12) << speedup
                  << std::setw(12) << r.compile_ms
                  << std::endl;
    }

    std::cout << "\n=== 关键洞见 ===" << std::endl;
    if (results[2].stats.p50_us > results[1].stats.p50_us * 1.2) {
        std::cout << "🔴 子图合并 (Merged-Seq) 相比 PerLayer-Compile 慢 " << std::setprecision(1)
                  << (100.0 * (results[2].stats.p50_us - results[1].stats.p50_us) / results[1].stats.p50_us)
                  << "%，验证反直觉假设（cache miss ROI 负）。\n";
    } else if (results[2].stats.p50_us < results[1].stats.p50_us * 0.8) {
        std::cout << "🟢 子图合并 (Merged-Seq) 相比 PerLayer-Compile 快 "
                  << std::setprecision(1)
                  << (100.0 * (results[1].stats.p50_us - results[2].stats.p50_us) / results[1].stats.p50_us)
                  << "%，证伪反直觉假设（kernel launch overhead 主导）。\n";
    } else {
        std::cout << "🟡 差异 < 20%，结果不确定。建议扩大样本或换 shape 重测。\n";
    }
    if (results[3].stats.p50_us > 0 && results[2].stats.p50_us > 0) {
        std::cout << "   PGO 包装 vs compileMerged P50: "
                  << (results[3].stats.p50_us < results[2].stats.p50_us ? "PGO 更快" : "PGO 持平/更慢")
                  << std::endl;
    }

    // 清理
    C3Engine::getInstance().shutdown();
    C3Engine::getInstance().clearCache();
    return 0;
}
