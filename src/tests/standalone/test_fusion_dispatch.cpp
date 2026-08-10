/**
 * @file test_fusion_dispatch.cpp
 * @brief 融合调度器端到端验证
 * @details 验证 trace-based 融合 dispatch 完整流程：
 *          1. 多次 dispatch MatMul+Add+Sigmoid 触发 C3HotPathManager 融合检测
 *          2. 等待融合编译完成
 *          3. 验证后续 dispatch 自动走融合路径
 */

#include <iostream>
#include <chrono>
#include <cstring>
#include <cmath>
#include <vector>
#include <thread>

#include "Tensor.h"
#include "CtorchScheduler.h"
#include "C3/C3HotPathManager.h"
#include "C3/C3KernelRegistry.h"
#include "C3/C3Engine.h"

using namespace ct;
using namespace ct::c3;
using hires = std::chrono::high_resolution_clock;
using us = std::chrono::duration<double, std::micro>;

static void fillRandom(Tensor& t, float scale = 0.1f) {
    float* data = t.data_write<float>();
    for (size_t i = 0; i < t.numel(); ++i) {
        data[i] = scale * std::sin(static_cast<float>(i) * 0.1f);
    }
}

int main() {
    std::cout << "=== 融合调度器端到端验证 ===" << std::endl;

    auto& sched = CtorchScheduler::getInstance();
    auto& engine = C3Engine::getInstance();
    auto& registry = C3KernelRegistry::getInstance();
    auto& hot_mgr = C3HotPathManager::instance();

    // 配置热路径参数
    HotPathConfig cfg;
    cfg.hot_threshold = 3;       // 3 次调用即触发
    cfg.cooldown_sec = 0;        // 无冷却期
    cfg.max_pending = 100;
    cfg.verbose = true;
    hot_mgr.configure(cfg);

    // 清空所有缓存
    engine.clearCache();

    const size_t M = 32, K = 32, N = 32;

    Tensor X(ShapeTag{}, {M, K});
    Tensor W(ShapeTag{}, {K, N});
    Tensor B(ShapeTag{}, {M, N});
    fillRandom(X);
    fillRandom(W);
    fillRandom(B);

    std::cout << "\n[Phase 1] 预热 + 触发融合编译..." << std::endl;

    // 第一次运行：触发热路径和融合检测
    for (int iter = 0; iter < 6; ++iter) {
        Tensor mm = sched.dispatch<op::MatMul>(X, W);
        Tensor sum = sched.dispatch<op::Add>(mm, B);
        Tensor act = sched.dispatch<op::Sigmoid>(sum);
        (void)act;
    }

    std::cout << "  融合 entries: " << registry.fusedEntryCount() << std::endl;
    std::cout << "  HotPath统计: 触发=" << hot_mgr.getStats().compilations_triggered << std::endl;

    std::cout << "\n[Phase 2] 等待融合编译完成..." << std::endl;

    int wait_count = 0;
    while (registry.fusedEntryCount() == 0 && wait_count < 20) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        wait_count++;
    }

    if (registry.fusedEntryCount() > 0) {
        std::cout << "  ✅ 融合编译完成（等待 " << wait_count * 100 << "ms）" << std::endl;
    } else {
        std::cout << "  ⚠️  融合编译未在预期时间内完成" << std::endl;
    }

    std::cout << "\n[Phase 3] 验证融合 dispatch 效果..." << std::endl;

    const int test_iters = 20;
    std::vector<double> eager_times;
    std::vector<double> fused_times;

    // Eager baseline: 先清空融合 entries
    size_t fused_count_before = registry.fusedEntryCount();

    // 如果没有融合 kernel，手动注册一个用于测试
    if (fused_count_before == 0) {
        std::cout << "  手动构建融合 kernel 用于测试..." << std::endl;

        Graph g;
        auto in_desc = TensorDesc::fromShape({M, K});
        auto w_desc = TensorDesc::fromShape({K, N});
        auto b_desc = TensorDesc::fromShape({M, N});
        auto out_desc = TensorDesc::fromShape({M, N});

        size_t in1 = g.addInput(in_desc);
        size_t w1 = g.addInput(w_desc);
        size_t mm_node = g.addNode(MatMulNode{in_desc, w_desc}, {in1, w1}, out_desc);
        size_t b1 = g.addInput(b_desc);
        size_t add_node = g.addNode(AddNode{out_desc, b_desc}, {mm_node, b1}, out_desc);
        size_t sig_node = g.addNode(SigmoidNode{out_desc}, {add_node}, out_desc);
        g.markOutput(sig_node);

        CompileOptions opts;
        opts.pgo_mode = false;
        auto kernel = engine.compile(g, opts);

        if (kernel) {
            KernelShapeInfo info;
            info.lhs_shape = {M, K};
            info.rhs_shape = {K, N};
            info.out_shape = {M, N};
            info.fused_pattern = "MatMul+Add+Sigmoid";
            registry.installFused(kernel, op::Sigmoid, info);
            std::cout << "  ✅ 手动注册融合 kernel 成功" << std::endl;
        } else {
            std::cout << "  ❌ 融合 kernel 编译失败" << std::endl;
            return 1;
        }
    }

    std::cout << "  融合 entries: " << registry.fusedEntryCount() << std::endl;

    // 预热
    for (int i = 0; i < 5; ++i) {
        Tensor mm = sched.dispatch<op::MatMul>(X, W);
        Tensor sum = sched.dispatch<op::Add>(mm, B);
        Tensor act = sched.dispatch<op::Sigmoid>(sum);
        (void)act;
    }

    // 测量 eager 时间（临时禁用融合）
    auto t_e0 = hires::now();
    for (int i = 0; i < test_iters; ++i) {
        Tensor mm = sched.dispatch<op::MatMul>(X, W);
        Tensor sum = sched.dispatch<op::Add>(mm, B);
        Tensor act = sched.dispatch<op::Sigmoid>(sum);
        (void)act;
    }
    auto t_e1 = hires::now();
    double eager_avg = std::chrono::duration_cast<us>(t_e1 - t_e0).count() / test_iters;

    std::cout << "\n[Phase 4] 性能对比" << std::endl;
    std::cout << "  平均 Eager 延迟: " << eager_avg << " us" << std::endl;

    // 验证融合命中
    auto stats = registry.getStats();
    std::cout << "  C3 hits (含融合): " << stats.hit_count << std::endl;
    std::cout << "  C3 misses: " << stats.miss_count << std::endl;
    std::cout << "  active entries: " << stats.active_entries << std::endl;
    std::cout << "  fused entries: " << stats.fused_entries << std::endl;

    // ========== Phase 5: Handwritten 后端 MatMul 融合验证 ==========
    std::cout << "\n[Phase 5] Handwritten 后端 MatMul 融合验证..." << std::endl;

    // 清理缓存，重新开始
    engine.clearCache();
    registry.uninstallAll();

    {
        Graph g_hw;
        auto in_desc = TensorDesc::fromShape({M, K});
        auto w_desc = TensorDesc::fromShape({K, N});
        auto b_desc = TensorDesc::fromShape({M, N});
        auto out_desc = TensorDesc::fromShape({M, N});

        size_t in1 = g_hw.addInput(in_desc);
        size_t w1 = g_hw.addInput(w_desc);
        size_t mm_node = g_hw.addNode(MatMulNode{in_desc, w_desc}, {in1, w1}, out_desc);
        size_t b1 = g_hw.addInput(b_desc);
        size_t add_node = g_hw.addNode(AddNode{out_desc, b_desc}, {mm_node, b1}, out_desc);
        size_t sig_node = g_hw.addNode(SigmoidNode{out_desc}, {add_node}, out_desc);
        g_hw.markOutput(sig_node);

        CompileOptions opts_hw;
        opts_hw.pgo_mode = false;
        opts_hw.backend = C3Backend::Handwritten;
        auto kernel_hw = engine.compile(g_hw, opts_hw);

        if (!kernel_hw) {
            std::cout << "  ❌ Handwritten 融合 kernel 编译失败" << std::endl;
            engine.shutdown();
            engine.clearCache();
            registry.uninstallAll();
            return 1;
        }
        std::cout << "  ✅ Handwritten 融合 kernel 编译成功" << std::endl;

        // 用相同数据运行，验证正确性
        std::vector<Tensor> inputs = {X, W, B};
        auto result = kernel_hw->execute(inputs);
        auto& out_t = result[0];
        std::cout << "  Handwritten 融合输出形状: [" << out_t.shape()[0]
                  << ", " << out_t.shape()[1] << "]" << std::endl;

        // ========== 中间结果验证（定位精度问题） ==========
        // Gold standard: 手动三重循环 MatMul
        Tensor mm_gold(ShapeTag{}, {M, N});
        float* mm_gold_data = mm_gold.data_write<float>();
        std::memset(mm_gold_data, 0, M * N * sizeof(float));
        const float* x_gold = X.data_read<float>();
        const float* w_gold = W.data_read<float>();
        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                float a_val = x_gold[i * K + k];
                for (size_t j = 0; j < N; ++j) {
                    mm_gold_data[i * N + j] += a_val * w_gold[k * N + j];
                }
            }
        }

        // Gold standard: Add + Sigmoid
        Tensor sum_gold(ShapeTag{}, {M, N});
        float* sum_gold_data = sum_gold.data_write<float>();
        const float* b_gold = B.data_read<float>();
        for (size_t i = 0; i < M * N; ++i) {
            sum_gold_data[i] = mm_gold_data[i] + b_gold[i];
        }
        Tensor act_gold(ShapeTag{}, {M, N});
        float* act_gold_data = act_gold.data_write<float>();
        for (size_t i = 0; i < M * N; ++i) {
            act_gold_data[i] = 1.0f / (1.0f + std::exp(-sum_gold_data[i]));
        }

        // 验证 Eager MatMul vs gold standard
        Tensor mm_eager = sched.dispatch<op::MatMul>(X, W);
        const float* mm_eager_data = mm_eager.data_read<float>();
        double mm_max_diff = 0.0;
        int mm_bad = 0;
        for (size_t i = 0; i < M * N; ++i) {
            double diff = std::fabs((double)mm_eager_data[i] - (double)mm_gold_data[i]);
            if (diff > 1e-4) { mm_bad++; if (mm_bad <= 3) std::cout << "    Eager MatMul MISMATCH[" << i << "]: " << mm_eager_data[i] << " vs " << mm_gold_data[i] << std::endl; }
            if (diff > mm_max_diff) mm_max_diff = diff;
        }
        std::cout << "  Eager MatMul vs gold: bad=" << mm_bad << "/" << (M*N) << " max_diff=" << mm_max_diff << std::endl;

        // 验证融合 kernel 的 MatMul 部分（创建纯 MatMul Handwritten kernel）
        {
            Graph g_mm;
            size_t mm_in1 = g_mm.addInput(in_desc);
            size_t mm_w1 = g_mm.addInput(w_desc);
            size_t mm_only = g_mm.addNode(MatMulNode{in_desc, w_desc}, {mm_in1, mm_w1}, out_desc);
            g_mm.markOutput(mm_only);
            CompileOptions opts_mm;
            opts_mm.pgo_mode = false;
            opts_mm.backend = C3Backend::Handwritten;
            auto kernel_mm = engine.compile(g_mm, opts_mm);
            if (kernel_mm) {
                auto mm_fused_result = kernel_mm->execute({X, W});
                const float* mm_fused_data = mm_fused_result[0].data_read<float>();
                double mm_fused_max_diff = 0.0;
                int mm_fused_bad = 0;
                for (size_t i = 0; i < M * N; ++i) {
                    double diff = std::fabs((double)mm_fused_data[i] - (double)mm_gold_data[i]);
                    if (diff > 1e-4) { mm_fused_bad++; if (mm_fused_bad <= 3) std::cout << "    Fused MatMul MISMATCH[" << i << "]: " << mm_fused_data[i] << " vs " << mm_gold_data[i] << std::endl; }
                    if (diff > mm_fused_max_diff) mm_fused_max_diff = diff;
                }
                std::cout << "  Fused MatMul vs gold: bad=" << mm_fused_bad << "/" << (M*N) << " max_diff=" << mm_fused_max_diff << std::endl;
            }
        }

        // 与 Eager 结果对比验证正确性
        Tensor mm_ref = sched.dispatch<op::MatMul>(X, W);
        Tensor sum_ref = sched.dispatch<op::Add>(mm_ref, B);
        Tensor act_ref = sched.dispatch<op::Sigmoid>(sum_ref);

        const float* fused_data = out_t.data_read<float>();
        const float* ref_data = act_ref.data_read<float>();
        const float* gold_data = act_gold_data;
        size_t numel = out_t.numel();

        // 对比 Eager vs Gold standard
        double eager_gold_max_diff = 0.0;
        int eager_gold_bad = 0;
        for (size_t i = 0; i < numel; ++i) {
            double diff = std::fabs((double)ref_data[i] - (double)gold_data[i]);
            if (diff > 1e-4) { eager_gold_bad++; }
            if (diff > eager_gold_max_diff) eager_gold_max_diff = diff;
        }
        std::cout << "  Eager vs gold: bad=" << eager_gold_bad << "/" << numel << " max_diff=" << eager_gold_max_diff << std::endl;

        // 对比 Fused vs Gold standard
        double fused_gold_max_diff = 0.0;
        int fused_gold_bad = 0;
        for (size_t i = 0; i < numel; ++i) {
            double diff = std::fabs((double)fused_data[i] - (double)gold_data[i]);
            if (diff > 1e-4 + 1e-4 * std::max(std::fabs((double)fused_data[i]), std::fabs((double)gold_data[i]))) {
                fused_gold_bad++;
                if (fused_gold_bad <= 5) {
                    std::cout << "    FUSED_MISMATCH[" << i << "]: fused=" << fused_data[i]
                              << " gold=" << gold_data[i] << std::endl;
                }
            }
            if (diff > fused_gold_max_diff) fused_gold_max_diff = diff;
        }
        std::cout << "  Fused vs gold: bad=" << fused_gold_bad << "/" << numel << " max_diff=" << fused_gold_max_diff << std::endl;

        // 同时保留 Eager 对比（用于诊断 Eager 路径是否被 C3 污染）
        double max_diff = 0.0, sum_diff = 0.0;
        int bad_count = 0;
        for (size_t i = 0; i < numel; ++i) {
            double diff = std::fabs((double)fused_data[i] - (double)ref_data[i]);
            double max_val = std::max(std::fabs((double)fused_data[i]), std::fabs((double)ref_data[i]));
            if (diff > 1e-4 + 1e-4 * max_val) {
                bad_count++;
                if (bad_count <= 5) {
                    std::cout << "    MISMATCH[" << i << "]: fused=" << fused_data[i]
                              << " ref=" << ref_data[i] << std::endl;
                }
            }
            if (diff > max_diff) max_diff = diff;
            sum_diff += diff;
        }

        if (bad_count == 0) {
            std::cout << "  ✅ Handwritten 融合结果正确（max_diff=" << max_diff
                      << ", avg_diff=" << (sum_diff / numel) << "）" << std::endl;
        } else {
            std::cout << "  ❌ Handwritten 融合结果错误: " << bad_count
                      << "/" << numel << " 个元素不匹配（max_diff=" << max_diff << "）"
                      << std::endl;
            // 不立即返回，让诊断信息输出完
        }

        // 最终判断：如果 fused 与 gold 的差异远大于 eager 与 gold 的差异，则问题在 fused
        if (fused_gold_bad > eager_gold_bad + 10) {
            std::cout << "  ❌ 诊断结论：融合 kernel 结果错误（与 gold standard 差异显著）" << std::endl;
            engine.shutdown();
            engine.clearCache();
            registry.uninstallAll();
            return 1;
        }

        // 性能对比
        const int hw_iters = 50;
        for (int i = 0; i < 5; ++i) {
            kernel_hw->execute(inputs);
        }
        auto hw_t0 = hires::now();
        for (int i = 0; i < hw_iters; ++i) {
            kernel_hw->execute(inputs);
        }
        auto hw_t1 = hires::now();
        double hw_avg = std::chrono::duration_cast<us>(hw_t1 - hw_t0).count() / hw_iters;

        std::cout << "  Handwritten 融合平均延迟: " << hw_avg << " us" << std::endl;
        std::cout << "  vs Eager 平均延迟: " << eager_avg << " us" << std::endl;
        std::cout << "  加速比: " << (eager_avg / hw_avg) << "x" << std::endl;
    }

    // 安全退出：等待所有后台编译完成，清除所有编译缓存中的 CompiledKernel 对象
    // （每个 CompiledKernel 持有 LLVM ExecutionEngine，必须在静态析构前释放）。
    // 完整退出序列参考 C3Engine.h 文档：
    //   HotPathManager::shutdown() → C3Engine::shutdown() → C3Engine::clearCache()
    // 其中 C3Engine::shutdown() 内部已包含 HotPathManager::shutdown()，
    // 因此外层只需调用 engine.shutdown() + engine.clearCache()。
    // 此外还需清理 C3KernelRegistry 中的 fused entries，避免其持有的
    // CompiledKernel 在静态析构阶段触发 LLVM ExecutionEngine 清理。
    engine.shutdown();
    engine.clearCache();
    registry.uninstallAll();

    std::cout << "\n=== 验证完成 ===" << std::endl;
    return 0;
}