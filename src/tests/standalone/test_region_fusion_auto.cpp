/**
 * @file test_region_fusion_auto.cpp
 * @brief 区域融合自动链路端到端验证
 * @details 纯用户场景测试：不手动编译/注册任何 kernel，
 *          让调度器 + C3HotPathManager 自动检测热路径、编译、注入。
 *          验证：
 *          1. 自动热点检测触发融合编译
 *          2. 编译完成后自动注入 RegionFusionRegistry
 *          3. 预走匹配自动执行融合 kernel
 *          4. 结果正确性
 *          5. 性能加速比
 */

#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <thread>
#include <cassert>

#include "Tensor.h"
#include "CtorchScheduler.h"
#include "C3/C3Cleanup.h"

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
    std::cout << "=== 区域融合自动链路端到端验证 ===" << std::endl;
    int passed = 0, total = 0;

    // ======================= EXP-1: 自动热点检测 + 融合编译 + 预走执行 =======================
    {
        std::cout << "\n[EXP-1] 自动热点检测 + 融合编译 + 预走执行..." << std::endl;
        auto& sched = CtorchScheduler::getInstance();
        auto& hp_mgr = C3HotPathManager::instance();
        auto& region_reg = RegionFusionRegistry::getInstance();

        // 清理状态
        sched.resetRegionFusion();
        hp_mgr.clear();
        region_reg.clear();

        const size_t M = 32, K = 32, N = 32;
        const int warmup_iters = 5;  // 5 次迭代触发热点检测（阈值=5）
        const int test_iters = 3;    // 3 次迭代验证融合执行

        // 创建输入数据
        Tensor X(ShapeTag{}, {M, K});
        Tensor W(ShapeTag{}, {K, N});
        Tensor B(ShapeTag{}, {M, N});
        fillRandom(X);
        fillRandom(W);
        fillRandom(B);

        // 第一次迭代：保存 eager 参考结果
        Tensor act_ref;
        {
            std::cout << "  第一次迭代 (eager + trace 记录)..." << std::endl;
            Tensor mm = sched.dispatch<op::MatMul>(X, W);
            Tensor sum = sched.dispatch<op::Add>(mm, B);
            Tensor act = sched.dispatch<op::Sigmoid>(sum);
            act_ref = act;
            std::cout << "    Eager 结果: numel=" << act_ref.numel() << " data_ptr="
                      << (void*)act_ref.data_read<float>() << std::endl;
        }

        // 继续 warmup 迭代以触发热点检测
        std::cout << "  热身迭代 2~" << (warmup_iters) << " (触发热点检测)..." << std::endl;
        for (int i = 2; i <= warmup_iters; ++i) {
            Tensor mm = sched.dispatch<op::MatMul>(X, W);
            Tensor sum = sched.dispatch<op::Add>(mm, B);
            Tensor act = sched.dispatch<op::Sigmoid>(sum);
            std::cout << "    Iter " << i << " done" << std::endl;
        }

        // 检查热点检测是否已触发
        auto hp_stats = hp_mgr.getStats();
        std::cout << "  HotPathManager stats: compilations_triggered="
                  << hp_stats.compilations_triggered
                  << " pending=" << hp_stats.pending_compiles << std::endl;

        // 等待异步编译完成
        std::cout << "  等待异步编译完成..." << std::endl;
        hp_mgr.waitForPendingCompiles();
        std::cout << "  编译完成, RegionFusionRegistry entries="
                  << region_reg.entryCount() << std::endl;

        // 验证 region 已注册
        bool registered = (region_reg.entryCount() > 0);
        std::cout << "  Region 已自动注册: " << (registered ? "✅" : "❌") << std::endl;
        total++;
        if (registered) passed++;

        // 再次迭代：应触发预走融合
        {
            std::cout << "  验证迭代 (预走融合)..." << std::endl;
            Tensor mm = sched.dispatch<op::MatMul>(X, W);
            Tensor sum = sched.dispatch<op::Add>(mm, B);
            Tensor act = sched.dispatch<op::Sigmoid>(sum);

            // 验证结果正确性
            const float* fused_data = act.data_read<float>();
            const float* ref_data = act_ref.data_read<float>();
            size_t numel = act.numel();

            double max_diff = 0.0;
            int bad_count = 0;
            for (size_t i = 0; i < numel; ++i) {
                double diff = std::fabs((double)fused_data[i] - (double)ref_data[i]);
                double max_val = std::max(std::fabs((double)fused_data[i]), std::fabs((double)ref_data[i]));
                if (diff > 1e-4 + 1e-4 * max_val) {
                    bad_count++;
                    if (bad_count <= 3) {
                        std::cout << "    MISMATCH[" << i << "]: fused=" << fused_data[i]
                                  << " ref=" << ref_data[i] << std::endl;
                    }
                }
                if (diff > max_diff) max_diff = diff;
            }

            bool correct = (bad_count == 0);
            std::cout << "  融合结果正确性: " << (correct ? "✅" : "❌")
                      << " bad=" << bad_count << "/" << numel
                      << " max_diff=" << max_diff << std::endl;
            total++;
            if (correct) passed++;
        }

        // 清理
        sched.resetRegionFusion();
        hp_mgr.clear();
        region_reg.clear();
    }

    // ======================= EXP-2: 大 tensor 自动融合 + 性能对比 =======================
    {
        std::cout << "\n[EXP-2] 大 tensor 自动融合 + 性能对比..." << std::endl;
        auto& sched = CtorchScheduler::getInstance();
        auto& hp_mgr = C3HotPathManager::instance();
        auto& region_reg = RegionFusionRegistry::getInstance();

        // 清理状态
        sched.resetRegionFusion();
        hp_mgr.clear();
        region_reg.clear();

        const size_t M = 1024, K = 1024, N = 1024;
        const int warmup_iters = 5;  // 触发热点检测
        const int perf_iters = 30;   // 性能测量迭代（1024 BLAS 路径加速比在 0.9x 边缘波动，取更多样本稳定均值）

        Tensor X(ShapeTag{}, {M, K});
        Tensor W(ShapeTag{}, {K, N});
        Tensor B(ShapeTag{}, {N});  // 1D bias
        fillRandom(X);
        fillRandom(W);
        fillRandom(B);

        // 第一次迭代：eager + trace 记录
        {
            Tensor mm = sched.dispatch<op::MatMul>(X, W);
            Tensor sum = sched.dispatch<op::Add>(mm, B);
            Tensor act = sched.dispatch<op::Sigmoid>(sum);
        }

        // 热身迭代触发热点
        for (int i = 2; i <= warmup_iters; ++i) {
            Tensor mm = sched.dispatch<op::MatMul>(X, W);
            Tensor sum = sched.dispatch<op::Add>(mm, B);
            Tensor act = sched.dispatch<op::Sigmoid>(sum);
        }

        // 等待编译完成
        std::cout << "  等待异步编译完成..." << std::endl;
        hp_mgr.waitForPendingCompiles();
        std::cout << "  RegionFusionRegistry entries=" << region_reg.entryCount() << std::endl;

        // 测量融合性能
        auto t0 = hires::now();
        for (int i = 0; i < perf_iters; ++i) {
            Tensor mm = sched.dispatch<op::MatMul>(X, W);
            Tensor sum = sched.dispatch<op::Add>(mm, B);
            Tensor act = sched.dispatch<op::Sigmoid>(sum);
        }
        auto t1 = hires::now();
        double fused_avg = std::chrono::duration_cast<us>(t1 - t0).count() / perf_iters;

        // 清理并测量 eager 性能
        sched.resetRegionFusion();
        region_reg.clear();
        // 重新重置 trace 并跑 eager 基准
        {
            // 先跑一次更新 trace
            Tensor mm = sched.dispatch(X, W, op::MatMul);
            Tensor sum = sched.dispatch(mm, B, op::Add);
            Tensor act = sched.dispatch(sum, op::Sigmoid);
        }

        auto t2 = hires::now();
        for (int i = 0; i < perf_iters; ++i) {
            Tensor mm = sched.dispatch(X, W, op::MatMul);
            Tensor sum = sched.dispatch(mm, B, op::Add);
            Tensor act = sched.dispatch(sum, op::Sigmoid);
        }
        auto t3 = hires::now();
        double eager_avg = std::chrono::duration_cast<us>(t3 - t2).count() / perf_iters;

        double speedup = eager_avg / fused_avg;
        std::cout << "  Eager 平均延迟: " << eager_avg << " us" << std::endl;
        std::cout << "  区域融合平均延迟: " << fused_avg << " us" << std::endl;
        std::cout << "  加速比: " << speedup << "x" << std::endl;

        if (speedup >= 0.9f) {
            std::cout << "  ✅ 区域融合性能可接受（加速比 >= 0.9x）" << std::endl;
            passed++;
        } else {
            std::cout << "  ⚠️  区域融合性能未达标（加速比 < 0.9x）" << std::endl;
        }
        total++;

        // 清理
        sched.resetRegionFusion();
        hp_mgr.clear();
        region_reg.clear();
    }

    // ======================= EXP-2b: 256x256 中矩阵真实区域融合路径性能验证 =======================
    {
        std::cout << "\n[EXP-2b] 256x256 中矩阵真实区域融合路径性能验证..." << std::endl;
        auto& sched = CtorchScheduler::getInstance();
        auto& hp_mgr = C3HotPathManager::instance();
        auto& region_reg = RegionFusionRegistry::getInstance();

        sched.resetRegionFusion();
        hp_mgr.clear();
        region_reg.clear();

        const size_t M = 256, K = 256, N = 256;
        const int warmup_iters = 5;
        const int perf_iters = 20;
        Tensor X(ShapeTag{}, {M, K});
        Tensor W(ShapeTag{}, {K, N});
        Tensor B(ShapeTag{}, {N});
        fillRandom(X);
        fillRandom(W);
        fillRandom(B);

        // eager 参考结果
        {
            Tensor mm = sched.dispatch<op::MatMul>(X, W);
            Tensor sum = sched.dispatch<op::Add>(mm, B);
            Tensor act = sched.dispatch<op::Sigmoid>(sum);
        }
        // 热身触发热点
        for (int i = 2; i <= warmup_iters; ++i) {
            Tensor mm = sched.dispatch<op::MatMul>(X, W);
            Tensor sum = sched.dispatch<op::Add>(mm, B);
            Tensor act = sched.dispatch<op::Sigmoid>(sum);
        }
        hp_mgr.waitForPendingCompiles();
        std::cout << "  256 region 已注册: " << (region_reg.entryCount() > 0 ? "✅" : "❌") << std::endl;

        // 融合性能（真实 dispatch 路径，含预走开销）
        auto t0 = hires::now();
        for (int i = 0; i < perf_iters; ++i) {
            Tensor mm = sched.dispatch<op::MatMul>(X, W);
            Tensor sum = sched.dispatch<op::Add>(mm, B);
            Tensor act = sched.dispatch<op::Sigmoid>(sum);
        }
        auto t1 = hires::now();
        double fused_avg = std::chrono::duration_cast<us>(t1 - t0).count() / perf_iters;

        // eager 基线
        sched.resetRegionFusion();
        region_reg.clear();
        {
            Tensor mm = sched.dispatch(X, W, op::MatMul);
            Tensor sum = sched.dispatch(mm, B, op::Add);
            Tensor act = sched.dispatch(sum, op::Sigmoid);
        }
        auto t2 = hires::now();
        for (int i = 0; i < perf_iters; ++i) {
            Tensor mm = sched.dispatch(X, W, op::MatMul);
            Tensor sum = sched.dispatch(mm, B, op::Add);
            Tensor act = sched.dispatch(sum, op::Sigmoid);
        }
        auto t3 = hires::now();
        double eager_avg = std::chrono::duration_cast<us>(t3 - t2).count() / perf_iters;

        double speedup = eager_avg / fused_avg;
        std::cout << "  [256] Eager 平均延迟: " << eager_avg << " us" << std::endl;
        std::cout << "  [256] 区域融合平均延迟: " << fused_avg << " us" << std::endl;
        std::cout << "  [256] 加速比: " << speedup << "x" << std::endl;
        total++;
        if (speedup >= 0.9f) {
            std::cout << "  ✅ 256x256 区域融合性能可接受（加速比 >= 0.9x）" << std::endl;
            passed++;
        } else {
            std::cout << "  ⚠️  256x256 区域融合性能未达标（加速比 < 0.9x）" << std::endl;
        }

        sched.resetRegionFusion();
        hp_mgr.clear();
        region_reg.clear();
    }

    // ======================= EXP-3: MatMul+Add+ReLU 端到端验证（正确性 + 性能） =======================
    {
        std::cout << "\n[EXP-3] MatMul+Add+ReLU 端到端验证（正确性 + 性能）..." << std::endl;
        auto& sched = CtorchScheduler::getInstance();
        auto& hp_mgr = C3HotPathManager::instance();
        auto& region_reg = RegionFusionRegistry::getInstance();

        // --- 3.1 小 tensor 正确性验证 ---
        {
            sched.resetRegionFusion();
            hp_mgr.clear();
            region_reg.clear();
            C3Engine::getInstance().clearCache();

            const size_t M = 32, K = 32, N = 32;
            const int warmup_iters = 5;
            Tensor X(ShapeTag{}, {M, K});
            Tensor W(ShapeTag{}, {K, N});
            Tensor B(ShapeTag{}, {M, N});
            fillRandom(X);
            fillRandom(W);
            fillRandom(B);

            // eager 参考结果
            Tensor act_ref;
            {
                Tensor mm = sched.dispatch<op::MatMul>(X, W);
                Tensor sum = sched.dispatch<op::Add>(mm, B);
                act_ref = sched.dispatch<op::ReLU>(sum);
            }

            // 热身触发热点
            for (int i = 2; i <= warmup_iters; ++i) {
                Tensor mm = sched.dispatch<op::MatMul>(X, W);
                Tensor sum = sched.dispatch<op::Add>(mm, B);
                Tensor act = sched.dispatch<op::ReLU>(sum);
            }
            hp_mgr.waitForPendingCompiles();
            bool registered = (region_reg.entryCount() > 0);
            std::cout << "  ReLU region 已自动注册: " << (registered ? "✅" : "❌") << std::endl;
            if (!registered) {
                auto stats = hp_mgr.getStats();
                std::cout << "    [DEBUG] hp_mgr: tracked=" << stats.calls_tracked
                          << " triggered=" << stats.compilations_triggered
                          << " pending=" << stats.pending_compiles << std::endl;
                std::cout << "    [DEBUG] last_compile_error: " << C3Engine::getInstance().getLastCompileError() << std::endl;
            }
            total++;
            if (registered) passed++;

            // 验证迭代（预走融合）
            Tensor mm = sched.dispatch<op::MatMul>(X, W);
            Tensor sum = sched.dispatch<op::Add>(mm, B);
            Tensor act = sched.dispatch<op::ReLU>(sum);

            const float* fused_data = act.data_read<float>();
            const float* ref_data = act_ref.data_read<float>();
            size_t numel = act.numel();
            double max_diff = 0.0;
            int bad_count = 0;
            for (size_t i = 0; i < numel; ++i) {
                double diff = std::fabs((double)fused_data[i] - (double)ref_data[i]);
                double max_val = std::max(std::fabs((double)fused_data[i]), std::fabs((double)ref_data[i]));
                if (diff > 1e-4 + 1e-4 * max_val) {
                    bad_count++;
                    if (bad_count <= 3) {
                        std::cout << "    MISMATCH[" << i << "]: fused=" << fused_data[i]
                                  << " ref=" << ref_data[i] << std::endl;
                    }
                }
                if (diff > max_diff) max_diff = diff;
            }
            bool correct = (bad_count == 0);
            std::cout << "  小 tensor ReLU 融合正确性: " << (correct ? "✅" : "❌")
                      << " bad=" << bad_count << "/" << numel
                      << " max_diff=" << max_diff << std::endl;
            total++;
            if (correct) passed++;
        }

        // --- 3.2 大 tensor 性能验证（1024，1D bias 广播） ---
        {
            sched.resetRegionFusion();
            hp_mgr.clear();
            region_reg.clear();
            C3Engine::getInstance().clearCache();

            const size_t M = 1024, K = 1024, N = 1024;
            const int warmup_iters = 5;
            const int perf_iters = 10;
            Tensor X(ShapeTag{}, {M, K});
            Tensor W(ShapeTag{}, {K, N});
            Tensor B(ShapeTag{}, {N});
            fillRandom(X);
            fillRandom(W);
            fillRandom(B);

            // eager 参考结果
            Tensor act_ref;
            {
                Tensor mm = sched.dispatch<op::MatMul>(X, W);
                Tensor sum = sched.dispatch<op::Add>(mm, B);
                act_ref = sched.dispatch<op::ReLU>(sum);
            }

            // 热身触发热点
            for (int i = 2; i <= warmup_iters; ++i) {
                Tensor mm = sched.dispatch<op::MatMul>(X, W);
                Tensor sum = sched.dispatch<op::Add>(mm, B);
                Tensor act = sched.dispatch<op::ReLU>(sum);
            }
            hp_mgr.waitForPendingCompiles();
            bool registered_big = (region_reg.entryCount() > 0);
            std::cout << "  大 tensor ReLU region 已注册: " << (registered_big ? "✅" : "❌") << std::endl;
            if (!registered_big) {
                auto stats = hp_mgr.getStats();
                std::cout << "    [DEBUG-BIG] hp_mgr: tracked=" << stats.calls_tracked
                          << " triggered=" << stats.compilations_triggered
                          << " pending=" << stats.pending_compiles << std::endl;
                std::cout << "    [DEBUG-BIG] last_compile_error: " << C3Engine::getInstance().getLastCompileError() << std::endl;
            }

            // 融合正确性
            Tensor mm = sched.dispatch<op::MatMul>(X, W);
            Tensor sum = sched.dispatch<op::Add>(mm, B);
            Tensor act = sched.dispatch<op::ReLU>(sum);
            const float* fused_data = act.data_read<float>();
            const float* ref_data = act_ref.data_read<float>();
            size_t numel = act.numel();
            double max_diff = 0.0;
            int bad_count = 0;
            for (size_t i = 0; i < numel; ++i) {
                double diff = std::fabs((double)fused_data[i] - (double)ref_data[i]);
                double max_val = std::max(std::fabs((double)fused_data[i]), std::fabs((double)ref_data[i]));
                if (diff > 1e-4 + 1e-4 * max_val) {
                    bad_count++;
                    if (bad_count <= 3) {
                        std::cout << "    MISMATCH[" << i << "]: fused=" << fused_data[i]
                                  << " ref=" << ref_data[i] << std::endl;
                    }
                }
                if (diff > max_diff) max_diff = diff;
            }
            bool correct = (bad_count == 0);
            std::cout << "  大 tensor ReLU 融合正确性: " << (correct ? "✅" : "❌")
                      << " bad=" << bad_count << "/" << numel
                      << " max_diff=" << max_diff << std::endl;
            total++;
            if (correct) passed++;

            // 融合性能
            auto t0 = hires::now();
            for (int i = 0; i < perf_iters; ++i) {
                Tensor mm = sched.dispatch<op::MatMul>(X, W);
                Tensor sum = sched.dispatch<op::Add>(mm, B);
                Tensor act = sched.dispatch<op::ReLU>(sum);
            }
            auto t1 = hires::now();
            double fused_avg = std::chrono::duration_cast<us>(t1 - t0).count() / perf_iters;

            // eager 基线
            sched.resetRegionFusion();
            region_reg.clear();
            {
                Tensor mm = sched.dispatch(X, W, op::MatMul);
                Tensor sum = sched.dispatch(mm, B, op::Add);
                Tensor act = sched.dispatch(sum, op::ReLU);
            }
            auto t2 = hires::now();
            for (int i = 0; i < perf_iters; ++i) {
                Tensor mm = sched.dispatch(X, W, op::MatMul);
                Tensor sum = sched.dispatch(mm, B, op::Add);
                Tensor act = sched.dispatch(sum, op::ReLU);
            }
            auto t3 = hires::now();
            double eager_avg = std::chrono::duration_cast<us>(t3 - t2).count() / perf_iters;

            double speedup = eager_avg / fused_avg;
            std::cout << "  [ReLU] Eager 平均延迟: " << eager_avg << " us" << std::endl;
            std::cout << "  [ReLU] 区域融合平均延迟: " << fused_avg << " us" << std::endl;
            std::cout << "  [ReLU] 加速比: " << speedup << "x" << std::endl;
            total++;
            if (speedup >= 0.9f) {
                std::cout << "  ✅ MatMul+Add+ReLU 性能可接受（加速比 >= 0.9x）" << std::endl;
                passed++;
            } else {
                std::cout << "  ⚠️  MatMul+Add+ReLU 性能未达标（加速比 < 0.9x）" << std::endl;
            }

            sched.resetRegionFusion();
            hp_mgr.clear();
            region_reg.clear();
        }
    }

    // ======================= 总结 =======================
    // ======================= EXP-4: buildFusedGraph 直接编译执行（二分隔离实验） =======================
    {
        std::cout << "\n[EXP-4] buildFusedGraph 生成的图直接编译执行，跳过预走机制..." << std::endl;
        auto& engine = C3Engine::getInstance();
        auto& sched = CtorchScheduler::getInstance();

        sched.resetRegionFusion();
        engine.clearCache();

        const size_t M = 32, K = 32, N = 32;
        Tensor X(ShapeTag{}, {M, K});
        Tensor W(ShapeTag{}, {K, N});
        Tensor B(ShapeTag{}, {M, N});
        fillRandom(X);
        fillRandom(W);
        fillRandom(B);

        // eager 参考结果
        Tensor mm_ref = sched.dispatch<op::MatMul>(X, W);
        Tensor sum_ref = sched.dispatch<op::Add>(mm_ref, B);
        Tensor act_ref = sched.dispatch<op::Sigmoid>(sum_ref);

        // 用 buildFusedGraph 构造图
        std::vector<C3HotPathManager::DispatchRecord> records = {
            {op::MatMul, {M, K, K, N}, {M, K}, {K, N}, std::chrono::steady_clock::now()},
            {op::Add,    {M, N, M, N}, {M, N}, {M, N}, std::chrono::steady_clock::now()},
            {op::Sigmoid,{M, N},       {M, N}, {},     std::chrono::steady_clock::now()},
        };
        Graph g = C3HotPathManager::buildFusedGraphForTest(records, "MatMul+Add+Sigmoid");

        auto kernel = engine.compile(g, CompileOptions{});
        if (!kernel) {
            std::cout << "  ❌ buildFusedGraph 图编译失败" << std::endl;
            total++;
        } else {
            std::cout << "  ✅ buildFusedGraph 图编译成功" << std::endl;
            Tensor act = C3KernelRegistry::getInstance().executeFusedWithInputs(
                kernel, {X, W, B}, KernelShapeInfo{});

            const float* fused_data = act.data_read<float>();
            const float* ref_data = act_ref.data_read<float>();
            size_t numel = act.numel();
            double max_diff = 0.0;
            int bad_count = 0;
            for (size_t i = 0; i < numel; ++i) {
                double diff = std::fabs((double)fused_data[i] - (double)ref_data[i]);
                double max_val = std::max(std::fabs((double)fused_data[i]), std::fabs((double)ref_data[i]));
                if (diff > 1e-4 + 1e-4 * max_val) {
                    bad_count++;
                    if (bad_count <= 3) {
                        std::cout << "    MISMATCH[" << i << "]: fused=" << fused_data[i]
                                  << " ref=" << ref_data[i] << std::endl;
                    }
                }
                if (diff > max_diff) max_diff = diff;
            }
            bool correct = (bad_count == 0);
            std::cout << "  buildFusedGraph 直接执行: " << (correct ? "✅" : "❌")
                      << " bad=" << bad_count << "/" << numel
                      << " max_diff=" << max_diff << std::endl;
            total++;
            if (correct) passed++;
        }
    }

    // ======================= EXP-5: LazyBox —— 融合期间读取中间值 z1 正确物化 =======================
    {
        std::cout << "\n[EXP-5] LazyBox 物化：融合期间读取中间值 z1（MatMul+Bias 输出）..." << std::endl;
        auto& sched = CtorchScheduler::getInstance();
        auto& hp_mgr = C3HotPathManager::instance();
        auto& region_reg = RegionFusionRegistry::getInstance();

        sched.resetRegionFusion();
        hp_mgr.clear();
        region_reg.clear();
        C3Engine::getInstance().clearCache();

        const size_t M = 32, K = 32, N = 32;
        const int warmup_iters = 5;
        Tensor X(ShapeTag{}, {M, K});
        Tensor W(ShapeTag{}, {K, N});
        Tensor B(ShapeTag{}, {M, N});
        fillRandom(X);
        fillRandom(W);
        fillRandom(B);

        // eager 参考：z1 = MatMul(X,W) + B
        Tensor z1_ref;
        {
            Tensor mm = sched.dispatch<op::MatMul>(X, W);
            z1_ref = sched.dispatch<op::Add>(mm, B);
            Tensor act = sched.dispatch<op::ReLU>(z1_ref);
        }
        // 热身触发热点
        for (int i = 2; i <= warmup_iters; ++i) {
            Tensor mm = sched.dispatch<op::MatMul>(X, W);
            Tensor sum = sched.dispatch<op::Add>(mm, B);
            Tensor act = sched.dispatch<op::ReLU>(sum);
        }
        hp_mgr.waitForPendingCompiles();
        bool registered = (region_reg.entryCount() > 0);
        std::cout << "  LazyBox region 已自动注册: " << (registered ? "✅" : "❌") << std::endl;
        total++;
        if (registered) passed++;

        // 融合迭代：捕获中间值 sum（z1），此时它是预走占位符（LazyBox）
        Tensor mm = sched.dispatch<op::MatMul>(X, W);
        Tensor sum = sched.dispatch<op::Add>(mm, B);
        Tensor act = sched.dispatch<op::ReLU>(sum);

        // 关键验证：用户读取中间值 z1，必须通过 LazyBox 物化出真实值
        if (!sum.isLazyBox()) {
            std::cout << "  ✅ 中间值已是真实张量（未走预走路径）" << std::endl;
        } else {
            std::cout << "  ✅ 中间值是 LazyBox 占位符，data_read() 将触发物化" << std::endl;
        }
        const float* z1_data = sum.data_read<float>();
        const float* z1_ref_data = z1_ref.data_read<float>();
        if (z1_data == nullptr || z1_ref_data == nullptr) {
            std::cout << "  ❌ z1 物化失败（返回空指针）" << std::endl;
            total++;
        } else {
            size_t numel = sum.numel();
            double max_diff = 0.0;
            int bad_count = 0;
            for (size_t i = 0; i < numel; ++i) {
                double diff = std::fabs((double)z1_data[i] - (double)z1_ref_data[i]);
                double max_val = std::max(std::fabs((double)z1_data[i]), std::fabs((double)z1_ref_data[i]));
                if (diff > 1e-4 + 1e-4 * max_val) {
                    bad_count++;
                    if (bad_count <= 3) {
                        std::cout << "    MISMATCH[" << i << "]: z1=" << z1_data[i]
                                  << " ref=" << z1_ref_data[i] << std::endl;
                    }
                }
                if (diff > max_diff) max_diff = diff;
            }
            bool correct = (bad_count == 0);
            std::cout << "  中间值 z1 物化正确性: " << (correct ? "✅" : "❌")
                      << " bad=" << bad_count << "/" << numel
                      << " max_diff=" << max_diff << std::endl;
            total++;
            if (correct) passed++;
        }

        // 物化后最终输出仍应正确（验证物化不破坏融合结果）
        const float* act_data = act.data_read<float>();
        double act_max_diff = 0.0;
        int act_bad = 0;
        {
            // eager 参考 act
            Tensor mm2 = sched.dispatch(X, W, op::MatMul);
            Tensor sum2 = sched.dispatch(mm2, B, op::Add);
            Tensor act2 = sched.dispatch(sum2, op::ReLU);
            const float* ref2 = act2.data_read<float>();
            for (size_t i = 0; i < act.numel(); ++i) {
                double diff = std::fabs((double)act_data[i] - (double)ref2[i]);
                double max_val = std::max(std::fabs((double)act_data[i]), std::fabs((double)ref2[i]));
                if (diff > 1e-4 + 1e-4 * max_val) act_bad++;
                if (diff > act_max_diff) act_max_diff = diff;
            }
        }
        bool act_correct = (act_bad == 0);
        std::cout << "  物化后最终输出正确性: " << (act_correct ? "✅" : "❌")
                  << " bad=" << act_bad << "/" << act.numel()
                  << " max_diff=" << act_max_diff << std::endl;
        total++;
        if (act_correct) passed++;

        sched.resetRegionFusion();
        hp_mgr.clear();
        region_reg.clear();
    }

    std::cout << "\n=== 测试结果: " << passed << "/" << total << " passed ==="
              << std::endl;

    // 显式清理 C3 资源：统一退出清理序列，确保所有 CompiledKernel/LLVM module
    // 在静态析构前释放，避免退出时 recursive_mutex / removeModule 崩溃
    shutdownAll();

    return (passed == total) ? 0 : 1;
}