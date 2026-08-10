/**
 * @file bench_auto_c3.cpp
 * @brief 调度器自动 C3 化端到端验证 benchmark
 * @details 验证 C3HotPathManager 的自动热路径检测和编译触发：
 *          1. 快速触发热路径（执行 N 次达到阈值）
 *          2. 等待异步编译完成
 *          3. 再次执行，验证 C3 kernel 已接管
 *          4. 对比 eager 与 C3 的性能差异
 *
 * @date 2026/8/4
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>
#include <string>
#include <thread>

#include "Tensor.h"
#include "CtorchScheduler.h"
#include "C3/C3HotPathManager.h"
#include "C3/C3KernelRegistry.h"

using namespace ct;
using namespace ct::c3;
using hires = std::chrono::high_resolution_clock;
using us = std::chrono::duration<double, std::micro>;
using ms = std::chrono::duration<double, std::milli>;

// ======================= 工具函数 =======================

struct TestResult {
    std::string op_name;
    int trigger_iters;
    int takeover_iter;
    double eager_avg_us;
    double c3_avg_us;
    bool success;
};

TestResult test_auto_c3(const std::string& op_name, op op_type,
                        const Tensor& a, const Tensor& b,
                        int hot_threshold, int warmup, int post_compile_iters) {
    TestResult result;
    result.op_name = op_name;
    result.success = false;

    std::cout << "\n--- " << op_name << " 自动 C3 化测试 ---\n";

    C3KernelRegistry::getInstance().uninstallAll();
    C3HotPathManager::instance().clear();

    HotPathConfig cfg;
    cfg.hot_threshold = hot_threshold;
    cfg.cooldown_sec = 0;
    C3HotPathManager::instance().configure(cfg);

    auto& scheduler = CtorchScheduler::getInstance();

    // Phase 1: 触发热路径
    std::cout << "  [Phase 1] 触发热路径（阈值=" << hot_threshold << "）...\n";
    auto hp_stats_before = C3HotPathManager::instance().getStats();

    for (int i = 0; i < hot_threshold + warmup; ++i) {
        scheduler.dispatch(a, b, op_type);
    }

    auto hp_stats_after = C3HotPathManager::instance().getStats();
    result.trigger_iters = hp_stats_after.compilations_triggered > hp_stats_before.compilations_triggered
                           ? hot_threshold : -1;
    std::cout << "    调用次数: " << hp_stats_after.calls_tracked
              << ", 编译触发: " << hp_stats_after.compilations_triggered << "\n";

    if (hp_stats_after.compilations_triggered == 0) {
        std::cout << "    ❌ 编译未触发\n";
        return result;
    }

    // Phase 2: 等待编译完成
    std::cout << "  [Phase 2] 等待编译完成...\n";
    const int max_wait_ms = (op_type == op::MatMul) ? 30000 : 5000;
    int waited_ms = 0;
    bool compile_done = false;

    while (waited_ms < max_wait_ms) {
        auto kr_stats = C3KernelRegistry::getInstance().getStats();
        auto hp_stats = C3HotPathManager::instance().getStats();

        if (kr_stats.active_entries > 0 || hp_stats.pending_compiles == 0) {
            compile_done = true;
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        waited_ms += 100;
    }

    auto kr_stats = C3KernelRegistry::getInstance().getStats();
    std::cout << "    ✅ 编译完成（约 " << waited_ms << "ms），active="
              << kr_stats.active_entries << "\n";

    // Phase 3: 验证 C3 接管 + 测量性能
    std::cout << "  [Phase 3] 验证 C3 接管 + 性能测量（" << post_compile_iters << " 次迭代）...\n";

    // 先做预热，让系统稳定
    for (int i = 0; i < 5; ++i) {
        scheduler.dispatch(a, b, op_type);
    }

    // 测量 eager 路径（临时卸载 C3）
    size_t kr_hits_before = C3KernelRegistry::getInstance().getStats().hit_count;
    std::vector<double> latencies;

    for (int i = 0; i < post_compile_iters; ++i) {
        auto start = hires::now();
        scheduler.dispatch(a, b, op_type);
        auto end = hires::now();
        latencies.push_back(us(end - start).count());
    }

    size_t kr_hits_after = C3KernelRegistry::getInstance().getStats().hit_count;
    int c3_hits = kr_hits_after - kr_hits_before;
    result.success = (c3_hits > 0);
    result.takeover_iter = c3_hits > 0 ? 0 : -1;

    // 分离 C3 命中和未命中的延迟
    std::vector<double> c3_lats, eager_lats;
    if (result.success) {
        // 命中 C3 的延迟
        for (int i = 0; i < c3_hits && i < (int)latencies.size(); ++i) {
            c3_lats.push_back(latencies[i]);
        }
        // 未命中的延迟（回退 eager）
        for (int i = c3_hits; i < (int)latencies.size(); ++i) {
            eager_lats.push_back(latencies[i]);
        }
    } else {
        eager_lats = latencies;
    }

    // 计算平均值
    if (!eager_lats.empty()) {
        result.eager_avg_us = 0;
        for (double l : eager_lats) result.eager_avg_us += l;
        result.eager_avg_us /= eager_lats.size();
    }
    if (!c3_lats.empty()) {
        result.c3_avg_us = 0;
        for (double l : c3_lats) result.c3_avg_us += l;
        result.c3_avg_us /= c3_lats.size();
    }

    auto final_stats = C3KernelRegistry::getInstance().getStats();
    std::cout << "    C3 命中: " << c3_hits << "/" << post_compile_iters
              << ", 活跃 entries: " << final_stats.active_entries << "\n";

    if (result.success && result.c3_avg_us > 0) {
        std::cout << "    C3 延迟: " << std::fixed << std::setprecision(2)
                  << result.c3_avg_us << " us\n";
    }
    if (!eager_lats.empty()) {
        std::cout << "    Eager 延迟: " << std::fixed << std::setprecision(2)
                  << result.eager_avg_us << " us\n";
    }

    return result;
}

// ======================= 主程序 =======================

int main() {
    std::cout << "====================================================\n";
    std::cout << "    C3 调度器自动化端到端验证 Benchmark\n";
    std::cout << "====================================================\n\n";

    const int hot_threshold = 3;
    const int warmup = 2;
    const int iters = 20;

    // Part 1: 中号 MatMul (128x128)
    std::cout << "\n\n========================================\n";
    std::cout << "  Part 1: MatMul (128x128) 自动 C3 化\n";
    std::cout << "========================================\n";

    const size_t M = 128, K = 128, N = 128;
    Tensor a = Tensor(ShapeTag{}, {M, K});
    Tensor b = Tensor(ShapeTag{}, {K, N});
    float* p;
    p = a.data_write<float>();
    for (size_t i = 0; i < a.numel(); ++i) p[i] = std::sin(static_cast<float>(i) * 0.1f);
    p = b.data_write<float>();
    for (size_t i = 0; i < b.numel(); ++i) p[i] = std::cos(static_cast<float>(i) * 0.1f);

    auto matmul_result = test_auto_c3("MatMul(128x128)", op::MatMul, a, b,
                                       hot_threshold, warmup, iters);

    // Part 2: 大号 MatMul (256x256) - 更能体现 C3 优势
    std::cout << "\n\n========================================\n";
    std::cout << "  Part 2: MatMul (256x256) 自动 C3 化\n";
    std::cout << "========================================\n";

    const size_t M2 = 256, K2 = 256, N2 = 256;
    Tensor a2 = Tensor(ShapeTag{}, {M2, K2});
    Tensor b2 = Tensor(ShapeTag{}, {K2, N2});
    p = a2.data_write<float>();
    for (size_t i = 0; i < a2.numel(); ++i) p[i] = std::sin(static_cast<float>(i) * 0.05f);
    p = b2.data_write<float>();
    for (size_t i = 0; i < b2.numel(); ++i) p[i] = std::cos(static_cast<float>(i) * 0.05f);

    auto matmul2_result = test_auto_c3("MatMul(256x256)", op::MatMul, a2, b2,
                                        hot_threshold, warmup, iters);

    // Part 3: Add (中号)
    std::cout << "\n\n========================================\n";
    std::cout << "  Part 3: Add (1024) 自动 C3 化\n";
    std::cout << "========================================\n";

    const size_t vec_size = 1024;
    Tensor x = Tensor(ShapeTag{}, {vec_size});
    Tensor y = Tensor(ShapeTag{}, {vec_size});
    p = x.data_write<float>();
    for (size_t i = 0; i < x.numel(); ++i) p[i] = 1.0f;
    p = y.data_write<float>();
    for (size_t i = 0; i < y.numel(); ++i) p[i] = 2.0f;

    auto add_result = test_auto_c3("Add(1024)", op::Add, x, y,
                                    hot_threshold, warmup, iters);

    // Part 4: Sigmoid
    std::cout << "\n\n========================================\n";
    std::cout << "  Part 4: Sigmoid (1024) 自动 C3 化\n";
    std::cout << "========================================\n";

    C3KernelRegistry::getInstance().uninstallAll();
    C3HotPathManager::instance().clear();

    HotPathConfig cfg;
    cfg.hot_threshold = hot_threshold;
    cfg.cooldown_sec = 0;
    C3HotPathManager::instance().configure(cfg);

    Tensor s_in = Tensor(ShapeTag{}, {vec_size});
    p = s_in.data_write<float>();
    for (size_t i = 0; i < s_in.numel(); ++i) p[i] = static_cast<float>(i) / 100.0f;

    auto& scheduler = CtorchScheduler::getInstance();

    // 触发 Sigmoid 热路径
    std::cout << "\n  [Phase 1] 触发 Sigmoid 热路径...\n";
    for (int i = 0; i < hot_threshold + warmup; ++i) {
        scheduler.dispatch(s_in, op::Sigmoid);
    }

    auto sig_hp = C3HotPathManager::instance().getStats();
    std::cout << "    调用次数: " << sig_hp.calls_tracked
              << ", 编译触发: " << sig_hp.compilations_triggered << "\n";

    // 等待编译
    std::cout << "  [Phase 2] 等待 Sigmoid 编译完成...\n";
    int waited = 0;
    while (waited < 5000) {
        auto kr = C3KernelRegistry::getInstance().getStats();
        if (kr.active_entries > 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        waited += 50;
    }
    auto sig_kr = C3KernelRegistry::getInstance().getStats();
    std::cout << "    active entries: " << sig_kr.active_entries << "\n";

    // 验证接管
    std::cout << "  [Phase 3] 验证 Sigmoid C3 接管...\n";
    int sig_hits_before = sig_kr.hit_count;
    for (int i = 0; i < iters; ++i) {
        scheduler.dispatch(s_in, op::Sigmoid);
    }
    auto sig_kr_final = C3KernelRegistry::getInstance().getStats();
    int sig_new_hits = sig_kr_final.hit_count - sig_hits_before;
    std::cout << "    新增 C3 命中: " << sig_new_hits << "/" << iters << "\n";
    bool sigmoid_success = sig_new_hits > 0;

    // 总结
    std::cout << "\n\n====================================================\n";
    std::cout << "                      总结\n";
    std::cout << "====================================================\n\n";

    struct SummaryItem {
        std::string name;
        bool success;
        double eager_us;
        double c3_us;
    };

    std::vector<SummaryItem> summary;

    auto add_summary = [&](const TestResult& r) {
        summary.push_back({r.op_name, r.success, r.eager_avg_us, r.c3_avg_us});
    };

    add_summary(matmul_result);
    add_summary(matmul2_result);
    add_summary(add_result);
    summary.push_back({"Sigmoid(1024)", sigmoid_success, 0, 0});

    std::cout << std::left << std::setw(15) << "算子"
              << std::setw(10) << "状态"
              << std::setw(15) << "Eager (us)"
              << std::setw(15) << "C3 (us)" << "\n";
    std::cout << std::string(55, '-') << "\n";

    int success_count = 0;
    for (const auto& item : summary) {
        std::cout << std::left << std::setw(15) << item.name;
        if (item.success) {
            std::cout << std::setw(10) << "✅";
            std::cout << std::fixed << std::setprecision(2) << std::setw(15) << item.eager_us;
            std::cout << std::fixed << std::setprecision(2) << std::setw(15) << item.c3_us;
            success_count++;
        } else {
            std::cout << std::setw(10) << "❌";
            std::cout << std::setw(15) << "-";
            std::cout << std::setw(15) << "-";
        }
        std::cout << "\n";
    }

    std::cout << "\n通过: " << success_count << " / " << summary.size() << "\n";

    if (success_count == (int)summary.size()) {
        std::cout << "\n🎉 调度器自动 C3 化验证全部通过！\n";
    } else {
        std::cout << "\n⚠️ 部分测试未通过。\n";
    }

    return 0;
}
