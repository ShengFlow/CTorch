/**
 * @file bench_arena.cpp
 * @author 苏璃珞
 * @brief Arena 对象池的分配开销基准
 *
 * @details 目的：判断 Arena 当前实现（每对象一条 `std::function<void()>` 析构记录 +
 *          每次分配走 std::align + 全局一把大锁）相对标准堆分配是否有实际优势，
 *          以及改进析构记录形式能带来多少收益。
 *
 * 负载形态刻意对齐真实用法：CTorch 的一次训练迭代里批量创建图节点，反向结束调用
 * `Arena::reset()` 释放，下一轮复用同一批内存块。因此基准按「若干轮 x 每轮若干次
 * 创建 + 一次 reset」组织，而不是一次性创建海量对象。
 *
 * 对照项：
 *  A. Arena::invoke<T>（当前实现）
 *  B. std::make_shared<T>（标准堆分配，含引用计数分配）
 *  C. Arena::reset()（每轮一次，单独计时）
 *
 * 用法：./bench_arena [每轮对象数] [轮数]
 *
 * @date 2026/9/17
 **/

#include "Arena.h"
#include "AutoGrad/Node.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

namespace {

/// 一个最小但非平凡可析构的节点（有 vector 成员，析构不是 no-op）
struct DummyNode final : Node {
    std::vector<int> payload{1, 2, 3, 4};
    std::vector<GradPack> backward(const std::vector<Tensor> &) override { return {}; }
};

/// 平凡可析构的载荷：用于分离「分配 + 锁」与「非平凡析构记录」两类开销
struct PodObject {
    int a[16]{};
};

using Clock = std::chrono::steady_clock;

double msSince(const Clock::time_point &t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

} // namespace

int main(int argc, char **argv) {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    int per_round = 2000;
    int rounds = 200;
    if (argc > 1) {
        per_round = std::atoi(argv[1]);
    }
    if (argc > 2) {
        rounds = std::atoi(argv[2]);
    }
    const long total = static_cast<long>(per_round) * rounds;

    std::cout << "========================================\n";
    std::cout << "Arena 分配开销基准\n";
    std::cout << "每轮对象数 = " << per_round << "，轮数 = " << rounds
              << "，合计 " << total << " 次创建\n";
    std::cout << "========================================\n";

    // ---- 预热 ----
    // 性能测量必须先预热：首次分配要向内核申请新页，若让被测项 A 冷启动、被测项 B
    // 复用 A 释放的内存，会得到「A 比 B 慢数倍」的假象（本基准早期版本即如此，
    // 顺序一换结论就翻转）。这里对两条路径各跑一轮不计时的负载。
    {
        std::vector<std::shared_ptr<DummyNode>> warm;
        warm.reserve(per_round);
        for (int i = 0; i < per_round; ++i) {
            warm.push_back(Arena::getInstance().invoke<DummyNode>());
        }
        warm.clear();
        Arena::getInstance().reset();

        std::vector<std::shared_ptr<DummyNode>> warm2;
        warm2.reserve(per_round);
        for (int i = 0; i < per_round; ++i) {
            warm2.push_back(std::make_shared<DummyNode>());
        }
        warm2.clear();
    }

    // ---- A. Arena::invoke ----
    double arena_alloc_ms = 0.0;
    double arena_reset_ms = 0.0;
    {
        for (int r = 0; r < rounds; ++r) {
            // 与对照项保持同样的持有模式：创建后持有到本轮结束再统一释放。
            // （早期版本在这里每次创建后立即释放，malloc/free 交错会让本段显得
            //  慢 4 倍 —— 那是分配器行为的差异，不是 invoke 的开销。）
            auto t0 = Clock::now();
            std::vector<std::shared_ptr<DummyNode>> keep;
            keep.reserve(per_round);
            for (int i = 0; i < per_round; ++i) {
                auto p = Arena::getInstance().invoke<DummyNode>();
                if (!p) {
                    std::cout << "分配失败，提前结束\n";
                    return 1;
                }
                keep.push_back(std::move(p));
            }
            arena_alloc_ms += msSince(t0);

            auto t1 = Clock::now();
            Arena::getInstance().reset();
            arena_reset_ms += msSince(t1);

            auto t2 = Clock::now();
            keep.clear();
            arena_reset_ms += msSince(t2);
        }
        std::cout << "\n[A] Arena::invoke  创建 " << total << " 个对象: " << arena_alloc_ms
                  << " ms  (" << (arena_alloc_ms * 1000.0 / total) << " ns/个)\n";
        std::cout << "    Arena::reset() 共 " << rounds << " 次:            "
                  << arena_reset_ms << " ms  ("
                  << (arena_reset_ms * 1000.0 / rounds) << " us/次)\n";
        std::cout << "    合计 " << (arena_alloc_ms + arena_reset_ms) << " ms\n";
    }

    // ---- B. std::make_shared ----
    double shared_ms = 0.0;
    {
        auto t_all = Clock::now();
        for (int r = 0; r < rounds; ++r) {
            auto t0 = Clock::now();
            std::vector<std::shared_ptr<DummyNode>> keep;
            keep.reserve(per_round);
            for (int i = 0; i < per_round; ++i) {
                keep.push_back(std::make_shared<DummyNode>());
            }
            shared_ms += msSince(t0);
            keep.clear(); // 显式释放，与 Arena::reset 对应
        }
        std::cout << "\n[B] make_shared     创建 " << total << " 个对象: " << shared_ms
                  << " ms  (" << (shared_ms * 1000.0 / total) << " ns/个)\n";
    }

    // ---- A2. 缓存 Arena 引用后的 invoke（分离 getInstance() 的开销）----
    double cached_ms = 0.0;
    {
        Arena &ar = Arena::getInstance();
        for (int r = 0; r < rounds; ++r) {
            auto t0 = Clock::now();
            std::vector<std::shared_ptr<DummyNode>> keep;
            keep.reserve(per_round);
            for (int i = 0; i < per_round; ++i) {
                keep.push_back(ar.invoke<DummyNode>());
            }
            cached_ms += msSince(t0);
            keep.clear();
            ar.reset();
        }
        std::cout << "\n[A2] 缓存引用后的 invoke  创建 " << total << " 个对象: " << cached_ms
                  << " ms\n";
        std::cout << "     与 [A] 之差（= getInstance() 的每次调用开销）: "
                  << (arena_alloc_ms - cached_ms) << " ms\n";
    }

    // ---- C. 细分：相同负载下分别测「分配+锁」「非平凡析构记录」「纯字节分配」----
    double pod_ms = 0.0;
    {
        for (int r = 0; r < rounds; ++r) {
            auto t0 = Clock::now();
            std::vector<std::shared_ptr<PodObject>> keep;
            keep.reserve(per_round);
            for (int i = 0; i < per_round; ++i) {
                keep.push_back(Arena::getInstance().invoke<PodObject>());
            }
            pod_ms += msSince(t0);
            keep.clear();
            Arena::getInstance().reset();
        }
        std::cout << "\n[C] invoke<平凡类型>  创建 " << total << " 个对象: " << pod_ms
                  << " ms  (" << (pod_ms * 1000.0 / total) << " us/个)\n";
        std::cout << "    与 [A] 之差（非平凡析构记录开销）: "
                  << (arena_alloc_ms - pod_ms) << " ms\n";
    }

    double bytes_ms = 0.0;
    {
        const size_t bytes = sizeof(PodObject);
        for (int r = 0; r < rounds; ++r) {
            auto t0 = Clock::now();
            for (int i = 0; i < per_round; ++i) {
                volatile char *p = Arena::getInstance().allocBytes(bytes, alignof(PodObject));
                (void)p;
            }
            bytes_ms += msSince(t0);
            Arena::getInstance().reset();
        }
        std::cout << "\n[D] allocBytes        分配 " << total << " 次: " << bytes_ms
                  << " ms  (" << (bytes_ms * 1000.0 / total) << " us/次)\n";
    }

    // ---- 对比 ----
    const double arena_total = arena_alloc_ms + arena_reset_ms;
    std::cout << "\n========================================\n";
    std::cout << "Arena 合计 " << arena_total << " ms vs make_shared " << shared_ms
              << " ms\n";
    if (shared_ms > 0.0) {
        std::cout << "Arena / make_shared = " << (arena_total / shared_ms) << "\n";
    }
    std::cout << "========================================\n";
    return 0;
}
