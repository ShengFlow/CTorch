/**
 * @file test_arena_lifecycle.cpp
 * @author 苏璃珞
 * @brief Arena 生命周期与水位回落回归测试
 *
 * @details Arena 提供两类能力，二者的生命周期语义**不同**，本文件分别覆盖：
 *
 *  1. **字节级池**（allocBytes / allocShared / reset / clear）：块不归还操作系统，
 *     因此必须按水位回落 —— 否则一次内存尖峰之后进程长期占住那份内存。
 *  2. **对象构造**（invoke）：自 2026-09-18 起改用标准堆分配，返回**拥有所有权**的
 *     shared_ptr，对象在引用归零时析构。此前的「空删除器 + reset() 强杀 + 地址复用」
 *     会让跨 reset 存活的 Tensor::_node 悬垂并指向复用后的新对象（类型混淆），
 *     多轮建图-反传稳定崩溃（见 test_multi_epoch_backward）。
 *
 * 覆盖：
 *  A. 字节池：多轮分配/复位后块数稳定，不随轮数增长
 *  B. 尖峰回落：一次超大需求把池撑大后，reset() 应把块数回落到保留水位
 *  C. invoke 的所有权语义：持有期间不析构、引用归零时析构且只析构一次
 *  D. clear() 释放全部块
 *
 * @date 2026/9/17
 **/

#include "Arena.h"
#include "AutoGrad/Node.h"

#include <iostream>
#include <memory>
#include <vector>

namespace {

int g_checks = 0;
int g_failed = 0;

void checkTrue(const char *name, bool cond) {
    ++g_checks;
    if (!cond) {
        ++g_failed;
    }
    std::cout << "  [" << (cond ? " ok " : "FAIL") << "] " << name << "\n";
}

/// 池中对象析构次数的计数器：用于验证 reset() 真的析构了对象
int g_destroyCount = 0;

/// 带可观测副作用析构的节点
struct TrackedNode final : Node {
    std::vector<int> payload{1, 2, 3};
    TrackedNode() = default;
    ~TrackedNode() override { ++g_destroyCount; }
    std::vector<GradPack> backward(const std::vector<Tensor> &) override { return {}; }
};

/// 分配一个远超单块容量（1 MB）的字节数，用于把池撑大
constexpr size_t kHugeBytes = 4 * 1024 * 1024;

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    Arena &arena = Arena::getInstance();

    std::cout << "========================================\n";
    std::cout << "Arena 生命周期与水位回落\n";
    std::cout << "========================================\n";

    // ---- A. 常规多轮迭代后块数稳定 ----
    std::cout << "\n[A] 多轮迭代的块数稳定性\n";
    {
        arena.clear();
        const size_t blocks_before = arena.blockCount();

        size_t blocks_after_round1 = 0;
        for (int round = 0; round < 20; ++round) {
            for (int i = 0; i < 500; ++i) {
                char *p = arena.allocBytes(256);
                if (p == nullptr) {
                    checkTrue("分配成功", false);
                    return 1;
                }
            }
            arena.reset();
            if (round == 0) {
                blocks_after_round1 = arena.blockCount();
            }
        }
        const size_t blocks_after = arena.blockCount();
        std::cout << "    起始块数 = " << blocks_before << "，第 1 轮后 = "
                  << blocks_after_round1 << "，20 轮后 = " << blocks_after << "\n";
        checkTrue("20 轮后块数不超过第 1 轮（稳态不增长）",
                  blocks_after <= blocks_after_round1);
        checkTrue("块数处于保留水位之内", blocks_after <= Arena::KEEP_BLOCKS);
    }

    // ---- B. 尖峰回落 ----
    std::cout << "\n[B] 内存尖峰后的回落\n";
    {
        const size_t before = arena.blockCount();
        // 连续切出远超单块容量的大块，把池撑到**保留水位之上**，
        // 否则 reset 无需释放任何块，回落行为得不到验证。
        bool all_ok = true;
        for (int i = 0; i < 2 * Arena::KEEP_BLOCKS; ++i) {
            char *p = arena.allocBytes(kHugeBytes);
            if (p == nullptr) {
                all_ok = false;
            }
        }
        checkTrue("大块分配全部成功", all_ok);
        const size_t peak = arena.blockCount();
        std::cout << "    尖峰前块数 = " << before << "，尖峰后 = " << peak << "\n";
        checkTrue("尖峰确实撑大了池", peak > before);

        arena.reset();
        const size_t after = arena.blockCount();
        std::cout << "    reset 后块数 = " << after << "（保留水位 "
                  << Arena::KEEP_BLOCKS << "）\n";
        checkTrue("reset 把块数回落到保留水位", after <= Arena::KEEP_BLOCKS);
        checkTrue("回落确实释放了块", after < peak);
    }

    // ---- C. invoke 的所有权语义 ----
    std::cout << "\n[C] invoke 的所有权语义\n";
    {
        g_destroyCount = 0;
        {
            std::vector<std::shared_ptr<TrackedNode>> keep;
            keep.reserve(100);
            for (int i = 0; i < 100; ++i) {
                auto p = arena.invoke<TrackedNode>();
                if (!p) {
                    checkTrue("invoke 分配成功", false);
                    break;
                }
                keep.push_back(p);
            }
            std::cout << "    持有 100 个引用期间析构次数 = " << g_destroyCount
                      << "（期望 0）\n";
            checkTrue("持有期间对象不被析构", g_destroyCount == 0);

            keep.clear();
            std::cout << "    释放全部引用后析构次数 = " << g_destroyCount << "（期望 100）\n";
            checkTrue("引用归零时全部析构", g_destroyCount == 100);
        }
        // 再取一次，确认不会被重复析构
        const int before = g_destroyCount;
        auto p = arena.invoke<TrackedNode>();
        p.reset();
        std::cout << "    二次分配后析构增量 = " << (g_destroyCount - before)
                  << "（期望 1）\n";
        checkTrue("每次分配只析构一次", g_destroyCount - before == 1);
    }

    // ---- D. clear() 释放全部块 ----
    std::cout << "\n[D] clear() 释放全部块\n";
    {
        for (int i = 0; i < 4; ++i) {
            (void)arena.allocBytes(kHugeBytes);
        }
        checkTrue("clear 前池非空", arena.blockCount() > 0);
        arena.clear();
        std::cout << "    clear 后块数 = " << arena.blockCount() << "\n";
        checkTrue("clear() 释放全部块", arena.blockCount() == 0);

        // clear 之后仍可正常工作（块会在首次分配时重建）
        auto p = arena.invoke<TrackedNode>();
        checkTrue("clear 后仍可分配", p != nullptr);
        arena.clear();
    }

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
