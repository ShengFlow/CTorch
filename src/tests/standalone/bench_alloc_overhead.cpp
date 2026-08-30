// src/tests/standalone/bench_alloc_overhead.cpp
// 微基准：量化 Eager 内存分配路径的固定开销，分离三个环节占比：
//   ① AllocatorManager 锁 + unordered_map 查找
//   ② CPUAllocator malloc
//   ③ Storage 内 shared_ptr 控制块构造（含 Deleter）
// 结论归因用：明确"分配"到底慢在哪一环，指导后续优化优先级。
#include <cstdio>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "Tensor.h"
#include "DeviceAllocator.h"

using Clock = std::chrono::steady_clock;

static double ns_per(long long iters, double total_ns) {
    return total_ns / (double)iters;
}

int main() {
    const int ITERS = 200000;
    const size_t N = 128 * 256;   // 与 MNIST 一层对齐的元素规模

    // ===== A: 裸 malloc + free（防优化：写一个字节到分配的内存，再读回加总）=====
    {
        volatile unsigned long sink = 0;
        auto t0 = Clock::now();
        for (int it = 0; it < ITERS; ++it) {
            char* p = (char*)std::malloc(N * sizeof(float));
            if (p) { p[0] = 1; sink += (unsigned char)p[0]; std::free(p); }
        }
        auto t1 = Clock::now();
        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        printf("[A] 裸 malloc+w/free (防优化) : %.3f ns/次\n", ns_per(ITERS, ns));
        printf("    (sink=%lu)\n", sink);
    }

    // ===== B: AllocatorManager 锁+查找（不分配）=====
    {
        auto& mgr = AllocatorManager::getInstance();
        // 预热：确保 CPU allocator 已创建，测纯 get 路径
        std::shared_ptr<DeviceAllocator> sink;
        auto t0 = Clock::now();
        for (int it = 0; it < ITERS; ++it) {
            sink = mgr.getAllocator(DeviceType::kCPU);
        }
        auto t1 = Clock::now();
        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        printf("[B] AllocatorManager 锁+查找  : %.3f ns/次\n", ns_per(ITERS, ns));
        printf("    (sink=%p)\n", (void*)sink.get());
    }

    // ===== C: CPUAllocator.allocate（malloc，无锁）=====
    {
        CPUAllocator alloc;
        auto t0 = Clock::now();
        for (int it = 0; it < ITERS; ++it) {
            void* p = alloc.allocate(N * sizeof(float), DeviceType::kCPU);
            if (p) alloc.deallocate(p, DeviceType::kCPU);
        }
        auto t1 = Clock::now();
        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        printf("[C] CPUAllocator allocate     : %.3f ns/次\n", ns_per(ITERS, ns));
    }

    // ===== D: Storage 完整构造+析构（共享指针 + Deleter + allocator 查找）=====
    {
        auto t0 = Clock::now();
        for (int it = 0; it < ITERS; ++it) {
            Storage s(N, DType::kFloat, DeviceType::kCPU);
        }
        auto t1 = Clock::now();
        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        printf("[D] Storage 完整构造+析构     : %.3f ns/次\n", ns_per(ITERS, ns));
    }

    // ===== E: Tensor 完整构造（Storage + shape + strides + autograd meta）=====
    {
        auto t0 = Clock::now();
        for (int it = 0; it < ITERS; ++it) {
            Tensor t(ShapeTag{}, std::vector<size_t>{N}, DType::kFloat, DeviceType::kCPU);
        }
        auto t1 = Clock::now();
        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        printf("[E] Tensor 完整构造          : %.3f ns/次\n", ns_per(ITERS, ns));
    }

    // ===== E1: Storage 在 Tensor 外的市值（等价于 [D]，对齐对照）=====
    // ===== F: 仅 autograd meta 控制块（shared_ptr<Tensor>(this)）=====
    {
        // 用独立对象模拟 autograd meta 的 shared_ptr 控制块分配
        struct Dummy { std::shared_ptr<Dummy> self; };
        auto t0 = Clock::now();
        for (int it = 0; it < ITERS; ++it) {
            Dummy* d = new Dummy();
            d->self = std::shared_ptr<Dummy>(d, [](Dummy* p){ delete p; });
            d->self.reset();
        }
        auto t1 = Clock::now();
        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        printf("[F] autograd meta 控制块      : %.3f ns/次\n", ns_per(ITERS, ns));
    }

    // ===== G: memset（zero() 的核心，128KB，防优化读回一个字节）=====
    {
        std::vector<char> buf(N * sizeof(float));
        volatile unsigned char s = 0;
        auto t0 = Clock::now();
        for (int it = 0; it < ITERS; ++it) {
            std::memset(buf.data(), 0, buf.size());
            s ^= (unsigned char)buf[it % 16];
        }
        auto t1 = Clock::now();
        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        printf("[G] memset 128KB (zero核心)   : %.3f ns/次\n", ns_per(ITERS, ns));
        printf("    (s=%u)\n", (unsigned)s);
    }

    return 0;
}