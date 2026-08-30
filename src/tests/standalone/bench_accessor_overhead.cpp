// src/tests/standalone/bench_accessor_overhead.cpp
// 微基准：量化 data_read()/data_write() 每次调用的固定开销（accessor 层）。
// 对比基线：裸 std::vector<float> 直接下标访问。
// 结论归因用：若 [Tensor data_read/data_write] 远慢于 [裸 vector 访问]，说明 accessor 层是固定开销大头。
#include <cstdio>
#include <chrono>
#include <vector>
#include "Tensor.h"

using Clock = std::chrono::steady_clock;

static double ns_per_it(long long iters, double total_ns) {
    return total_ns / (double)iters;
}

int main() {
    const int ITERS = 2000000;   // 访问次数
    const int N = 128 * 784;     // 与 MNIST 一层对齐的元素规模

    // ===== A: 裸 vector 下标访问（复用）=====
    {
        std::vector<float> a(N, 1.0f), b(N, 0.5f), c(N);
        auto t0 = Clock::now();
        for (int it = 0; it < ITERS; ++it) {
            for (int i = 0; i < N; ++i) c[i] = a[i] + b[i];
        }
        auto t1 = Clock::now();
        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        printf("[A] 裸 vector 下标访问        : %.3f ns/次\n", ns_per_it(ITERS, ns));
    }

    // ===== B: Tensor data_read/data_write 访问（复用 output）=====
    {
        Tensor a(ShapeTag{}, std::vector<size_t>{N}, DType::kFloat, DeviceType::kCPU);

        for (int i = 0; i < N; ++i) a.data_write<float>()[i] = 1.0f;
        Tensor b(ShapeTag{}, std::vector<size_t>{N}, DType::kFloat, DeviceType::kCPU);
        for (int i = 0; i < N; ++i) b.data_write<float>()[i] = 0.5f;
        Tensor c(ShapeTag{}, std::vector<size_t>{N}, DType::kFloat, DeviceType::kCPU);

        auto t0 = Clock::now();
        for (int it = 0; it < ITERS; ++it) {
            const float* ap = a.data_read<float>();
            const float* bp = b.data_read<float>();
            float* cp = c.data_write<float>();
            for (int i = 0; i < N; ++i) cp[i] = ap[i] + bp[i];
        }
        auto t1 = Clock::now();
        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        printf("[B] Tensor data_read/write     : %.3f ns/次\n", ns_per_it(ITERS, ns));
    }

    // ===== C: Tensor data_read/data_write 仅取指针（不带头元素循环，测纯 accessor 开销）=====
    {
        Tensor a(ShapeTag{}, std::vector<size_t>{N}, DType::kFloat, DeviceType::kCPU);
        for (int i = 0; i < N; ++i) a.data_write<float>()[i] = 1.0f;
        const float* sink = nullptr;
        auto t0 = Clock::now();
        for (int it = 0; it < ITERS; ++it) {
            sink = a.data_read<float>();
        }
        auto t1 = Clock::now();
        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        printf("[C] 纯 data_read 取指针       : %.3f ns/次\n", ns_per_it(ITERS, ns));
        printf("    (sink=%p 防优化)\n", (const void*)sink);
    }

    return 0;
}