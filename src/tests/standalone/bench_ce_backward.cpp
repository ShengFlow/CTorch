/**
 * @file bench_ce_backward.cpp
 * @brief Micro-benchmark: eager CrossEntropy backward cost (ROI probe for C3 fusion)
 * @details 量化 eager CE 反向（grad * (softmax(logits,1) - target)）在若干规模下的
 *          每次成本，用于判断把 CE 反向融合进 C3 是否值得（性能 ROI）。
 *          同时给出 MNIST(128,10) 按 468 batch/epoch 折算的每-epoch 天花板。
 * @date 2026-08-28
 */
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>
#include "Tensor.h"

using namespace ct;
using hires = std::chrono::high_resolution_clock;
using us = std::chrono::duration<double, std::micro>;

// 简单确定性填充：避免随机开销进入计时，仅需同等工作量
static void fill(Tensor& t, float seed) {
    float* p = t.data_write<float>();
    size_t n = t.numel();
    for (size_t i = 0; i < n; ++i) p[i] = seed + (float)(i % 7) * 0.01f;
}

static void bench_shape(size_t B, size_t C, int iters) {
    Tensor logits(ShapeTag{}, {B, C});
    Tensor target(ShapeTag{}, {B, C});
    Tensor grad(ShapeTag{}, {1});
    fill(logits, 0.5f);
    fill(target, 0.1f);
    grad.data_write<float>()[0] = 1.0f;

    // warmup
    for (int i = 0; i < 5; ++i) { Tensor p = logits.softmax(1); Tensor d = p - target; Tensor g = grad * d; (void)g; }

    auto t0 = hires::now();
    volatile float sink = 0.f;
    for (int i = 0; i < iters; ++i) {
        Tensor p = logits.softmax(1);
        Tensor d = p - target;
        Tensor g = grad * d;
        sink += g.data_read<float>()[0];
    }
    auto t1 = hires::now();
    double us_per = std::chrono::duration_cast<us>(t1 - t0).count() / (double)iters;

    // MNIST: 468 batch/epoch → 每 epoch 折算
    double mnist_per_epoch = us_per * 468.0 / 1000.0; // ms

    std::cout << "  CE-backward [" << B << "x" << C << "]  per-call=" << std::fixed << std::setprecision(2)
              << us_per << " us   (" << iters << " iters)   ~" << std::setprecision(2) << mnist_per_epoch
              << " ms/epoch @468batch  (sink=" << sink << ")" << std::endl;
}

int main() {
    std::cout << "=== eager CrossEntropy backward micro-benchmark ===\n";
    std::cout << "  (softmax + sub + mul, per-call cost, ROI probe for C3 fusion)\n";
    bench_shape(128, 10, 2000);
    bench_shape(128, 100, 2000);
    bench_shape(256, 100, 2000);
    bench_shape(256, 1000, 1000);
    bench_shape(512, 1000, 1000);
    bench_shape(1024, 1000, 500);
    return 0;
}
