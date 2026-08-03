/**
 * @file bench_simd_math.cpp
 * @brief SIMDMath vs 标量版本微基准
 * @details 量化向量化数学函数在不同 size 下的加速比
 *
 * 用法：./bench_simd_math
 * 输出：表格形式展示 exp/log/tanh/sigmoid/gelu 在 N=1K/16K/256K/1M 下的加速比
 */

#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "kernels/SIMDMath.h"

using clk = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;

struct BenchResult {
    const char* name;
    size_t N;
    double scalar_ns;
    double vec_ns;
    double speedup;
};

template<typename ScalarFn, typename VecFn>
BenchResult run_bench(const char* name, size_t N, ScalarFn sf, VecFn vf, int trials = 200) {
    std::vector<float> in(N), out(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    for (size_t i = 0; i < N; ++i) in[i] = dist(rng);

    // warmup
    for (size_t i = 0; i < N; ++i) out[i] = sf(in[i]);
    vf(in.data(), out.data(), N);

    // 标量
    auto t0 = clk::now();
    for (int t = 0; t < trials; ++t) {
        for (size_t i = 0; i < N; ++i) out[i] = sf(in[i]);
    }
    auto t1 = clk::now();
    double scalar_ns = std::chrono::duration_cast<ns>(t1 - t0).count() / (double)trials;

    // 向量
    auto t2 = clk::now();
    for (int t = 0; t < trials; ++t) {
        vf(in.data(), out.data(), N);
    }
    auto t3 = clk::now();
    double vec_ns = std::chrono::duration_cast<ns>(t3 - t2).count() / (double)trials;

    return {name, N, scalar_ns, vec_ns, scalar_ns / std::max(vec_ns, 1.0)};
}

int main() {
    std::cout << "=== SIMDMath 性能 bench ===" << std::endl;
    std::cout << "Platform: ";
#if defined(__AVX2__) && defined(__FMA__)
    std::cout << "x86_64 AVX2+FMA";
#elif defined(__aarch64__)
    std::cout << "aarch64 NEON";
#else
    std::cout << "scalar fallback";
#endif
    std::cout << std::endl << std::endl;

    std::vector<size_t> sizes = {1024, 16384, 262144, 1048576};
    std::vector<BenchResult> results;

    for (size_t N : sizes) {
        std::cout << "--- N = " << N << " ---" << std::endl;

        results.push_back(run_bench("exp", N,
            [](float x) { return std::exp(x); },
            [](const float* in, float* out, size_t n) {
                ct::kernels::simd::vexp(in, out, n);
            }));

        results.push_back(run_bench("log", N,
            [](float x) { return std::log(x); },
            [](const float* in, float* out, size_t n) {
                ct::kernels::simd::vlog(in, out, n);
            }));

        results.push_back(run_bench("tanh", N,
            [](float x) { return std::tanh(x); },
            [](const float* in, float* out, size_t n) {
                ct::kernels::simd::vtanh(in, out, n);
            }));

        results.push_back(run_bench("sigmoid", N,
            [](float x) { return 1.0f / (1.0f + std::exp(-x)); },
            [](const float* in, float* out, size_t n) {
                ct::kernels::simd::vsigmoid(in, out, n);
            }));

        results.push_back(run_bench("gelu", N,
            [](float x) {
                float v = 0.7978845608f * (x + 0.044715f * x * x * x);
                return 0.5f * x * (1.0f + std::tanh(v));
            },
            [](const float* in, float* out, size_t n) {
                ct::kernels::simd::vgelu(in, out, n);
            }));
    }

    // 打印汇总
    std::cout << "\n=== 加速比汇总 ===" << std::endl;
    std::cout << std::left
              << std::setw(12) << "fn"
              << std::setw(12) << "N"
              << std::setw(15) << "scalar (us)"
              << std::setw(15) << "vec (us)"
              << std::setw(12) << "speedup"
              << "\n";
    std::cout << std::string(70, '-') << "\n";
    for (const auto& r : results) {
        std::cout << std::left
                  << std::setw(12) << r.name
                  << std::setw(12) << r.N
                  << std::setw(15) << std::fixed << std::setprecision(2) << (r.scalar_ns / 1000.0)
                  << std::setw(15) << (r.vec_ns / 1000.0)
                  << std::setw(12) << std::setprecision(2) << r.speedup << "x"
                  << "\n";
    }

    // 计算平均加速比
    double sum = 0;
    for (const auto& r : results) sum += r.speedup;
    std::cout << "\n平均加速比: " << (sum / results.size()) << "x\n";

    return 0;
}
