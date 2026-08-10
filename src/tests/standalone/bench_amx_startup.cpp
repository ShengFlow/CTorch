/**
 * @file bench_amx_startup.cpp
 * @brief AMX 启动开销基准测试
 * @details 实验目的：验证 cblas_sgemm (AMX) 在小矩阵上是否存在显著的启动开销，
 *          确认朴素三重循环在何种矩阵规模下更快。
 *
 *          实验设计（遵循 main.md 假设-实验-裁决闭环）：
 *          (H) cblas_sgemm 在小矩阵上有显著的 AMX 启动开销，三重循环可能更快
 *          (PREDICTION) 存在 crossover point，低于该阈值时三重循环占优
 *          (EXP) 测试 4×4×4 → 256×256×256 的矩阵规模，对比两种实现
 *          (OBSERVATION) 收集每个规模下的平均执行时间
 *          (VERDICT) 确认 crossover point 的位置
 *
 * @date 2026/8/4
 */

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <Accelerate/Accelerate.h>

using hires = std::chrono::high_resolution_clock;
using us = std::chrono::duration<double, std::micro>;

// ======================= 朴素三重循环 MatMul =======================

static void naive_matmul(const float* a, const float* b, float* c,
                          int M, int K, int N) {
    // 初始化 c 为 0
    for (int i = 0; i < M * N; ++i) c[i] = 0.0f;

    // 三重循环: for i → for j → for k
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

// ======================= 分块三重循环 MatMul（tile 32） =======================

static void tiled_matmul_impl(const float* a, const float* b, float* c,
                               int M, int K, int N, int tile_m = 32, int tile_n = 32) {
    for (int i = 0; i < M * N; ++i) c[i] = 0.0f;

    for (int i0 = 0; i0 < M; i0 += tile_m) {
        int i1 = std::min(i0 + tile_m, M);
        for (int j0 = 0; j0 < N; j0 += tile_n) {
            int j1 = std::min(j0 + tile_n, N);
            for (int i = i0; i < i1; ++i) {
                for (int j = j0; j < j1; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += a[i * K + k] * b[k * N + j];
                    }
                    c[i * N + j] = sum;
                }
            }
        }
    }
}

// 无默认参数的包装，用于函数指针
static void tiled_matmul(const float* a, const float* b, float* c,
                          int M, int K, int N) {
    tiled_matmul_impl(a, b, c, M, K, N, 32, 32);
}

// ======================= cblas_sgemm 包装 =======================

static void blas_matmul(const float* a, const float* b, float* c,
                         int M, int K, int N) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, a, K, b, N, 0.0f, c, N);
#pragma clang diagnostic pop
}

// ======================= 基准测试工具 =======================

struct BenchResult {
    std::string label;
    double avg_us;
    double min_us;
    double max_us;
    double cold_us;  // 首次调用（冷启动）时间
};

static BenchResult run_bench(const std::string& label,
                             void (*kernel)(const float*, const float*, float*, int, int, int),
                             const float* a, const float* b, float* c,
                             int M, int K, int N,
                             int warmup = 5, int iterations = 50) {
    BenchResult r;
    r.label = label;

    // 冷启动测量（单独分配 buffer 避免 cache 影响）
    float* cold_c = new float[M * N];
    auto t0 = hires::now();
    kernel(a, b, cold_c, M, K, N);
    auto t1 = hires::now();
    r.cold_us = std::chrono::duration_cast<us>(t1 - t0).count();
    delete[] cold_c;

    // 预热
    for (int i = 0; i < warmup; ++i) {
        kernel(a, b, c, M, K, N);
    }

    // 稳态测量
    r.min_us = 1e9;
    r.max_us = 0;
    double sum = 0;
    for (int i = 0; i < iterations; ++i) {
        t0 = hires::now();
        kernel(a, b, c, M, K, N);
        t1 = hires::now();
        double elapsed = std::chrono::duration_cast<us>(t1 - t0).count();
        sum += elapsed;
        if (elapsed < r.min_us) r.min_us = elapsed;
        if (elapsed > r.max_us) r.max_us = elapsed;
    }
    r.avg_us = sum / iterations;

    return r;
}

// ======================= 主程序 =======================

struct TestSize {
    int M, K, N;
    const char* label;
};

int main() {
    std::cout << "================================================" << std::endl;
    std::cout << " AMX 启动开销基准测试" << std::endl;
    std::cout << " 对比: cblas_sgemm vs 朴素三重循环 vs 分块三重循环" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << std::endl;

    // 测试矩阵规模（覆盖 MLIR 后端的三个阈值区间）
    // 小矩阵: total_ops < 4096
    // 中等矩阵: 4096 ≤ total_ops < 131072
    // 大矩阵: total_ops ≥ 131072
    std::vector<TestSize> sizes = {
        // 小矩阵区间
        {4, 4, 4, "4×4×4"},
        {8, 8, 8, "8×8×8"},
        {12, 12, 12, "12×12×12"},
        {16, 16, 16, "16×16×16"},
        // 阈值附近
        {20, 20, 20, "20×20×20"},
        {24, 24, 24, "24×24×24"},
        // 中等矩阵区间
        {32, 32, 32, "32×32×32"},
        {40, 40, 40, "40×40×40"},
        {48, 48, 48, "48×48×48"},
        {64, 64, 64, "64×64×64"},
        {96, 96, 96, "96×96×96"},
        // 大矩阵区间
        {128, 128, 128, "128×128×128"},
        {192, 192, 192, "192×192×192"},
        {256, 256, 256, "256×256×256"},
        // 非对称矩阵（常见 MLP 形状）
        {32, 128, 64, "32×128×64"},
        {64, 256, 128, "64×256×128"},
        {128, 512, 256, "128×512×256"},
    };

    // 表头
    std::cout << std::left
              << std::setw(18) << "矩阵规模"
              << std::setw(12) << "total_ops"
              << std::setw(14) << "cblas_sgemm"
              << std::setw(14) << "cblas(冷启动)"
              << std::setw(14) << "朴素三重循环"
              << std::setw(14) << "分块三重循环"
              << std::setw(12) << "最优方案"
              << std::endl;
    std::cout << std::string(18 + 12 + 14*4 + 12, '-') << std::endl;

    for (const auto& sz : sizes) {
        int M = sz.M, K = sz.K, N = sz.N;
        int64_t total_ops = (int64_t)M * K * N;

        // 分配矩阵
        float* a = new float[M * K];
        float* b = new float[K * N];
        float* c_blas = new float[M * N];
        float* c_naive = new float[M * N];
        float* c_tiled = new float[M * N];

        // 随机初始化
        srand(42);
        for (int i = 0; i < M * K; ++i) a[i] = (float)(rand() % 1000) / 1000.0f;
        for (int i = 0; i < K * N; ++i) b[i] = (float)(rand() % 1000) / 1000.0f;

        // 运行基准测试
        auto r_blas = run_bench("cblas_sgemm", blas_matmul, a, b, c_blas, M, K, N);
        auto r_naive = run_bench("naive", naive_matmul, a, b, c_naive, M, K, N);
        auto r_tiled = run_bench("tiled", tiled_matmul, a, b, c_tiled, M, K, N);

        // 验证正确性（blas vs naive）
        bool correct = true;
        for (int i = 0; i < M * N; ++i) {
            float diff = std::fabs(c_blas[i] - c_naive[i]);
            float max_val = std::max(std::fabs(c_blas[i]), std::fabs(c_naive[i]));
            if (diff > 1e-4f + 1e-4f * max_val) {
                correct = false;
                break;
            }
        }

        // 确定最优方案
        std::string best;
        double best_time = std::min({r_blas.avg_us, r_naive.avg_us, r_tiled.avg_us});
        if (best_time == r_blas.avg_us) best = "cblas_sgemm";
        else if (best_time == r_naive.avg_us) best = "naive";
        else best = "tiled";

        // 输出
        std::cout << std::left
                  << std::setw(18) << sz.label
                  << std::setw(12) << total_ops
                  << std::setw(14) << std::fixed << std::setprecision(3) << r_blas.avg_us
                  << std::setw(14) << std::fixed << std::setprecision(3) << r_blas.cold_us
                  << std::setw(14) << std::fixed << std::setprecision(3) << r_naive.avg_us
                  << std::setw(14) << std::fixed << std::setprecision(3) << r_tiled.avg_us
                  << std::setw(12) << best
                  << (correct ? "" : " ❌ MISMATCH")
                  << std::endl;

        // 冷启动开销分析
        if (r_blas.cold_us > r_blas.avg_us * 3) {
            std::cout << "  ⚠️  cblas_sgemm 冷启动: " << r_blas.cold_us
                      << " us (稳态的 " << (r_blas.cold_us / r_blas.avg_us)
                      << "x)" << std::endl;
        }

        delete[] a;
        delete[] b;
        delete[] c_blas;
        delete[] c_naive;
        delete[] c_tiled;
    }

    std::cout << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << " 实验完成" << std::endl;
    std::cout << "================================================" << std::endl;

    return 0;
}