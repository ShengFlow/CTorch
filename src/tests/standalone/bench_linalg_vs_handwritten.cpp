/**
 * @file bench_linalg_vs_handwritten.cpp
 * @brief 性能对比：linalg.generic JIT kernel vs 手写 C++ 循环（同样 O3 编译）
 *
 * 验证 STATUS_CONTEXT 4.9 结论「linalg.generic 声明式逐元素是否足以替代手写分支」：
 *   - linalg：LinalgElementwiseGen 编译出的 JIT kernel
 *   - scalar：普通 for 循环（依赖 LLVM 自动向量化，与手写 MLIR IR 同源优化）
 *   - seg：显式 VL=8 分段循环（模拟主库 buildReLU/buildElementwiseBinaryVectorized 的结构）
 *
 * 公平性说明：三者最终都是同一套 LLVM O3 后端，对比的是「声明式 linalg 是否不输手写」。
 *
 * 编译/运行：见 CMake target `bench_linalg_vs_handwritten`
 * @date 2026/08/15
 */

#include "C3/LinalgElementwiseGen.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace ct::c3;

// ======================= 手写参考实现 =======================

static void scalar_relu(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = in[i] > 0.f ? in[i] : 0.f;
}
static void scalar_sigmoid(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = 1.f / (1.f + std::exp(-in[i]));
}
static void scalar_add(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

// 显式 VL=8 分段循环（模拟主库手写 vectorized 的结构）
static void seg_relu(const float* in, float* out, size_t n) {
    constexpr size_t VL = 8;
    size_t n_vec = n - (n % VL);
    for (size_t base = 0; base < n_vec; base += VL) {
        for (size_t k = 0; k < VL; ++k) {
            float v = in[base + k];
            out[base + k] = v > 0.f ? v : 0.f;
        }
    }
    for (size_t i = n_vec; i < n; ++i) {
        float v = in[i];
        out[i] = v > 0.f ? v : 0.f;
    }
}
static void seg_add(const float* a, const float* b, float* out, size_t n) {
    constexpr size_t VL = 8;
    size_t n_vec = n - (n % VL);
    for (size_t base = 0; base < n_vec; base += VL) {
        for (size_t k = 0; k < VL; ++k) out[base + k] = a[base + k] + b[base + k];
    }
    for (size_t i = n_vec; i < n; ++i) out[i] = a[i] + b[i];
}

// ======================= 计时 =======================

template <typename F>
static double benchNsPerElem(F&& fn, size_t n, size_t min_iters, double target_ms = 200.0) {
    // 预热
    fn();
    // 迭代数自适应：保证总耗时接近 target_ms
    size_t iters = min_iters;
    double elapsed = 0.0;
    for (int round = 0; round < 3; ++round) {
        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < iters; ++i) fn();
        auto t1 = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double per_iter = elapsed / static_cast<double>(iters);
        if (per_iter > 0.0 && elapsed < target_ms) {
            size_t new_iters = static_cast<size_t>(target_ms / per_iter) + 1;
            if (new_iters > iters * 8) iters = new_iters; // 最多扩 8 倍
        } else {
            break;
        }
    }
    return (elapsed * 1e6) / static_cast<double>(iters * n); // ns/elem
}

static volatile float g_sink = 0.f;

// ======================= 单个 case =======================

struct Case {
    ElementwiseOp op;
    const char* name;
    size_t num_in;
};

static void benchCase(const Case& c, size_t n) {
    std::vector<std::vector<float>> inputs(c.num_in, std::vector<float>(n));
    for (size_t i = 0; i < n; ++i) {
        inputs[0][i] = (static_cast<float>(std::rand()) / RAND_MAX) * 2.f - 1.f;
        if (c.num_in > 1) {
            inputs[1][i] = (static_cast<float>(std::rand()) / RAND_MAX) * 2.f - 1.f;
        }
    }
    std::vector<float> out(n);
    std::vector<const float*> in_ptrs;
    for (const auto& in : inputs) in_ptrs.push_back(in.data());

    // linalg kernel（编译一次，多次执行）
    LinalgElementwiseKernel linalg_kernel(c.op);
    const size_t min_iters = (n >= 1048576) ? 3 : 30;

    double linalg_ns = 0.0, scalar_ns = 0.0, seg_ns = 0.0;
    switch (c.op) {
    case ElementwiseOp::ReLU:
        linalg_ns = benchNsPerElem(
            [&] { linalg_kernel.execute(in_ptrs.data(), out.data(), n); g_sink += out[n-1]; }, n, min_iters);
        scalar_ns = benchNsPerElem(
            [&] { scalar_relu(inputs[0].data(), out.data(), n); g_sink += out[n-1]; }, n, min_iters);
        seg_ns = benchNsPerElem(
            [&] { seg_relu(inputs[0].data(), out.data(), n); g_sink += out[n-1]; }, n, min_iters);
        break;
    case ElementwiseOp::Sigmoid:
        linalg_ns = benchNsPerElem(
            [&] { linalg_kernel.execute(in_ptrs.data(), out.data(), n); g_sink += out[n-1]; }, n, min_iters);
        scalar_ns = benchNsPerElem(
            [&] { scalar_sigmoid(inputs[0].data(), out.data(), n); g_sink += out[n-1]; }, n, min_iters);
        seg_ns = scalar_ns; // sigmoid 无显式分段参考，复用标量
        break;
    case ElementwiseOp::Add:
        linalg_ns = benchNsPerElem(
            [&] { linalg_kernel.execute(in_ptrs.data(), out.data(), n); g_sink += out[n-1]; }, n, min_iters);
        scalar_ns = benchNsPerElem(
            [&] { scalar_add(inputs[0].data(), inputs[1].data(), out.data(), n); g_sink += out[n-1]; }, n, min_iters);
        seg_ns = benchNsPerElem(
            [&] { seg_add(inputs[0].data(), inputs[1].data(), out.data(), n); g_sink += out[n-1]; }, n, min_iters);
        break;
    default:
        return;
    }

    std::printf("  %-8s n=%-9zu  linalg=%8.3f  scalar=%8.3f  seg=%8.3f  (ns/elem)\n",
                c.name, n, linalg_ns, scalar_ns, seg_ns);
}

// ======================= 主入口 =======================

int main() {
    std::srand(42);
    const Case cases[] = {
        {ElementwiseOp::ReLU, "ReLU", 1},
        {ElementwiseOp::Sigmoid, "Sigmoid", 1},
        {ElementwiseOp::Add, "Add", 2},
    };
    const size_t sizes[] = {1024, 65536, 1048576, 4194304};
    constexpr int num_cases = static_cast<int>(sizeof(cases) / sizeof(cases[0]));
    constexpr int num_sizes = static_cast<int>(sizeof(sizes) / sizeof(sizes[0]));

    std::printf("==========================================================\n");
    std::printf("  linalg.generic vs 手写 C++ 循环 (同 LLVM O3)  单位: ns/elem\n");
    std::printf("==========================================================\n");
    for (int c = 0; c < num_cases; ++c) {
        for (int s = 0; s < num_sizes; ++s) {
            benchCase(cases[c], sizes[s]);
        }
    }
    std::printf("==========================================================\n");
    std::printf("  (g_sink=%f 防止死代码消除)\n", (float)g_sink);
    return 0;
}
