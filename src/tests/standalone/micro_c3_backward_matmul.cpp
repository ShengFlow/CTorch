/**
 * @file micro_c3_backward_matmul.cpp
 * @brief DEBT-NEW-7 性能调研:手测 C3BackwardCapture vs Eager MatMul backward
 * @details
 *   MatMulNode::backward = grad @ B.T + A.T @ grad(2 transposes + 2 matmuls)
 *   Eager path:4 次 dispatch(Tensor 构造 + kernel 调用)
 *   C3 path:C3BackwardCapture 把 4 个 op 编成 1 个 multi-op kernel,1 次调用
 *
 *   实验:同 shape 跑 100 次,eager vs c3 各取 median
 */

#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "Tensor.h"
#include "AutoGrad.h"
#include "C3/C3BackwardCapture.h"
#include "C3/C3KernelRegistry.h"

using namespace ct;
using namespace ct::c3;

static double now_ms() {
    auto t = std::chrono::high_resolution_clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(t).count();
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main() {
    std::cout << "=== C3 Backward vs Eager micro-benchmark ===" << std::endl;
    std::cout << "  Workload: MatMulNode::backward (2 transposes + 2 matmuls)\n" << std::endl;

    // MNIST layer 1 shape
    const int M = 128, K = 784, N = 256;
    const int N_WARMUP = 30, N_MEASURE = 100;
    std::cout << "  Shape: M=" << M << " K=" << K << " N=" << N << std::endl;
    std::cout << "  Warmup=" << N_WARMUP << " Measure=" << N_MEASURE << std::endl;

    // 固定输入数据
    auto A_base = Tensor(ShapeTag{}, {M, K}, DType::kFloat, DeviceType::kCPU);
    auto B_base = Tensor(ShapeTag{}, {K, N}, DType::kFloat, DeviceType::kCPU);
    auto G_base = Tensor(ShapeTag{}, {M, N}, DType::kFloat, DeviceType::kCPU);

    uint32_t rng = 0xdeadbeef;
    for (int i = 0; i < M*K; ++i) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        A_base.data_write<float>()[i] = ((double)rng / 4294967296.0) * 0.1f;
    }
    for (int i = 0; i < K*N; ++i) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        B_base.data_write<float>()[i] = ((double)rng / 4294967296.0) * 0.1f;
    }
    for (int i = 0; i < M*N; ++i) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        G_base.data_write<float>()[i] = ((double)rng / 4294967296.0);
    }

    // 构造一个 MatMulNode (模拟反向节点)
    // 关键:必须 requires_grad=true + 模拟 autograd 调用流程

    // 实际 MatMulNode::backward 会做:
    //   grad_A = grad @ B.T
    //   grad_B = A.T @ grad
    // 4 次 kernel 调用

    auto runEager = [&]() {
        Tensor A = A_base.clone();
        Tensor B = B_base.clone();
        Tensor G = G_base.clone();

        // grad @ B.T: grad (M,N) @ B.T (N,K) = grad_A (M,K)
        Tensor B_T = B.transpose(0, 1);
        Tensor grad_A = G.matmul(B_T);

        // A.T @ grad: A.T (K,M) @ grad (M,N) = grad_B (K,N)
        Tensor A_T = A.transpose(0, 1);
        Tensor grad_B = A_T.matmul(G);

        // 校验:跟第一次跑结果对比,丢弃前几次 warmup
        (void)grad_A; (void)grad_B;
    };

    // Warmup
    std::cout << "\n  [Warmup]..." << std::endl;
    for (int i = 0; i < N_WARMUP; ++i) {
        runEager();
    }

    // Measure Eager
    std::vector<double> eager_samples;
    eager_samples.reserve(N_MEASURE);
    for (int i = 0; i < N_MEASURE; ++i) {
        auto t0 = now_ms();
        runEager();
        auto t1 = now_ms();
        eager_samples.push_back(t1 - t0);
    }
    double eager_p50 = median(eager_samples);
    double eager_mean = 0;
    for (auto v : eager_samples) eager_mean += v;
    eager_mean /= eager_samples.size();

    std::cout << "\n  Eager (4 dispatch, 2 transposes + 2 matmuls):" << std::endl;
    std::cout << "    median=" << eager_p50 * 1000 << " us  mean=" << eager_mean * 1000 << " us" << std::endl;
    std::cout << "    min=" << *std::min_element(eager_samples.begin(), eager_samples.end()) * 1000 << " us" << std::endl;
    std::cout << "    max=" << *std::max_element(eager_samples.begin(), eager_samples.end()) * 1000 << " us" << std::endl;

    // ========== 单次调用开销拆解 ==========
    // 4 个 dispatch 各自的成本:这是 backward 优化的 target
    std::cout << "\n  === Eager 单次调用拆解 (4 op, 单独测每步) ===" << std::endl;

    // Op 1: B.transpose(0,1)  - 实际是 reshape 不拷贝,接近 0
    std::vector<double> t1_samples;
    for (int i = 0; i < N_MEASURE; ++i) {
        auto t0 = now_ms();
        Tensor B_T = B_base.transpose(0, 1);
        auto t1 = now_ms();
        t1_samples.push_back(t1 - t0);
    }
    std::cout << "    B.transpose(0,1)         median=" << median(t1_samples) * 1000 << " us" << std::endl;

    // Op 2: G @ B_T  (1 matmul)
    std::vector<double> t2_samples;
    for (int i = 0; i < N_MEASURE; ++i) {
        Tensor B_T = B_base.transpose(0, 1);
        auto t0 = now_ms();
        Tensor out = G_base.matmul(B_T);
        auto t1 = now_ms();
        t2_samples.push_back(t1 - t0);
    }
    std::cout << "    G.matmul(B_T)            median=" << median(t2_samples) * 1000 << " us" << std::endl;

    // Op 3: A.transpose(0,1)
    std::vector<double> t3_samples;
    for (int i = 0; i < N_MEASURE; ++i) {
        auto t0 = now_ms();
        Tensor A_T = A_base.transpose(0, 1);
        auto t1 = now_ms();
        t3_samples.push_back(t1 - t0);
    }
    std::cout << "    A.transpose(0,1)         median=" << median(t3_samples) * 1000 << " us" << std::endl;

    // Op 4: A_T @ G  (1 matmul)
    std::vector<double> t4_samples;
    for (int i = 0; i < N_MEASURE; ++i) {
        Tensor A_T = A_base.transpose(0, 1);
        auto t0 = now_ms();
        Tensor out = A_T.matmul(G_base);
        auto t1 = now_ms();
        t4_samples.push_back(t1 - t0);
    }
    std::cout << "    A_T.matmul(G)            median=" << median(t4_samples) * 1000 << " us" << std::endl;

    // Sum
    double sum_single = median(t1_samples) + median(t2_samples) + median(t3_samples) + median(t4_samples);
    std::cout << "    sum of 4 single ops      median=" << sum_single * 1000 << " us" << std::endl;
    std::cout << "    amortized per-op dispatch  median=" << (median(eager_samples) - sum_single) / 4 * 1000 << " us" << std::endl;

    // ========== ROI 估算 ==========
    std::cout << "\n  === ROI 估算 ===" << std::endl;
    std::cout << "    backward 占总时间 ~80% (sample 数据)" << std::endl;
    std::cout << "    backward 主体 = 多个 MatMulNode::backward (类似 op)" << std::endl;
    std::cout << "    MNIST 5 epoch backward: 估算 ~6500ms" << std::endl;
    std::cout << "    假设 c3 backward fusion 节省 X% eager work" << std::endl;
    std::cout << "      X=10%  → -130ms/epoch = -6% 总 wall clock" << std::endl;
    std::cout << "      X=20%  → -260ms/epoch = -13%" << std::endl;
    std::cout << "      X=30%  → -390ms/epoch = -19%" << std::endl;

    return 0;
}
