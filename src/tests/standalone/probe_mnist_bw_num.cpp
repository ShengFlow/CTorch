/**
 * @file probe_mnist_bw_num.cpp
 * @brief 临时探针:用 MNIST 精确形状验证 C3 backward 单 kernel 数值正确性
 * @details 对 MatMul backward 与 ReLU backward 用 MNIST 的真实形状
 *          ([128,784]x[784,256] / ReLU [128,256] 等) 对比 eager 与 C3 梯度。
 * @date 2026-08-11
 */

#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>

#include "Tensor.h"
#include "AutoGrad.h"
#include "C3/C3BackwardCapture.h"
#include "C3/C3KernelRegistry.h"
#include "C3/C3Engine.h"

using namespace ct;
using namespace ct::c3;

static double maxDiff(const Tensor& a, const Tensor& b) {
    if (a.numel() != b.numel()) return 1e9;
    const float* ap = a.data_read<float>();
    const float* bp = b.data_read<float>();
    double m = 0;
    for (size_t i = 0; i < a.numel(); ++i) {
        double d = std::fabs((double)ap[i] - (double)bp[i]);
        if (d > m) m = d;
    }
    return m;
}

// 跑一次 MatMul 前向反向, 返回 [grad_A, grad_B] (按 eager order)
static std::vector<Tensor> runMatMulBwd(
    size_t M, size_t K, size_t N, bool c3_mode, unsigned seed) {
    Tensor x(ShapeTag{}, {M, K}, DType::kFloat, DeviceType::kCPU);
    Tensor w(ShapeTag{}, {K, N}, DType::kFloat, DeviceType::kCPU);
    float* xp = x.data_write<float>();
    float* wp = w.data_write<float>();
    unsigned s = seed;
    for (size_t i = 0; i < M * K; ++i) { s = s * 1103515245 + 12345; xp[i] = (float)((s >> 16) & 0x7fff) / 32768.0f; }
    for (size_t i = 0; i < K * N; ++i) { s = s * 1103515245 + 12345; wp[i] = (float)((s >> 16) & 0x7fff) / 32768.0f; }
    x.requires_grad(true); w.requires_grad(true);
    Tensor y = x.matmul(w);
    AutoGrad::backward(y.getRelatedNode(), false);
    return {x.grad(), w.grad()};
}

// 跑一次 ReLU 反向
static Tensor runReluBwd(size_t M, size_t N, bool c3_mode, unsigned seed) {
    Tensor x(ShapeTag{}, {M, N}, DType::kFloat, DeviceType::kCPU);
    float* xp = x.data_write<float>();
    unsigned s = seed;
    for (size_t i = 0; i < M * N; ++i) { s = s * 1103515245 + 12345; xp[i] = (float)(((int)((s >> 16) & 0x7fff) % 2001) - 1000) / 1000.0f; }
    x.requires_grad(true);
    Tensor y = x.relu();
    AutoGrad::backward(y.getRelatedNode(), false);
    return x.grad();
}

int main() {
    std::cout << "=== PROBE: MNIST 形状 C3 backward 数值正确性 ===" << std::endl;

    // 触发 C3 编译 (异步)
    std::cout << "触发 C3 编译并等待 3s..." << std::endl;

    // 逐场景:先 eager 参考, 再 c3 (多次以命中)
    // 注意:ref 与 got 必须用同一 seed 生成「完全相同」的 x/w 输入,
    //   否则比较的是不同输入的梯度, maxDiff 是假阳性。
    //   ref 是第一个调用(编译未完成 → eager), got 是后续调用(C3 命中)。
    auto testMatMul = [&](size_t M, size_t K, size_t N, int name) {
        auto ref = runMatMulBwd(M, K, N, false, name);
        double gx_ref = maxDiff(ref[0], ref[0]);
        double gx_max = 0, gw_max = 0;
        // 触发编译
        for (int it = 0; it < 3; ++it) runMatMulBwd(M, K, N, true, name + it * 1000);
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        for (int it = 0; it < 6; ++it) {
            auto got = runMatMulBwd(M, K, N, true, name); // 同一 seed → 同一输入
            gx_max = std::max(gx_max, maxDiff(ref[0], got[0]));
            gw_max = std::max(gw_max, maxDiff(ref[1], got[1]));
        }
        std::cout << "[MatMul " << name << "] shape=[" << M << "," << K << "]x[" << K << "," << N
                  << "]  grad_x_max=" << gx_max << " grad_w_max=" << gw_max
                  << (gx_max < 1e-3 && gw_max < 1e-3 ? "  OK" : "  BAD") << std::endl;
        // 打印 C3 命中后的前几个值做诊断
        {
            auto got = runMatMulBwd(M, K, N, true, name);
            std::cout << "  diag ref[0]=";
            for (int i = 0; i < 4; ++i) std::cout << ref[0].data_read<float>()[i] << " ";
            std::cout << " got[0]=";
            for (int i = 0; i < 4; ++i) std::cout << got[0].data_read<float>()[i] << " ";
            std::cout << " ref[1]=";
            for (int i = 0; i < 4; ++i) std::cout << ref[1].data_read<float>()[i] << " ";
            std::cout << " got[1]=";
            for (int i = 0; i < 4; ++i) std::cout << got[1].data_read<float>()[i] << " " << std::endl;
        }
    };
    auto testRelu = [&](size_t M, size_t N, int name) {
        auto ref = runReluBwd(M, N, false, name);
        for (int it = 0; it < 3; ++it) runReluBwd(M, N, true, name + it * 1000);
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        double gx_max = 0;
        for (int it = 0; it < 6; ++it) {
            auto got = runReluBwd(M, N, true, name); // 同一 seed → 同一输入
            gx_max = std::max(gx_max, maxDiff(ref, got));
        }
        std::cout << "[ReLU " << name << "] shape=[" << M << "," << N << "]  grad_x_max=" << gx_max
                  << (gx_max < 1e-3 ? "  OK" : "  BAD") << std::endl;
    };

    testMatMul(128, 784, 256, 1);
    testMatMul(128, 256, 128, 2);
    testMatMul(128, 128, 10, 3);
    testRelu(128, 256, 1);
    testRelu(128, 128, 2);

    auto stats = C3BackwardCapture::getInstance().getStats();
    std::cout << "\nC3 stats: hits=" << stats.cache_hit_count
              << " misses=" << stats.cache_miss_count
              << " compiles=" << stats.compile_count << std::endl;

    C3KernelRegistry::getInstance().uninstallAll();
    C3Engine::getInstance().shutdown();
    C3Engine::getInstance().clearCache();
    std::cout << "=== PROBE done ===" << std::endl;
    return 0;
}