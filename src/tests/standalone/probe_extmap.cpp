/**
 * @file probe_extmap.cpp
 * @brief 临时探针:实证 MatMul backward 的 ext_map 协议
 * @details 直接驱动 C3BackwardCapture::compileBackwardAsync 编译 MatMul backward kernel,
 *          触发 HandwrittenKernelGen::generateMultiNodeKernel 的 [PROBE-5D] 探针,
 *          观察 graph.inputs() 顺序 + external_input_map + 每个 compute node 输入指针映射,
 *          确认建图端(graph inputs) 与执行端(kernel inputs 组装)是否对齐。
 * @date 2026-08-11
 */

#include <iostream>
#include <thread>
#include <chrono>

#include "Tensor.h"
#include "AutoGrad.h"
#include "C3/C3BackwardCapture.h"
#include "C3/C3KernelRegistry.h"
#include "C3/C3Engine.h"

using namespace ct;
using namespace ct::c3;

int main() {
    std::cout << "=== PROBE-5D: MatMul backward ext_map 协议实证 ===" << std::endl;

    const size_t M = 4, K = 3, N = 5;

    // 构造 MatMul 前向节点 y = x @ w
    Tensor x(ShapeTag{}, {M, K}, DType::kFloat, DeviceType::kCPU);
    float* xp = x.data_write<float>();
    for (size_t i = 0; i < M * K; ++i) xp[i] = static_cast<float>(i % 7) * 0.1f;
    Tensor w(ShapeTag{}, {K, N}, DType::kFloat, DeviceType::kCPU);
    float* wp = w.data_write<float>();
    for (size_t i = 0; i < K * N; ++i) wp[i] = static_cast<float>(i % 5) * 0.2f;
    x.requires_grad(true);
    w.requires_grad(true);
    Tensor y = x.matmul(w);
    auto node = y.getRelatedNode();

    // dL/dY = ones(M,N)
    Tensor grad(ShapeTag{}, {M, N}, DType::kFloat, DeviceType::kCPU);
    float* gp = grad.data_write<float>();
    for (size_t i = 0; i < M * N; ++i) gp[i] = 1.0f;

    std::cout << "MatMul node inputs: " << node->getInputs().size() << " 条" << std::endl;
    std::cout << "forward_inputs 顺序: [x(M,K), w(K,N)] → kernel 端组装为 [grad, x, w]" << std::endl;

    std::cout << "触发 compileBackwardAsync(grad=" << M << "x" << N << ")..." << std::endl;
    C3BackwardCapture::getInstance().compileBackwardAsync(node.get(), grad);

    // 等待异步编译(含探针打印)完成
    std::cout << "等待异步编译 3s..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));

    auto stats = C3BackwardCapture::getInstance().getStats();
    std::cout << "\n编译统计: compiles=" << stats.compile_count
              << " hits=" << stats.cache_hit_count
              << " misses=" << stats.cache_miss_count << std::endl;

    // 看 key 是否安装(Transpose 无 handler 时编译会 throw,预期不安装)
    std::cout << "MatMul backward key 是否安装: "
              << "(见上方 [PROBE-5D] 探针输出确认 ext_map 协议)" << std::endl;

    C3KernelRegistry::getInstance().uninstallAll();
    C3Engine::getInstance().shutdown();
    C3Engine::getInstance().clearCache();
    std::cout << "=== PROBE-5D done ===" << std::endl;
    return 0;
}