/**
 * @file test_c3_compile_and_inject.cpp
 * @brief 验证 C3 compileAndInject 热注入流程
 * @details 编译 C3 kernel → 自动注入注册表 → 调度器自动使用 C3 kernel
 * @date 2026/8/2
 */

#include <iostream>
#include <cmath>
#include <cstring>

#include "Tensor.h"
#include "CtorchScheduler.h"
#include "C3/Graph.h"
#include "C3/C3Engine.h"
#include "C3/C3Cleanup.h"
#include "C3/C3KernelRegistry.h"
#include "kernels/kernels.h"

using namespace ct;
using namespace ct::c3;

static void fillTensor(Tensor& t, const std::vector<float>& vals) {
    float* data = t.data_write<float>();
    size_t n = std::min(vals.size(), t.numel());
    for (size_t i = 0; i < n; ++i) data[i] = vals[i];
}

static bool tensorsAllClose(const Tensor& a, const Tensor& b, float eps = 1e-5f) {
    if (a.shape() != b.shape()) return false;
    const float* pa = a.data_read<float>();
    const float* pb = b.data_read<float>();
    for (size_t i = 0; i < a.numel(); ++i) {
        if (std::fabs(pa[i] - pb[i]) > eps) return false;
    }
    return true;
}

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);
    CtorchScheduler::getInstance();

    std::cout << "=== C3 compileAndInject 热注入测试 ===" << std::endl;

    int passed = 0, failed = 0;

    // ======================= 测试 1: Add 热注入 =======================
    {
        auto desc = TensorDesc::fromShape({4});
        Graph g;
        size_t x = g.addInput(desc);
        size_t y = g.addInput(desc);
        size_t a = g.addNode(AddNode{desc, desc}, {x, y}, desc);
        g.markOutput(a);

        auto& engine = C3Engine::getInstance();
        auto kernel = engine.compileAndInject(g, {});
        if (!kernel) {
            std::cout << "  FAIL: compileAndInject returned nullptr\n";
            ++failed;
        } else {
            // 通过调度器 dispatch
            Tensor ta(ShapeTag{}, {4});
            Tensor tb(ShapeTag{}, {4});
            fillTensor(ta, {1.0f, 2.0f, 3.0f, 4.0f});
            fillTensor(tb, {5.0f, 6.0f, 7.0f, 8.0f});

            auto& sched = CtorchScheduler::getInstance();
            Tensor c3_result = sched.dispatch<op::Add>(ta, tb);
            Tensor eager = ta + tb;

            if (tensorsAllClose(c3_result, eager)) {
                std::cout << "  PASS: Add 热注入 + 调度器自动 dispatch\n";
                ++passed;
            } else {
                std::cout << "  FAIL: Add 结果不匹配\n";
                ++failed;
            }
        }

        C3KernelRegistry::getInstance().uninstall(op::Add, DeviceType::kCPU);
    }

    // ======================= 测试 2: ReLU 热注入（unary） =======================
    {
        auto desc = TensorDesc::fromShape({4});
        Graph g;
        size_t x = g.addInput(desc);
        size_t r = g.addNode(ReLUNode{desc}, {x}, desc);
        g.markOutput(r);

        auto& engine = C3Engine::getInstance();
        auto kernel = engine.compileAndInject(g, {});
        if (!kernel) {
            std::cout << "  FAIL: ReLU compileAndInject returned nullptr\n";
            ++failed;
        } else {
            Tensor ta(ShapeTag{}, {4});
            fillTensor(ta, {-1.0f, 2.0f, -3.0f, 4.0f});

            auto& sched = CtorchScheduler::getInstance();
            Tensor c3_result = sched.dispatch<op::ReLU>(ta);
            Tensor eager = ta.relu();

            if (tensorsAllClose(c3_result, eager)) {
                std::cout << "  PASS: ReLU 热注入 + 调度器自动 dispatch\n";
                ++passed;
            } else {
                std::cout << "  FAIL: ReLU 结果不匹配\n";
                ++failed;
            }
        }

        C3KernelRegistry::getInstance().uninstall(op::ReLU, DeviceType::kCPU);
    }

    // ======================= 测试 3: MatMul 热注入 =======================
    {
        auto lhs_desc = TensorDesc::fromShape({2, 3});
        auto rhs_desc = TensorDesc::fromShape({3, 4});
        auto out_desc = TensorDesc::fromShape({2, 4});
        Graph g;
        size_t x = g.addInput(lhs_desc);
        size_t y = g.addInput(rhs_desc);
        size_t mm = g.addNode(MatMulNode{lhs_desc, rhs_desc}, {x, y}, out_desc);
        g.markOutput(mm);

        auto& engine = C3Engine::getInstance();
        auto kernel = engine.compileAndInject(g, {});
        if (!kernel) {
            std::cout << "  FAIL: MatMul compileAndInject returned nullptr\n";
            ++failed;
        } else {
            Tensor ta(ShapeTag{}, {2, 3});
            Tensor tb(ShapeTag{}, {3, 4});
            fillTensor(ta, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
            fillTensor(tb, {1.0f, 0.0f, 0.0f, 1.0f,
                            0.0f, 1.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 1.0f, 1.0f});

            auto& sched = CtorchScheduler::getInstance();
            Tensor c3_result = sched.dispatch<op::MatMul>(ta, tb);
            Tensor eager = ta.matmul(tb);

            if (tensorsAllClose(c3_result, eager)) {
                std::cout << "  PASS: MatMul 热注入 + 调度器自动 dispatch\n";
                ++passed;
            } else {
                std::cout << "  FAIL: MatMul 结果不匹配\n";
                ++failed;
            }
        }

        C3KernelRegistry::getInstance().uninstall(op::MatMul, DeviceType::kCPU);
    }

    // ======================= 测试 4: 注册表统计 =======================
    {
        auto desc = TensorDesc::fromShape({4});
        Graph g;
        size_t x = g.addInput(desc);
        size_t y = g.addInput(desc);
        size_t a = g.addNode(AddNode{desc, desc}, {x, y}, desc);
        g.markOutput(a);

        // 编译并安装 3 次（不同形状），验证统计
        auto& engine = C3Engine::getInstance();
        auto& registry = C3KernelRegistry::getInstance();

        // 安装 Add(4,4)
        Graph g1 = g;
        engine.compileAndInject(g1, {});
        // 执行一次以增加 hit
        Tensor ta(ShapeTag{}, {4});
        Tensor tb(ShapeTag{}, {4});
        fillTensor(ta, {1.0f, 2.0f, 3.0f, 4.0f});
        fillTensor(tb, {5.0f, 6.0f, 7.0f, 8.0f});
        CtorchScheduler::getInstance().dispatch<op::Add>(ta, tb);

        auto stats = registry.getStats();
        std::cout << "  Stats: installs=" << stats.install_count
                  << " hits=" << stats.hit_count
                  << " active=" << stats.active_entries << std::endl;

        if (stats.install_count >= 1 && stats.hit_count >= 1 && stats.active_entries >= 1) {
            std::cout << "  PASS: 注册表统计正确\n";
            ++passed;
        } else {
            std::cout << "  FAIL: 注册表统计异常\n";
            ++failed;
        }

        registry.uninstall(op::Add, DeviceType::kCPU);
    }

    // ======================= 小结 =======================
    std::cout << "\n结果: " << passed << " passed, " << failed << " failed" << std::endl;

    // 退出清理：统一释放所有 CompiledKernel/LLVM module，避免静态析构时
    // recursive_mutex / removeModule 崩溃（与 test_c3_mnist_train 一致）。
    c3::shutdownAll();

    return failed > 0 ? 1 : 0;
}