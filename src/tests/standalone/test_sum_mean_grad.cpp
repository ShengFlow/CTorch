/**
 * @file test_sum_mean_grad.cpp
 * @brief sum()/mean() 反向传播梯度回归测试
 * @details 修复 2026-09-06: sum() 经 dot(ones) 挂 DotNode(从未实现) / mean() 裸循环无节点,
 *          两者 backward 均静默断链。修复后 sum()/mean() 挂 SumNode/MeanNode。
 *          本测试验证: 全 reduce sum/mean 的梯度正确广播、链式传播正常。
 * @date 2026-09-06
 */
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "Tensor.h"
#include "AutoGrad.h"

using namespace ct;

static int g_fails = 0;
#define CHECK(cond, msg)                                    \
    do {                                                    \
        if (!(cond)) {                                      \
            std::cerr << "FAIL: " << msg << "\n";           \
            ++g_fails;                                      \
        } else {                                            \
            std::cout << "PASS: " << msg << "\n";           \
        }                                                   \
    } while (0)

int main() {
    // Test 1: sum 前向值 + 梯度全 1
    {
        Tensor x(ShapeTag{}, {3, 4}, DType::kFloat, DeviceType::kCPU);
        float* p = x.data_write<float>();
        for (int i = 0; i < 12; ++i) p[i] = i * 0.5f;  // sum = 0..5.5 = 33
        x.requires_grad(true);
        Tensor l = x.sum();
        CHECK(std::abs(l.item<float>() - 33.0f) < 1e-4f, "sum 前向值 = 33");
        AutoGrad::backward(l.getRelatedNode(), false);
        const float* g = x.grad_ptr();
        CHECK(g != nullptr, "sum backward 填了梯度");
        bool ok = (g != nullptr);
        for (int i = 0; ok && i < 12; ++i) ok = (g[i] == 1.0f);
        CHECK(ok, "sum 梯度全 1");
    }

    // Test 2: mean 梯度全 1/n
    {
        Tensor x(ShapeTag{}, {3, 4}, DType::kFloat, DeviceType::kCPU);
        float* p = x.data_write<float>();
        for (int i = 0; i < 12; ++i) p[i] = 1.0f + i;
        x.requires_grad(true);
        Tensor l = x.mean();
        AutoGrad::backward(l.getRelatedNode(), false);
        const float* g = x.grad_ptr();
        CHECK(g != nullptr, "mean backward 填了梯度");
        bool ok = (g != nullptr);
        for (int i = 0; ok && i < 12; ++i) ok = (std::abs(g[i] - 1.0f / 12.0f) < 1e-6f);
        CHECK(ok, "mean 梯度全 1/12");
    }

    // Test 3: 链式 relu→sum, 梯度 = relu mask
    {
        Tensor x(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
        float* p = x.data_write<float>();
        const float vals[6] = {-2.0f, -0.5f, 0.3f, 1.5f, -1.0f, 2.0f};
        for (int i = 0; i < 6; ++i) p[i] = vals[i];
        x.requires_grad(true);
        Tensor y = x.relu();
        Tensor l = y.sum();
        AutoGrad::backward(l.getRelatedNode(), false);
        const float* g = x.grad_ptr();
        CHECK(g != nullptr, "链式 relu→sum 梯度存在");
        bool ok = (g != nullptr);
        for (int i = 0; ok && i < 6; ++i) {
            float expect = (vals[i] > 0.0f) ? 1.0f : 0.0f;
            if (std::abs(g[i] - expect) > 1e-6f) ok = false;
        }
        CHECK(ok, "relu→sum 梯度 = relu mask");
    }

    // Test 4: sum 作 loss 后参数 SGD 更新真实发生(grad 非空且可更新)
    {
        Tensor w(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU);
        float* wp = w.data_write<float>();
        for (int i = 0; i < 4; ++i) wp[i] = 1.0f;
        w.requires_grad(true);
        Tensor x(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU);
        float* xp = x.data_write<float>();
        for (int i = 0; i < 4; ++i) xp[i] = 2.0f;
        Tensor out = x * w;          // MulNode, 梯度流到 w
        Tensor l = out.sum();
        AutoGrad::backward(l.getRelatedNode(), false);
        float* gp = w.grad_ptr();
        CHECK(gp != nullptr, "w 收到梯度(x=2 全 → grad 全 2)");
        bool ok = (gp != nullptr);
        for (int i = 0; ok && i < 4; ++i) ok = (std::abs(gp[i] - 2.0f) < 1e-6f);
        CHECK(ok, "w 梯度全 2");
    }

    std::cout << (g_fails == 0 ? "=== ALL PASS ===" : "=== HAS FAIL ===") << "\n";
    return g_fails;
}
