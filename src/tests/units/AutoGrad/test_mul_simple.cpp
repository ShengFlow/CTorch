#include "AutoGrad.h"
#include "Tensor.h"
#include "CtorchError.h"
#include <iostream>
#include <cmath>

/**
 * @brief 简单的 MulNode 测试
 * @details 测试 c = a * b 的梯度计算
 */
bool test_mul_node_simple() {
    try {
        std::cout << "=== 测试 MulNode 简单案例 ===" << std::endl;

        AutoGrad::EnableGrad = true;

        Tensor a(2.0f);
        Tensor b(3.0f);

        a.requires_grad(true);
        b.requires_grad(true);

        std::cout << "输入: a = " << a.item<float>() << ", b = " << b.item<float>() << std::endl;

        Tensor c = a * b;
        std::cout << "输出: c = a * b = " << c.item<float>() << std::endl;

        AutoGrad::backward(c.getRelatedNode(), false);

        Tensor grad_a = a.grad();
        Tensor grad_b = b.grad();

        std::cout << "梯度: grad_a = " << grad_a.item<float>() << ", grad_b = " << grad_b.item<float>() << std::endl;

        if (std::abs(grad_a.item<float>() - 3.0f) > 1e-6) {
            std::cout << "❌ grad_a 错误: 期望 3.0, 实际 " << grad_a.item<float>() << std::endl;
            return false;
        }

        if (std::abs(grad_b.item<float>() - 2.0f) > 1e-6) {
            std::cout << "❌ grad_b 错误: 期望 2.0, 实际 " << grad_b.item<float>() << std::endl;
            return false;
        }

        std::cout << "✅ MulNode 反向传播测试通过!" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cout << "!!! 测试异常: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    std::cout << "🚀 开始测试" << std::endl;
    std::cout << "====================================" << std::endl;

    bool passed = test_mul_node_simple();

    std::cout << "\n====================================" << std::endl;
    if (passed) {
        std::cout << "🎉 测试通过!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ 测试失败" << std::endl;
        return 1;
    }
}