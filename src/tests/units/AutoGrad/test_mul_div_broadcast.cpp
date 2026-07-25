#include "AutoGrad.h"
#include "Tensor.h"
#include "CtorchError.h"
#include <iostream>
#include <cmath>

bool test_mul_broadcast_scalar_vector() {
    try {
        std::cout << "=== 测试 MulNode 广播: 标量 × 向量 ===" << std::endl;
        
        Tensor a(2.0f);
        Tensor b(ShapeTag{}, {3}, DType::kFloat, DeviceType::kCPU);
        b.data<float>()[0] = 1.0f;
        b.data<float>()[1] = 2.0f;
        b.data<float>()[2] = 3.0f;
        
        a.requires_grad(true);
        b.requires_grad(true);
        
        std::cout << "输入: a = " << a.item<float>() << ", b = [" << b.data<float>()[0] << ", " << b.data<float>()[1] << ", " << b.data<float>()[2] << "]" << std::endl;
        
        Tensor c = a * b;
        
        std::cout << "输出: c = [" << c.data<float>()[0] << ", " << c.data<float>()[1] << ", " << c.data<float>()[2] << "]" << std::endl;
        
        if (std::abs(c.data<float>()[0] - 2.0f) > 1e-6 ||
            std::abs(c.data<float>()[1] - 4.0f) > 1e-6 ||
            std::abs(c.data<float>()[2] - 6.0f) > 1e-6) {
            std::cout << "❌ 前向计算错误" << std::endl;
            return false;
        }
        
        AutoGrad::backward(c.getRelatedNode(), false);
        
        Tensor grad_a = a.grad();
        Tensor grad_b = b.grad();
        
        std::cout << "梯度: grad_a = " << grad_a.item<float>() << ", grad_b = [" << grad_b.data<float>()[0] << ", " << grad_b.data<float>()[1] << ", " << grad_b.data<float>()[2] << "]" << std::endl;
        
        float expected_grad_a = 1.0f + 2.0f + 3.0f;
        if (std::abs(grad_a.item<float>() - expected_grad_a) > 1e-6) {
            std::cout << "❌ grad_a 错误: 期望 " << expected_grad_a << ", 实际 " << grad_a.item<float>() << std::endl;
            return false;
        }
        
        if (std::abs(grad_b.data<float>()[0] - 2.0f) > 1e-6 ||
            std::abs(grad_b.data<float>()[1] - 2.0f) > 1e-6 ||
            std::abs(grad_b.data<float>()[2] - 2.0f) > 1e-6) {
            std::cout << "❌ grad_b 错误" << std::endl;
            return false;
        }
        
        std::cout << "✅ MulNode 标量×向量广播测试通过!" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "!!! 测试异常: " << e.what() << std::endl;
        return false;
    }
}

bool test_mul_broadcast_scalar_matrix() {
    try {
        std::cout << "\n=== 测试 MulNode 广播: 标量 × 矩阵 ===" << std::endl;
        
        Tensor a(2.0f);
        
        Tensor b(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
        b.data<float>()[0] = 1.0f; b.data<float>()[1] = 2.0f; b.data<float>()[2] = 3.0f;
        b.data<float>()[3] = 4.0f; b.data<float>()[4] = 5.0f; b.data<float>()[5] = 6.0f;
        
        a.requires_grad(true);
        b.requires_grad(true);
        
        std::cout << "输入: a = " << a.item<float>() << std::endl;
        std::cout << "输入: b = [[1,2,3],[4,5,6]]" << std::endl;
        
        Tensor c = a * b;
        
        std::cout << "输出: c = [[2,4,6],[8,10,12]]" << std::endl;
        
        if (std::abs(c.data<float>()[0] - 2.0f) > 1e-6 ||
            std::abs(c.data<float>()[1] - 4.0f) > 1e-6 ||
            std::abs(c.data<float>()[2] - 6.0f) > 1e-6 ||
            std::abs(c.data<float>()[3] - 8.0f) > 1e-6 ||
            std::abs(c.data<float>()[4] - 10.0f) > 1e-6 ||
            std::abs(c.data<float>()[5] - 12.0f) > 1e-6) {
            std::cout << "❌ 前向计算错误" << std::endl;
            return false;
        }
        
        AutoGrad::backward(c.getRelatedNode(), false);
        
        Tensor grad_a = a.grad();
        Tensor grad_b = b.grad();
        
        std::cout << "梯度: grad_a = " << grad_a.item<float>() << std::endl;
        std::cout << "梯度: grad_b = [[2,2,2],[2,2,2]]" << std::endl;
        
        float expected_grad_a = 1.0f + 2.0f + 3.0f + 4.0f + 5.0f + 6.0f;
        
        if (std::abs(grad_a.item<float>() - expected_grad_a) > 1e-6) {
            std::cout << "❌ grad_a 错误: 期望 " << expected_grad_a << ", 实际 " << grad_a.item<float>() << std::endl;
            return false;
        }
        
        for (int i = 0; i < 6; i++) {
            if (std::abs(grad_b.data<float>()[i] - 2.0f) > 1e-6) {
                std::cout << "❌ grad_b[" << i << "] 错误: 期望 2.0, 实际 " << grad_b.data<float>()[i] << std::endl;
                return false;
            }
        }
        
        std::cout << "✅ MulNode 标量×矩阵广播测试通过!" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "!!! 测试异常: " << e.what() << std::endl;
        return false;
    }
}

bool test_div_broadcast_scalar_vector() {
    try {
        std::cout << "\n=== 测试 DivNode 广播: 向量 ÷ 标量 ===" << std::endl;
        
        Tensor a(ShapeTag{}, {3}, DType::kFloat, DeviceType::kCPU);
        a.data<float>()[0] = 6.0f;
        a.data<float>()[1] = 8.0f;
        a.data<float>()[2] = 10.0f;
        
        Tensor b(2.0f);
        
        a.requires_grad(true);
        b.requires_grad(true);
        
        std::cout << "输入: a = [" << a.data<float>()[0] << ", " << a.data<float>()[1] << ", " << a.data<float>()[2] << "], b = " << b.item<float>() << std::endl;
        
        Tensor c = a / b;
        
        std::cout << "输出: c = [" << c.data<float>()[0] << ", " << c.data<float>()[1] << ", " << c.data<float>()[2] << "]" << std::endl;
        
        if (std::abs(c.data<float>()[0] - 3.0f) > 1e-6 ||
            std::abs(c.data<float>()[1] - 4.0f) > 1e-6 ||
            std::abs(c.data<float>()[2] - 5.0f) > 1e-6) {
            std::cout << "❌ 前向计算错误" << std::endl;
            return false;
        }
        
        AutoGrad::backward(c.getRelatedNode(), false);
        
        Tensor grad_a = a.grad();
        Tensor grad_b = b.grad();
        
        std::cout << "梯度: grad_a = [" << grad_a.data<float>()[0] << ", " << grad_a.data<float>()[1] << ", " << grad_a.data<float>()[2] << "]" << std::endl;
        std::cout << "梯度: grad_b = " << grad_b.item<float>() << std::endl;
        
        float expected_grad_b = -(6.0f/(2.0f*2.0f) + 8.0f/(2.0f*2.0f) + 10.0f/(2.0f*2.0f));
        
        for (int i = 0; i < 3; i++) {
            if (std::abs(grad_a.data<float>()[i] - 0.5f) > 1e-6) {
                std::cout << "❌ grad_a[" << i << "] 错误: 期望 0.5, 实际 " << grad_a.data<float>()[i] << std::endl;
                return false;
            }
        }
        
        if (std::abs(grad_b.item<float>() - expected_grad_b) > 1e-6) {
            std::cout << "❌ grad_b 错误: 期望 " << expected_grad_b << ", 实际 " << grad_b.item<float>() << std::endl;
            return false;
        }
        
        std::cout << "✅ DivNode 向量÷标量广播测试通过!" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "!!! 测试异常: " << e.what() << std::endl;
        return false;
    }
}

bool test_div_broadcast_vector_scalar() {
    try {
        std::cout << "\n=== 测试 DivNode 广播: 标量 ÷ 向量 ===" << std::endl;
        
        Tensor a(12.0f);
        Tensor b(ShapeTag{}, {3}, DType::kFloat, DeviceType::kCPU);
        b.data<float>()[0] = 2.0f;
        b.data<float>()[1] = 3.0f;
        b.data<float>()[2] = 4.0f;
        
        a.requires_grad(true);
        b.requires_grad(true);
        
        std::cout << "输入: a = " << a.item<float>() << ", b = [" << b.data<float>()[0] << ", " << b.data<float>()[1] << ", " << b.data<float>()[2] << "]" << std::endl;
        
        Tensor c = a / b;
        
        std::cout << "输出: c = [" << c.data<float>()[0] << ", " << c.data<float>()[1] << ", " << c.data<float>()[2] << "]" << std::endl;
        
        if (std::abs(c.data<float>()[0] - 6.0f) > 1e-6 ||
            std::abs(c.data<float>()[1] - 4.0f) > 1e-6 ||
            std::abs(c.data<float>()[2] - 3.0f) > 1e-6) {
            std::cout << "❌ 前向计算错误" << std::endl;
            return false;
        }
        
        AutoGrad::backward(c.getRelatedNode(), false);
        
        Tensor grad_a = a.grad();
        Tensor grad_b = b.grad();
        
        std::cout << "梯度: grad_a = " << grad_a.item<float>() << std::endl;
        std::cout << "梯度: grad_b = [" << grad_b.data<float>()[0] << ", " << grad_b.data<float>()[1] << ", " << grad_b.data<float>()[2] << "]" << std::endl;
        
        float expected_grad_a = 1.0f/2.0f + 1.0f/3.0f + 1.0f/4.0f;
        
        if (std::abs(grad_a.item<float>() - expected_grad_a) > 1e-6) {
            std::cout << "❌ grad_a 错误: 期望 " << expected_grad_a << ", 实际 " << grad_a.item<float>() << std::endl;
            return false;
        }
        
        if (std::abs(grad_b.data<float>()[0] + 12.0f/(2.0f*2.0f)) > 1e-6 ||
            std::abs(grad_b.data<float>()[1] + 12.0f/(3.0f*3.0f)) > 1e-6 ||
            std::abs(grad_b.data<float>()[2] + 12.0f/(4.0f*4.0f)) > 1e-6) {
            std::cout << "❌ grad_b 错误" << std::endl;
            return false;
        }
        
        std::cout << "✅ DivNode 标量÷向量广播测试通过!" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "!!! 测试异常: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);
    
    std::cout << "🚀 开始 MulNode 和 DivNode 广播梯度测试" << std::endl;
    std::cout << "====================================" << std::endl;
    
    int passed = 0;
    int total = 4;
    
    if (test_mul_broadcast_scalar_vector()) {
        passed++;
    }
    
    if (test_mul_broadcast_scalar_matrix()) {
        passed++;
    }
    
    if (test_div_broadcast_scalar_vector()) {
        passed++;
    }
    
    if (test_div_broadcast_vector_scalar()) {
        passed++;
    }
    
    std::cout << "\n====================================" << std::endl;
    std::cout << "测试结果: " << passed << "/" << total << " 通过" << std::endl;
    
    if (passed == total) {
        std::cout << "🎉 所有测试通过!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ 部分测试失败" << std::endl;
        return 1;
    }
}