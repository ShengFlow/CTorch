#include "AutoGrad.h"
#include "Tensor.h"
#include "CtorchError.h"
#include <iostream>
#include <cmath>

/**
 * @brief 测试 MulNode 反向传播
 * @details 测试 c = a * b 的梯度计算
 */
bool test_mul_node_backward() {
    try {
        std::cout << "=== 测试 MulNode 反向传播 ===" << std::endl;
        
        // 创建输入张量
        Tensor a(2.0f, DeviceType::kMPS);
        Tensor b(3.0f, DeviceType::kMPS);
        a.requires_grad(true);
        b.requires_grad(true);
        
        std::cout << "输入: a = " << a.item<float>() << ", b = " << b.item<float>() << std::endl;
        
        // 执行乘法操作
        Tensor c = a * b;
        std::cout << "输出: c = a * b = " << c.item<float>() << std::endl;
        
        // 验证前向计算结果
        if (std::abs(c.item<float>() - 6.0f) > 1e-6) {
            std::cout << "❌ 前向计算错误: 期望 6.0, 实际 " << c.item<float>() << std::endl;
            return false;
        }
        
        // 执行反向传播
        AutoGrad::backward(c.getRelatedNode(), false);
        
        // 获取梯度
        Tensor grad_a = a.grad();
        Tensor grad_b = b.grad();
        
        std::cout << "梯度: grad_a = " << grad_a.item<float>() << ", grad_b = " << grad_b.item<float>() << std::endl;
        
        // 验证梯度
        // 理论值: ∂c/∂a = b = 3.0, ∂c/∂b = a = 2.0
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

/**
 * @brief 测试 MatMul 反向传播
 * @details 测试矩阵乘法的梯度计算
 */
bool test_matmul_node_backward() {
    try {
        std::cout << "\n=== 测试 MatMul 反向传播 ===" << std::endl;
        
        // 创建矩阵
        Tensor A(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kMPS);
        Tensor B(ShapeTag{}, {3, 2}, DType::kFloat, DeviceType::kMPS);
        
        // 填充数据
        float* A_data = A.data<float>();
        float* B_data = B.data<float>();
        
        // A = [[1, 2, 3], [4, 5, 6]]
        A_data[0] = 1.0f; A_data[1] = 2.0f; A_data[2] = 3.0f;
        A_data[3] = 4.0f; A_data[4] = 5.0f; A_data[5] = 6.0f;
        
        // B = [[7, 8], [9, 10], [11, 12]]
        B_data[0] = 7.0f; B_data[1] = 8.0f;
        B_data[2] = 9.0f; B_data[3] = 10.0f;
        B_data[4] = 11.0f; B_data[5] = 12.0f;
        
        A.requires_grad(true);
        B.requires_grad(true);
        
        std::cout << "矩阵 A: [[1, 2, 3], [4, 5, 6]]" << std::endl;
        std::cout << "矩阵 B: [[7, 8], [9, 10], [11, 12]]" << std::endl;
        
        // 执行矩阵乘法
        Tensor C = A.matmul(B);
        
        std::cout << "矩阵 C = A * B:" << std::endl;
        std::cout << "[[" << C.data<float>()[0] << ", " << C.data<float>()[1] << "]" << std::endl;
        std::cout << " [" << C.data<float>()[2] << ", " << C.data<float>()[3] << "]]" << std::endl;
        
        // 验证前向计算结果
        // 期望: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //      = [[7+18+33, 8+20+36], [28+45+66, 32+50+72]]
        //      = [[58, 64], [139, 154]]
        if (std::abs(C.data<float>()[0] - 58.0f) > 1e-6) {
            std::cout << "❌ 前向计算错误: C[0,0] 期望 58.0, 实际 " << C.data<float>()[0] << std::endl;
            return false;
        }
        
        if (std::abs(C.data<float>()[1] - 64.0f) > 1e-6) {
            std::cout << "❌ 前向计算错误: C[0,1] 期望 64.0, 实际 " << C.data<float>()[1] << std::endl;
            return false;
        }
        
        if (std::abs(C.data<float>()[2] - 139.0f) > 1e-6) {
            std::cout << "❌ 前向计算错误: C[1,0] 期望 139.0, 实际 " << C.data<float>()[2] << std::endl;
            return false;
        }
        
        if (std::abs(C.data<float>()[3] - 154.0f) > 1e-6) {
            std::cout << "❌ 前向计算错误: C[1,1] 期望 154.0, 实际 " << C.data<float>()[3] << std::endl;
            return false;
        }
        
        // 执行反向传播
        AutoGrad::backward(C.getRelatedNode(), false);
        
        // 获取梯度
        Tensor grad_A = A.grad();
        Tensor grad_B = B.grad();
        
        std::cout << "\n梯度 grad_A:" << std::endl;
        std::cout << "[[" << grad_A.data<float>()[0] << ", " << grad_A.data<float>()[1] << ", " << grad_A.data<float>()[2] << "]" << std::endl;
        std::cout << " [" << grad_A.data<float>()[3] << ", " << grad_A.data<float>()[4] << ", " << grad_A.data<float>()[5] << "]]" << std::endl;
        
        std::cout << "梯度 grad_B:" << std::endl;
        std::cout << "[[" << grad_B.data<float>()[0] << ", " << grad_B.data<float>()[1] << "]" << std::endl;
        std::cout << " [" << grad_B.data<float>()[2] << ", " << grad_B.data<float>()[3] << "]" << std::endl;
        std::cout << " [" << grad_B.data<float>()[4] << ", " << grad_B.data<float>()[5] << "]]" << std::endl;
        
        // 验证梯度
        // 理论值: grad_A = grad_C * B^T, grad_B = A^T * grad_C
        // grad_C 是全 1 的 2x2 矩阵
        // grad_A (2x3) = [[15, 19, 23], [15, 19, 23]]
        // grad_B (3x2) = [[5, 5], [7, 7], [9, 9]]
        
        // 验证 grad_A
        float expected_grad_A[] = {15, 19, 23, 15, 19, 23};
        for (int i = 0; i < 6; i++) {
            if (std::abs(grad_A.data<float>()[i] - expected_grad_A[i]) > 1e-6) {
                std::cout << "❌ grad_A 错误: grad_A[" << i << "] 期望 " << expected_grad_A[i] << ", 实际 " << grad_A.data<float>()[i] << std::endl;
                return false;
            }
        }
        
        // 验证 grad_B
        float expected_grad_B[] = {5, 5, 7, 7, 9, 9};
        for (int i = 0; i < 6; i++) {
            if (std::abs(grad_B.data<float>()[i] - expected_grad_B[i]) > 1e-6) {
                std::cout << "❌ grad_B 错误: grad_B[" << i << "] 期望 " << expected_grad_B[i] << ", 实际 " << grad_B.data<float>()[i] << std::endl;
                return false;
            }
        }
        
        std::cout << "✅ MatMul 反向传播测试通过!" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cout << "!!! 测试异常: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "🚀 开始 MulNode 和 MatMul 反向传播测试" << std::endl;
    std::cout << "====================================" << std::endl;
    
    int passed = 0;
    int total = 2;
    
    if (test_mul_node_backward()) {
        passed++;
    }
    
    if (test_matmul_node_backward()) {
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