#include "Tensor.h"
#include <iostream>
#include <vector>

// 测试张量梯度计算
void test_tensor_gradient() {
    std::cout << "=== 测试：张量梯度计算 ===" << std::endl;
    try {
        AutoDiff ctx;
        AutoDiffContext::Guard guard(&ctx);

        // 创建输入张量
        std::vector<size_t> shape = {2, 3};
        Tensor a(ShapeTag{}, shape, DType::kFloat, DeviceType::kCPU);
        Tensor b(ShapeTag{}, shape, DType::kFloat, DeviceType::kCPU);
        
        // 初始化张量数据
        float* a_data = a.data<float>();
        float* b_data = b.data<float>();
        for (size_t i = 0; i < a.numel(); ++i) {
            a_data[i] = static_cast<float>(i);
            b_data[i] = static_cast<float>(a.numel() - i);
        }
        
        // 设置需要梯度
        a.requires_grad(true);
        b.requires_grad(true);

        // 执行张量操作
        Tensor c = a + b;
        Tensor d = c * 2.0f;
        
        // 输出中间结果
        std::cout << "a = " << a << std::endl;
        std::cout << "b = " << b << std::endl;
        std::cout << "c = a + b = " << c << std::endl;
        std::cout << "d = c * 2 = " << d << std::endl;

        // 反向传播 - 直接使用d作为根节点
        backward(d);

        // 获取梯度
        Tensor grad_a = grad(a);
        Tensor grad_b = grad(b);

        // 输出梯度结果
        std::cout << "∂c/∂a = " << grad_a << std::endl;
        std::cout << "∂c/∂b = " << grad_b << std::endl;

        // 验证梯度结果
        bool passed = true;
        float* grad_a_data = grad_a.data<float>();
        float* grad_b_data = grad_b.data<float>();
        
        // 期望梯度：对于d = 2*(a + b)，梯度应该是全2
        for (size_t i = 0; i < grad_a.numel(); ++i) {
            if (std::abs(grad_a_data[i] - 2.0f) > 1e-6 || std::abs(grad_b_data[i] - 2.0f) > 1e-6) {
                passed = false;
                break;
            }
        }
        
        if (passed) {
            std::cout << "✅ 张量梯度计算测试通过" << std::endl;
        } else {
            std::cout << "❌ 张量梯度计算测试失败" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "!!! 张量梯度测试异常: " << e.what() << std::endl;
    }
}

// 测试广播操作的梯度计算
void test_broadcast_gradient() {
    std::cout << "\n=== 测试：广播操作的梯度计算 ===" << std::endl;
    try {
        AutoDiff ctx;
        AutoDiffContext::Guard guard(&ctx);

        // 创建输入张量
        Tensor a(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
        Tensor b(ShapeTag{}, {3}, DType::kFloat, DeviceType::kCPU);
        
        // 初始化张量数据
        float* a_data = a.data<float>();
        float* b_data = b.data<float>();
        for (size_t i = 0; i < a.numel(); ++i) {
            a_data[i] = static_cast<float>(i);
        }
        for (size_t i = 0; i < b.numel(); ++i) {
            b_data[i] = static_cast<float>(i + 1);
        }
        
        // 设置需要梯度
        a.requires_grad(true);
        b.requires_grad(true);

        // 执行广播操作：a (2,3) + b (3) -> c (2,3)
        Tensor c = a + b;
        
        // 输出中间结果
        std::cout << "a = " << a << std::endl;
        std::cout << "b = " << b << std::endl;
        std::cout << "c = a + b = " << c << std::endl;

        // 反向传播 - 直接使用c作为根节点
        backward(c);

        // 获取梯度
        Tensor grad_a = grad(a);
        Tensor grad_b = grad(b);

        // 输出梯度结果
        std::cout << "∂c/∂a = " << grad_a << std::endl;
        std::cout << "∂c/∂b = " << grad_b << std::endl;

        // 验证梯度结果
        bool passed = true;
        float* grad_a_data = grad_a.data<float>();
        float* grad_b_data = grad_b.data<float>();
        
        // 期望梯度：∂c/∂a 应该是全1
        for (size_t i = 0; i < grad_a.numel(); ++i) {
            if (std::abs(grad_a_data[i] - 1.0f) > 1e-6) {
                passed = false;
                break;
            }
        }
        
        // 期望梯度：∂c/∂b 应该是全1（因为我们直接对c反向传播）
        for (size_t i = 0; i < grad_b.numel(); ++i) {
            if (std::abs(grad_b_data[i] - 1.0f) > 1e-6) {
                passed = false;
                break;
            }
        }
        
        if (passed) {
            std::cout << "✅ 广播操作梯度测试通过" << std::endl;
        } else {
            std::cout << "❌ 广播操作梯度测试失败" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "!!! 广播梯度测试异常: " << e.what() << std::endl;
    }
}

// 测试复杂计算图的张量梯度
void test_complex_graph_gradient() {
    std::cout << "\n=== 测试：复杂计算图的张量梯度 ===" << std::endl;
    try {
        AutoDiff ctx;
        AutoDiffContext::Guard guard(&ctx);

        // 创建输入张量
        std::vector<size_t> shape = {2, 2};
        Tensor x(ShapeTag{}, shape, DType::kFloat, DeviceType::kCPU);
        Tensor y(ShapeTag{}, shape, DType::kFloat, DeviceType::kCPU);
        
        // 初始化张量数据
        float* x_data = x.data<float>();
        float* y_data = y.data<float>();
        x_data[0] = 1.0f; x_data[1] = 2.0f;
        x_data[2] = 3.0f; x_data[3] = 4.0f;
        
        y_data[0] = 5.0f; y_data[1] = 6.0f;
        y_data[2] = 7.0f; y_data[3] = 8.0f;
        
        // 设置需要梯度
        x.requires_grad(true);
        y.requires_grad(true);

        // 执行复杂操作
        Tensor z1 = x * y;
        Tensor z2 = z1 + x;
        
        // 输出中间结果
        std::cout << "x = " << x << std::endl;
        std::cout << "y = " << y << std::endl;
        std::cout << "z1 = x * y = " << z1 << std::endl;
        std::cout << "z2 = z1 + x = " << z2 << std::endl;

        // 反向传播 - 直接使用z2作为根节点
        backward(z2);

        // 获取梯度
        Tensor grad_x = grad(x);
        Tensor grad_y = grad(y);

        // 输出梯度结果
        std::cout << "∂z2/∂x = " << grad_x << std::endl;
        std::cout << "∂z2/∂y = " << grad_y << std::endl;

        // 验证梯度结果
        bool passed = true;
        float* grad_x_data = grad_x.data<float>();
        float* grad_y_data = grad_y.data<float>();
        
        // 期望梯度：∂z3/∂x = y + 1
        float expected_grad_x[] = {6.0f, 7.0f, 8.0f, 9.0f};
        // 期望梯度：∂z3/∂y = x
        float expected_grad_y[] = {1.0f, 2.0f, 3.0f, 4.0f};
        
        for (size_t i = 0; i < grad_x.numel(); ++i) {
            if (std::abs(grad_x_data[i] - expected_grad_x[i]) > 1e-6 || 
                std::abs(grad_y_data[i] - expected_grad_y[i]) > 1e-6) {
                passed = false;
                break;
            }
        }
        
        if (passed) {
            std::cout << "✅ 复杂计算图梯度测试通过" << std::endl;
        } else {
            std::cout << "❌ 复杂计算图梯度测试失败" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "!!! 复杂计算图梯度测试异常: " << e.what() << std::endl;
    }
}

int main() {
    // 执行所有测试
    test_tensor_gradient();
    test_broadcast_gradient();
    test_complex_graph_gradient();
    
    std::cout << "\n=== 所有张量梯度测试完成 ===" << std::endl;
    return 0;
}
