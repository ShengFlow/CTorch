#include "Tensor.h"
#include "Ctorch_Error.h"
#include <iostream>
#include <vector>

// 测试张量梯度计算
void test_tensor_gradient() {
    Ctorch_Error::trace(ErrorPlatform::kCPU, "=== 测试：张量梯度计算 ===");
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
        Ctorch_Error::trace(ErrorPlatform::kCPU, "a = Tensor(shape=[2, 3], dtype=float)");
        Ctorch_Error::trace(ErrorPlatform::kCPU, "b = Tensor(shape=[2, 3], dtype=float)");
        Ctorch_Error::trace(ErrorPlatform::kCPU, "c = a + b = Tensor(shape=[2, 3], dtype=float)");
        Ctorch_Error::trace(ErrorPlatform::kCPU, "d = c * 2 = Tensor(shape=[2, 3], dtype=float)");

        // 反向传播 - 直接使用d作为根节点
        backward(d);

        // 获取梯度
        Tensor grad_a = grad(a);
        Tensor grad_b = grad(b);

        // 输出梯度结果
        Ctorch_Error::trace(ErrorPlatform::kCPU, "∂c/∂a = Tensor(shape=[2, 3], dtype=float)");
        Ctorch_Error::trace(ErrorPlatform::kCPU, "∂c/∂b = Tensor(shape=[2, 3], dtype=float)");

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
            Ctorch_Error::trace(ErrorPlatform::kCPU, "✅ 张量梯度计算测试通过");
        } else {
            Ctorch_Error::trace(ErrorPlatform::kCPU, "❌ 张量梯度计算测试失败");
        }
        
    } catch (const std::exception& e) {
        Ctorch_Error::trace(ErrorPlatform::kCPU, "!!! 张量梯度测试异常: " + std::string(e.what()));
    }
}

// 测试广播操作的梯度计算
void test_broadcast_gradient() {
    Ctorch_Error::trace(ErrorPlatform::kCPU, "\n=== 测试：广播操作的梯度计算 ===");
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
        Ctorch_Error::trace(ErrorPlatform::kCPU, "a = Tensor(shape=[2, 3], dtype=float)");
        Ctorch_Error::trace(ErrorPlatform::kCPU, "b = Tensor(shape=[3], dtype=float)");
        Ctorch_Error::trace(ErrorPlatform::kCPU, "c = a + b = Tensor(shape=[2, 3], dtype=float)");

        // 反向传播 - 直接使用c作为根节点
        backward(c);

        // 获取梯度
        Tensor grad_a = grad(a);
        Tensor grad_b = grad(b);

        // 输出梯度结果
        Ctorch_Error::trace(ErrorPlatform::kCPU, "∂c/∂a = Tensor(shape=[2, 3], dtype=float)");
        Ctorch_Error::trace(ErrorPlatform::kCPU, "∂c/∂b = Tensor(shape=[3], dtype=float)");

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
            Ctorch_Error::trace(ErrorPlatform::kCPU, "✅ 广播操作梯度测试通过");
        } else {
            Ctorch_Error::trace(ErrorPlatform::kCPU, "❌ 广播操作梯度测试失败");
        }
        
    } catch (const std::exception& e) {
        Ctorch_Error::trace(ErrorPlatform::kCPU, "!!! 广播梯度测试异常: " + std::string(e.what()));
    }
}

// 测试复杂计算图的张量梯度
void test_complex_graph_gradient() {
    Ctorch_Error::trace(ErrorPlatform::kCPU, "\n=== 测试：复杂计算图的张量梯度 ===");
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
        Ctorch_Error::trace(ErrorPlatform::kCPU, "x = Tensor(shape=[2, 2], dtype=float)");
        Ctorch_Error::trace(ErrorPlatform::kCPU, "y = Tensor(shape=[2, 2], dtype=float)");
        Ctorch_Error::trace(ErrorPlatform::kCPU, "z1 = x * y = Tensor(shape=[2, 2], dtype=float)");
        Ctorch_Error::trace(ErrorPlatform::kCPU, "z2 = z1 + x = Tensor(shape=[2, 2], dtype=float)");

        // 反向传播 - 直接使用z2作为根节点
        backward(z2);

        // 获取梯度
        Tensor grad_x = grad(x);
        Tensor grad_y = grad(y);

        // 输出梯度结果
        Ctorch_Error::trace(ErrorPlatform::kCPU, "∂z2/∂x = Tensor(shape=[2, 2], dtype=float)");
        Ctorch_Error::trace(ErrorPlatform::kCPU, "∂z2/∂y = Tensor(shape=[2, 2], dtype=float)");

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
            Ctorch_Error::trace(ErrorPlatform::kCPU, "✅ 复杂计算图梯度测试通过");
        } else {
            Ctorch_Error::trace(ErrorPlatform::kCPU, "❌ 复杂计算图梯度测试失败");
        }
        
    } catch (const std::exception& e) {
        Ctorch_Error::trace(ErrorPlatform::kCPU, "!!! 复杂计算图梯度测试异常: " + std::string(e.what()));
    }
}

int main() {
    // 执行所有测试
    test_tensor_gradient();
    test_broadcast_gradient();
    test_complex_graph_gradient();
    
    Ctorch_Error::trace(ErrorPlatform::kCPU, "\n=== 所有张量梯度测试完成 ===");
    return 0;
}
