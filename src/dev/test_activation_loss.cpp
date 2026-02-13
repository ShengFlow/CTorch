/**
 * @file test_activation_loss.cpp
 * @brief 测试激活函数和损失函数
 * @author GhostFace
 * @date 2026/02/09
 */

#include "Tensor.h"
#include "Ctorch_Error.h"
#include <iostream>

void test_activation_functions() {
    Ctorch_Error::trace(ErrorPlatform::kCPU, "=== 测试激活函数 ===");
    
    // 创建上下文
    AutoDiff ctx;
    AutoDiffContext::Guard guard(&ctx);
    Ctorch_Error::setPrintLevel(PrintLevel::MINIUM);
    
    // 创建测试张量
    Tensor x({-1.0f, 0.0f, 1.0f, 2.0f});
    Ctorch_Error::trace(ErrorPlatform::kCPU, "原始张量: Tensor(shape=[4], dtype=float)");
    
    // 测试 Sigmoid 激活函数
    Tensor sigmoid_result = x.sigmoid();
    Ctorch_Error::trace(ErrorPlatform::kCPU, "Sigmoid 结果: Tensor(shape=[4], dtype=float)");
    
    // 测试 Tanh 激活函数
    Tensor tanh_result = x.tanh();
    Ctorch_Error::trace(ErrorPlatform::kCPU, "Tanh 结果: Tensor(shape=[4], dtype=float)");
    
    // 测试 Softmax 激活函数
    Tensor softmax_result = x.softmax(0);
    Ctorch_Error::trace(ErrorPlatform::kCPU, "Softmax 结果: Tensor(shape=[4], dtype=float)");
    
    // 测试带自动微分的激活函数
    Ctorch_Error::trace(ErrorPlatform::kCPU, "\n=== 测试带自动微分的激活函数 ===");
    Tensor y({-0.5f, 0.5f});
    y.requires_grad(true);
    
    Tensor sigmoid_with_grad = y.sigmoid();
    Tensor tanh_with_grad = y.tanh();
    
    Ctorch_Error::trace(ErrorPlatform::kCPU, "带自动微分的 Sigmoid 结果: Tensor(shape=[2], dtype=float)");
    Ctorch_Error::trace(ErrorPlatform::kCPU, "带自动微分的 Tanh 结果: Tensor(shape=[2], dtype=float)");
    
    // 测试反向传播
    Tensor loss = sigmoid_with_grad.sum() + tanh_with_grad.sum();
    backward(loss);
    
    Tensor grad_y = grad(y);
    Ctorch_Error::trace(ErrorPlatform::kCPU, "y 的梯度: Tensor(shape=[2], dtype=float)");
}

void test_loss_functions() {
    Ctorch_Error::trace(ErrorPlatform::kCPU, "\n=== 测试损失函数 ===");
    
    // 创建上下文
    AutoDiff ctx;
    AutoDiffContext::Guard guard(&ctx);
    Ctorch_Error::setPrintLevel(PrintLevel::MINIUM);
    
    // 创建预测值和目标值张量
    Tensor y_pred({0.8f, 0.2f, 0.6f});
    Tensor y_true({1.0f, 0.0f, 0.5f});
    
    Ctorch_Error::trace(ErrorPlatform::kCPU, "预测值: Tensor(shape=[3], dtype=float)");
    Ctorch_Error::trace(ErrorPlatform::kCPU, "目标值: Tensor(shape=[3], dtype=float)");
    
    // 测试 MSE 损失函数
    Tensor mse_result = y_pred.mse_loss(y_true);
    Ctorch_Error::trace(ErrorPlatform::kCPU, "MSE 损失: " + std::to_string(mse_result.item<float>()));
    
    // 测试 MAE 损失函数
    Tensor mae_result = y_pred.mae_loss(y_true);
    Ctorch_Error::trace(ErrorPlatform::kCPU, "MAE 损失: " + std::to_string(mae_result.item<float>()));
    
    // 测试 CrossEntropy 损失函数
    // 注意：CrossEntropy 通常用于分类问题，这里使用简单的示例
    Tensor logits({2.0f, 1.0f, 0.1f});
    Tensor targets({1.0f, 0.0f, 0.0f}); // 独热编码
    
    Tensor ce_result = logits.softmax(0).cross_entropy(targets);
    Ctorch_Error::trace(ErrorPlatform::kCPU, "CrossEntropy 损失: " + std::to_string(ce_result.item<float>()));
    
    // 测试带自动微分的损失函数
    Ctorch_Error::trace(ErrorPlatform::kCPU, "\n=== 测试带自动微分的损失函数 ===");
    Tensor y_pred_with_grad({0.7f, 0.3f});
    Tensor y_true_with_grad({1.0f, 0.0f});
    
    y_pred_with_grad.requires_grad(true);
    
    Tensor mse_with_grad = y_pred_with_grad.mse_loss(y_true_with_grad);
    Ctorch_Error::trace(ErrorPlatform::kCPU, "带自动微分的 MSE 损失: " + std::to_string(mse_with_grad.item<float>()));
    
    // 测试反向传播
    backward(mse_with_grad);
    
    Tensor grad_y_pred = grad(y_pred_with_grad);
    Ctorch_Error::trace(ErrorPlatform::kCPU, "y_pred 的梯度: Tensor(shape=[2], dtype=float)");
}

void test_more_activation_functions() {
    Ctorch_Error::trace(ErrorPlatform::kCPU, "\n=== 测试更多激活函数 ===");
    
    // 创建上下文
    AutoDiff ctx;
    AutoDiffContext::Guard guard(&ctx);
    Ctorch_Error::setPrintLevel(PrintLevel::MINIUM);
    
    // 创建测试张量
    Tensor x({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
    Ctorch_Error::trace(ErrorPlatform::kCPU, "原始张量: Tensor(shape=[5], dtype=float)");
    
    // 测试 ReLU 激活函数
    Tensor relu_result = x.relu();
    Ctorch_Error::trace(ErrorPlatform::kCPU, "ReLU 结果: Tensor(shape=[5], dtype=float)");
    
    // 测试带自动微分的 ReLU
    Ctorch_Error::trace(ErrorPlatform::kCPU, "\n=== 测试带自动微分的 ReLU ===");
    Tensor y({-1.0f, 0.0f, 1.0f});
    y.requires_grad(true);
    
    Tensor relu_with_grad = y.relu();
    Ctorch_Error::trace(ErrorPlatform::kCPU, "带自动微分的 ReLU 结果: Tensor(shape=[3], dtype=float)");
    
    // 测试反向传播
    Tensor loss = relu_with_grad.sum();
    backward(loss);
    
    Tensor grad_y = grad(y);
    Ctorch_Error::trace(ErrorPlatform::kCPU, "y 的梯度: Tensor(shape=[3], dtype=float)");
}

void test_tensor_operations() {
    Ctorch_Error::trace(ErrorPlatform::kCPU, "\n=== 测试张量操作 ===");
    
    // 创建上下文
    AutoDiff ctx;
    AutoDiffContext::Guard guard(&ctx);
    Ctorch_Error::setPrintLevel(PrintLevel::MINIUM);
    
    // 测试不同形状的张量
    Tensor a({1.0f, 2.0f, 3.0f});
    Tensor b({4.0f, 5.0f, 6.0f});
    Ctorch_Error::trace(ErrorPlatform::kCPU, "张量 a: Tensor(shape=[3], dtype=float)");
    Ctorch_Error::trace(ErrorPlatform::kCPU, "张量 b: Tensor(shape=[3], dtype=float)");
    
    // 测试基本运算
    Tensor add_result = a + b;
    Tensor sub_result = a - b;
    Tensor mul_result = a * b;
    Tensor div_result = a / b;
    
    Ctorch_Error::trace(ErrorPlatform::kCPU, "a + b: Tensor(shape=[3], dtype=float)");
    Ctorch_Error::trace(ErrorPlatform::kCPU, "a - b: Tensor(shape=[3], dtype=float)");
    Ctorch_Error::trace(ErrorPlatform::kCPU, "a * b: Tensor(shape=[3], dtype=float)");
    Ctorch_Error::trace(ErrorPlatform::kCPU, "a / b: Tensor(shape=[3], dtype=float)");
    
    // 测试广播操作
    Ctorch_Error::trace(ErrorPlatform::kCPU, "\n=== 测试广播操作 ===");
    Tensor scalar(2.0f);
    Tensor broadcast_result = a * scalar;
    Ctorch_Error::trace(ErrorPlatform::kCPU, "a * 2.0: Tensor(shape=[3], dtype=float)");
}

void test_random_tensors() {
    Ctorch_Error::trace(ErrorPlatform::kCPU, "\n=== 测试随机数生成 ===");
    
    // 创建上下文
    AutoDiff ctx;
    AutoDiffContext::Guard guard(&ctx);
    Ctorch_Error::setPrintLevel(PrintLevel::MINIUM);
    
    // 测试随机张量生成
    Tensor rand_tensor({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    rand_tensor.rand();
    Tensor reshaped_rand = rand_tensor.reshape({3, 3});
    Ctorch_Error::trace(ErrorPlatform::kCPU, "随机张量: Tensor(shape=[3, 3], dtype=float)");
}

int main() {
    Ctorch_Error::trace(ErrorPlatform::kCPU, "开始测试激活函数和损失函数...");
    
    // 测试激活函数
    test_activation_functions();
    
    // 测试损失函数
    test_loss_functions();
    
    // 测试更多激活函数
    test_more_activation_functions();
    
    // 测试张量操作
    test_tensor_operations();
    
    // 测试随机数生成
    test_random_tensors();
    
    Ctorch_Error::trace(ErrorPlatform::kCPU, "\n所有测试完成！");
    return 0;
}