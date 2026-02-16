/**
 * @file test_activation_loss.cpp
 * @brief 测试激活函数和损失函数
 * @author GhostFace
 * @date 2026/02/09
 */

#include "Tensor.h"
#include "Ctorch_Error.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <algorithm>

static void expect_close(float a, float b, float eps, const char* msg) {
    if (std::fabs(a - b) > eps) {
        std::cerr << "[TEST FAIL] " << msg << " | a=" << a << " b=" << b << " eps=" << eps << std::endl;
        std::exit(1);
    }
}

static void expect_true(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[TEST FAIL] " << msg << std::endl;
        std::exit(1);
    }
}

static float row_sum_2d(const Tensor& t, size_t row, size_t cols) {
    const float* p = t.data<float>();
    float s = 0.0f;
    for (size_t j = 0; j < cols; ++j) s += p[row * cols + j];
    return s;
}

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
    // 单元测试：softmax 概率和为 1
    {
        const float* s = softmax_result.data<float>();
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) sum += s[i];
        expect_close(sum, 1.0f, 1e-5f, "1D softmax sum == 1");
    }

    // 测试带自动微分的激活函数
    Ctorch_Error::trace(ErrorPlatform::kCPU, "\n=== 测试带自动微分的激活函数 ===");
    Tensor y({-0.5f, 0.5f});
    y.requires_grad(true);

    Tensor sigmoid_with_grad = y.sigmoid();
    Tensor tanh_with_grad    = y.tanh();

    Ctorch_Error::trace(ErrorPlatform::kCPU,
                        "带自动微分的 Sigmoid 结果: Tensor(shape=[2], dtype=float)");
    Ctorch_Error::trace(ErrorPlatform::kCPU,
                        "带自动微分的 Tanh 结果: Tensor(shape=[2], dtype=float)");

    // 测试反向传播
    Tensor loss = sigmoid_with_grad.sum() + tanh_with_grad.sum();
    backward(loss);

    Tensor grad_y = grad(y);
    Ctorch_Error::trace(ErrorPlatform::kCPU, "y 的梯度: Tensor(shape=[2], dtype=float)");
}

static void test_softmax_dim_and_backward() {
    Ctorch_Error::trace(ErrorPlatform::kCPU, "\n=== 单元测试: softmax(dim) 前向/反向 ===");
    AutoDiff ctx;
    AutoDiffContext::Guard guard(&ctx);
    Ctorch_Error::setPrintLevel(PrintLevel::MINIUM);

    // 2D softmax dim=1：每行和为1
    Tensor m(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
    float* md = m.data<float>();
    // row0: [1,2,3], row1: [0,-1,1]
    md[0]=1.0f; md[1]=2.0f; md[2]=3.0f;
    md[3]=0.0f; md[4]=-1.0f; md[5]=1.0f;
    Tensor p = m.softmax(1);
    expect_close(row_sum_2d(p, 0, 3), 1.0f, 1e-5f, "2D softmax(dim=1) row0 sum == 1");
    expect_close(row_sum_2d(p, 1, 3), 1.0f, 1e-5f, "2D softmax(dim=1) row1 sum == 1");

    // 2D softmax dim=0：每列和为1
    Tensor pc = m.softmax(0);
    {
        const float* d = pc.data<float>();
        // col0: d[0], d[3]  col1: d[1], d[4]  col2: d[2], d[5]
        expect_close(d[0] + d[3], 1.0f, 1e-5f, "2D softmax(dim=0) col0 sum == 1");
        expect_close(d[1] + d[4], 1.0f, 1e-5f, "2D softmax(dim=0) col1 sum == 1");
        expect_close(d[2] + d[5], 1.0f, 1e-5f, "2D softmax(dim=0) col2 sum == 1");
    }

    // softmax backward：sum(softmax) 对输入的梯度应为0（因为每行 sum=1 是常数；两行总和=2常数）
    Tensor x(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
    float* xd = x.data<float>();
    for (int i = 0; i < 6; ++i) xd[i] = md[i];
    x.requires_grad(true);
    Tensor s = x.softmax(1);
    Tensor L = s.sum(); // 常数 2
    backward(L);
    Tensor gx = grad(x);
    const float* g = gx.data<float>();
    float max_abs = 0.0f;
    for (int i = 0; i < 6; ++i) max_abs = std::max(max_abs, std::fabs(g[i]));
    expect_true(max_abs < 1e-4f, "softmax backward: grad of sum(softmax) ~ 0");
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
    Ctorch_Error::trace(ErrorPlatform::kCPU,
                        "MSE 损失: " + std::to_string(mse_result.item<float>()));

    // 测试 MAE 损失函数
    Tensor mae_result = y_pred.mae_loss(y_true);
    Ctorch_Error::trace(ErrorPlatform::kCPU,
                        "MAE 损失: " + std::to_string(mae_result.item<float>()));

    // 测试 CrossEntropy 损失函数
    // 注意：CrossEntropy 通常用于分类问题，这里使用简单的示例
    Tensor logits({2.0f, 1.0f, 0.1f});
    Tensor targets({1.0f, 0.0f, 0.0f}); // 独热编码

    // CrossEntropy(kernel) 内部会对 logits 做 softmax，因此这里直接对 logits 计算 CE
    Tensor ce_result = logits.cross_entropy(targets);
    Ctorch_Error::trace(ErrorPlatform::kCPU,
                        "CrossEntropy 损失: " + std::to_string(ce_result.item<float>()));

    // 单元测试：CE forward 数值（1D one-hot）应等于 -log(softmax(z)[0])
    {
        Tensor sm = logits.softmax(0);
        const float* smp = sm.data<float>();
        float expected = -std::log(std::max(smp[0], 1e-10f));
        expect_close(ce_result.item<float>(), expected, 1e-4f, "CE forward matches -log(p_true) (1D)");
    }

    // 单元测试：CE backward 基本正确性（1D one-hot）：grad(logits) = softmax(logits) - target
    {
        AutoDiff ctx2;
        AutoDiffContext::Guard guard2(&ctx2);
        Tensor z({2.0f, 1.0f, 0.1f});
        Tensor t({1.0f, 0.0f, 0.0f});
        z.requires_grad(true);
        Tensor ce = z.cross_entropy(t);
        backward(ce);
        Tensor gz = grad(z);
        Tensor sm = z.softmax(0);
        const float* gzp = gz.data<float>();
        const float* smp = sm.data<float>();
        const float* tp = t.data<float>();
        for (int i = 0; i < 3; ++i) {
            expect_close(gzp[i], smp[i] - tp[i], 1e-4f, "CE grad matches softmax-target (1D)");
        }
    }

    // 测试带自动微分的损失函数
    Ctorch_Error::trace(ErrorPlatform::kCPU, "\n=== 测试带自动微分的损失函数 ===");
    Tensor y_pred_with_grad({0.7f, 0.3f});
    Tensor y_true_with_grad({1.0f, 0.0f});

    y_pred_with_grad.requires_grad(true);

    Tensor mse_with_grad = y_pred_with_grad.mse_loss(y_true_with_grad);
    Ctorch_Error::trace(ErrorPlatform::kCPU,
                        "带自动微分的 MSE 损失: " + std::to_string(mse_with_grad.item<float>()));

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
    Ctorch_Error::trace(ErrorPlatform::kCPU,
                        "带自动微分的 ReLU 结果: Tensor(shape=[3], dtype=float)");

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
    test_more_activation_functions();
    test_softmax_dim_and_backward();
    Ctorch_Error::info(ErrorPlatform::kCPU,"激活函数测试完毕");

    // 测试损失函数
    test_loss_functions();
    Ctorch_Error::info(ErrorPlatform::kCPU,"损失函数测试完毕");

    // 测试张量操作
    test_tensor_operations();
    Ctorch_Error::info(ErrorPlatform::kCPU,"张量操作测试完毕");

    // 测试随机数生成
    test_random_tensors();
    Ctorch_Error::info(ErrorPlatform::kCPU,"随机数测试完毕");

    Ctorch_Error::info(ErrorPlatform::kCPU,"所有测试完成！");
    return 0;
}
