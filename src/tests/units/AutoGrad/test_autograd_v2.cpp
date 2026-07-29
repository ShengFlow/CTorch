/**
 * @file test_autograd_v2.cpp
 * @brief v2 AD 引擎（AutoGrad 命名空间）的单元测试
 * @details 覆盖加/减/乘/除/矩阵乘/激活函数/损失函数的反向传播正确性
 */

#include "AutoGrad.h"
#include "Tensor.h"
#include "Arena.h"
#include "AutoGrad/Nodes/AddNode.h"
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

namespace {

constexpr float kEps = 1e-5f;

int g_passed = 0;
int g_failed = 0;

#define EXPECT(cond, msg) do { \
    if (cond) { ++g_passed; } else { ++g_failed; std::cerr << "[FAIL] " << msg << std::endl; } \
} while (0)

#define EXPECT_NEAR_F(a, b, eps) do { \
    float av = (a), bv = (b); \
    if (std::fabs(av - bv) <= (eps)) { ++g_passed; } \
    else { ++g_failed; std::cerr << "[FAIL] expected " << av << " ≈ " << bv << " (|diff| > " << (eps) << ")" << std::endl; } \
} while (0)

Tensor makeTensor(std::initializer_list<float> values) {
    Tensor t(values, DeviceType::kMPS);
    return t;
}

Tensor makeTensor2D(const std::vector<float>& values, size_t rows, size_t cols) {
    Tensor t(ShapeTag{}, {rows, cols}, DType::kFloat, DeviceType::kMPS);
    std::copy(values.begin(), values.end(), t.data<float>());
    return t;
}

void test_add_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({2.0f, 3.0f, 4.0f});
    Tensor b = makeTensor({10.0f, 20.0f, 30.0f});
    a.requires_grad(true);
    b.requires_grad(true);
    Tensor c = a + b;
    AutoGrad::backward(c.getRelatedNode(), false);
    Tensor ga = a.grad();
    Tensor gb = b.grad();
    const float* ga_p = ga.data<float>();
    const float* gb_p = gb.data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR_F(ga_p[i], 1.0f, kEps);
        EXPECT_NEAR_F(gb_p[i], 1.0f, kEps);
    }
}

void test_mul_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({2.0f, 3.0f});
    Tensor b = makeTensor({4.0f, 5.0f});
    a.requires_grad(true);
    b.requires_grad(true);
    Tensor c = a * b;
    AutoGrad::backward(c.getRelatedNode(), false);
    Tensor ga = a.grad();
    Tensor gb = b.grad();
    const float* ga_p = ga.data<float>();
    const float* gb_p = gb.data<float>();
    EXPECT_NEAR_F(ga_p[0], 4.0f, kEps); EXPECT_NEAR_F(ga_p[1], 5.0f, kEps);
    EXPECT_NEAR_F(gb_p[0], 2.0f, kEps); EXPECT_NEAR_F(gb_p[1], 3.0f, kEps);
}

void test_sub_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({5.0f, 7.0f});
    Tensor b = makeTensor({2.0f, 3.0f});
    a.requires_grad(true);
    b.requires_grad(true);
    Tensor c = a - b;
    AutoGrad::backward(c.getRelatedNode(), false);
    Tensor ga = a.grad();
    Tensor gb = b.grad();
    const float* ga_p = ga.data<float>();
    const float* gb_p = gb.data<float>();
    EXPECT_NEAR_F(ga_p[0], 1.0f, kEps); EXPECT_NEAR_F(ga_p[1], 1.0f, kEps);
    EXPECT_NEAR_F(gb_p[0], -1.0f, kEps); EXPECT_NEAR_F(gb_p[1], -1.0f, kEps);
}

void test_div_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({6.0f, 8.0f});
    Tensor b = makeTensor({2.0f, 4.0f});
    a.requires_grad(true);
    b.requires_grad(true);
    Tensor c = a / b;
    AutoGrad::backward(c.getRelatedNode(), false);
    Tensor ga = a.grad();
    Tensor gb = b.grad();
    const float* ga_p = ga.data<float>();
    const float* gb_p = gb.data<float>();
    EXPECT_NEAR_F(ga_p[0], 0.5f, kEps);
    EXPECT_NEAR_F(ga_p[1], 0.25f, kEps);
    EXPECT_NEAR_F(gb_p[0], -1.5f, kEps);
    EXPECT_NEAR_F(gb_p[1], -0.5f, kEps);
}

void test_matmul_grad() {
    AutoGrad::EnableGrad = true;
    Tensor A = makeTensor2D({1, 2, 3, 4, 5, 6}, 2, 3);
    Tensor B = makeTensor2D({7, 8, 9, 10, 11, 12}, 3, 2);
    A.requires_grad(true);
    B.requires_grad(true);
    Tensor C = A.matmul(B);
    AutoGrad::backward(C.getRelatedNode(), false);

    // C = A(2x3) * B(3x2) = 2x2
    // grad_C 全 1 (2x2)
    // grad_A = grad_C * B^T = (2x2) * (2x3) = 2x3
    //   = [[15, 19, 23], [15, 19, 23]]
    // grad_B = A^T * grad_C = (3x2) * (2x2) = 3x2
    //   = [[5, 5], [7, 7], [9, 9]]
    const float expected_grad_A[] = {15, 19, 23, 15, 19, 23};
    const float expected_grad_B[] = {5, 5, 7, 7, 9, 9};
    Tensor ga = A.grad();
    Tensor gb = B.grad();
    const float* ga_p = ga.data<float>();
    const float* gb_p = gb.data<float>();
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR_F(ga_p[i], expected_grad_A[i], kEps);
        EXPECT_NEAR_F(gb_p[i], expected_grad_B[i], kEps);
    }
}

// 关键回归测试：两个输入都不需要梯度时，dispatch 不应再无条件 requires_grad(true)
void test_scheduler_no_grad_propagation() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({1.0f, 2.0f});
    Tensor b = makeTensor({3.0f, 4.0f});
    Tensor c = a + b;
    EXPECT(!c.requires_grad(), "no-grad inputs should not produce grad output");
}

// 至少一个输入需要梯度时，应该记录计算图
void test_scheduler_grad_propagation() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({1.0f, 2.0f});
    Tensor b = makeTensor({3.0f, 4.0f});
    a.requires_grad(true);
    Tensor c = a + b;
    EXPECT(c.requires_grad(), "any grad input should propagate to output");
    EXPECT(c.getRelatedNode().get() != nullptr, "graph node should be attached");
}

void test_relu_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({-2.0f, 0.0f, 3.0f, -1.0f, 5.0f});
    a.requires_grad(true);
    Tensor b = a.relu();
    AutoGrad::backward(b.getRelatedNode(), false);
    Tensor ga = a.grad();
    const float* ga_p = ga.data<float>();
    EXPECT_NEAR_F(ga_p[0], 0.0f, kEps);
    EXPECT_NEAR_F(ga_p[1], 0.0f, kEps);
    EXPECT_NEAR_F(ga_p[2], 1.0f, kEps);
    EXPECT_NEAR_F(ga_p[3], 0.0f, kEps);
    EXPECT_NEAR_F(ga_p[4], 1.0f, kEps);
}

void test_neg_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({2.0f, -3.0f, 5.0f});
    a.requires_grad(true);
    Tensor b = -a;
    AutoGrad::backward(b.getRelatedNode(), false);
    Tensor ga = a.grad();
    const float* ga_p = ga.data<float>();
    EXPECT_NEAR_F(ga_p[0], -1.0f, kEps);
    EXPECT_NEAR_F(ga_p[1], -1.0f, kEps);
    EXPECT_NEAR_F(ga_p[2], -1.0f, kEps);
}

void test_sin_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({0.0f, 1.5707963f, 3.1415926f});
    a.requires_grad(true);
    Tensor b = a.sin();
    AutoGrad::backward(b.getRelatedNode(), false);
    Tensor ga = a.grad();
    const float* ga_p = ga.data<float>();
    EXPECT_NEAR_F(ga_p[0], 1.0f, kEps);
    EXPECT_NEAR_F(ga_p[1], 0.0f, kEps);
    EXPECT_NEAR_F(ga_p[2], -1.0f, kEps);
}

void test_cos_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({0.0f, 1.5707963f, 3.1415926f});
    a.requires_grad(true);
    Tensor b = a.cos();
    AutoGrad::backward(b.getRelatedNode(), false);
    Tensor ga = a.grad();
    const float* ga_p = ga.data<float>();
    EXPECT_NEAR_F(ga_p[0], 0.0f, kEps);
    EXPECT_NEAR_F(ga_p[1], -1.0f, kEps);
    EXPECT_NEAR_F(ga_p[2], 0.0f, kEps);
}

void test_tanh_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({0.0f, 1.0f, -1.0f});
    a.requires_grad(true);
    Tensor b = a.tanh();
    AutoGrad::backward(b.getRelatedNode(), false);
    Tensor ga = a.grad();
    const float* ga_p = ga.data<float>();
    float t0 = std::tanh(0.0f);
    float t1 = std::tanh(1.0f);
    float t2 = std::tanh(-1.0f);
    
    EXPECT_NEAR_F(ga_p[0], 1.0f - t0 * t0, kEps);
    EXPECT_NEAR_F(ga_p[1], 1.0f - t1 * t1, kEps);
    EXPECT_NEAR_F(ga_p[2], 1.0f - t2 * t2, kEps);
}

void test_sigmoid_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({0.0f, 1.0f, -1.0f});
    a.requires_grad(true);
    Tensor b = a.sigmoid();
    AutoGrad::backward(b.getRelatedNode(), false);
    Tensor ga = a.grad();
    const float* ga_p = ga.data<float>();
    float s0 = 1.0f / (1.0f + std::exp(-0.0f));
    float s1 = 1.0f / (1.0f + std::exp(-1.0f));
    float s2 = 1.0f / (1.0f + std::exp(1.0f));
    EXPECT_NEAR_F(ga_p[0], s0 * (1.0f - s0), kEps);
    EXPECT_NEAR_F(ga_p[1], s1 * (1.0f - s1), kEps);
    EXPECT_NEAR_F(ga_p[2], s2 * (1.0f - s2), kEps);
}

void test_memory_grad_accumulator_safety() {
    AutoGrad::EnableGrad = true;
    std::shared_ptr<Tensor> a_ptr = std::make_shared<Tensor>(makeTensor({2.0f, 3.0f}));
    a_ptr->requires_grad(true);
    Tensor b = makeTensor({10.0f, 20.0f});
    b.requires_grad(true);
    Tensor c = (*a_ptr) + b;
    a_ptr.reset();
    AutoGrad::backward(c.getRelatedNode(), false);
    EXPECT(true, "GradAccumulator with weak_ptr should not crash");
}

void test_memory_tensor_copy_grad_independence() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({1.0f, 2.0f});
    a.requires_grad(true);
    Tensor b = makeTensor({10.0f, 20.0f});
    b.requires_grad(true);
    Tensor c = a + b;
    AutoGrad::backward(c.getRelatedNode(), false);
    Tensor a_copy = a;
    auto grad_before = a.grad().data<float>();
    auto grad_copy_before = a_copy.grad().data<float>();
    EXPECT(grad_before[0] == grad_copy_before[0], "Initial grads should be equal");
    Tensor d = a_copy * makeTensor({2.0f, 3.0f});
    AutoGrad::backward(d.getRelatedNode(), false);
    auto grad_after = a.grad().data<float>();
    auto grad_copy_after = a_copy.grad().data<float>();
    EXPECT(grad_after[0] != grad_copy_after[0], "Grads should be independent after copy");
}

void test_memory_arena_clear() {
    Arena& arena = Arena::getInstance();
    auto node = arena.invoke<AddNode>(std::vector<std::shared_ptr<Node>>(), std::vector<Tensor>());
    EXPECT(node != nullptr, "Arena should allocate node");
    arena.clear();
    EXPECT(arena.invoke<AddNode>(std::vector<std::shared_ptr<Node>>(), std::vector<Tensor>()) != nullptr, 
           "Arena should work after clear");
}

void test_memory_tensor_move_grad() {
    AutoGrad::EnableGrad = true;
    Tensor a = makeTensor({1.0f, 2.0f});
    a.requires_grad(true);
    Tensor b = makeTensor({10.0f, 20.0f});
    b.requires_grad(true);
    Tensor c = a + b;
    AutoGrad::backward(c.getRelatedNode(), false);
    Tensor d = std::move(a);
    EXPECT(d.grad().numel() == 2, "Moved tensor should retain grad");
}

} // namespace

int main() {
    std::cout << "=== v2 AD 引擎单元测试 ===" << std::endl;
    test_add_grad();
    test_mul_grad();
    test_sub_grad();
    test_div_grad();
    test_matmul_grad();
    test_scheduler_no_grad_propagation();
    test_scheduler_grad_propagation();
    test_relu_grad();
    test_neg_grad();
    test_sin_grad();
    test_cos_grad();
    test_tanh_grad();
    test_sigmoid_grad();
    test_memory_grad_accumulator_safety();
    test_memory_tensor_copy_grad_independence();
    test_memory_arena_clear();
    test_memory_tensor_move_grad();
    std::cout << "\n通过: " << g_passed << " / 失败: " << g_failed << std::endl;
    return g_failed == 0 ? 0 : 1;
}
