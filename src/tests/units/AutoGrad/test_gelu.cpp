/**
 * @file test_gelu.cpp
 * @brief GELU 算子单元测试
 * @details 覆盖 CPU/MPS 前向/反向正确性、跨设备一致性、CPU 反向与数值梯度对比
 */

#include "AutoGrad.h"
#include "Tensor.h"
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

namespace {

constexpr float kSqrt2OverPi = 0.7978845608f;
constexpr float kGeluCoeff = 0.044715f;
constexpr float kEps = 1e-4f;
constexpr float kGradEps = 5e-4f;

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

inline float gelu_reference(float x) {
    float v = kSqrt2OverPi * (x + kGeluCoeff * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(v));
}

inline float gelu_derivative_reference(float x) {
    float v = kSqrt2OverPi * (x + kGeluCoeff * x * x * x);
    float tanh_v = std::tanh(v);
    float term1 = 0.5f * (1.0f + tanh_v);
    float term2 = 0.5f * x * (1.0f - tanh_v * tanh_v) * kSqrt2OverPi *
                  (1.0f + 3.0f * kGeluCoeff * x * x);
    return term1 + term2;
}

Tensor make_tensor_cpu(const std::vector<float>& values) {
    Tensor t(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    std::copy(values.begin(), values.end(), t.data<float>());
    return t;
}

Tensor make_tensor_mps(const std::vector<float>& values) {
    Tensor t(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kMPS);
    std::copy(values.begin(), values.end(), t.data<float>());
    MPS_markBufferModified(static_cast<void*>(t.data<float>()), values.size() * sizeof(float));
    return t;
}

void test_cpu_forward_reference() {
    std::vector<float> inputs = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f};
    Tensor a = make_tensor_cpu(inputs);
    Tensor b = a.gelu();

    const float* out = b.data<float>();
    for (size_t i = 0; i < inputs.size(); ++i) {
        EXPECT_NEAR_F(out[i], gelu_reference(inputs[i]), kEps);
    }
}

void test_cpu_backward_reference() {
    std::vector<float> inputs = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f};
    Tensor a = make_tensor_cpu(inputs);
    a.requires_grad(true);
    Tensor b = a.gelu();
    AutoGrad::backward(b.getRelatedNode(), false);

    Tensor ga = a.grad();
    const float* grad = ga.data<float>();
    for (size_t i = 0; i < inputs.size(); ++i) {
        EXPECT_NEAR_F(grad[i], gelu_derivative_reference(inputs[i]), kEps);
    }
}

void test_cpu_backward_numerical() {
    std::vector<float> inputs = {-1.5f, -0.5f, 0.0f, 0.3f, 1.2f};
    const float delta = 1e-3f;

    Tensor a = make_tensor_cpu(inputs);
    a.requires_grad(true);
    Tensor b = a.gelu();
    AutoGrad::backward(b.getRelatedNode(), false);

    Tensor ga = a.grad();
    const float* grad = ga.data<float>();
    for (size_t i = 0; i < inputs.size(); ++i) {
        float x = inputs[i];
        float num_grad = (gelu_reference(x + delta) - gelu_reference(x - delta)) / (2.0f * delta);
        EXPECT_NEAR_F(grad[i], num_grad, kGradEps);
    }
}

void test_mps_forward_vs_cpu() {
    std::vector<float> inputs = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f};
    Tensor a_cpu = make_tensor_cpu(inputs);
    Tensor a_mps = make_tensor_mps(inputs);

    Tensor b_cpu = a_cpu.gelu();
    Tensor b_mps = a_mps.gelu();

    // (SYNC) 读取 MPS 结果前 flush accumulator
    MPS_flush_wait(true);

    const float* cpu_out = b_cpu.data<float>();
    const float* mps_out = b_mps.data<float>();
    for (size_t i = 0; i < inputs.size(); ++i) {
        EXPECT_NEAR_F(mps_out[i], cpu_out[i], kEps);
    }
}

void test_mps_backward_vs_cpu() {
    std::vector<float> inputs = {-2.0f, -1.0f, 0.0f, 0.5f, 1.0f, 2.0f};
    Tensor a_cpu = make_tensor_cpu(inputs);
    a_cpu.requires_grad(true);
    Tensor b_cpu = a_cpu.gelu();
    AutoGrad::backward(b_cpu.getRelatedNode(), false);

    Tensor a_mps = make_tensor_mps(inputs);
    a_mps.requires_grad(true);
    Tensor b_mps = a_mps.gelu();
    AutoGrad::backward(b_mps.getRelatedNode(), false);

    // (SYNC) 读取 MPS 梯度前 flush accumulator
    MPS_flush_wait(true);

    Tensor ga_cpu = a_cpu.grad();
    Tensor ga_mps = a_mps.grad();
    const float* cpu_grad = ga_cpu.data<float>();
    const float* mps_grad = ga_mps.data<float>();
    for (size_t i = 0; i < inputs.size(); ++i) {
        EXPECT_NEAR_F(mps_grad[i], cpu_grad[i], kEps);
    }
}

void test_mps_backward_through_matmul() {
    // 验证 GELU 梯度在 MPS 上能正确传递给下游 MatMul/Add
    std::vector<float> inputs = {-1.0f, 0.5f, 1.0f, -0.5f};
    Tensor a = make_tensor_mps(inputs);
    a.requires_grad(true);

    Tensor b = a.gelu();
    Tensor c = b * 2.0f;  // 标量乘法，调度到 MPS
    AutoGrad::backward(c.getRelatedNode(), false);

    // (SYNC) 读取 MPS 梯度前 flush accumulator
    MPS_flush_wait(true);

    Tensor ga = a.grad();
    const float* grad = ga.data<float>();
    for (size_t i = 0; i < inputs.size(); ++i) {
        float expected = gelu_derivative_reference(inputs[i]) * 2.0f;
        EXPECT_NEAR_F(grad[i], expected, kEps);
    }
}

void test_cpu_random_consistency() {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> inputs(64);
    for (auto& v : inputs) v = dist(gen);

    Tensor a = make_tensor_cpu(inputs);
    a.requires_grad(true);
    Tensor b = a.gelu();
    AutoGrad::backward(b.getRelatedNode(), false);

    const float* out = b.data<float>();
    const float* grad = a.grad().data<float>();
    for (size_t i = 0; i < inputs.size(); ++i) {
        EXPECT_NEAR_F(out[i], gelu_reference(inputs[i]), kEps);
        EXPECT_NEAR_F(grad[i], gelu_derivative_reference(inputs[i]), kEps);
    }
}

} // namespace

int main() {
    std::cout << "=== GELU 算子单元测试 ===" << std::endl;
    test_cpu_forward_reference();
    test_cpu_backward_reference();
    test_cpu_backward_numerical();
    test_mps_forward_vs_cpu();
    test_mps_backward_vs_cpu();
    test_mps_backward_through_matmul();
    test_cpu_random_consistency();
    std::cout << "\n通过: " << g_passed << " / 失败: " << g_failed << std::endl;
    return g_failed == 0 ? 0 : 1;
}
