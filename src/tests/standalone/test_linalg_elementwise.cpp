/**
 * @file test_linalg_elementwise.cpp
 * @brief 端到端正确性验证：LinalgElementwiseGen（linalg.generic 声明式逐元素 kernel）
 *
 * 对 8 个算子（ReLU/Sigmoid/Tanh/Exp/Log/Add/Sub/Mul）× 4 个数据规模
 * （16/128/1024/1048576），验证 JIT 输出与手写参考逐元素一致。
 * 这是 STATUS_CONTEXT 4.7-2「用 linalg.generic 统一覆盖逐元素算子」的移植验收测试。
 *
 * 编译/运行：见 CMake target `test_linalg_elementwise`
 * @date 2026/08/15
 */

#include "C3/LinalgElementwiseGen.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace ct::c3;

// ======================= 参考实现 =======================

static float ref_relu(float x) { return x > 0.f ? x : 0.f; }
static float ref_sigmoid(float x) { return 1.f / (1.f + std::exp(-x)); }
static float ref_tanh(float x) { return std::tanh(x); }
static float ref_exp(float x) { return std::exp(x); }
static float ref_log(float x) { return std::log(x); }

// ======================= 验证逻辑 =======================

struct TestCase {
    ElementwiseOp op;
    const char* name;
};

static bool verifyApproxEqual(const float* expected, const float* actual, size_t n,
                              float eps) {
    for (size_t i = 0; i < n; ++i) {
        float diff = std::fabs(expected[i] - actual[i]);
        if (diff > eps) {
            std::fprintf(stderr, "  MISMATCH [%zu]: expected=%f actual=%f (diff=%f > %f)\n",
                         i, expected[i], actual[i], diff, eps);
            return false;
        }
    }
    return true;
}

static bool runCase(const TestCase& tc, size_t n) {
    const size_t num_in = elementwiseOpNumInputs(tc.op);
    const bool unary = isUnaryElementwiseOp(tc.op);
    float eps = (tc.op == ElementwiseOp::Sigmoid || tc.op == ElementwiseOp::Tanh ||
                 tc.op == ElementwiseOp::Exp || tc.op == ElementwiseOp::Log)
                    ? 1e-4f
                    : 1e-5f;

    // 生成输入
    std::vector<std::vector<float>> inputs(num_in, std::vector<float>(n));
    for (size_t i = 0; i < n; ++i) {
        float x = (static_cast<float>(std::rand()) / RAND_MAX) * 2.f - 1.f;
        if (tc.op == ElementwiseOp::Log) x = std::fabs(x) + 0.1f; // log 输入必须 > 0
        inputs[0][i] = x;
        if (num_in > 1) {
            inputs[1][i] = (static_cast<float>(std::rand()) / RAND_MAX) * 2.f - 1.f;
        }
    }

    // 参考输出
    std::vector<float> expected(n);
    for (size_t i = 0; i < n; ++i) {
        float a = inputs[0][i];
        float b = num_in > 1 ? inputs[1][i] : 0.f;
        switch (tc.op) {
        case ElementwiseOp::ReLU: expected[i] = ref_relu(a); break;
        case ElementwiseOp::Sigmoid: expected[i] = ref_sigmoid(a); break;
        case ElementwiseOp::Tanh: expected[i] = ref_tanh(a); break;
        case ElementwiseOp::Exp: expected[i] = ref_exp(a); break;
        case ElementwiseOp::Log: expected[i] = ref_log(a); break;
        case ElementwiseOp::Add: expected[i] = a + b; break;
        case ElementwiseOp::Sub: expected[i] = a - b; break;
        case ElementwiseOp::Mul: expected[i] = a * b; break;
        }
    }

    // 编译 + 执行 linalg kernel
    LinalgElementwiseKernel kernel(tc.op);
    std::vector<const float*> in_ptrs;
    for (const auto& in : inputs) in_ptrs.push_back(in.data());
    std::vector<float> actual(n, -1.f);
    kernel.execute(in_ptrs.data(), actual.data(), n);

    // 验证
    bool ok = verifyApproxEqual(expected.data(), actual.data(), n, eps);
    std::printf("  [%s] n=%-9zu unary=%-5s => %s\n", tc.name, n,
                unary ? "yes" : "no", ok ? "PASSED" : "FAILED");
    return ok;
}

// ======================= 主入口 =======================

int main() {
    std::srand(42);
    const TestCase cases[] = {
        {ElementwiseOp::ReLU, "ReLU"},
        {ElementwiseOp::Sigmoid, "Sigmoid"},
        {ElementwiseOp::Tanh, "Tanh"},
        {ElementwiseOp::Exp, "Exp"},
        {ElementwiseOp::Log, "Log"},
        {ElementwiseOp::Add, "Add"},
        {ElementwiseOp::Sub, "Sub"},
        {ElementwiseOp::Mul, "Mul"},
    };
    const size_t sizes[] = {16, 128, 1024, 1048576};
    constexpr int num_cases = static_cast<int>(sizeof(cases) / sizeof(cases[0]));
    constexpr int num_sizes = static_cast<int>(sizeof(sizes) / sizeof(sizes[0]));

    std::printf("==================================================\n");
    std::printf("  LinalgElementwiseGen 端到端正确性验证 (%d ops x %d sizes)\n",
                num_cases, num_sizes);
    std::printf("==================================================\n");

    int passed = 0;
    int total = 0;
    for (int c = 0; c < num_cases; ++c) {
        for (int s = 0; s < num_sizes; ++s) {
            ++total;
            if (runCase(cases[c], sizes[s])) ++passed;
        }
    }

    std::printf("==================================================\n");
    std::printf("  RESULT: %d/%d passed\n", passed, total);
    std::printf("==================================================\n");
    return (passed == total) ? 0 : 1;
}
