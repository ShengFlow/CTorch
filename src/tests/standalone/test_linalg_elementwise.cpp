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
#include "C3/JITCache.h"

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

    // ======================= 标量广播正确性验证 (新管线) =======================
    std::printf("\n--- 标量广播 (rhs_broadcast, linalg indexing map d0->0) ---\n");
    int bc_passed = 0, bc_total = 0;
    const ElementwiseOp bc_ops[] = {ElementwiseOp::Add, ElementwiseOp::Sub, ElementwiseOp::Mul};
    const char* bc_names[] = {"Add(bc)", "Sub(bc)", "Mul(bc)"};
    const size_t bc_sizes[] = {16, 128, 1024, 1048576};
    for (int c = 0; c < 3; ++c) {
        for (int s = 0; s < 4; ++s) {
            ++bc_total;
            size_t n = bc_sizes[s];
            // lhs 随机 n 个元素，rhs 仅 1 个标量
            std::vector<float> lhs(n);
            float rhs_val = (static_cast<float>(std::rand()) / RAND_MAX) * 2.f - 1.f;
            for (size_t i = 0; i < n; ++i) {
                lhs[i] = (static_cast<float>(std::rand()) / RAND_MAX) * 2.f - 1.f;
            }
            std::vector<float> expected(n);
            for (size_t i = 0; i < n; ++i) {
                switch (bc_ops[c]) {
                case ElementwiseOp::Add: expected[i] = lhs[i] + rhs_val; break;
                case ElementwiseOp::Sub: expected[i] = lhs[i] - rhs_val; break;
                case ElementwiseOp::Mul: expected[i] = lhs[i] * rhs_val; break;
                default: break;
                }
            }

            // 走缓存工厂，验证标量广播 kernel
            auto kernel = getCachedLinalgKernel(bc_ops[c], 3, true);
            const float* in_ptrs[2] = {lhs.data(), &rhs_val};
            std::vector<float> actual(n, -1.f);
            kernel->execute(in_ptrs, actual.data(), n);

            bool ok = verifyApproxEqual(expected.data(), actual.data(), n, 1e-5f);
            std::printf("  [%s] n=%-9zu => %s\n", bc_names[c], n, ok ? "PASSED" : "FAILED");
            if (ok) ++bc_passed;
        }
    }
    std::printf("--- Broadcast RESULT: %d/%d passed ---\n", bc_passed, bc_total);

    // ======================= 缓存工厂验证 =======================
    std::printf("\n--- 缓存工厂 (getCachedLinalgKernel, C3_LINALG_CACHE=0 逃生) ---\n");
    auto k1 = getCachedLinalgKernel(ElementwiseOp::ReLU);
    auto k2 = getCachedLinalgKernel(ElementwiseOp::ReLU);
    bool cache_hit = (k1.get() == k2.get());
    std::printf("  [Cache] 同一 (ReLU,3,0) 两次 getCachedLinalgKernel: %s\n",
                cache_hit ? "HIT (shared)" : "miss (different)");
    // 不同 op 应不同指针
    auto k3 = getCachedLinalgKernel(ElementwiseOp::Sigmoid);
    bool cache_miss = (k1.get() != k3.get());
    std::printf("  [Cache] ReLU vs Sigmoid 不同指针: %s\n",
                cache_miss ? "OK (different)" : "UNEXPECTED (same!)");
    int cache_passed = (cache_hit && cache_miss) ? 2 : 0;

    // ======================= 周期广播正确性验证 (管线②) =======================
    std::printf("\n--- 周期广播 (rhs_mod=k>1, linalg indexing map d0 -> d0 mod k) ---\n");
    const ElementwiseOp pb_ops[] = {ElementwiseOp::Add, ElementwiseOp::Sub, ElementwiseOp::Mul};
    const char* pb_names[] = {"Add(pb)", "Sub(pb)", "Mul(pb)"};
    const size_t pb_k = 4;                       // 周期模数
    const size_t pb_sizes[] = {16, 64, 1024};    // n 均为 k 的倍数
    int pb_passed = 0, pb_total = 0;
    for (int c = 0; c < 3; ++c) {
        for (int s = 0; s < 3; ++s) {
            ++pb_total;
            size_t n = pb_sizes[s];
            std::vector<float> lhs(n);
            std::vector<float> rhs(pb_k);
            for (size_t i = 0; i < n; ++i)
                lhs[i] = (static_cast<float>(std::rand()) / RAND_MAX) * 2.f - 1.f;
            for (size_t i = 0; i < pb_k; ++i)
                rhs[i] = (static_cast<float>(std::rand()) / RAND_MAX) * 2.f - 1.f;
            std::vector<float> expected(n);
            for (size_t i = 0; i < n; ++i) {
                float b = rhs[i % pb_k];
                switch (pb_ops[c]) {
                case ElementwiseOp::Add: expected[i] = lhs[i] + b; break;
                case ElementwiseOp::Sub: expected[i] = lhs[i] - b; break;
                case ElementwiseOp::Mul: expected[i] = lhs[i] * b; break;
                default: break;
                }
            }
            // 走缓存工厂，rhs_mod=k 周期广播 kernel
            auto kernel = getCachedLinalgKernel(pb_ops[c], 3, static_cast<int>(pb_k));
            const float* in_ptrs[2] = {lhs.data(), rhs.data()};
            std::vector<float> actual(n, -1.f);
            kernel->execute(in_ptrs, actual.data(), n);
            bool ok = verifyApproxEqual(expected.data(), actual.data(), n, 1e-5f);
            std::printf("  [%s] n=%-6zu k=%zu => %s\n", pb_names[c], n, pb_k,
                        ok ? "PASSED" : "FAILED");
            if (ok) ++pb_passed;
        }
    }
    std::printf("--- Periodic-broadcast RESULT: %d/%d passed ---\n", pb_passed, pb_total);

    // ======================= 多维张量广播正确性验证 =======================
    std::printf("\n--- 多维张量广播 (Multi-dimensional broadcast, LinalgBroadcastingKernel) ---\n");
    int md_passed = 0, md_total = 0;
    const ElementwiseOp md_ops[] = {ElementwiseOp::Add, ElementwiseOp::Sub, ElementwiseOp::Mul};
    const char* md_names[] = {"Add(md)", "Sub(md)", "Mul(md)"};
    
    // 模拟常见的多维广播场景：
    // Scenario 1: [2, 3] + [1, 3] -> [2, 3] (按行广播)
    // Scenario 2: [2, 3] + [2, 1] -> [2, 3] (按列广播)
    // Scenario 3: [1, 3] + [2, 1] -> [2, 3] (双向广播)
    struct BroadcastScenario {
        std::vector<size_t> lhs;
        std::vector<size_t> rhs;
        std::vector<size_t> out;
    } scenarios[] = {
        {{2, 3}, {1, 3}, {2, 3}},
        {{2, 3}, {2, 1}, {2, 3}},
        {{1, 3}, {2, 1}, {2, 3}},
        {{1, 2, 3}, {4, 1, 3}, {4, 2, 3}}
    };

    for (int c = 0; c < 3; ++c) {
        for (const auto& sc : scenarios) {
            ++md_total;
            size_t n_lhs = 1, n_rhs = 1, n_out = 1;
            for (size_t d : sc.lhs) n_lhs *= d;
            for (size_t d : sc.rhs) n_rhs *= d;
            for (size_t d : sc.out) n_out *= d;

            std::vector<float> lhs(n_lhs);
            std::vector<float> rhs(n_rhs);
            for (size_t i = 0; i < n_lhs; ++i) lhs[i] = (static_cast<float>(std::rand()) / RAND_MAX) * 2.f - 1.f;
            for (size_t i = 0; i < n_rhs; ++i) rhs[i] = (static_cast<float>(std::rand()) / RAND_MAX) * 2.f - 1.f;

            std::vector<float> expected(n_out);
            // 简单循环参考计算
            size_t rank = sc.out.size();
            std::vector<size_t> padded_lhs = sc.lhs;
            std::vector<size_t> padded_rhs = sc.rhs;
            while (padded_lhs.size() < rank) padded_lhs.insert(padded_lhs.begin(), 1);
            while (padded_rhs.size() < rank) padded_rhs.insert(padded_rhs.begin(), 1);

            for (size_t o_idx = 0; o_idx < n_out; ++o_idx) {
                // 计算高维多维坐标
                size_t temp = o_idx;
                std::vector<size_t> coord(rank);
                for (int i = (int)rank - 1; i >= 0; --i) {
                    coord[i] = temp % sc.out[i];
                    temp /= sc.out[i];
                }

                // 映射回输入坐标
                size_t l_flat = 0, r_flat = 0;
                for (size_t i = 0; i < rank; ++i) {
                    size_t l_c = (padded_lhs[i] == 1) ? 0 : coord[i];
                    size_t r_c = (padded_rhs[i] == 1) ? 0 : coord[i];
                    l_flat = l_flat * padded_lhs[i] + l_c;
                    r_flat = r_flat * padded_rhs[i] + r_c;
                }

                float lv = lhs[l_flat];
                float rv = rhs[r_flat];
                switch (md_ops[c]) {
                case ElementwiseOp::Add: expected[o_idx] = lv + rv; break;
                case ElementwiseOp::Sub: expected[o_idx] = lv - rv; break;
                case ElementwiseOp::Mul: expected[o_idx] = lv * rv; break;
                default: break;
                }
            }

            // 编译多维广播 Kernel
            auto kernel = getCachedLinalgBroadcastKernel(md_ops[c], 3, sc.lhs, sc.rhs, sc.out);
            const float* in_ptrs[2] = {lhs.data(), rhs.data()};
            std::vector<float> actual(n_out, -1.f);
            kernel->execute(in_ptrs, actual.data(), sc.lhs, sc.rhs, sc.out);

            bool ok = verifyApproxEqual(expected.data(), actual.data(), n_out, 1e-5f);
            if (!ok) {
                std::printf("  [DEBUG MISMATCH]\n");
                std::printf("  LHS: "); for (float x : lhs) std::printf("%f, ", x); std::printf("\n");
                std::printf("  RHS: "); for (float x : rhs) std::printf("%f, ", x); std::printf("\n");
                std::printf("  EXP: "); for (float x : expected) std::printf("%f, ", x); std::printf("\n");
                std::printf("  ACT: "); for (float x : actual) std::printf("%f, ", x); std::printf("\n");
            }
            std::printf("  [%s] shapes: ", md_names[c]);
            for (size_t d : sc.lhs) std::printf("%zu,", d);
            std::printf(" + ");
            for (size_t d : sc.rhs) std::printf("%zu,", d);
            std::printf(" -> ");
            for (size_t d : sc.out) std::printf("%zu,", d);
            std::printf(" => %s\n", ok ? "PASSED" : "FAILED");
            if (ok) ++md_passed;
        }
    }
    std::printf("--- Multi-dimensional broadcast RESULT: %d/%d passed ---\n", md_passed, md_total);

    // 缓存 key 区分 rhs_mod：不同 rhs_mod 应不同实例，相同 rhs_mod 应命中
    auto pm0 = getCachedLinalgKernel(ElementwiseOp::Add, 3, 0);
    auto pm1 = getCachedLinalgKernel(ElementwiseOp::Add, 3, 1);
    auto pm4 = getCachedLinalgKernel(ElementwiseOp::Add, 3, 4);
    auto pm4b = getCachedLinalgKernel(ElementwiseOp::Add, 3, 4);
    bool mod_key_ok = (pm0.get() != pm1.get()) && (pm1.get() != pm4.get()) &&
                      (pm0.get() != pm4.get()) && (pm4.get() == pm4b.get());
    std::printf("  [Cache] rhs_mod 区分 (0/1/4 不同, 4==4 命中): %s\n",
                mod_key_ok ? "OK" : "FAIL");

    // ======================= AOT 磁盘持久化缓存验证 (管线①) =======================
    std::printf("\n--- AOT 持久化缓存 (JITCache 2.0 read path, llvmModuleBuilder 注入) ---\n");
    bool aot_ok = true;
    if (JITCache::isEnabled()) {
        JITCache& jc = JITCache::getInstance();
        jc.evict();  // 清空，保证冷启动
        const uint64_t hits0 = jc.hits();
        const uint64_t stores0 = jc.stores();
        {
            // 冷启动：完整 build + lowering + translate → store bitcode
            LinalgElementwiseKernel cold(ElementwiseOp::ReLU, 3, 0);
        }
        const uint64_t stores1 = jc.stores();
        {
            // 热启动：磁盘命中 → loadBitcode + JIT（跳过 build/lowering/translate）
            LinalgElementwiseKernel warm(ElementwiseOp::ReLU, 3, 0);
        }
        const uint64_t hits1 = jc.hits();
        const bool store_ok = (stores1 == stores0 + 1);
        const bool hit_ok = (hits1 == hits0 + 1);
        aot_ok = store_ok && hit_ok;
        std::printf("  [AOT] 冷启动 stores +%llu (%s), 热启动 hits +%llu (%s)\n",
                    static_cast<unsigned long long>(stores1 - stores0),
                    store_ok ? "OK" : "FAIL",
                    static_cast<unsigned long long>(hits1 - hits0),
                    hit_ok ? "OK" : "FAIL");
    } else {
        std::printf("  [AOT] C3_JIT_CACHE_DISABLE=1，跳过\n");
    }

    return ((passed == total) && (bc_passed == bc_total) && (cache_passed == 2) &&
            (pb_passed == pb_total) && mod_key_ok && aot_ok)
               ? 0 : 1;
}
