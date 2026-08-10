/**
 * @file test_h2_registry_override.cpp
 * @brief H2 缺陷实证/回归：C3 单 kernel registry 槽位按 (op, dev, shape_hash) 索引，
 *          多形状 kernel 互不覆盖。
 * @details 把两个不同形状的 MatMul kernel install 到同一 (MatMul, CPU)，
 *          验证：
 *          1) 修复前：按 (op,dev) 覆盖，旧形状丢失 → A-afterB MISS 或误用 B；
 *          2) 修复后：按 (op,dev,shape_hash) 索引，A/B 各自独立命中。
 *          本测试断言修复后行为：A 安装后不被 B 覆盖，A/B 均命中正确 kernel。
 * @date 2026/08/07
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#ifndef CT_DISABLE_C3
#include "Tensor.h"
#include "CtorchScheduler.h"
#include "C3/C3KernelRegistry.h"
#include "C3/C3Cleanup.h"
#endif

using namespace ct;

#ifndef CT_DISABLE_C3

// 可观测 kernel：输出 = 常数（不读输入，避免 MatMul 循环次数 M*N 与输入 numel M*K 不一致导致的越界读）
// 常数可区分 kernel 身份：ALPHA=1.0，BETA=100.0
// 用 C3KernelFunc 签名：void(*)(const float*, const float*, float*, size_t, size_t, size_t, size_t)
static void fakeKernelAlpha(const float*, const float*, float* out,
                            size_t, size_t M, size_t K, size_t N) {
    (void)K;
    for (size_t i = 0; i < M * N; ++i) out[i] = 1.0f;
}
static void fakeKernelBeta(const float*, const float*, float* out,
                           size_t, size_t M, size_t K, size_t N) {
    (void)K;
    for (size_t i = 0; i < M * N; ++i) out[i] = 100.0f;
}

static double checkOut(const Tensor& t, float scale) {
    const float* d = t.data_read<float>();
    size_t n = t.numel();
    double maxd = 0;
    for (size_t i = 0; i < n; ++i) {
        // 期望值: 输入是 1.0，期望 out = 1.0 * scale
        double expect = 1.0 * scale;
        double dd = std::abs((double)d[i] - expect);
        if (dd > maxd) maxd = dd;
    }
    return maxd;
}

int main() {
    CtorchScheduler::getInstance();
    fprintf(stderr, "=== H2 registry 槽位覆盖实证 ===\n");

    auto& reg = c3::C3KernelRegistry::getInstance();
    reg.uninstallAll();

    // ---- 形状 A: MatMul {M=2,K=3,N=4} ----
    {
        c3::KernelShapeInfo sa;
        sa.is_matmul = true;
        sa.M = 2; sa.K = 3; sa.N = 4;
        sa.lhs_shape = {2, 3};
        sa.rhs_shape = {3, 4};
        sa.out_shape = {2, 4};
        reg.install(op::MatMul, DeviceType::kCPU, fakeKernelAlpha, sa);
        fprintf(stderr, "[install] 形状 A {2x3x4} kernel=ALPHA(x1)\n");
    }

    // 形状 A 调用：应命中 A（x1）
    {
        Tensor a(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
        Tensor b(ShapeTag{}, {3, 4}, DType::kFloat, DeviceType::kCPU);
        { float* p = a.data_write<float>(); for (size_t i=0;i<a.numel();++i) p[i]=1.0f; }
        { float* p = b.data_write<float>(); for (size_t i=0;i<b.numel();++i) p[i]=1.0f; }
        auto r = reg.tryExecute(op::MatMul, a, b);
        if (r.has_value())
            fprintf(stderr, "[tryExecute A] hit, max_diff_vs_ALPHA=%.1e\n", checkOut(r.value(), 1.0f));
        else
            fprintf(stderr, "[tryExecute A] MISS(回退eager)\n");
    }

    // ---- 形状 B: MatMul {M=5,K=6,N=7} 安装到同一 (MatMul, CPU) 槽位 ----
    {
        c3::KernelShapeInfo sb;
        sb.is_matmul = true;
        sb.M = 5; sb.K = 6; sb.N = 7;
        sb.lhs_shape = {5, 6};
        sb.rhs_shape = {6, 7};
        sb.out_shape = {5, 7};
        reg.install(op::MatMul, DeviceType::kCPU, fakeKernelBeta, sb);
        fprintf(stderr, "[install] 形状 B {5x6x7} kernel=BETA(x100) 覆盖同一槽位\n");
    }

    // 再次调用形状 A：修复后应仍命中 A（key 含 shape_hash，不被 B 覆盖）
    {
        Tensor a(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
        Tensor b(ShapeTag{}, {3, 4}, DType::kFloat, DeviceType::kCPU);
        { float* p = a.data_write<float>(); for (size_t i=0;i<a.numel();++i) p[i]=1.0f; }
        { float* p = b.data_write<float>(); for (size_t i=0;i<b.numel();++i) p[i]=1.0f; }
        auto r = reg.tryExecute(op::MatMul, a, b);
        if (r.has_value()) {
            double d = checkOut(r.value(), 1.0f); // 期望 A(x1)
            if (d < 1e-6f) {
                fprintf(stderr, "[PASS] tryExecute A-afterB HIT shape A kernel(ALPHA x1), max_diff=%.1e\n", d);
            } else {
                fprintf(stderr, "[FAIL] tryExecute A-afterB HIT but result polluted by B, max_diff_vs_ALPHA=%.1e\n", d);
                return 1;
            }
        } else {
            fprintf(stderr, "[FAIL] tryExecute A-afterB MISS(回退eager)——A 被 B 覆盖丢失\n");
            return 1;
        }
    }

    // ---- 形状 B 调用：应命中 B ----
    {
        Tensor a(ShapeTag{}, {5, 6}, DType::kFloat, DeviceType::kCPU);
        Tensor b(ShapeTag{}, {6, 7}, DType::kFloat, DeviceType::kCPU);
        { float* p = a.data_write<float>(); for (size_t i=0;i<a.numel();++i) p[i]=1.0f; }
        { float* p = b.data_write<float>(); for (size_t i=0;i<b.numel();++i) p[i]=1.0f; }
        auto r = reg.tryExecute(op::MatMul, a, b);
        if (r.has_value()) {
            double d = checkOut(r.value(), 100.0f);
            if (d < 1e-6f) {
                fprintf(stderr, "[PASS] tryExecute B HIT shape B kernel(BETA x100), max_diff=%.1e\n", d);
            } else {
                fprintf(stderr, "[FAIL] tryExecute B HIT but wrong kernel, max_diff_vs_BETA=%.1e\n", d);
                return 1;
            }
        } else {
            fprintf(stderr, "[FAIL] tryExecute B MISS(回退eager)\n");
            return 1;
        }
    }

    // 统计槽位数量：修复后 A/B 应共存（active_entries=2，各占独立形状槽位）
    auto st = reg.getStats();
    fprintf(stderr, "=== active_entries=%zu install=%zu ===\n", st.active_entries, st.install_count);
    if (st.active_entries < 2) {
        fprintf(stderr, "[FAIL] 期望 A/B 两形状 kernel 共存 (active_entries>=2)，实际=%zu\n", st.active_entries);
        return 1;
    }
    fprintf(stderr, "[PASS] active_entries=%zu：多形状 kernel 共存，互不覆盖\n", st.active_entries);

    c3::shutdownAll();
    fprintf(stderr, "=== done ===\n");
    return 0;
#else
    fprintf(stderr, "CT_DISABLE_C3 定义，跳过 C3 测试\n");
#endif
    return 0;
}