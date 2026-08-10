/**
 * @file test_h2_matmul_precision.cpp
 * @brief H2 缺陷排查：C3 单 kernel hotpath 注入前后，训练真实形状 MatMul 的位级数值差异
 * @details 用训练真实形状 {M,K,K,N} + Xavier 初始化输入，
 *          对比 eager 与调度器注入的 C3 kernel 的精确 max_diff / diff_count。
 *          关键：使用 waitForPendingCompiles + hit_count 确保真的命中了 C3 kernel，
 *          避免 testMatMulEquivalence 的"编译未完成"假阳性。
 * @date 2026/08/07
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#ifndef CT_DISABLE_C3
#include "Tensor.h"
#include "CtorchScheduler.h"
#include "C3/C3HotPathManager.h"
#include "C3/C3KernelRegistry.h"
#include "C3/C3Cleanup.h"
#include "ctQALS/Random.h"
#endif

using namespace ct;

#ifndef CT_DISABLE_C3
static void xavierInit(Tensor& t, size_t fan_in, size_t fan_out) {
    static ctQALS::rng::Xoshiro256PlusPlus rng(42);
    float std = std::sqrt(2.0f / (float)(fan_in + fan_out));
    float* p = t.data_write<float>();
    for (size_t i = 0; i < t.numel(); ++i) {
        float r = 2.0f * rng.uniform_f32() - 1.0f;
        p[i] = r * std;
    }
}

/// 对一个训练形状做位级对比
/// @param M,K,N 训练 MatMul 维度
/// @param name 形状名
static void compareShape(size_t M, size_t K, size_t N, const char* name) {
    // 输入接近训练真实分布：a 接近输入激活(0-1)，b 接近 Xavier 权重
    Tensor a(ShapeTag{}, {M, K}, DType::kFloat, DeviceType::kCPU);
    Tensor b(ShapeTag{}, {K, N}, DType::kFloat, DeviceType::kCPU);
    {
        static ctQALS::rng::Xoshiro256PlusPlus rng(7);
        float* pa = a.data_write<float>();
        for (size_t i = 0; i < a.numel(); ++i) pa[i] = rng.uniform_f32(); // 0-1 激活
        xavierInit(b, K, N); // Xavier 权重
    }

    // 1) eager 基线
    Tensor eager_out = a.matmul(b);
    const float* ed = eager_out.data_read<float>();
    size_t out_numel = eager_out.numel();

    c3::C3KernelRegistry::getInstance().uninstallAll(); // 清空上次注册，确保本次是新鲜注入
    c3::C3HotPathManager::instance().clear();

    // 2) 触发 C3 编译（hot_threshold=5，跑 10 次确保触发）
    for (int i = 0; i < 10; ++i) {
        Tensor t = a.matmul(b);
        (void)t;
    }
    // 3) 等待编译完成（消除假阳性）
    c3::C3HotPathManager::instance().waitForPendingCompiles();

    // 4) 记录注入后 hit 增量
    auto s0 = c3::C3KernelRegistry::getInstance().getStats();

    // 5) 这一步应命中 C3 kernel
    Tensor c3_out = a.matmul(b);
    const float* cd = c3_out.data_read<float>();

    auto s1 = c3::C3KernelRegistry::getInstance().getStats();
    size_t hit_delta = s1.hit_count - s0.hit_count;

    // 6) 位级对比
    double max_diff = 0.0;
    size_t diff_cnt = 0;
    size_t nan_cnt = 0;
    for (size_t i = 0; i < out_numel; ++i) {
        if (std::isnan(cd[i])) ++nan_cnt;
        double d = std::abs((double)ed[i] - (double)cd[i]);
        if (d > max_diff) max_diff = d;
        if (d != 0.0) ++diff_cnt;
        if (d > 1e-5) {} // 阈值统计在下方
    }
    size_t over_1e5 = 0, over_1e4 = 0, over_1e3 = 0;
    for (size_t i = 0; i < out_numel; ++i) {
        double d = std::abs((double)ed[i] - (double)cd[i]);
        if (d > 1e-5) ++over_1e5;
        if (d > 1e-4) ++over_1e4;
        if (d > 1e-3) ++over_1e3;
    }

    fprintf(stderr, "[H2] shape=%s (%zux%zux%zu) hit_delta=%zu max_diff=%.6e nonbit_diff=%zu/%zu over1e-5=%zu over1e-4=%zu over1e-3=%zu nan=%zu\n",
            name, M, K, N, hit_delta, max_diff, diff_cnt, out_numel, over_1e5, over_1e4, over_1e3, nan_cnt);
}
#endif

int main() {
#ifndef CT_DISABLE_C3
    CtorchScheduler::getInstance();
    fprintf(stderr, "=== H2 MatMul 位级精度排查 ===\n");
    // 训练真实形状：前向三层 + backward 常用的 [128,10]@[10,128]
    compareShape(128, 784, 256, "{128,784,784,256} fwd-L1");
    compareShape(128, 256, 128, "{128,256,256,128} fwd-L2");
    compareShape(128, 128, 10,  "{128,128,128,10}  fwd-L3");
    compareShape(128, 10, 128,  "{128,10,10,128}  bwd-grad");
    compareShape(128, 128, 128, "{128,128,128,128} neut");
    c3::shutdownAll();
    fprintf(stderr, "=== done ===\n");
#else
    fprintf(stderr, "CT_DISABLE_C3 定义，跳过 C3 测试\n");
#endif
    return 0;
}