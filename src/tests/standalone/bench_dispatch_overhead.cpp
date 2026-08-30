// src/tests/standalone/bench_dispatch_overhead.cpp
// 微基准：隔离量化 dispatch 框架固定开销（不含 kernel 执行）。
// 对比基线：
//   [A] 纯 kernel 直调（Add_SIMD_kernel / ReLU_SIMD_kernel / MatMul）
//       —— 只含 kernel 自身执行，不含任何调度框架。
//   [B] 完整 dispatch（CtorchScheduler::dispatch<op>(a,b)）
//       —— 含 dtype/shape 检查 + region fusion 检查 + C3 单 kernel 检查 +
//          recordCall/hotpath 记账 + region trace 记录 + kernel 查找表 + kernel 执行。
//   [C] dispatch 但 in_autograd scope（requires_grad=true）
//       —— 训练期真实路径：recordCall 被短路，但 region trace 记录仍在。
// 差值 (B-A)/(C-A) = dispatch 框架固定开销。
// 用法：在 C3 ON 的 Release+LTO build 下运行，得到真实 dispatch 固定开销。
#include <cstdio>
#include <chrono>
#include <vector>
#include "Tensor.h"
#include "CtorchScheduler.h"
#include "kernels/kernels.h"
#include "C3/C3Cleanup.h"

using Clock = std::chrono::steady_clock;
using namespace ct;

static double ns_per_it(long long iters, double total_ns) {
    return total_ns / (double)iters;
}

// 防优化：把结果地址累积进 sink，避免编译器把循环提升出计时区间
static volatile const void* g_sink = nullptr;

template <typename F>
static double bench(const char* name, int iters, F&& fn) {
    (void)name;
    // 预热
    for (int i = 0; i < std::min(iters, 100); ++i) fn();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) fn();
    auto t1 = Clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

int main() {
    const int ITERS = 200000;
    const size_t M = 128, K = 784, N = 256;   // MNIST 第一层规模

    auto& sched = CtorchScheduler::getInstance();

    // ===== 输入构造（CPU）=====
    Tensor a(ShapeTag{}, {M, K}, DType::kFloat, DeviceType::kCPU);
    Tensor b(ShapeTag{}, {M, K}, DType::kFloat, DeviceType::kCPU);
    float* ap = a.data_write<float>();
    float* bp = b.data_write<float>();
    for (size_t i = 0; i < a.numel(); ++i) { ap[i] = 1.0f; bp[i] = 2.0f; }

    // 元素级 op 用 1D 视角（与 SIMD kernel 对齐）
    Tensor x(ShapeTag{}, {M * K}, DType::kFloat, DeviceType::kCPU);
    float* xp = x.data_write<float>();
    for (size_t i = 0; i < x.numel(); ++i) xp[i] = 1.0f;

    printf("===== Add (元素级) dispatck 固定开销 =====\n");
    {
        // [A] 纯 kernel 直调
        double A = bench("A", ITERS, [&]() {
            Tensor r = Add_SIMD_kernel(a, b);
            g_sink = r.data_read<float>();
        });

        // [B] 完整 dispatch（非 autograd）
        double B = bench("B", ITERS, [&]() {
            Tensor r = sched.template dispatch<op::Add>(a, b);
            g_sink = r.data_read<float>();
        });

        // [C] dispatch，autograd scope（requires_grad=true）
        Tensor a_g = a;
        a_g.set_requires_grad(true);
        double C = bench("C", ITERS, [&]() {
            Tensor r = sched.template dispatch<op::Add>(a_g, b);
            g_sink = r.data_read<float>();
        });

        printf("[A] 纯 kernel 直调        : %.3f ns/次\n", ns_per_it(ITERS, A));
        printf("[B] dispatch(非autograd)  : %.3f ns/次  (框架固定开销=%.3f ns)\n",
               ns_per_it(ITERS, B), ns_per_it(ITERS, B - A));
        printf("[C] dispatch(autograd)    : %.3f ns/次  (框架固定开销=%.3f ns)\n",
               ns_per_it(ITERS, C), ns_per_it(ITERS, C - A));

        // [D] 直接调 tryExecute（跳过 dispatch 框架，只走单 kernel 查询+execute）
        auto& reg2 = ct::c3::C3KernelRegistry::getInstance();
        double D = bench("D", ITERS, [&]() {
            auto r = reg2.tryExecute(op::Add, a, b);
            if (r) g_sink = r->data_read<float>();
        });
        printf("[D] tryExecute直接(Add)   : %.3f ns/次  (相对纯kernel+%.3f ns)\n",
               ns_per_it(ITERS, D), ns_per_it(ITERS, D - A));
    }

    printf("\n===== ReLU (一元) dispatch 固定开销 =====\n");
    {
        double A = bench("A", ITERS, [&]() {
            Tensor r = ReLU_SIMD_kernel(x);
            g_sink = r.data_read<float>();
        });
        double B = bench("B", ITERS, [&]() {
            Tensor r = sched.template dispatch<op::ReLU>(x);
            g_sink = r.data_read<float>();
        });
        Tensor x_g = x;
        x_g.set_requires_grad(true);
        double C = bench("C", ITERS, [&]() {
            Tensor r = sched.template dispatch<op::ReLU>(x_g);
            g_sink = r.data_read<float>();
        });
        printf("[A] 纯 kernel 直调        : %.3f ns/次\n", ns_per_it(ITERS, A));
        printf("[B] dispatch(非autograd)  : %.3f ns/次  (框架固定开销=%.3f ns)\n",
               ns_per_it(ITERS, B), ns_per_it(ITERS, B - A));
        printf("[C] dispatch(autograd)    : %.3f ns/次  (框架固定开销=%.3f ns)\n",
               ns_per_it(ITERS, C), ns_per_it(ITERS, C - A));
        // [D] tryExecuteUnary 直接调（隔离：一元单 kernel 查找/执行 vs eager selectBestUnary）
        auto& reg2u = ct::c3::C3KernelRegistry::getInstance();
        double D = bench("D", ITERS, [&]() {
            auto r = reg2u.tryExecuteUnary(op::ReLU, x);
            if (r) g_sink = r->data_read<float>();
        });
        printf("[D] tryExecuteUnary直接  : %.3f ns/次  (相对纯kernel+%.3f ns)\n",
               ns_per_it(ITERS, D), ns_per_it(ITERS, D - A));
    }

    printf("(sink=%p 防优化)\n", (const void*)g_sink);

    // ===== C3 单 kernel 命中统计 =====
    auto st = ct::c3::C3KernelRegistry::getInstance().getStats();
    printf("\n===== C3 single-kernel 统计 =====\n");
    printf("hit=%zu miss=%zu bypass=%zu active_entries=%zu\n",
           st.hit_count, st.miss_count, st.bypass_count, st.active_entries);
    printf("fused_hit=%zu region_dispatch=%zu region_match=%zu\n",
           st.fused_hit_count, st.region_dispatch_count, st.region_match_count);

    // 退出清理：避免 C3Engine 全局静态析构时 recursive_mutex/removeModule 崩溃
    ct::c3::shutdownAll();
    return 0;
}