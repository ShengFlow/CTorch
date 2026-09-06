// PEL25 Stage 5.2 inference bench: 测 region fusion MatMul+Add+SiLU 加速
//
// 跑 200 次同一 [MatMul → Add → SiLU] 序列 (无 autograd, 推理路径):
//   - 前 ~50 次: cold (region fusion 编译中, 走单 op 路径)
//   - 后 ~150 次: warm (region fusion fused kernel 已就绪, 走 fused 路径)
// 对比 cold vs warm timing, 测 region fusion 收益
//
// 编译 (build 目录内, 复用 test_swiglu link 命令, 替换 .o):
//   1) g++ -O3 -ffast-math -march=native -std=c++17 -DCT_ENABLE_MLIR=1 \
//        -I /Users/ghostface/CTorch-optimize-AutoDiff/include \
//        -I /Users/ghostface/CTorch-optimize-AutoDiff/src \
//        -I /Users/ghostface/CTorch-optimize-AutoDiff/c3/include \
//        -I /Users/ghostface/CTorch-optimize-AutoDiff/c3/src \
//        -I /opt/homebrew/Cellar/llvm/22.1.8/include \
//        /Users/ghostface/CTorch-optimize-AutoDiff/bench/bench_region_fusion_silu.cpp \
//        -c -o /tmp/bench_region_fusion_silu.o
//   2) 用 test_swiglu link 命令, 替换 .o, 输出 /tmp/bench_region_fusion_silu
// 跑: /tmp/bench_region_fusion_silu

#include <cstdio>
#include <chrono>
#include <vector>
#include <random>
#include <string>

#include "Tensor.h"
#include "AutoGrad.h"

namespace {

Tensor make_tensor(const std::vector<float>& data, const std::vector<size_t>& shape) {
    Tensor t(ShapeTag{}, shape, DType::kFloat, DeviceType::kCPU);
    std::copy(data.begin(), data.end(), t.data_write<float>());
    return t;
}

// 1 层 MLP: y = silu(x @ W + b)
// 跑 200 次, 让 C3 region fusion 触发 MatMul+Add+SiLU 编译并 fused
double bench_mlp_inference(int iters, int cold_window = 50) {
    using clk = std::chrono::high_resolution_clock;
    // 构造固定输入
    const size_t M = 64;       // batch
    const size_t K = 1024;     // input dim
    const size_t N = 1024;     // hidden dim

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> x_data(M * K), w_data(K * N), b_data(N);
    for (auto& v : x_data) v = dist(rng);
    for (auto& v : w_data) v = dist(rng);
    for (auto& v : b_data) v = dist(rng);

    Tensor x = make_tensor(x_data, {M, K});
    Tensor W = make_tensor(w_data, {K, N});
    Tensor b = make_tensor(b_data, {N});

    // warm-up (让 C3 启动)
    {
        Tensor h1 = x.matmul(W);
        Tensor h2 = h1 + b;
        Tensor y = h2.silu();
        (void)y.data_read<float>()[0];
    }

    // 测每一次 timing
    std::vector<double> per_iter_ms(iters);
    volatile float sink = 0.0f;
    for (int it = 0; it < iters; ++it) {
        auto t0 = clk::now();
        Tensor h1 = x.matmul(W);    // MatMul
        Tensor h2 = h1 + b;         // Add
        Tensor y = h2.silu();       // SiLU  (Stage 5.1 走 dispatch 路径)
        sink += y.data_read<float>()[0];
        auto t1 = clk::now();
        per_iter_ms[it] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    (void)sink;

    // 统计 cold (前 cold_window, 跳过 compile jitter 离群点) vs warm (后 iters - cold_window)
    // 编译 MLIR 大概要 200-500ms, 那一次不能算在 fused 加速统计里
    auto is_jitter = [](double ms) { return ms > 50.0; };  // >50ms 算 compile jitter

    // cold: 前 cold_window 中, 排除 jitter
    double cold_total = 0.0;
    int cold_count = 0;
    for (int i = 0; i < std::min(cold_window, iters); ++i) {
        if (!is_jitter(per_iter_ms[i])) {
            cold_total += per_iter_ms[i];
            cold_count++;
        }
    }
    double cold_avg = (cold_count > 0) ? cold_total / cold_count : 0.0;

    // warm: 后 iters - cold_window, 排除 jitter
    double warm_total = 0.0;
    int warm_count = 0;
    for (int i = cold_window; i < iters; ++i) {
        if (!is_jitter(per_iter_ms[i])) {
            warm_total += per_iter_ms[i];
            warm_count++;
        }
    }
    double warm_avg = (warm_count > 0) ? warm_total / warm_count : 0.0;

    // 找 compile iter (单次最大 timing, 通常是 MLIR compile)
    int compile_iter = -1;
    double compile_ms = 0.0;
    for (int i = 0; i < iters; ++i) {
        if (per_iter_ms[i] > compile_ms) {
            compile_ms = per_iter_ms[i];
            compile_iter = i;
        }
    }

    printf("========================================================\n");
    printf(" PEL25 Stage 5.2: Region Fusion MatMul+Add+SiLU bench\n");
    printf("========================================================\n");
    printf(" MLP shape: x=[%zu, %zu] @ W=[%zu, %zu] + b=[%zu] -> silu\n", M, K, K, N, N);
    printf(" iters: %d (cold window = first %d, warm = rest)\n", iters, cold_window);
    printf(" platform: ARM64 NEON, g++ -O3 -ffast-math -march=native\n");
    printf("--------------------------------------------------------\n");
    if (compile_iter >= 0 && compile_ms > 50.0) {
        printf(" MLIR compile jitter: iter %d 耗时 %.2f ms (一次性 region fusion 编译)\n",
               compile_iter, compile_ms);
    }
    printf(" cold avg (前 %d 次, 排除 jitter, n=%d):  %.4f ms / iter\n", cold_window, cold_count, cold_avg);
    printf(" warm avg (后 %d 次, 排除 jitter, n=%d):  %.4f ms / iter\n", iters - cold_window, warm_count, warm_avg);
    printf(" --- 加速比 (cold / warm): %.2fx\n", (warm_avg > 0) ? cold_avg / warm_avg : 0.0);
    printf("--------------------------------------------------------\n");
    printf(" 逐 iter timing (前 20 次 + 末 10 次):\n");
    printf("   iter |    ms\n");
    printf("  ------|--------\n");
    int preview = std::min(20, iters);
    for (int i = 0; i < preview; ++i) {
        printf("  %4d  | %.4f %s\n", i, per_iter_ms[i],
               (is_jitter(per_iter_ms[i]) ? "[COMPILE-JITTER]" :
                (i == cold_window - 1) ? "<-- cold/warm 边界" : ""));
    }
    if (iters > 30) {
        printf("  ...   | (省略中间)\n");
        for (int i = iters - 10; i < iters; ++i) {
            printf("  %4d  | %.4f %s\n", i, per_iter_ms[i],
                   is_jitter(per_iter_ms[i]) ? "[JITTER]" : "");
        }
    }
    printf("========================================================\n");

    return warm_avg;
}

}  // namespace

int main(int argc, char** argv) {
    int iters = (argc > 1) ? std::atoi(argv[1]) : 200;
    int cold_window = (argc > 2) ? std::atoi(argv[2]) : 50;
    bench_mlp_inference(iters, cold_window);
    return 0;
}
