// PEL25 Stage 4 microbench: SiLU/SwiGLU 性能对比
// 跨平台: 用 vsilu / vswiglu 跨平台 wrapper (自动 AVX2/AVX-512/NEON/fallback)
//
// 对比:
//   - Stage 1 伪 SIMD (Stage 1 ops/SwiGLU.cpp 实际走的路径: SIMD+标量 exp)
//   - Stage 4 真 SIMD (vsilu / vswiglu 跨平台 wrapper, 走 polynomial sigmoid)
//
// 编译 (build 目录内):
//   g++ -O3 -ffast-math -march=native -std=c++17 \
//       -I /Users/ghostface/CTorch-optimize-AutoDiff/include \
//       -I /Users/ghostface/CTorch-optimize-AutoDiff/src \
//       -I /opt/homebrew/Cellar/llvm/22.1.8/include \
//       bench/bench_swiglu_simd.cpp \
//       CMakeFiles/CTorch.dir/src/kernels/CPU-SIMD/SIMDMath.cpp.o \
//       -o bench_swiglu_simd
//
// 跑: ./bench_swiglu_simd [N] [iters]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <vector>
#include <random>
#include <string>

#include "kernels/SIMDMath.h"
#include "Tensor.h"
#include "AutoGrad.h"
#include "ops/SiLU.h"
#include "ops/SwiGLU.h"

namespace {

// ============== Stage 1 伪 SIMD: 平台对齐 SIMD lane + 标量 std::exp 回退 ==============
// 复制自 ops/SiLU.cpp + ops/SwiGLU.cpp 的 internal kernel
// 在 ARM64 上走 4-wide NEON store/load, x86 上走 8-wide AVX2 store/load
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
inline void silu_pseudo_neon(const float* in, float* out, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(in + i);
        float xs[4];
        vst1q_f32(xs, vx);
        xs[0] = xs[0] / (1.0f + std::exp(-xs[0]));
        xs[1] = xs[1] / (1.0f + std::exp(-xs[1]));
        xs[2] = xs[2] / (1.0f + std::exp(-xs[2]));
        xs[3] = xs[3] / (1.0f + std::exp(-xs[3]));
        vst1q_f32(out + i, vld1q_f32(xs));
    }
    for (; i < n; ++i) {
        out[i] = in[i] / (1.0f + std::exp(-in[i]));
    }
}

inline void swiglu_pseudo_neon(const float* x_in, const float* g_in, float* out, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t vx = vld1q_f32(x_in + i);
        float32x4_t vg = vld1q_f32(g_in + i);
        float xs[4], gs[4];
        vst1q_f32(xs, vx);
        vst1q_f32(gs, vg);
        for (int k = 0; k < 4; ++k) {
            xs[k] = xs[k] / (1.0f + std::exp(-xs[k])) * gs[k];
        }
        vst1q_f32(out + i, vld1q_f32(xs));
    }
    for (; i < n; ++i) {
        float s = x_in[i] / (1.0f + std::exp(-x_in[i]));
        out[i] = s * g_in[i];
    }
}
#endif

// ============== Bench 函数 ==============

double bench_silu_pseudo(const std::vector<float>& in, std::vector<float>& out, int iters) {
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    float sum = 0.0f;
    for (int it = 0; it < iters; ++it) {
#if defined(__ARM_NEON) || defined(__aarch64__)
        silu_pseudo_neon(in.data(), out.data(), in.size());
#else
        for (size_t i = 0; i < in.size(); ++i) {
            out[i] = in[i] / (1.0f + std::exp(-in[i]));
        }
#endif
        sum += out[0] + out[in.size() - 1];
    }
    auto t1 = clk::now();
    (void)sum;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

double bench_silu_real(const std::vector<float>& in, std::vector<float>& out, int iters) {
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    float sum = 0.0f;
    for (int it = 0; it < iters; ++it) {
        ct::kernels::simd::vsilu(in.data(), out.data(), in.size());
        sum += out[0] + out[in.size() - 1];
    }
    auto t1 = clk::now();
    (void)sum;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

double bench_swiglu_pseudo(const std::vector<float>& in, const std::vector<float>& gate, std::vector<float>& out, int iters) {
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    float sum = 0.0f;
    for (int it = 0; it < iters; ++it) {
#if defined(__ARM_NEON) || defined(__aarch64__)
        swiglu_pseudo_neon(in.data(), gate.data(), out.data(), in.size());
#else
        for (size_t i = 0; i < in.size(); ++i) {
            float s = in[i] / (1.0f + std::exp(-in[i]));
            out[i] = s * gate[i];
        }
#endif
        sum += out[0] + out[in.size() - 1];
    }
    auto t1 = clk::now();
    (void)sum;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

double bench_swiglu_real(const std::vector<float>& in, const std::vector<float>& gate, std::vector<float>& out, int iters) {
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    float sum = 0.0f;
    for (int it = 0; it < iters; ++it) {
        ct::kernels::simd::vswiglu(in.data(), gate.data(), out.data(), in.size());
        sum += out[0] + out[in.size() - 1];
    }
    auto t1 = clk::now();
    (void)sum;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

double bench_silu_pure_scalar(const std::vector<float>& in, std::vector<float>& out, int iters) {
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    float sum = 0.0f;
    for (int it = 0; it < iters; ++it) {
        for (size_t i = 0; i < in.size(); ++i) {
            out[i] = in[i] / (1.0f + std::exp(-in[i]));
        }
        sum += out[0] + out[in.size() - 1];
    }
    auto t1 = clk::now();
    (void)sum;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ============== 端到端 (用户代码) bench ==============
// Stage 1: ct::ops::silu_forward (旧 ops 路径, 伪 SIMD)
// Stage 5.1: Tensor::silu() (新 dispatch 路径, 真 SIMD via C3)
//
// e2e bench 在循环外创建 Tensor, 只测 forward 调用本身, 避免分配开销

Tensor make_tensor_from(const std::vector<float>& data) {
    Tensor t(ShapeTag{}, {data.size()}, DType::kFloat, DeviceType::kCPU);
    std::copy(data.begin(), data.end(), t.data_write<float>());
    return t;
}

double bench_silu_e2e_stage1(const std::vector<float>& in, int iters) {
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    volatile float sink = 0.0f;
    for (int it = 0; it < iters; ++it) {
        Tensor x = make_tensor_from(in);
        Tensor y = ct::ops::silu_forward(x);
        sink += y.data_read<float>()[0];
    }
    auto t1 = clk::now();
    (void)sink;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

double bench_silu_e2e_stage51(const std::vector<float>& in, int iters) {
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    volatile float sink = 0.0f;
    for (int it = 0; it < iters; ++it) {
        Tensor x = make_tensor_from(in);
        Tensor y = x.silu();
        sink += y.data_read<float>()[0];
    }
    auto t1 = clk::now();
    (void)sink;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

double bench_swiglu_e2e_stage1(const std::vector<float>& in, const std::vector<float>& gate, int iters) {
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    volatile float sink = 0.0f;
    for (int it = 0; it < iters; ++it) {
        Tensor x = make_tensor_from(in);
        Tensor g = make_tensor_from(gate);
        Tensor y = ct::ops::swiglu_forward(x, g);
        sink += y.data_read<float>()[0];
    }
    auto t1 = clk::now();
    (void)sink;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

double bench_swiglu_e2e_stage51(const std::vector<float>& in, const std::vector<float>& gate, int iters) {
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();
    volatile float sink = 0.0f;
    for (int it = 0; it < iters; ++it) {
        Tensor x = make_tensor_from(in);
        Tensor g = make_tensor_from(gate);
        Tensor y = x.swiglu(g);
        sink += y.data_read<float>()[0];
    }
    auto t1 = clk::now();
    (void)sink;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    size_t N = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : (1u << 20);  // 1M
    int iters = (argc > 2) ? std::atoi(argv[2]) : 100;

    // 对齐到 16
    if (N % 16 != 0) N = (N / 16 + 1) * 16;

    std::vector<float> in(N), gate(N), out(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    for (size_t i = 0; i < N; ++i) {
        in[i] = dist(rng);
        gate[i] = dist(rng);
    }

    // warm-up
    bench_silu_pseudo(in, out, 5);
    bench_silu_real(in, out, 5);
    bench_swiglu_pseudo(in, gate, out, 5);
    bench_swiglu_real(in, gate, out, 5);
    bench_silu_pure_scalar(in, out, 5);

#if defined(__ARM_NEON) || defined(__aarch64__)
    const char* platform = "ARM64 (NEON, Apple Silicon)";
#else
    const char* platform = "x86_64 (AVX2)";
#endif

    printf("========================================================\n");
    printf(" PEL25 Stage 4 SiLU/SwiGLU SIMD microbench\n");
    printf("========================================================\n");
    printf(" platform      : %s\n", platform);
    printf(" N             : %zu elements (%.2f MB per buffer)\n", N, (double)(N * sizeof(float)) / (1024 * 1024));
    printf(" iters         : %d (per kernel)\n", iters);
    printf(" compiler      : g++ -O3 -ffast-math -march=native\n");
    printf("========================================================\n\n");

    auto fmt = [](double ms, size_t n, int iters) {
        double total_elems = (double)n * iters;
        double sec = ms / 1000.0;
        return std::to_string(ms) + " ms  |  " +
               std::to_string(total_elems / sec / 1e6) + " M elem/s  |  " +
               std::to_string((double)total_elems * 8.0 / sec / 1e9) + " GB/s (2x32-bit)";
    };

    // ============== SiLU ==============
    printf("[SiLU forward]\n");
    double t_scalar   = bench_silu_pure_scalar(in, out, iters);
    double t_pseudo   = bench_silu_pseudo(in, out, iters);
    double t_real     = bench_silu_real(in, out, iters);
    printf("  scalar pure (std::exp)     : %s\n", fmt(t_scalar, N, iters).c_str());
    printf("  Stage1 SIMD+std::exp(pseudo): %s\n", fmt(t_pseudo, N, iters).c_str());
    printf("  Stage4 SIMD+poly  (real)   : %s\n", fmt(t_real,   N, iters).c_str());
    printf("  --- 加速比 (real vs pure scalar): %.2fx\n", t_scalar / t_real);
    printf("  --- 加速比 (real vs pseudo    ): %.2fx\n\n", t_pseudo / t_real);

    // ============== SwiGLU ==============
    printf("[SwiGLU forward] (双输入 fused)\n");
    double t_sw_pseudo = bench_swiglu_pseudo(in, gate, out, iters);
    double t_sw_real   = bench_swiglu_real(in, gate, out, iters);
    printf("  Stage1 SIMD+std::exp(pseudo): %s\n", fmt(t_sw_pseudo, N, iters).c_str());
    printf("  Stage4 SIMD+poly  (real)   : %s\n", fmt(t_sw_real,   N, iters).c_str());
    printf("  --- 加速比 (real vs pseudo): %.2fx\n\n", t_sw_pseudo / t_sw_real);

    // ============== 数值精度 sanity ==============
    printf("[数值精度 sanity] 前 8 个元素 (scalar vs real):\n");
    (void)N; (void)iters;  // 防止 unused warning
    (void)gate; (void)out;
    printf("  idx |     in     | scalar silu  |  real silu  | |err|\n");
    for (size_t i = 0; i < 8; ++i) {
        float scalar_v = in[i] / (1.0f + std::exp(-in[i]));
        out.assign(N, 0.0f);
        ct::kernels::simd::vsilu(in.data(), out.data(), N);
        printf("  %2zu  | %9.4f | %12.6f | %12.6f | %.2e\n",
               i, in[i], scalar_v, out[i], std::abs(scalar_v - out[i]));
    }

    // ============== 端到端 (用户代码) bench ==============
    // 对比 Stage 1 (ct::ops::silu_forward) vs Stage 5.1 (Tensor::silu() 走 C3 dispatch)
    // e2e 包含: Tensor 构造 + forward kernel 调用 (alloc 开销外)
    int e2e_iters = std::min(iters, 20);  // e2e 慢, 20 次足够稳定
    printf("\n[端到端用户代码 bench] (含 Tensor 分配, alloc 外开销):\n");
    // warm-up
    bench_silu_e2e_stage1(in, 3);
    bench_silu_e2e_stage51(in, 3);
    bench_swiglu_e2e_stage1(in, gate, 3);
    bench_swiglu_e2e_stage51(in, gate, 3);

    auto fmt_e2e = [](double ms, int iters) {
        return std::to_string(ms) + " ms  |  " +
               std::to_string((double)iters * 1000.0 / ms) + " call/s";
    };

    double t_s1_silu = bench_silu_e2e_stage1(in, e2e_iters);
    double t_s5_silu = bench_silu_e2e_stage51(in, e2e_iters);
    printf("  SiLU Stage 1  (ct::ops::silu_forward) : %s\n", fmt_e2e(t_s1_silu, e2e_iters).c_str());
    printf("  SiLU Stage 5.1 (Tensor::silu dispatch): %s\n", fmt_e2e(t_s5_silu, e2e_iters).c_str());
    printf("  --- 加速比 (e2e Stage 5.1 vs Stage 1): %.2fx\n", t_s1_silu / t_s5_silu);

    double t_s1_swg = bench_swiglu_e2e_stage1(in, gate, e2e_iters);
    double t_s5_swg = bench_swiglu_e2e_stage51(in, gate, e2e_iters);
    printf("  SwiGLU Stage 1  (ct::ops::swiglu_forward) : %s\n", fmt_e2e(t_s1_swg, e2e_iters).c_str());
    printf("  SwiGLU Stage 5.1 (Tensor::swiglu dispatch): %s\n", fmt_e2e(t_s5_swg, e2e_iters).c_str());
    printf("  --- 加速比 (e2e Stage 5.1 vs Stage 1): %.2fx\n", t_s1_swg / t_s5_swg);

    printf("========================================================\n");

    return 0;
}
