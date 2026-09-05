/**
 * @file test_simd_avx512.cpp
 * @brief AVX-512 向量化单元测试（16-wide float32）
 * @details 在 x86-64 AVX-512 (F+DQ) 机器上直接验证 SIMDMath 的 512 实现，
 *          并校验 SIMDConfig 的编译期架构检测在 512 机器上正确选择 Avx512/16 lanes。
 *
 * 构建（在 AVX-512 机器上，<repo>=CTorch-optimize-AutoDiff）：
 *   clang++ -O2 -mavx512f -mavx512dq -mfma -I<repo>/include \\
 *       test_simd_avx512.cpp <repo>/src/kernels/CPU-SIMD/SIMDMath.cpp -o test512
 *   ./test512
 * 也可用 g++。gcc 需同 flag（gcc 用 -mavx512f -mavx512dq -mfma）。
 *
 * @date 2026/09/05
 */
#include "kernels/SIMDMath.h"
#include "kernels/SIMDConfig.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int g_pass = 0, g_fail = 0;
#define CHECK(c, m) do { if (c) { ++g_pass; } else { ++g_fail; std::printf("  FAIL: %s\n", m); } } while (0)

static inline uint32_t as_uint(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static int32_t ulp_diff(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) return 0x7fffffff;
    if (a == b) return 0;
    int32_t ai = (int32_t)as_uint(a), bi = (int32_t)as_uint(b);
    if ((ai < 0) != (bi < 0)) { if (ai < 0) ai = (1u << 31) - ai; if (bi < 0) bi = (1u << 31) - bi; }
    return std::abs(ai - bi);
}

// 在随机输入上校验单条 512 函数的 max ULP / max rel error
template <typename F>
static void check512(const char* name, F simd_fn, float (*ref_fn)(float), float lo, float hi,
                     int seed, bool use_log_space = false) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    const int N = 8192;
    float in[N], ref[N], got[N];
    for (int i = 0; i < N; ++i) {
        float v = dist(rng);
        if (use_log_space) v = std::exp(v);   // 对数空间采样正数域
        in[i] = v;
        ref[i] = ref_fn(v);
    }
    int32_t maxu = 0; double maxre = 0.0;
    for (int i = 0; i + 16 <= N; i += 16) {
        __m512 x = _mm512_loadu_ps(&in[i]);
        __m512 y = simd_fn(x);
        float t[16]; _mm512_storeu_ps(t, y);
        for (int k = 0; k < 16; ++k) {
            int u = ulp_diff(ref[i + k], t[k]); maxu = std::max(maxu, u);
            if (std::abs(ref[i + k]) > 1e-6f) { double re = std::abs((double)t[k] - (double)ref[i + k]) / std::abs((double)ref[i + k]); maxre = std::max(maxre, re); }
        }
    }
    // 标量尾段
    for (int i = (N / 16) * 16; i < N; ++i) got[i] = std::nanf("");  // unused placeholder
    std::printf("  %-10s maxULP=%d maxRelErr=%.3e\n", name, maxu, maxre);
    CHECK(maxu <= 4, "ULP<=4 (algorithm tolerance)"); CHECK(maxre < 1e-4, "rel<1e-4");
}

int main() {
    std::printf("=== SIMDMath AVX-512 test ===\n");
    std::printf("kSimdArch=%d (Avx512=%d) kSimdFloatLanes=%zu bits=%d\n",
                (int)ct::kernels::simd::kSimdArch, (int)ct::kernels::simd::SimdArch::Avx512,
                ct::kernels::simd::kSimdFloatLanes, ct::kernels::simd::kVecWidthBits);

#if defined(__AVX512F__) && defined(__AVX512DQ__)
    CHECK(ct::kernels::simd::kSimdArch == ct::kernels::simd::SimdArch::Avx512, "kSimdArch==Avx512");
    CHECK(ct::kernels::simd::kSimdFloatLanes == 16, "kSimdFloatLanes==16");
    CHECK(ct::kernels::simd::kVecWidthBits == 512, "bits==512");

    check512("exp512", [](__m512 x){ return ct::kernels::simd::exp512_ps(x); }, std::exp, -10, 10, 1);
    check512("log512", [](__m512 x){ return ct::kernels::simd::log512_ps(x); }, std::log, 0.001f, 1.0f, 2, /*log_space*/true);
    check512("tanh512", [](__m512 x){ return ct::kernels::simd::tanh512_ps(x); }, std::tanh, -8, 8, 3);
    check512("sigmoid512", [](__m512 x){ return ct::kernels::simd::sigmoid512_ps(x); },
             [](float x){ return 1.0f/(1.0f+std::exp(-x)); }, -12, 12, 4);
    check512("gelu512", [](__m512 x){ return ct::kernels::simd::gelu512_ps(x); },
             [](float x){ float v=0.7978845608f*(x+0.044715f*x*x*x); return 0.5f*x*(1.0f+std::tanh(v)); }, -6, 6, 5);
    check512("rsqrt512", [](__m512 x){ return ct::kernels::simd::rsqrt512_ps(x); },
             [](float x){ return 1.0f/std::sqrt(x); }, 1.0f, 100.0f, 6, /*log_space*/true);
#else
    std::printf("NOTE: not compiled with AVX-512 (need -mavx512f -mavx512dq); wrapper path only.\n");
#endif

    // 跨平台 wrapper 一致性（在 512 机器上走 512 分发路径）
    {
        const int N = 4096;
        std::vector<float> in(N), o(N);
        float base = 0.5f;
        for (int i = 0; i < N; ++i) in[i] = base + i * 0.004f;
        ct::kernels::simd::vexp(in.data(), o.data(), N);
        float me = 0; for (int i = 0; i < N; ++i) me = std::max(me, (float)std::fabs(o[i] - std::exp(in[i])));
        std::printf("  vexp (wrapper)  maxAbsErr=%.3e\n", me);
        CHECK(me < 1e-4f, "vexp wrapper abs err < 1e-4");
    }

    std::printf("\n=== %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
