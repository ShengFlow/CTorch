/**
 * @file test_simd_math.cpp
 * @brief SIMDMath 单元测试：精度 + 性能
 * @details
 *   1. 精度测试：ULP error vs std::expf / std::logf / std::tanhf
 *   2. 数值稳定性：大/小输入不溢出
 *   3. 边界条件：clamp 行为正确
 *   4. 一致性：vec vs scalar (wrapper) 结果一致
 *   5. 性能：vs 标量版本
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "kernels/SIMDMath.h"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

using clk = std::chrono::high_resolution_clock;
using ns = std::chrono::nanoseconds;

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++g_passed; } \
    else { ++g_failed; std::cout << "  FAIL [" << __LINE__ << "]: " << msg << "\n"; } \
} while(0)

// ======================= ULP 计算 =======================
//
// 参考: https://blog.regehr.org/archives/1064
// ULP 是 two adjacent floats 之间的距离
//
static inline uint32_t as_uint(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

static inline float as_float(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

static int32_t ulp_diff(float a, float b) {
    if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<int32_t>::max();
    if (a == b) return 0;
    // Sign-aware ULP
    int32_t ai = static_cast<int32_t>(as_uint(a));
    int32_t bi = static_cast<int32_t>(as_uint(b));
    // 负数时 bit pattern 是 "反序"的，需要处理
    if ((ai < 0) != (bi < 0)) {
        // 不同符号：转换为正数后比较
        if (ai < 0) ai = (1u << 31) - ai;
        if (bi < 0) bi = (1u << 31) - bi;
    }
    return std::abs(ai - bi);
}

// ======================= 精度测试 =======================

static int platform_width() {
#if defined(__AVX2__)
    return 8;
#elif defined(__aarch64__)
    return 4;
#else
    return 0;  // scalar fallback
#endif
}

static void test_exp_precision() {
    std::cout << "=== exp 精度测试 ===" << std::endl;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    const int N = 10000;
    int W = platform_width();
    std::vector<float> in(N), ref(N), got(N);
    for (int i = 0; i < N; ++i) in[i] = dist(rng);

    // 参考：标量 std::expf
    for (int i = 0; i < N; ++i) ref[i] = std::exp(in[i]);

#if defined(__AVX2__)
    // AVX2 8-wide
    for (int i = 0; i < N; i += 8) {
        __m256 x = _mm256_loadu_ps(&in[i]);
        _mm256_storeu_ps(&got[i], ct::kernels::simd::exp256_ps(x));
    }
    for (int i = (N / 8) * 8; i < N; ++i) got[i] = std::exp(in[i]);
#elif defined(__aarch64__)
    // NEON 4-wide
    for (int i = 0; i < N; i += 4) {
        float32x4_t x = vld1q_f32(&in[i]);
        vst1q_f32(&got[i], ct::kernels::simd::exp_neon_f32(x));
    }
    for (int i = (N / 4) * 4; i < N; ++i) got[i] = std::exp(in[i]);
#else
    for (int i = 0; i < N; ++i) got[i] = std::exp(in[i]);
#endif

    int32_t max_ulp = 0;
    double max_rel_err = 0.0;
    for (int i = 0; i < N; ++i) {
        int32_t u = ulp_diff(ref[i], got[i]);
        max_ulp = std::max(max_ulp, u);
        if (std::abs(ref[i]) > 1e-6f) {
            double rel = std::abs(double(got[i]) - double(ref[i])) / std::abs(double(ref[i]));
            max_rel_err = std::max(max_rel_err, rel);
        }
    }
    std::cout << "  max ULP error: " << max_ulp
              << " | max relative error: " << max_rel_err << std::endl;
    CHECK(max_ulp <= 2, "exp max ULP error <= 2");
    CHECK(max_rel_err < 1e-5, "exp max relative error < 1e-5");
}

static void test_log_precision() {
    std::cout << "=== log 精度测试 ===" << std::endl;
    std::mt19937 rng(43);
    std::uniform_real_distribution<float> dist(1e-3f, 100.0f);

    const int N = 10000;
    std::vector<float> in(N), ref(N), got(N);
    for (int i = 0; i < N; ++i) in[i] = dist(rng);
    for (int i = 0; i < N; ++i) ref[i] = std::log(in[i]);

#ifdef __AVX__
    for (int i = 0; i < N; i += 8) {
        __m256 x = _mm256_loadu_ps(&in[i]);
        _mm256_storeu_ps(&got[i], ct::kernels::simd::log256_ps(x));
    }
    for (int i = (N / 8) * 8; i < N; ++i) got[i] = std::log(in[i]);
#elif defined(__aarch64__)
    for (int i = 0; i < N; i += 4) {
        float32x4_t x = vld1q_f32(&in[i]);
        vst1q_f32(&got[i], ct::kernels::simd::log_neon_f32(x));
    }
    for (int i = (N / 4) * 4; i < N; ++i) got[i] = std::log(in[i]);
#else
    for (int i = 0; i < N; ++i) got[i] = std::log(in[i]);
#endif

    int32_t max_ulp = 0;
    for (int i = 0; i < N; ++i) {
        int32_t u = ulp_diff(ref[i], got[i]);
        max_ulp = std::max(max_ulp, u);
    }
    std::cout << "  max ULP error: " << max_ulp << std::endl;
    CHECK(max_ulp <= 2, "log max ULP error <= 2");
}

static void test_tanh_precision() {
    std::cout << "=== tanh 精度测试 ===" << std::endl;
    std::mt19937 rng(44);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    const int N = 10000;
    std::vector<float> in(N), ref(N), got(N);
    for (int i = 0; i < N; ++i) in[i] = dist(rng);
    for (int i = 0; i < N; ++i) ref[i] = std::tanh(in[i]);

#ifdef __AVX__
    for (int i = 0; i < N; i += 8) {
        __m256 x = _mm256_loadu_ps(&in[i]);
        _mm256_storeu_ps(&got[i], ct::kernels::simd::tanh256_ps(x));
    }
    for (int i = (N / 8) * 8; i < N; ++i) got[i] = std::tanh(in[i]);
#elif defined(__aarch64__)
    for (int i = 0; i < N; i += 4) {
        float32x4_t x = vld1q_f32(&in[i]);
        vst1q_f32(&got[i], ct::kernels::simd::tanh_neon_f32(x));
    }
    for (int i = (N / 4) * 4; i < N; ++i) got[i] = std::tanh(in[i]);
#else
    for (int i = 0; i < N; ++i) got[i] = std::tanh(in[i]);
#endif

    int32_t max_ulp = 0;
    for (int i = 0; i < N; ++i) {
        int32_t u = ulp_diff(ref[i], got[i]);
        max_ulp = std::max(max_ulp, u);
    }
    std::cout << "  max ULP error: " << max_ulp << std::endl;
    CHECK(max_ulp <= 4, "tanh max ULP error <= 4 (more lenient for transcendental)");
}

static void test_sigmoid_precision() {
    std::cout << "=== sigmoid 精度测试 ===" << std::endl;
    std::mt19937 rng(45);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    const int N = 10000;
    std::vector<float> in(N), ref(N), got(N);
    for (int i = 0; i < N; ++i) in[i] = dist(rng);
    for (int i = 0; i < N; ++i) ref[i] = 1.0f / (1.0f + std::exp(-in[i]));

#ifdef __AVX2__
    for (int i = 0; i < N; i += 8) {
        __m256 x = _mm256_loadu_ps(&in[i]);
        _mm256_storeu_ps(&got[i], ct::kernels::simd::sigmoid256_ps(x));
    }
    for (int i = (N / 8) * 8; i < N; ++i) got[i] = 1.0f / (1.0f + std::exp(-in[i]));
#elif defined(__aarch64__)
    for (int i = 0; i < N; i += 4) {
        float32x4_t x = vld1q_f32(&in[i]);
        vst1q_f32(&got[i], ct::kernels::simd::sigmoid_neon_f32(x));
    }
    for (int i = (N / 4) * 4; i < N; ++i) got[i] = 1.0f / (1.0f + std::exp(-in[i]));
#else
    for (int i = 0; i < N; ++i) got[i] = 1.0f / (1.0f + std::exp(-in[i]));
#endif

    int32_t max_ulp = 0;
    for (int i = 0; i < N; ++i) {
        int32_t u = ulp_diff(ref[i], got[i]);
        max_ulp = std::max(max_ulp, u);
    }
    std::cout << "  max ULP error: " << max_ulp << std::endl;
    CHECK(max_ulp <= 4, "sigmoid max ULP error <= 4");
}

// ======================= 边界条件测试 =======================

static void test_exp_boundaries() {
    std::cout << "=== exp 边界条件 ===" << std::endl;
#if defined(__AVX2__)
    // exp(0) = 1
    __m256 zero = _mm256_setzero_ps();
    __m256 r = ct::kernels::simd::exp256_ps(zero);
    float buf[8];
    _mm256_storeu_ps(buf, r);
    CHECK(std::abs(buf[0] - 1.0f) < 1e-6f, "exp(0) ≈ 1");

    // exp(large) 不溢出
    __m256 large = _mm256_set1_ps(80.0f);
    r = ct::kernels::simd::exp256_ps(large);
    _mm256_storeu_ps(buf, r);
    CHECK(std::isfinite(buf[0]), "exp(80) is finite");

    // exp(-large) → 0
    __m256 neg_large = _mm256_set1_ps(-80.0f);
    r = ct::kernels::simd::exp256_ps(neg_large);
    _mm256_storeu_ps(buf, r);
    CHECK(buf[0] >= 0.0f && buf[0] < 1e-30f, "exp(-80) ≈ 0");
#elif defined(__aarch64__)
    float32x4_t zero = vdupq_n_f32(0.0f);
    float32x4_t r = ct::kernels::simd::exp_neon_f32(zero);
    float buf[4];
    vst1q_f32(buf, r);
    CHECK(std::abs(buf[0] - 1.0f) < 1e-6f, "exp(0) ≈ 1");

    float32x4_t large = vdupq_n_f32(80.0f);
    r = ct::kernels::simd::exp_neon_f32(large);
    vst1q_f32(buf, r);
    CHECK(std::isfinite(buf[0]), "exp(80) is finite");

    float32x4_t neg_large = vdupq_n_f32(-80.0f);
    r = ct::kernels::simd::exp_neon_f32(neg_large);
    vst1q_f32(buf, r);
    CHECK(buf[0] >= 0.0f && buf[0] < 1e-30f, "exp(-80) ≈ 0");
#endif
}

static void test_tanh_saturation() {
    std::cout << "=== tanh 饱和行为 ===" << std::endl;
#if defined(__AVX2__)
    __m256 big = _mm256_set1_ps(50.0f);
    __m256 r = ct::kernels::simd::tanh256_ps(big);
    float buf[8];
    _mm256_storeu_ps(buf, r);
    CHECK(std::abs(buf[0] - 1.0f) < 1e-6f, "tanh(50) ≈ 1");

    __m256 neg_big = _mm256_set1_ps(-50.0f);
    r = ct::kernels::simd::tanh256_ps(neg_big);
    _mm256_storeu_ps(buf, r);
    CHECK(std::abs(buf[0] - (-1.0f)) < 1e-6f, "tanh(-50) ≈ -1");
#elif defined(__aarch64__)
    float32x4_t big = vdupq_n_f32(50.0f);
    float32x4_t r = ct::kernels::simd::tanh_neon_f32(big);
    float buf[4];
    vst1q_f32(buf, r);
    CHECK(std::abs(buf[0] - 1.0f) < 1e-6f, "tanh(50) ≈ 1");

    float32x4_t neg_big = vdupq_n_f32(-50.0f);
    r = ct::kernels::simd::tanh_neon_f32(neg_big);
    vst1q_f32(buf, r);
    CHECK(std::abs(buf[0] - (-1.0f)) < 1e-6f, "tanh(-50) ≈ -1");
#endif
}

static void test_sigmoid_saturation() {
    std::cout << "=== sigmoid 饱和行为 ===" << std::endl;
#if defined(__AVX2__)
    __m256 big = _mm256_set1_ps(50.0f);
    __m256 r = ct::kernels::simd::sigmoid256_ps(big);
    float buf[8];
    _mm256_storeu_ps(buf, r);
    CHECK(std::abs(buf[0] - 1.0f) < 1e-6f, "sigmoid(50) ≈ 1");

    __m256 neg_big = _mm256_set1_ps(-50.0f);
    r = ct::kernels::simd::sigmoid256_ps(neg_big);
    _mm256_storeu_ps(buf, r);
    CHECK(std::abs(buf[0]) < 1e-6f, "sigmoid(-50) ≈ 0");
#elif defined(__aarch64__)
    float32x4_t big = vdupq_n_f32(50.0f);
    float32x4_t r = ct::kernels::simd::sigmoid_neon_f32(big);
    float buf[4];
    vst1q_f32(buf, r);
    CHECK(std::abs(buf[0] - 1.0f) < 1e-6f, "sigmoid(50) ≈ 1");

    float32x4_t neg_big = vdupq_n_f32(-50.0f);
    r = ct::kernels::simd::sigmoid_neon_f32(neg_big);
    vst1q_f32(buf, r);
    CHECK(std::abs(buf[0]) < 1e-6f, "sigmoid(-50) ≈ 0");
#endif
}

// ======================= Wrapper 一致性测试 =======================

static void test_wrapper_consistency() {
    std::cout << "=== wrapper (vexp/vlog/vtanh/vsigmoid/vgelu) 一致性 ===" << std::endl;
    const int N = 1024;
    std::vector<float> in(N), out(N);
    std::mt19937 rng(46);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    for (int i = 0; i < N; ++i) in[i] = dist(rng);

    // vexp
    ct::kernels::simd::vexp(in.data(), out.data(), N);
    for (int i = 0; i < N; ++i) {
        float ref = std::exp(in[i]);
        if (std::abs(ref - out[i]) > 1e-5f) {
            ++g_failed;
            std::cout << "  FAIL: vexp mismatch at " << i
                      << " in=" << in[i] << " ref=" << ref << " got=" << out[i] << "\n";
            return;
        }
    }
    ++g_passed;
    std::cout << "  vexp OK\n";

    // vlog
    for (int i = 0; i < N; ++i) in[i] = std::abs(dist(rng)) + 0.1f;
    ct::kernels::simd::vlog(in.data(), out.data(), N);
    for (int i = 0; i < N; ++i) {
        float ref = std::log(in[i]);
        if (std::abs(ref - out[i]) > 1e-5f) {
            ++g_failed;
            std::cout << "  FAIL: vlog mismatch at " << i << "\n";
            return;
        }
    }
    ++g_passed;
    std::cout << "  vlog OK\n";

    // vtanh
    for (int i = 0; i < N; ++i) in[i] = dist(rng);
    ct::kernels::simd::vtanh(in.data(), out.data(), N);
    for (int i = 0; i < N; ++i) {
        float ref = std::tanh(in[i]);
        if (std::abs(ref - out[i]) > 1e-5f) {
            ++g_failed;
            std::cout << "  FAIL: vtanh mismatch at " << i
                      << " in=" << in[i] << " ref=" << ref << " got=" << out[i] << "\n";
            return;
        }
    }
    ++g_passed;
    std::cout << "  vtanh OK\n";

    // vsigmoid
    ct::kernels::simd::vsigmoid(in.data(), out.data(), N);
    for (int i = 0; i < N; ++i) {
        float ref = 1.0f / (1.0f + std::exp(-in[i]));
        if (std::abs(ref - out[i]) > 1e-5f) {
            ++g_failed;
            std::cout << "  FAIL: vsigmoid mismatch at " << i
                      << " in=" << in[i] << " ref=" << ref << " got=" << out[i] << "\n";
            return;
        }
    }
    ++g_passed;
    std::cout << "  vsigmoid OK\n";

    // vgelu
    ct::kernels::simd::vgelu(in.data(), out.data(), N);
    for (int i = 0; i < N; ++i) {
        float v = 0.7978845608f * (in[i] + 0.044715f * in[i] * in[i] * in[i]);
        float ref = 0.5f * in[i] * (1.0f + std::tanh(v));
        if (std::abs(ref - out[i]) > 1e-4f) {  // gelu 累计误差稍大
            ++g_failed;
            std::cout << "  FAIL: vgelu mismatch at " << i
                      << " in=" << in[i] << " ref=" << ref << " got=" << out[i] << "\n";
            return;
        }
    }
    ++g_passed;
    std::cout << "  vgelu OK\n";
}

// ======================= 性能测试 =======================

template<typename ScalarFn, typename VecFn>
double benchmark(const char* name, ScalarFn scalar_fn, VecFn vec_fn,
                size_t N, int trials = 100) {
    std::vector<float> in(N), out(N);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    for (size_t i = 0; i < N; ++i) in[i] = dist(rng);

    // 防 DCE：用 volatile sink 强制编译器不优化掉写操作
    volatile float sink = 0.0f;

    // warmup
    for (size_t i = 0; i < N; ++i) out[i] = scalar_fn(in[i]);
    vec_fn(in.data(), out.data(), N);
    sink += out[0];

    // 标量
    auto t0 = clk::now();
    for (int t = 0; t < trials; ++t) {
        for (size_t i = 0; i < N; ++i) out[i] = scalar_fn(in[i]);
        sink += out[t % N];
    }
    auto t1 = clk::now();
    double scalar_ns = std::chrono::duration_cast<ns>(t1 - t0).count() / (double)trials;

    // 向量
    auto t2 = clk::now();
    for (int t = 0; t < trials; ++t) {
        vec_fn(in.data(), out.data(), N);
        sink += out[t % N];
    }
    auto t3 = clk::now();
    double vec_ns = std::chrono::duration_cast<ns>(t3 - t2).count() / (double)trials;

    // 防止 sink 被完全优化掉
    if (sink == 123.456f) std::cout << "";  // never happens

    double speedup = scalar_ns / std::max(vec_ns, 1.0);
    std::cout << "  " << name
              << " | scalar: " << std::fixed << std::setprecision(2) << (scalar_ns / 1000.0) << " us"
              << " | vec: "    << std::setprecision(2) << (vec_ns / 1000.0) << " us"
              << " | speedup: " << std::setprecision(2) << speedup << "x"
              << std::endl;
    return speedup;
}

static void test_performance() {
    std::cout << "=== 性能 bench (N=65536, 100 trials) ===" << std::endl;
    const size_t N = 65536;
    const int trials = 100;

    double s_exp = benchmark("exp", [](float x) { return std::exp(x); },
                             [](const float* in, float* out, size_t n) {
                                 ct::kernels::simd::vexp(in, out, n);
                             }, N, trials);

    double s_log = benchmark("log", [](float x) { return std::log(x); },
                             [](const float* in, float* out, size_t n) {
                                 ct::kernels::simd::vlog(in, out, n);
                             }, N, trials);

    double s_tanh = benchmark("tanh", [](float x) { return std::tanh(x); },
                              [](const float* in, float* out, size_t n) {
                                  ct::kernels::simd::vtanh(in, out, n);
                              }, N, trials);

    double s_sigmoid = benchmark("sigmoid",
        [](float x) { return 1.0f / (1.0f + std::exp(-x)); },
        [](const float* in, float* out, size_t n) {
            ct::kernels::simd::vsigmoid(in, out, n);
        }, N, trials);

    double s_gelu = benchmark("gelu",
        [](float x) {
            float v = 0.7978845608f * (x + 0.044715f * x * x * x);
            return 0.5f * x * (1.0f + std::tanh(v));
        },
        [](const float* in, float* out, size_t n) {
            ct::kernels::simd::vgelu(in, out, n);
        }, N, trials);

    // 性能目标：>= 2x 平均加速
    double avg = (s_exp + s_log + s_tanh + s_sigmoid + s_gelu) / 5.0;
    std::cout << "  平均加速比: " << avg << "x" << std::endl;
    CHECK(avg >= 2.0, "average speedup >= 2x");
}

int main() {
    std::cout << "=== SIMDMath 单元测试 (ADR-009) ===" << std::endl;
    std::cout << "Platform: ";
#if defined(__AVX2__) && defined(__FMA__)
    std::cout << "x86_64 AVX2+FMA";
#elif defined(__aarch64__)
    std::cout << "aarch64 NEON";
#else
    std::cout << "scalar fallback";
#endif
    std::cout << std::endl << std::endl;

    test_exp_precision();
    test_log_precision();
    test_tanh_precision();
    test_sigmoid_precision();

    test_exp_boundaries();
    test_tanh_saturation();
    test_sigmoid_saturation();

    test_wrapper_consistency();
    test_performance();

    std::cout << "\n=== 总计: " << g_passed << " passed, " << g_failed << " failed ===" << std::endl;
    return g_failed == 0 ? 0 : 1;
}
