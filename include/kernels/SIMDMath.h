/**
 * @file SIMDMath.h
 * @brief 向量化超越函数库（AVX2 + NEON）
 * @details 提供 exp / log / tanh / sigmoid / gelu 的 SIMD 实现，目标：
 *          - 性能：单函数 4-8x 加速（vs 标量 std::expf）
 *          - 精度：max ULP error < 2 (相对 libc++ std::expf 等)
 *          - 跨平台：x86_64 (AVX2+FMA) + aarch64 (NEON)
 *
 * 算法：
 *   - exp256:  Cephes 风格：x = k*ln2 + r，r 用 5 次 Padé 多项式逼近
 *   - log256:  范围缩减 + 9 次多项式逼近（Cephes 风格）
 *   - tanh256: Padé [4/4] 逼近 + 大参数饱和到 ±1
 *   - sigmoid256: 1/(1+exp(-x))，用 exp256 + 对称性 sigmoid(-x) = 1 - sigmoid(x)
 *   - gelu256:  0.5*x*(1+tanh(sqrt(2/pi)*(x + 0.044715*x^3)))，复用 tanh256
 *
 * 适用场景：
 *   - 深度学习推理（transformer、MLP、CNN）
 *   - 任何用 exp/log/tanh 的热路径
 *   - 替代 kernel 中 #pragma omp simd + std::exp（编译器无法真正向量化 libc++）
 *
 * @date 2026/08/03
 * @see ADR-009-vectorized-transcendentals
 */

#ifndef CTORCH_KERNELS_SIMD_MATH_H
#define CTORCH_KERNELS_SIMD_MATH_H

#include <cstddef>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace ct {
namespace kernels {
namespace simd {

// ======================= AVX2 (x86_64) API =======================

#ifdef __AVX__

/**
 * @brief 向量化 expf（8-wide AVX2）
 * @param x 输入向量（任意实数）
 * @return exp(x) 逐元素
 * @note 输入被 clamp 到 [-87, 87] 避免 overflow
 * @note max ULP error < 2 vs std::expf
 */
__m256 exp256_ps(__m256 x);

/**
 * @brief 向量化 logf（8-wide AVX2）
 * @param x 输入向量（必须 > 0）
 * @return log(x) 逐元素
 * @note x <= 0 行为未定义（调用方需保证）
 * @note max ULP error < 2 vs std::logf
 */
__m256 log256_ps(__m256 x);

/**
 * @brief 向量化 tanhf（8-wide AVX2）
 * @param x 输入向量（任意实数）
 * @return tanh(x) 逐元素，|x| 很大时饱和到 ±1
 * @note max ULP error < 2 vs std::tanhf
 */
__m256 tanh256_ps(__m256 x);

/**
 * @brief 向量化 sigmoid（8-wide AVX2）
 * @param x 输入向量
 * @return 1/(1+exp(-x))
 */
__m256 sigmoid256_ps(__m256 x);

/**
 * @brief 向量化 GELU（tanh 近似版本，8-wide AVX2）
 * @param x 输入向量
 * @return 0.5*x*(1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
 */
__m256 gelu256_ps(__m256 x);

/**
 * @brief 向量化 1/sqrtf（8-wide AVX2）
 * @param x 输入向量（必须 > 0）
 * @return 1/sqrt(x)
 */
__m256 rsqrt256_ps(__m256 x);

#endif // __AVX__

// ======================= NEON (aarch64) API =======================

#ifdef __aarch64__

/**
 * @brief 向量化 expf（4-wide NEON）
 */
float32x4_t exp_neon_f32(float32x4_t x);

/**
 * @brief 向量化 logf（4-wide NEON）
 */
float32x4_t log_neon_f32(float32x4_t x);

/**
 * @brief 向量化 tanhf（4-wide NEON）
 */
float32x4_t tanh_neon_f32(float32x4_t x);

/**
 * @brief 向量化 sigmoid（4-wide NEON）
 */
float32x4_t sigmoid_neon_f32(float32x4_t x);

/**
 * @brief 向量化 GELU（4-wide NEON）
 */
float32x4_t gelu_neon_f32(float32x4_t x);

#endif // __aarch64__

// ======================= 跨平台 wrapper =======================

/**
 * @brief 跨平台向量化 exp（自动选择 AVX2 或 NEON）
 * @param in 输入数组（连续内存）
 * @param out 输出数组（连续内存）
 * @param n 元素数
 */
void vexp(const float* in, float* out, size_t n);

/**
 * @brief 跨平台向量化 log
 */
void vlog(const float* in, float* out, size_t n);

/**
 * @brief 跨平台向量化 tanh
 */
void vtanh(const float* in, float* out, size_t n);

/**
 * @brief 跨平台向量化 sigmoid
 */
void vsigmoid(const float* in, float* out, size_t n);

/**
 * @brief 跨平台向量化 GELU
 */
void vgelu(const float* in, float* out, size_t n);

}  // namespace simd
}  // namespace kernels
}  // namespace ct

#endif  // CTORCH_KERNELS_SIMD_MATH_H
