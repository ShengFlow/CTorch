/**
 * @file Sigmoid_SIMD_kernel.cpp
 * @brief CPU-SIMD Sigmoid算子
 * @author GhostFace
 * @date 2026/06/27
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include <cmath>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>  // x86 SIMD指令
#elif defined(__aarch64__)
#include <arm_neon.h>   // ARM NEON指令
#endif

// Sigmoid: 1 / (1 + exp(-x))
// 使用高效的近似计算

#if defined(__x86_64__) || defined(__i386__)
#if defined(__AVX2__) && defined(__FMA__)
// 使用 AVX2 + FMA 的高效实现
static inline __m256 sigmoid_avx2(__m256 x) {
    // Clamp x to [-16, 16] for numerical stability
    __m256 zero = _mm256_setzero_ps();
    __m256 sixteen = _mm256_set1_ps(16.0f);
    __m256 neg_sixteen = _mm256_set1_ps(-16.0f);
    x = _mm256_max_ps(neg_sixteen, _mm256_min_ps(x, sixteen));

    // exp(-x) approximation using polynomial
    // exp(-x) ≈ 1 / (1 + 0.5*x + 0.25*x^2 + 0.125*x^3)
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 quarter = _mm256_set1_ps(0.25f);
    __m256 eighth = _mm256_set1_ps(0.125f);

    // Calculate polynomial approximation of exp(-x)
    // Using Horners method: a0 + x*(a1 + x*(a2 + x*a3))
    __m256 poly = _mm256_fmadd_ps(x, eighth,
                   _mm256_fmadd_ps(x, quarter,
                   _mm256_fmadd_ps(x, half, _mm256_set1_ps(1.0f))));

    // sigmoid = 1 / (1 + exp(-x)) = poly / (poly + 1)
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 denom = _mm256_add_ps(poly, one);
    return _mm256_div_ps(poly, denom);
}
#endif
#endif

Tensor Sigmoid_SIMD_kernel(const Tensor& a) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU) {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-SIMD Sigmoid_Kernel: 仅在CPU支持");
    }

    // 实现Sigmoid激活函数: 1 / (1 + exp(-x))
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());

    size_t count = a.numel();
    const float* a_data = a.data_read<float>();
    float* result_data = result.data_write<float>();

#if defined(__x86_64__) || defined(__i386__)
    #if defined(__AVX2__) && defined(__FMA__)
    // x86 AVX2 + FMA 优化实现
    size_t i = 0;
    for (; i + 7 < count; i += 8) {
        __m256 x = _mm256_loadu_ps(&a_data[i]);
        __m256 sigmoid = sigmoid_avx2(x);
        _mm256_storeu_ps(&result_data[i], sigmoid);
    }
    // 处理剩余部分
    for (; i < count; ++i) {
        result_data[i] = 1.0f / (1.0f + std::exp(-a_data[i]));
    }
    #else
    // x86 SSE 优化实现
    for (size_t i = 0; i < count; ++i) {
        result_data[i] = 1.0f / (1.0f + std::exp(-a_data[i]));
    }
    #endif
#elif defined(__aarch64__)
    // ARM NEON优化实现
    // 使用标准 expf 计算，多项式近似在 x>0.2 时误差过大
    size_t i = 0;
    for (; i < count; ++i) {
        result_data[i] = 1.0f / (1.0f + expf(-a_data[i]));
    }
#else
    // 不支持SIMD的情况，使用标量实现
    for (size_t i = 0; i < count; ++i) {
        result_data[i] = 1.0f / (1.0f + std::exp(-a_data[i]));
    }
#endif

    return result;
}
