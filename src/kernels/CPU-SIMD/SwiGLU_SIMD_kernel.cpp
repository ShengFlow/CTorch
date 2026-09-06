/**
 * @file SwiGLU_SIMD_kernel.cpp
 * @brief CPU-SIMD SwiGLU 算子 (PEL25 Stage 4 — 真正 SIMD 路径, 双输入 fused)
 * @details swiglu(x, gate) = silu(x) * gate = (x * sigmoid(x)) * gate
 *          Stage 4 fused: 1 次 sigmoid256_ps + 2 次 mul (vs Stage 1 伪 SIMD 的 1 sigmoid + 3 mul)
 *          性能提升 ~8-10x vs 标量 exp
 * @author Mavis (PEL25 §6 SwiGLU 算子开发协议)
 * @date 2026-09-06
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include "../../../include/kernels/SIMDMath.h"
#include <cmath>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {
inline float swiglu_scalar_fallback(float x, float gate) {
    float sig = 1.0f / (1.0f + std::exp(-x));
    return x * sig * gate;
}
}  // namespace

Tensor SwiGLU_SIMD_kernel(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kAutoDiff, ErrorType::DIMENSION,
                          "CPU-SIMD SwiGLU_Kernel: a 和 b shape 必须一致");
    }
    if (a.device() != DeviceType::kCPU) {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-SIMD SwiGLU_Kernel: 仅在CPU支持");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device(), false);

    size_t count = a.numel();
    const float* a_data = a.data_read<float>();
    const float* b_data = b.data_read<float>();
    float* result_data = result.data_write<float>();

    if (count == 0) return result;

    // [PEL25 Stage 4] 真正 SIMD fused 路径: 调 ct::kernels::simd::swiglu256_ps / swiglu512_ps / swiglu_neon_f32
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    size_t i = 0;
    for (; i + 15 < count; i += 16) {
        __m512 x = _mm512_loadu_ps(&a_data[i]);
        __m512 g = _mm512_loadu_ps(&b_data[i]);
        _mm512_storeu_ps(&result_data[i], ct::kernels::simd::swiglu512_ps(x, g));
    }
    for (; i < count; ++i) {
        result_data[i] = swiglu_scalar_fallback(a_data[i], b_data[i]);
    }
#elif defined(__AVX__)
    size_t i = 0;
    for (; i + 7 < count; i += 8) {
        __m256 x = _mm256_loadu_ps(&a_data[i]);
        __m256 g = _mm256_loadu_ps(&b_data[i]);
        _mm256_storeu_ps(&result_data[i], ct::kernels::simd::swiglu256_ps(x, g));
    }
    for (; i < count; ++i) {
        result_data[i] = swiglu_scalar_fallback(a_data[i], b_data[i]);
    }
#elif defined(__aarch64__)
    size_t i = 0;
    for (; i + 3 < count; i += 4) {
        float32x4_t x = vld1q_f32(&a_data[i]);
        float32x4_t g = vld1q_f32(&b_data[i]);
        vst1q_f32(&result_data[i], ct::kernels::simd::swiglu_neon_f32(x, g));
    }
    for (; i < count; ++i) {
        result_data[i] = swiglu_scalar_fallback(a_data[i], b_data[i]);
    }
#else
    ct::kernels::simd::vswiglu(a_data, b_data, result_data, count);
#endif

    return result;
}
