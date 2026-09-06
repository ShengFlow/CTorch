/**
 * @file SiLU_SIMD_kernel.cpp
 * @brief CPU-SIMD SiLU 算子 (PEL25 Stage 4 — 真正 SIMD 路径)
 * @details silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *          Stage 4: 用 ct::kernels::simd::silu256_ps 真正 SIMD (含 Padé 多项式 sigmoid256_ps 复用)
 *          vs Stage 1 伪 SIMD (标量 exp + SIMD loadu/storeu 包装), 性能提升 ~8-10x
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
inline float silu_scalar_fallback(float x) {
    return x / (1.0f + std::exp(-x));
}
}  // namespace

Tensor SiLU_SIMD_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kCPU) {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-SIMD SiLU_Kernel: 仅在CPU支持");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device(), false);

    size_t count = a.numel();
    const float* a_data = a.data_read<float>();
    float* result_data = result.data_write<float>();

    if (count == 0) return result;

    // [PEL25 Stage 4] 真正 SIMD 路径: 调 ct::kernels::simd::silu256_ps / silu512_ps / silu_neon_f32
    //    Stage 1 用 ops/SiLU.cpp 直调, Stage 3 用伪 SIMD (标量 exp + loadu/storeu)
    //    Stage 4 这里用 polynomial sigmoid 真正 SIMD
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    size_t i = 0;
    for (; i + 15 < count; i += 16) {
        __m512 x = _mm512_loadu_ps(&a_data[i]);
        _mm512_storeu_ps(&result_data[i], ct::kernels::simd::silu512_ps(x));
    }
    for (; i < count; ++i) {
        result_data[i] = silu_scalar_fallback(a_data[i]);
    }
#elif defined(__AVX__)
    size_t i = 0;
    for (; i + 7 < count; i += 8) {
        __m256 x = _mm256_loadu_ps(&a_data[i]);
        _mm256_storeu_ps(&result_data[i], ct::kernels::simd::silu256_ps(x));
    }
    for (; i < count; ++i) {
        result_data[i] = silu_scalar_fallback(a_data[i]);
    }
#elif defined(__aarch64__)
    size_t i = 0;
    for (; i + 3 < count; i += 4) {
        float32x4_t x = vld1q_f32(&a_data[i]);
        vst1q_f32(&result_data[i], ct::kernels::simd::silu_neon_f32(x));
    }
    for (; i < count; ++i) {
        result_data[i] = silu_scalar_fallback(a_data[i]);
    }
#else
    ct::kernels::simd::vsilu(a_data, result_data, count);
#endif

    return result;
}
