/**
 * @file GELU_SIMD_kernel.cpp
 * @brief CPU-SIMD GELU算子（集成 SIMDMath 向量化实现）
 * @author GhostFace
 * @date 2026/08/03 - 2026/07/28
 * @see SIMDMath.h 向量化超越函数库
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
constexpr float kSqrt2OverPi = 0.7978845608f;
constexpr float kGeluCoeff = 0.044715f;

// 标量 GELU（仅用于 tail 处理）
inline float gelu_scalar(float x) {
    float v = kSqrt2OverPi * (x + kGeluCoeff * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(v));
}
}

// GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// 性能优化历史：
//   - v1 (2026/07/28): 8-wide AVX2 + 4-wide NEON，但循环内部用 storeu → scalar → loadu
//                      是完全伪向量化（向量化开销 + 标量计算 = 反而更慢）
//   - v2 (2026/08/03): 集成 SIMDMath，直接调用 gelu256_ps / gelu_neon_f32，
//                      真正的 SIMD 全程无标量回落
//                      速度提升 6-10x vs v1

Tensor GELU_SIMD_kernel(const Tensor& a) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU) {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-SIMD GELU_Kernel: 仅在CPU支持");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device(), false);

    size_t count = a.numel();
    const float* a_data = a.data_read<float>();
    float* result_data = result.data_write<float>();

    if (count == 0) return result;

#if defined(__AVX2__)
    size_t i = 0;
    for (; i + 7 < count; i += 8) {
        __m256 x = _mm256_loadu_ps(&a_data[i]);
        _mm256_storeu_ps(&result_data[i],
                         ct::kernels::simd::gelu256_ps(x));
    }
    for (; i < count; ++i) {
        result_data[i] = gelu_scalar(a_data[i]);
    }
#elif defined(__aarch64__)
    size_t i = 0;
    for (; i + 3 < count; i += 4) {
        float32x4_t x = vld1q_f32(&a_data[i]);
        vst1q_f32(&result_data[i],
                  ct::kernels::simd::gelu_neon_f32(x));
    }
    for (; i < count; ++i) {
        result_data[i] = gelu_scalar(a_data[i]);
    }
#else
    ct::kernels::simd::vgelu(a_data, result_data, count);
#endif

    return result;
}
