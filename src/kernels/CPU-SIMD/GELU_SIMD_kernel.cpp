/**
 * @file GELU_SIMD_kernel.cpp
 * @brief CPU-SIMD GELU算子
 * @author GhostFace
 * @date 2026/07/28
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include <cmath>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {
constexpr float kSqrt2OverPi = 0.7978845608f;
constexpr float kGeluCoeff = 0.044715f;

inline float gelu_scalar(float x) {
    float v = kSqrt2OverPi * (x + kGeluCoeff * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(v));
}
}

Tensor GELU_SIMD_kernel(const Tensor& a) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU) {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-SIMD GELU_Kernel: 仅在CPU支持");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());

    size_t count = a.numel();
    const float* a_data = a.data_read<float>();
    float* result_data = result.data_write<float>();

#if defined(__x86_64__) || defined(__i386__)
    #if defined(__AVX2__)
    size_t i = 0;
    for (; i + 7 < count; i += 8) {
        __m256 x = _mm256_loadu_ps(&a_data[i]);
        alignas(32) float buf[8];
        _mm256_storeu_ps(buf, x);
        for (int j = 0; j < 8; ++j) {
            buf[j] = gelu_scalar(buf[j]);
        }
        _mm256_storeu_ps(&result_data[i], _mm256_loadu_ps(buf));
    }
    for (; i < count; ++i) {
        result_data[i] = gelu_scalar(a_data[i]);
    }
    #else
    for (size_t i = 0; i < count; ++i) {
        result_data[i] = gelu_scalar(a_data[i]);
    }
    #endif
#elif defined(__aarch64__)
    size_t i = 0;
    for (; i + 3 < count; i += 4) {
        float32x4_t x = vld1q_f32(&a_data[i]);
        alignas(16) float buf[4];
        vst1q_f32(buf, x);
        for (int j = 0; j < 4; ++j) {
            buf[j] = gelu_scalar(buf[j]);
        }
        vst1q_f32(&result_data[i], vld1q_f32(buf));
    }
    for (; i < count; ++i) {
        result_data[i] = gelu_scalar(a_data[i]);
    }
#else
    for (size_t i = 0; i < count; ++i) {
        result_data[i] = gelu_scalar(a_data[i]);
    }
#endif

    return result;
}
