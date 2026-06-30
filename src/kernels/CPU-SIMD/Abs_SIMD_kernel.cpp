/**
 * @file Abs_SIMD_kernel.cpp
 * @brief CPU-SIMD Abs算子（绝对值）
 * @author GhostFace
 * @date 2026/06/30
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include "../../../include/CoreDefs.h"
#include <cmath>

#ifdef __x86_64__
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

CT_HOT Tensor Abs_SIMD_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-SIMD Abs_Kernel: 仅在CPU支持");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
    size_t count = a.numel();
    const float* CT_RESTRICT a_data = a.data<float>();
    float* CT_RESTRICT result_data = result.data<float>();

#ifdef __x86_64__
    // x86 AVX: 使用 AND 操作清除符号位实现 abs
    size_t i = 0;
    __m256 sign_mask = _mm256_set1_ps(-0.0f);  // 符号位掩码 (0x80000000)
    for (; i + 7 < count; i += 8) {
        __m256 x = _mm256_loadu_ps(&a_data[i]);
        __m256 abs_x = _mm256_andnot_ps(sign_mask, x);  // 清除符号位
        _mm256_storeu_ps(&result_data[i], abs_x);
    }
    for (; i < count; ++i) {
        result_data[i] = std::abs(a_data[i]);
    }
#elif defined(__aarch64__)
    // ARM NEON: 使用 vabsq_f32
    size_t i = 0;
    for (; i + 3 < count; i += 4) {
        float32x4_t x = vld1q_f32(&a_data[i]);
        float32x4_t abs_x = vabsq_f32(x);
        vst1q_f32(&result_data[i], abs_x);
    }
    for (; i < count; ++i) {
        result_data[i] = std::abs(a_data[i]);
    }
#else
    for (size_t i = 0; i < count; ++i) {
        result_data[i] = std::abs(a_data[i]);
    }
#endif

    return result;
}