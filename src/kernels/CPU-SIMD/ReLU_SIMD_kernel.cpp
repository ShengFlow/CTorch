/**
 * @file ReLU_SIMD_kernel.cpp
 * @brief CPU-SIMD ReLU算子
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

Tensor ReLU_SIMD_kernel(const Tensor& a) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU) {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-SIMD ReLU_Kernel: 仅在CPU支持");
    }

    // 实现ReLU激活函数: max(x, 0)
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());

    size_t count = a.numel();
    const float* a_data = a.data_read<float>();
    float* result_data = result.data_write<float>();

#if defined(__x86_64__) || defined(__i386__)
    // x86 SIMD优化实现 (AVX)
    size_t i = 0;
    __m256 zero = _mm256_setzero_ps();
    for (; i + 7 < count; i += 8) {
        __m256 x = _mm256_loadu_ps(&a_data[i]);
        // ReLU: max(x, 0) - 使用 AVX2 的 max
        #if defined(__AVX2__)
        __m256 relu = _mm256_max_ps(x, zero);
        #else
        // AVX1 fallback: 使用比较 + 掩码
        __m256 mask = _mm256_cmp_ps(x, zero, _CMP_GE_OQ);
        __m256 relu = _mm256_and_ps(mask, x);
        #endif
        _mm256_storeu_ps(&result_data[i], relu);
    }
    // 处理剩余部分
    for (; i < count; ++i) {
        result_data[i] = std::max(0.0f, a_data[i]);
    }
#elif defined(__aarch64__)
    // ARM NEON优化实现
    size_t i = 0;
    float32x4_t zero_vec = vdupq_n_f32(0.0f);
    for (; i + 3 < count; i += 4) {
        float32x4_t x = vld1q_f32(&a_data[i]);
        float32x4_t relu = vmaxq_f32(x, zero_vec);
        vst1q_f32(&result_data[i], relu);
    }
    // 处理剩余部分
    for (; i < count; ++i) {
        result_data[i] = std::max(0.0f, a_data[i]);
    }
#else
    // 不支持SIMD的情况，使用标量实现
    for (size_t i = 0; i < count; ++i) {
        result_data[i] = std::max(0.0f, a_data[i]);
    }
#endif

    return result;
}
