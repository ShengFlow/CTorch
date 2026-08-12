/**
 * @file Neg_SIMD_kernel.cpp
 * @brief CPU-SIMD 取负算子
 * @author GhostFace
 * @date 2026/06/27
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include "../../../include/CoreDefs.h"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>  // x86 SIMD指令
#elif defined(__aarch64__)
#include <arm_neon.h>   // ARM NEON指令
#endif

CT_HOT Tensor Neg_SIMD_kernel(const Tensor& a) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-SIMD Neg_Kernel: 仅在CPU支持");
    }

    // 实现取负操作: -a
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device(), false);

    size_t count = a.numel();
    const float* CT_RESTRICT a_data = a.data_read<float>();
    float* CT_RESTRICT result_data = result.data_write<float>();

#if defined(__x86_64__) || defined(__i386__)
    // x86 SIMD优化实现
    size_t i = 0;
    __m256 neg_zero = _mm256_set1_ps(-0.0f);  // 仅符号位为1
    for (; i + 7 < count; i += 8) {
        __m256 x = _mm256_loadu_ps(&a_data[i]);
        __m256 neg = _mm256_xor_ps(x, neg_zero);  // 取反符号位
        _mm256_storeu_ps(&result_data[i], neg);
    }
    // 处理剩余部分
    for (; i < count; ++i) {
        result_data[i] = -a_data[i];
    }
#elif defined(__aarch64__)
    // ARM NEON优化实现
    size_t i = 0;
    for (; i + 3 < count; i += 4) {
        float32x4_t x = vld1q_f32(&a_data[i]);
        float32x4_t neg = vnegq_f32(x);  // NEON 有直接的取负指令
        vst1q_f32(&result_data[i], neg);
    }
    // 处理剩余部分
    for (; i < count; ++i) {
        result_data[i] = -a_data[i];
    }
#else
    // 不支持SIMD的情况，使用标量实现
    for (size_t i = 0; i < count; ++i) {
        result_data[i] = -a_data[i];
    }
#endif

    return result;
}
