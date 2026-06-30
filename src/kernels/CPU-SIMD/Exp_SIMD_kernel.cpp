/**
 * @file Exp_SIMD_kernel.cpp
 * @brief CPU-SIMD Exp算子（指数函数）
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

CT_HOT Tensor Exp_SIMD_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-SIMD Exp_Kernel: 仅在CPU支持");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
    size_t count = a.numel();
    const float* CT_RESTRICT a_data = a.data<float>();
    float* CT_RESTRICT result_data = result.data<float>();

#ifdef __x86_64__
    // x86: SVML 向量化 exp（如果可用）
    size_t i = 0;
    #if defined(__INTEL_COMPILER) || (defined(__GNUC__) && defined(__AVX__) && defined(__SVML__))
    for (; i + 7 < count; i += 8) {
        __m256 x = _mm256_loadu_ps(&a_data[i]);
        __m256 exp_x = _mm256_exp_ps(x);
        _mm256_storeu_ps(&result_data[i], exp_x);
    }
    #endif
    // 处理剩余部分（或无 SVML 时全部标量）
    for (; i < count; ++i) {
        result_data[i] = std::exp(a_data[i]);
    }
#elif defined(__aarch64__)
    // ARM NEON: 无内置 exp，使用标量循环
    for (size_t i = 0; i < count; ++i) {
        result_data[i] = std::exp(a_data[i]);
    }
#else
    for (size_t i = 0; i < count; ++i) {
        result_data[i] = std::exp(a_data[i]);
    }
#endif

    return result;
}