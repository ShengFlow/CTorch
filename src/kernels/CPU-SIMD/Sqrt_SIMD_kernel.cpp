/**
 * @file Sqrt_SIMD_kernel.cpp
 * @author 苏璃珞
 * @brief CPU-SIMD 平方根算子
 * @date 2026/9/17
 *
 * @note sqrt 是 IEEE-754 要求正确舍入的基本运算，各架构都有单指令支持
 *       （x86 sqrtps / ARM fsqrt）。因此这里直接使用硬件指令，不像 exp/log 那样
 *       需要 SIMDMath 里的多项式近似 —— 精度与标量路径逐位一致。
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

CT_HOT Tensor Sqrt_SIMD_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-SIMD Sqrt_Kernel: 仅在CPU支持");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device(), false);
    const size_t count = a.numel();
    if (count == 0) return result;

    const float* CT_RESTRICT in = a.data_read<float>();
    float* CT_RESTRICT out = result.data_write<float>();

#if defined(__x86_64__) && defined(__AVX__)
    size_t i = 0;
    for (; i + 7 < count; i += 8) {
        __m256 x = _mm256_loadu_ps(&in[i]);
        _mm256_storeu_ps(&out[i], _mm256_sqrt_ps(x));
    }
    for (; i < count; ++i) {
        out[i] = std::sqrt(in[i]);
    }
#elif defined(__aarch64__)
    size_t i = 0;
    for (; i + 3 < count; i += 4) {
        float32x4_t x = vld1q_f32(&in[i]);
        vst1q_f32(&out[i], vsqrtq_f32(x));
    }
    for (; i < count; ++i) {
        out[i] = std::sqrt(in[i]);
    }
#else
    for (size_t i = 0; i < count; ++i) {
        out[i] = std::sqrt(in[i]);
    }
#endif

    return result;
}
