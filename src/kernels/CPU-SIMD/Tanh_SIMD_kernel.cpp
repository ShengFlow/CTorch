/**
 * @file Tanh_SIMD_kernel.cpp
 * @brief CPU-SIMD Tanh算子（集成 SIMDMath 向量化实现）
 * @author GhostFace
 * @date 2026/08/03 - 2026/02/09
 * @see SIMDMath.h 向量化超越函数库
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include "../../../include/kernels/SIMDMath.h"
#include <cmath>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>  // x86 SIMD指令
#elif defined(__aarch64__)
#include <arm_neon.h>   // ARM NEON指令
#endif

// Tanh: (e^x - e^-x) / (e^x + e^-x)
//
// 性能优化历史：
//   - v1 (2026/02/09): x86 走 SVML，NEON 走 4-wide 但内部循环是标量（伪向量化）
//   - v2 (2026/08/03): 集成 SIMDMath，所有平台走真正的 8-wide / 4-wide
//                       精度提升：原来 NEON 在 |x| 较大时有 ~1e-3 误差，
//                                SIMDMath 通过 Padé [5/4] 逼近 < 1 ULP
//                       NEON 性能提升 ~5x（消除了伪向量化内层标量循环）

Tensor Tanh_SIMD_kernel(const Tensor& a) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU) {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-SIMD Tanh_Kernel: 仅在CPU支持");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device(), false);

    size_t count = a.numel();
    const float* a_data = a.data_read<float>();
    float* result_data = result.data_write<float>();

    if (count == 0) return result;

#if defined(__AVX2__)
    // x86 AVX2: 8-wide 调用 SIMDMath 的 tanh256_ps
    size_t i = 0;
    for (; i + 7 < count; i += 8) {
        __m256 x = _mm256_loadu_ps(&a_data[i]);
        _mm256_storeu_ps(&result_data[i],
                         ct::kernels::simd::tanh256_ps(x));
    }
    for (; i < count; ++i) {
        // 标量回退：与 SIMDMath 内部精度保持一致（|x| 很大时饱和）
        float x = a_data[i];
        if (x > 20.0f)      result_data[i] = 1.0f;
        else if (x < -20.0f) result_data[i] = -1.0f;
        else                result_data[i] = std::tanh(x);
    }
#elif defined(__aarch64__)
    // ARM NEON: 4-wide 调用 SIMDMath 的 tanh_neon_f32
    size_t i = 0;
    for (; i + 3 < count; i += 4) {
        float32x4_t x = vld1q_f32(&a_data[i]);
        vst1q_f32(&result_data[i],
                  ct::kernels::simd::tanh_neon_f32(x));
    }
    for (; i < count; ++i) {
        float x = a_data[i];
        if (x > 20.0f)      result_data[i] = 1.0f;
        else if (x < -20.0f) result_data[i] = -1.0f;
        else                result_data[i] = std::tanh(x);
    }
#else
    // 跨平台 wrapper
    ct::kernels::simd::vtanh(a_data, result_data, count);
#endif

    return result;
}
