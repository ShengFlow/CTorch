/**
 * @file Sigmoid_SIMD_kernel.cpp
 * @brief CPU-SIMD Sigmoid算子（集成 SIMDMath 向量化实现）
 * @author GhostFace
 * @date 2026/08/03 - 2026/06/27
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

// Sigmoid: 1 / (1 + exp(-x))
//
// 性能优化历史：
//   - v1 (2026/06/27): 4 阶多项式近似，AVX2 有效，NEON 走标量
//   - v2 (2026/08/03): 集成 SIMDMath，所有平台走 8-wide AVX2 / 4-wide NEON
//                       精度从 ~1e-3 提升到 < 1e-6
//                       速度提升 4-8x vs 标量 std::expf

Tensor Sigmoid_SIMD_kernel(const Tensor& a) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU) {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-SIMD Sigmoid_Kernel: 仅在CPU支持");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device(), false);

    size_t count = a.numel();
    const float* a_data = a.data_read<float>();
    float* result_data = result.data_write<float>();

    if (count == 0) return result;

#if defined(__AVX2__)
    // x86 AVX2: 8-wide 直接调用 SIMDMath 的 sigmoid256_ps
    size_t i = 0;
    for (; i + 7 < count; i += 8) {
        __m256 x = _mm256_loadu_ps(&a_data[i]);
        _mm256_storeu_ps(&result_data[i],
                         ct::kernels::simd::sigmoid256_ps(x));
    }
    // 处理剩余部分（标量 + clamp，参考 SIMDMath 内部精度）
    for (; i < count; ++i) {
        float x = a_data[i];
        // clamp 到 [-20, 20] 避免 exp 溢出（与 SIMDMath 内部一致）
        x = std::min(x, 20.0f);
        x = std::max(x, -20.0f);
        result_data[i] = 1.0f / (1.0f + std::exp(-x));
    }
#elif defined(__aarch64__)
    // ARM NEON: 4-wide 调用 SIMDMath 的 sigmoid_neon_f32
    size_t i = 0;
    for (; i + 3 < count; i += 4) {
        float32x4_t x = vld1q_f32(&a_data[i]);
        vst1q_f32(&result_data[i],
                  ct::kernels::simd::sigmoid_neon_f32(x));
    }
    for (; i < count; ++i) {
        float x = a_data[i];
        x = std::min(x, 20.0f);
        x = std::max(x, -20.0f);
        result_data[i] = 1.0f / (1.0f + std::exp(-x));
    }
#else
    // 不支持 SIMD 的情况，使用 SIMDMath 的跨平台 wrapper
    ct::kernels::simd::vsigmoid(a_data, result_data, count);
#endif

    return result;
}
