/**
 * @file Mul_SIMD_kernel.cpp
 * @brief CPU-SIMD 乘法算子
 * @author GhostFace
 * @date 2026/02/09
 */

#include "../kernels.h"
#include "../../Ctorch_Error.h"
#include "../../Tensor.h"

#ifdef __x86_64__
#include <immintrin.h>  // x86 SIMD指令
#elif defined(__aarch64__)
#include <arm_neon.h>   // ARM NEON指令
#endif

Tensor Mul_SIMD_kernel(const Tensor& a, const Tensor& b) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU || b.device() != DeviceType::kCPU) {
        Ctorch_Error::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-SIMD Mul_Kernel: 仅在CPU支持");
    }

    // 校验数据类型
    if (a.dtype() != b.dtype()) {
        Ctorch_Error::log(ErrorLevel::ERROR, ErrorPlatform::kCPU, ErrorType::DATATYPE,
                          "CPU-SIMD Mul_Kernel: Tensor数据类型不匹配");
    }

    // 检查是否需要广播
    if (a.sizes() != b.sizes()) {
        // 处理0D张量的情况
        if (a.dim() == 0) {
            // a是标量，广播到b的形状
            Tensor a_broadcasted = a.broadcast_to(b.sizes());
            return Mul_SIMD_kernel(a_broadcasted, b);
        } else if (b.dim() == 0) {
            // b是标量，广播到a的形状
            Tensor b_broadcasted = b.broadcast_to(a.sizes());
            return Mul_SIMD_kernel(a, b_broadcasted);
        } else {
            // 两个都是非0D张量，尝试广播到相同形状
            Tensor a_broadcasted = a.broadcast_to(b.sizes());
            Tensor b_broadcasted = b.broadcast_to(a.sizes());
            
            // 检查广播是否成功
            if (a_broadcasted.sizes() == b_broadcasted.sizes()) {
                return Mul_SIMD_kernel(a_broadcasted, b_broadcasted);
            } else {
                Ctorch_Error::log(ErrorLevel::ERROR, ErrorPlatform::kCPU, ErrorType::DIMENSION,
                                  "CPU-SIMD Mul_Kernel: Tensor形状不兼容，无法广播");
            }
        }
    }

    int elem_count = a.numel();

    // 创建结果Tensor
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());

    // 获取Tensor数据指针
    const float* a_data = a.data<float>();
    const float* b_data = b.data<float>();
    float* result_data = result.data<float>();

#ifdef __x86_64__
    // x86 SIMD优化实现
    size_t i = 0;
    // 处理可以向量化的部分
    for (; i + 7 < elem_count; i += 8) {
        // 加载8个float值
        __m256 a_vec = _mm256_loadu_ps(&a_data[i]);
        __m256 b_vec = _mm256_loadu_ps(&b_data[i]);
        
        // 执行乘法
        __m256 result_vec = _mm256_mul_ps(a_vec, b_vec);
        
        // 存储结果
        _mm256_storeu_ps(&result_data[i], result_vec);
    }
    // 处理剩余部分
    for (; i < elem_count; ++i) {
        result_data[i] = a_data[i] * b_data[i];
    }
#elif defined(__aarch64__)
    // ARM NEON优化实现
    size_t i = 0;
    // 处理可以向量化的部分
    for (; i + 3 < elem_count; i += 4) {
        // 加载4个float值
        float32x4_t a_vec = vld1q_f32(&a_data[i]);
        float32x4_t b_vec = vld1q_f32(&b_data[i]);
        
        // 执行乘法
        float32x4_t result_vec = vmulq_f32(a_vec, b_vec);
        
        // 存储结果
        vst1q_f32(&result_data[i], result_vec);
    }
    // 处理剩余部分
    for (; i < elem_count; ++i) {
        result_data[i] = a_data[i] * b_data[i];
    }
#else
    // 不支持SIMD的情况，使用标量实现
    for (size_t i = 0; i < elem_count; ++i) {
        result_data[i] = a_data[i] * b_data[i];
    }
#endif

    return result;
}
