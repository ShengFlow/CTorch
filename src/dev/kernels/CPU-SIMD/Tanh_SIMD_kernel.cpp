/**
 * @file Tanh_SIMD_kernel.cpp
 * @brief CPU-SIMD Tanh算子
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

Tensor Tanh_SIMD_kernel(const Tensor& a) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU) {
        Ctorch_Error::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-SIMD Tanh_Kernel: 仅在CPU支持");
    }

    // 实现Tanh激活函数
    Tensor result(a);

    size_t count = a.numel();
    float *data = result.data<float>();

#ifdef __x86_64__
    // x86 SIMD优化实现
    size_t i = 0;
    // 处理可以向量化的部分
    for (; i + 7 < count; i += 8) {
        // 加载8个float值
        __m256 x = _mm256_loadu_ps(&data[i]);
        
        // 计算exp(x)和exp(-x)
        __m256 exp_x = _mm256_exp_ps(x);
        __m256 exp_neg_x = _mm256_exp_ps(_mm256_sub_ps(_mm256_setzero_ps(), x));
        
        // 计算tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        __m256 numerator = _mm256_sub_ps(exp_x, exp_neg_x);
        __m256 denominator = _mm256_add_ps(exp_x, exp_neg_x);
        __m256 tanh_x = _mm256_div_ps(numerator, denominator);
        
        // 存储结果
        _mm256_storeu_ps(&data[i], tanh_x);
    }
    // 处理剩余部分
    for (; i < count; ++i) {
        float exp_x = std::exp(data[i]);
        float exp_neg_x = std::exp(-data[i]);
        data[i] = (exp_x - exp_neg_x) / (exp_x + exp_neg_x);
    }
#elif defined(__aarch64__)
    // ARM NEON优化实现
    size_t i = 0;
    // 处理可以向量化的部分
    for (; i + 3 < count; i += 4) {
        // 加载4个float值
        float32x4_t x = vld1q_f32(&data[i]);
        
        // 计算exp(x)和exp(-x)
        // 注意：ARM NEON没有内置的exp函数，需要使用外部实现或简化版本
        // 这里暂时使用标量实现
        for (int j = 0; j < 4; ++j) {
            float exp_x = std::exp(data[i + j]);
            float exp_neg_x = std::exp(-data[i + j]);
            data[i + j] = (exp_x - exp_neg_x) / (exp_x + exp_neg_x);
        }
    }
    // 处理剩余部分
    for (; i < count; ++i) {
        float exp_x = std::exp(data[i]);
        float exp_neg_x = std::exp(-data[i]);
        data[i] = (exp_x - exp_neg_x) / (exp_x + exp_neg_x);
    }
#else
    // 不支持SIMD的情况，使用标量实现
    for (size_t i = 0; i < count; ++i) {
        float exp_x = std::exp(data[i]);
        float exp_neg_x = std::exp(-data[i]);
        data[i] = (exp_x - exp_neg_x) / (exp_x + exp_neg_x);
    }
#endif

    return result;
}
