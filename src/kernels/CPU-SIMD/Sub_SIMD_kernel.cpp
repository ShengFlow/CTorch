/**
 * @file Sub_SIMD_kernel.cpp
 * @brief CPU-SIMD 减法算子
 * @author GhostFace
 * @date 2026/02/09
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include "../../../include/CoreDefs.h"

#ifdef __x86_64__
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

CT_HOT Tensor Sub_SIMD_kernel(const Tensor& a, const Tensor& b) {
    if (a.device() != DeviceType::kCPU || b.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-SIMD Sub_Kernel: 仅在CPU支持");
    }
    if (a.dtype() != b.dtype()) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kCPU, ErrorType::DATATYPE,
                          "CPU-SIMD Sub_Kernel: Tensor数据类型不匹配");
    }

    if (a.sizes() != b.sizes()) {
        if (a.dim() == 0) {
            Tensor a_broadcasted = a.broadcast_to(b.sizes());
            return Sub_SIMD_kernel(a_broadcasted, b);
        } else if (b.dim() == 0) {
            Tensor b_broadcasted = b.broadcast_to(a.sizes());
            return Sub_SIMD_kernel(a, b_broadcasted);
        } else {
            std::vector<size_t> broadcast_shape;
            size_t max_dims = std::max(a.sizes().size(), b.sizes().size());
            broadcast_shape.reserve(max_dims);
            
            for (size_t i = 0; i < max_dims; ++i) {
                size_t a_dim = i < a.sizes().size() ? a.sizes()[a.sizes().size() - 1 - i] : 1;
                size_t b_dim = i < b.sizes().size() ? b.sizes()[b.sizes().size() - 1 - i] : 1;
                if (a_dim != 1 && b_dim != 1 && a_dim != b_dim) {
                    CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kCPU, ErrorType::DIMENSION,
                                     "CPU-SIMD Sub_Kernel: Tensor形状不兼容，无法广播");
                    return Tensor();
                }
                broadcast_shape.push_back(std::max(a_dim, b_dim));
            }
            
            std::reverse(broadcast_shape.begin(), broadcast_shape.end());
            
            Tensor a_broadcasted = a.broadcast_to(broadcast_shape);
            Tensor b_broadcasted = b.broadcast_to(broadcast_shape);
            
            return Sub_SIMD_kernel(a_broadcasted, b_broadcasted);
        }
    }
    
    size_t elem_count = a.numel();
    const float* CT_RESTRICT a_data = a.data_read<float>();
    const float* CT_RESTRICT b_data = b.data_read<float>();
    
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device(), false);
    float* CT_RESTRICT result_data = result.data_write<float>();

#ifdef __x86_64__
    size_t i = 0;
    for (; i + 7 < elem_count; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(&a_data[i]);
        __m256 b_vec = _mm256_loadu_ps(&b_data[i]);
        __m256 res_vec = _mm256_sub_ps(a_vec, b_vec);
        _mm256_storeu_ps(&result_data[i], res_vec);
    }
    for (; i < elem_count; ++i) result_data[i] = a_data[i] - b_data[i];
#elif defined(__aarch64__)
    size_t i = 0;
    for (; i + 3 < elem_count; i += 4) {
        float32x4_t a_vec = vld1q_f32(&a_data[i]);
        float32x4_t b_vec = vld1q_f32(&b_data[i]);
        float32x4_t res_vec = vsubq_f32(a_vec, b_vec);
        vst1q_f32(&result_data[i], res_vec);
    }
    for (; i < elem_count; ++i) result_data[i] = a_data[i] - b_data[i];
#else
    for (size_t i = 0; i < elem_count; ++i) result_data[i] = a_data[i] - b_data[i];
#endif
    return result;
}