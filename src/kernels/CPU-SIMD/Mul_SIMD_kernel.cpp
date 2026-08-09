/**
 * @file Mul_SIMD_kernel.cpp
 * @brief CPU-SIMD 乘法算子（零拷贝广播优化）
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

static void computeBroadcastStrides(
    const std::vector<size_t>& shape,
    const std::vector<size_t>& target_shape,
    std::vector<size_t>& strides) {
    
    size_t dims = target_shape.size();
    strides.resize(dims);
    
    std::vector<size_t> padded_shape(dims, 1);
    size_t offset = dims - shape.size();
    for (size_t i = 0; i < shape.size(); ++i) {
        padded_shape[offset + i] = shape[i];
    }
    
    strides[dims - 1] = (padded_shape[dims - 1] == 1) ? 0 : 1;
    for (int i = dims - 2; i >= 0; --i) {
        strides[i] = (padded_shape[i] == 1) ? 0 : strides[i + 1] * padded_shape[i + 1];
    }
}

CT_HOT Tensor Mul_SIMD_kernel(const Tensor& a, const Tensor& b) {
    if (a.device() != DeviceType::kCPU || b.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-SIMD Mul_Kernel: 仅在CPU支持");
    }
    if (a.dtype() != b.dtype()) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kCPU, ErrorType::DATATYPE,
                          "CPU-SIMD Mul_Kernel: Tensor数据类型不匹配");
    }

    // 形状相同：直接SIMD
    if (a.sizes() == b.sizes()) {
        size_t elem_count = a.numel();
        const float* CT_RESTRICT a_data = a.data_read<float>();
        const float* CT_RESTRICT b_data = b.data_read<float>();
        
        Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
        float* CT_RESTRICT result_data = result.data_write<float>();

#ifdef __aarch64__
        size_t i = 0;
        for (; i + 3 < elem_count; i += 4) {
            float32x4_t a_vec = vld1q_f32(&a_data[i]);
            float32x4_t b_vec = vld1q_f32(&b_data[i]);
            vst1q_f32(&result_data[i], vmulq_f32(a_vec, b_vec));
        }
        for (; i < elem_count; ++i) result_data[i] = a_data[i] * b_data[i];
#else
        for (size_t i = 0; i < elem_count; ++i) result_data[i] = a_data[i] * b_data[i];
#endif
        return result;
    }

    // 标量优化
    if (b.numel() == 1) {
        size_t elem_count = a.numel();
        float b_val = b.data_read<float>()[0];
        const float* CT_RESTRICT a_data = a.data_read<float>();
        
        Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
        float* CT_RESTRICT result_data = result.data_write<float>();
        
#ifdef __aarch64__
        size_t i = 0;
        float32x4_t b_vec = vdupq_n_f32(b_val);
        for (; i + 3 < elem_count; i += 4) {
            float32x4_t a_vec = vld1q_f32(&a_data[i]);
            vst1q_f32(&result_data[i], vmulq_f32(a_vec, b_vec));
        }
        for (; i < elem_count; ++i) result_data[i] = a_data[i] * b_val;
#else
        for (size_t i = 0; i < elem_count; ++i) result_data[i] = a_data[i] * b_val;
#endif
        return result;
    }

    if (a.numel() == 1) {
        size_t elem_count = b.numel();
        float a_val = a.data_read<float>()[0];
        const float* CT_RESTRICT b_data = b.data_read<float>();
        
        Tensor result(ShapeTag{}, b.sizes(), b.dtype(), b.device());
        float* CT_RESTRICT result_data = result.data_write<float>();
        
#ifdef __aarch64__
        size_t i = 0;
        float32x4_t a_vec = vdupq_n_f32(a_val);
        for (; i + 3 < elem_count; i += 4) {
            float32x4_t b_vec = vld1q_f32(&b_data[i]);
            vst1q_f32(&result_data[i], vmulq_f32(a_vec, b_vec));
        }
        for (; i < elem_count; ++i) result_data[i] = a_val * b_data[i];
#else
        for (size_t i = 0; i < elem_count; ++i) result_data[i] = a_val * b_data[i];
#endif
        return result;
    }

    // 零拷贝广播
    size_t max_dims = std::max(a.sizes().size(), b.sizes().size());
    std::vector<size_t> broadcast_shape(max_dims);
    
    for (size_t i = 0; i < max_dims; ++i) {
        size_t a_dim = i < a.sizes().size() ? a.sizes()[a.sizes().size() - 1 - i] : 1;
        size_t b_dim = i < b.sizes().size() ? b.sizes()[b.sizes().size() - 1 - i] : 1;
        if (a_dim != 1 && b_dim != 1 && a_dim != b_dim) {
#ifdef CT_DEBUG
            fprintf(stderr, "[DEBUG Mul] a_shape=[");
            for (size_t s : a.sizes()) fprintf(stderr, "%zu,", s);
            fprintf(stderr, "] b_shape=[");
            for (size_t s : b.sizes()) fprintf(stderr, "%zu,", s);
            fprintf(stderr, "] i=%zu a_dim=%zu b_dim=%zu\n", i, a_dim, b_dim);
#endif
            CtorchError::throwException(ErrorPlatform::kCPU, ErrorType::DIMENSION,
                "CPU-SIMD Mul_Kernel: Tensor形状不兼容");
        }
        broadcast_shape[max_dims - 1 - i] = std::max(a_dim, b_dim);
    }
    
    std::vector<size_t> a_strides, b_strides;
    computeBroadcastStrides(a.sizes(), broadcast_shape, a_strides);
    computeBroadcastStrides(b.sizes(), broadcast_shape, b_strides);
    
    Tensor result(ShapeTag{}, broadcast_shape, a.dtype(), a.device());
    float* CT_RESTRICT result_data = result.data_write<float>();
    const float* CT_RESTRICT a_data = a.data_read<float>();
    const float* CT_RESTRICT b_data = b.data_read<float>();
    
    size_t elem_count = result.numel();
    
    if (max_dims == 2) {
        size_t rows = broadcast_shape[0];
        size_t cols = broadcast_shape[1];
        
#ifdef __aarch64__
        for (size_t i = 0; i < rows; ++i) {
            size_t a_row_offset = i * a_strides[0];
            size_t b_row_offset = i * b_strides[0];
            
            size_t j = 0;
            for (; j + 3 < cols; j += 4) {
                size_t a_idx = a_row_offset + j * a_strides[1];
                size_t b_idx = b_row_offset + j * b_strides[1];
                
                float32x4_t a_vec = vld1q_f32(&a_data[a_idx]);
                float32x4_t b_vec = vld1q_f32(&b_data[b_idx]);
                vst1q_f32(&result_data[i * cols + j], vmulq_f32(a_vec, b_vec));
            }
            for (; j < cols; ++j) {
                size_t a_idx = a_row_offset + j * a_strides[1];
                size_t b_idx = b_row_offset + j * b_strides[1];
                result_data[i * cols + j] = a_data[a_idx] * b_data[b_idx];
            }
        }
#else
        for (size_t i = 0; i < rows; ++i) {
            size_t a_row_offset = i * a_strides[0];
            size_t b_row_offset = i * b_strides[0];
            
            for (size_t j = 0; j < cols; ++j) {
                size_t a_idx = a_row_offset + j * a_strides[1];
                size_t b_idx = b_row_offset + j * b_strides[1];
                result_data[i * cols + j] = a_data[a_idx] * b_data[b_idx];
            }
        }
#endif
    } else {
        std::vector<size_t> indices(max_dims);
        for (size_t i = 0; i < elem_count; ++i) {
            size_t temp = i;
            for (int j = max_dims - 1; j >= 0; --j) {
                indices[j] = temp % broadcast_shape[j];
                temp /= broadcast_shape[j];
            }
            
            size_t a_idx = 0, b_idx = 0;
            for (size_t j = 0; j < max_dims; ++j) {
                a_idx += indices[j] * a_strides[j];
                b_idx += indices[j] * b_strides[j];
            }
            
            result_data[i] = a_data[a_idx] * b_data[b_idx];
        }
    }
    
    return result;
}