/**
 * @file Add_SIMD_kernel.cpp
 * @brief CPU-SIMD 加法算子（零拷贝广播优化）
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

// 计算广播stride（零拷贝）
static void computeBroadcastStrides(
    const std::vector<size_t>& shape,
    const std::vector<size_t>& target_shape,
    std::vector<size_t>& strides) {
    
    size_t dims = target_shape.size();
    strides.resize(dims);
    
    // 补全shape维度
    std::vector<size_t> padded_shape(dims, 1);
    size_t offset = dims - shape.size();
    for (size_t i = 0; i < shape.size(); ++i) {
        padded_shape[offset + i] = shape[i];
    }
    
    // 计算stride：如果维度为1（广播维度），stride为0
    strides[dims - 1] = (padded_shape[dims - 1] == 1) ? 0 : 1;
    for (int i = dims - 2; i >= 0; --i) {
        if (padded_shape[i] == 1) {
            strides[i] = 0;  // 广播维度，stride为0
        } else {
            strides[i] = strides[i + 1] * padded_shape[i + 1];
        }
    }
}

CT_HOT Tensor Add_SIMD_kernel(const Tensor& a, const Tensor& b) {
    if (a.device() != DeviceType::kCPU || b.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-SIMD Add_Kernel: 仅在CPU支持");
    }
    if (a.dtype() != b.dtype()) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kCPU, ErrorType::DATATYPE,
                          "CPU-SIMD Add_Kernel: Tensor数据类型不匹配");
    }

    // 形状相同：直接SIMD
    if (a.sizes() == b.sizes()) {
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
            _mm256_storeu_ps(&result_data[i], _mm256_add_ps(a_vec, b_vec));
        }
        for (; i < elem_count; ++i) result_data[i] = a_data[i] + b_data[i];
#elif defined(__aarch64__)
        size_t i = 0;
        for (; i + 3 < elem_count; i += 4) {
            float32x4_t a_vec = vld1q_f32(&a_data[i]);
            float32x4_t b_vec = vld1q_f32(&b_data[i]);
            vst1q_f32(&result_data[i], vaddq_f32(a_vec, b_vec));
        }
        for (; i < elem_count; ++i) result_data[i] = a_data[i] + b_data[i];
#else
        for (size_t i = 0; i < elem_count; ++i) result_data[i] = a_data[i] + b_data[i];
#endif
        return result;
    }

    // [HPC 2026-08-11] 快速路径：2D + 1D 行广播（如 [M,N] + [N]，MLP bias 的典型形态）
    // 避免通用广播路径内层循环的逐元素 stride 乘法（a_row_offset + j*a_strides[1]），
    // 让 NEON/AVX 干净向量化。a 需为默认连续（stride [N,1]）。
    if (a.sizes().size() == 2 && b.sizes().size() == 1 &&
        a.sizes()[1] == b.sizes()[0] &&
        a.strides()[0] == a.sizes()[1] && a.strides()[1] == 1) {
        size_t rows = a.sizes()[0];
        size_t cols = a.sizes()[1];
        Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device(), false);
        float* CT_RESTRICT result_data = result.data_write<float>();
        const float* CT_RESTRICT a_data = a.data_read<float>();
        const float* CT_RESTRICT b_data = b.data_read<float>();
#if defined(__AVX2__)
        for (size_t i = 0; i < rows; ++i) {
            const float* CT_RESTRICT ar = a_data + i * cols;
            float* CT_RESTRICT rr = result_data + i * cols;
            size_t j = 0;
            for (; j + 7 < cols; j += 8) {
                __m256 av = _mm256_loadu_ps(&ar[j]);
                __m256 bv = _mm256_loadu_ps(&b_data[j]);
                _mm256_storeu_ps(&rr[j], _mm256_add_ps(av, bv));
            }
            for (; j < cols; ++j) rr[j] = ar[j] + b_data[j];
        }
#elif defined(__aarch64__)
        for (size_t i = 0; i < rows; ++i) {
            const float* CT_RESTRICT ar = a_data + i * cols;
            float* CT_RESTRICT rr = result_data + i * cols;
            size_t j = 0;
            for (; j + 3 < cols; j += 4) {
                float32x4_t av = vld1q_f32(&ar[j]);
                float32x4_t bv = vld1q_f32(&b_data[j]);
                vst1q_f32(&rr[j], vaddq_f32(av, bv));
            }
            for (; j < cols; ++j) rr[j] = ar[j] + b_data[j];
        }
#else
        for (size_t i = 0; i < rows; ++i) {
            const float* ar = a_data + i * cols;
            float* rr = result_data + i * cols;
            for (size_t j = 0; j < cols; ++j) rr[j] = ar[j] + b_data[j];
        }
#endif
        return result;
    }

    // 形状不同：计算公共广播形状和stride（零拷贝）
    size_t max_dims = std::max(a.sizes().size(), b.sizes().size());
    std::vector<size_t> broadcast_shape(max_dims);
    
    for (size_t i = 0; i < max_dims; ++i) {
        size_t a_dim = i < a.sizes().size() ? a.sizes()[a.sizes().size() - 1 - i] : 1;
        size_t b_dim = i < b.sizes().size() ? b.sizes()[b.sizes().size() - 1 - i] : 1;
        if (a_dim != 1 && b_dim != 1 && a_dim != b_dim) {
            CtorchError::throwException(ErrorPlatform::kCPU, ErrorType::DIMENSION,
                "CPU-SIMD Add_Kernel: Tensor形状不兼容");
        }
        broadcast_shape[max_dims - 1 - i] = std::max(a_dim, b_dim);
    }
    
    // 计算广播stride
    std::vector<size_t> a_strides, b_strides;
    computeBroadcastStrides(a.sizes(), broadcast_shape, a_strides);
    computeBroadcastStrides(b.sizes(), broadcast_shape, b_strides);
    
    // 创建结果Tensor
    Tensor result(ShapeTag{}, broadcast_shape, a.dtype(), a.device(), false);
    float* CT_RESTRICT result_data = result.data_write<float>();
    const float* CT_RESTRICT a_data = a.data_read<float>();
    const float* CT_RESTRICT b_data = b.data_read<float>();
    
    size_t elem_count = result.numel();
    
    // 零拷贝广播：用stride索引访问
    // 对于简单形状优化（如 [M, N] + [N] 或 [M, N] + [1, N]）
    if (max_dims == 2) {
        size_t rows = broadcast_shape[0];
        size_t cols = broadcast_shape[1];
        
#ifdef __aarch64__
        for (size_t i = 0; i < rows; ++i) {
            size_t a_row_offset = i * a_strides[0];
            size_t b_row_offset = i * b_strides[0];
            
            size_t j = 0;
            for (; j + 3 < cols; j += 4) {
                // 用stride计算索引
                size_t a_idx = a_row_offset + j * a_strides[1];
                size_t b_idx = b_row_offset + j * b_strides[1];
                
                float32x4_t a_vec = vld1q_f32(&a_data[a_idx]);
                float32x4_t b_vec = vld1q_f32(&b_data[b_idx]);
                vst1q_f32(&result_data[i * cols + j], vaddq_f32(a_vec, b_vec));
            }
            for (; j < cols; ++j) {
                size_t a_idx = a_row_offset + j * a_strides[1];
                size_t b_idx = b_row_offset + j * b_strides[1];
                result_data[i * cols + j] = a_data[a_idx] + b_data[b_idx];
            }
        }
#else
        for (size_t i = 0; i < rows; ++i) {
            size_t a_row_offset = i * a_strides[0];
            size_t b_row_offset = i * b_strides[0];
            
            for (size_t j = 0; j < cols; ++j) {
                size_t a_idx = a_row_offset + j * a_strides[1];
                size_t b_idx = b_row_offset + j * b_strides[1];
                result_data[i * cols + j] = a_data[a_idx] + b_data[b_idx];
            }
        }
#endif
    } else {
        // 通用广播（fallback到标量循环）
        std::vector<size_t> indices(max_dims);
        for (size_t i = 0; i < elem_count; ++i) {
            // 计算多维索引
            size_t temp = i;
            for (int j = max_dims - 1; j >= 0; --j) {
                indices[j] = temp % broadcast_shape[j];
                temp /= broadcast_shape[j];
            }
            
            // 用stride计算源索引
            size_t a_idx = 0, b_idx = 0;
            for (size_t j = 0; j < max_dims; ++j) {
                a_idx += indices[j] * a_strides[j];
                b_idx += indices[j] * b_strides[j];
            }
            
            result_data[i] = a_data[a_idx] + b_data[b_idx];
        }
    }
    
    return result;
}