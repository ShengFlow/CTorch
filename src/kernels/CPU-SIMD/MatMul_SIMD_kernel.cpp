/**
 * @file MatMul_SIMD_kernel.cpp
 * @brief CPU-SIMD 矩阵乘法算子（朴素向量化，作为 AMX 不可用时的 fallback）
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include "../../../include/CoreDefs.h"

Tensor MatMul_SIMD_kernel(const Tensor& a, const Tensor& b) {
    if (a.device() != DeviceType::kCPU || b.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                         ErrorType::DEVICE_COMPAT,
                         "CPU-SIMD MatMul_Kernel: 仅在CPU支持");
    }
    if (a.dtype() != b.dtype()) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                         "CPU-SIMD MatMul_Kernel: 张量数据类型不一致");
    }
    if (a.sizes().size() != 2 || b.sizes().size() != 2) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                         "CPU-SIMD MatMul_Kernel: 仅支持 2D 矩阵");
        return Tensor();
    }

    size_t m = a.sizes()[0];
    size_t k = a.sizes()[1];
    size_t n = b.sizes()[1];
    if (k != b.sizes()[0]) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                         "CPU-SIMD MatMul_Kernel: 矩阵维度不匹配");
        return Tensor();
    }

    Tensor result(ShapeTag{}, {m, n}, a.dtype(), a.device());
    result.zero();

    const float* CT_RESTRICT a_data = a.data_read<float>();
    const float* CT_RESTRICT b_data = b.data_read<float>();
    float* CT_RESTRICT r_data = result.data_write<float>();

    const auto& a_strides = a.strides();
    const auto& b_strides = b.strides();
    size_t a_stride0 = a_strides[0];
    size_t a_stride1 = a_strides[1];
    size_t b_stride0 = b_strides[0];
    size_t b_stride1 = b_strides[1];

    #pragma omp parallel for
    for (size_t i = 0; i < m; ++i) {
        for (size_t l = 0; l < k; ++l) {
            float a_val = a_data[i * a_stride0 + l * a_stride1];
            #pragma omp simd
            for (size_t j = 0; j < n; ++j) {
                r_data[i * n + j] += a_val * b_data[l * b_stride0 + j * b_stride1];
            }
        }
    }

    return result;
}
