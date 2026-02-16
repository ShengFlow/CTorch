/**
 * @file Add_BASIC_kernel.h
 * @brief CPU-BASIC MatMul算子
 * @author GhostFace
 * @date 2025/12/22
 */

#include "./../kernels.h"
#include "./../../../include/Ctorch_Error.h"
#include "./../../../include/Tensor.h"

// 全局的matMul函数
Tensor MatMul_BASIC_kernel(const Tensor& a, const Tensor& b) {
    
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU || b.device() != DeviceType::kCPU) {
        Ctorch_Error::log(ErrorLevel::ERROR,DeviceTypeToErrorPlatform(a.device()),ErrorType::DEVICE_COMPAT,"CPU-BASIC MatMul_Kernel: 仅在CPU支持");
    }
    // 校验数据类型
    if (a.dtype() != b.dtype()) {
        Ctorch_Error::log(ErrorLevel::ERROR,ErrorPlatform::kCPU,ErrorType::DATATYPE,"CPU-BASIC MatMul_Kernel: Tensor数据类型不匹配");
    }

    // 简单实现，仅支持2D张量
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DIMENSION, "MatMul仅支持2D张量");
    }
    if (a.shape()[1] != b.shape()[0]) {
        Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DIMENSION, "矩阵维度不匹配");
    }
    
    size_t m = a.shape()[0];
    size_t k = a.shape()[1];
    size_t n = b.shape()[1];
    
    Tensor result(ShapeTag{}, {m, n}, a.dtype(), a.device());
    // 考虑步长的矩阵乘法实现
    const std::vector<size_t>& a_strides = a.strides();
    const std::vector<size_t>& b_strides = b.strides();
    const std::vector<size_t>& result_strides = result.strides();
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; ++l) {
                // 使用步长访问数据，支持转置后的张量
                size_t a_idx = i * a_strides[0] + l * a_strides[1];
                size_t b_idx = l * b_strides[0] + j * b_strides[1];
                sum += a.data<float>()[a_idx] * b.data<float>()[b_idx];
            }
            size_t result_idx = i * result_strides[0] + j * result_strides[1];
            result.data<float>()[result_idx] = sum;
        }
    }
    
    return result;
}
