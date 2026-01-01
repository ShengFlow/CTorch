/**
 * @file Add_BASIC_kernel.h
 * @brief CPU-BASIC MatMul算子
 * @author GhostFace
 * @date 2025/12/22
 */

#include "./../kernels.h"
#include "./../../Ctorch_Error.h"
#include "./../../Tensor.h"

// 全局的matMul函数
Tensor MatMul_BASIC_kernel(const Tensor& a, const Tensor& b) {
    
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU || b.device() != DeviceType::kCPU) {
        Ctorch_Error::log(ErrorLevel::ERROR,DeviceTypeToErrorPlatform(a.device()),ErrorType::DEVICE_COMPAT,"CPU-BASIC MatMul_Kernel: 仅在CPU支持");
    }
    // 校验形状：必须一致
    if (a.sizes() != b.sizes()) {
        Ctorch_Error::log(ErrorLevel::ERROR,ErrorPlatform::kCPU,ErrorType::DIMENSION,"CPU-BASIC MatMul_Kernel: Tensor形状不匹配");
    }
    // 校验数据类型
    if (a.dtype() != b.dtype()) {
        Ctorch_Error::log(ErrorLevel::ERROR,ErrorPlatform::kCPU,ErrorType::DIMENSION,"CPU-BASIC MatMul_Kernel: Tensor数据类型不匹配");
    }

    // 简单实现，仅支持2D张量
    if (a.shape().size() != 2 || b.shape().size() != 2) {
        throw std::invalid_argument("matMul仅支持2D张量");
    }
    if (a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument("矩阵维度不匹配");
    }
    
    size_t m = a.shape()[0];
    size_t k = a.shape()[1];
    size_t n = b.shape()[1];
    
    Tensor result(ShapeTag{}, {m, n}, a.dtype(), a.device());
    // 简单的矩阵乘法实现
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; ++l) {
                sum += a.data<float>()[i * k + l] * b.data<float>()[l * n + j];
            }
            result.data<float>()[i * n + j] = sum;
        }
    }
    
    return result;
}
