/**
 * @file MSE_BASIC_kernel.cpp
 * @brief CPU-BASIC MSE（均方误差）算子
 * @author GhostFace
 * @date 2026/02/09
 */

#include "./../kernels.h"
#include "./../../../include/CtorchError.h"
#include "./../../../include/Tensor.h"
#include "./../../../include/CoreDefs.h"

CT_HOT Tensor MSE_BASIC_kernel(const Tensor& a, const Tensor& b) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU || b.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-BASIC MSE_Kernel: 仅在CPU支持");
    }
    
    // 校验形状和数据类型
    if (a.sizes() != b.sizes()) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                          "CPU-BASIC MSE_Kernel: 张量形状不一致");
    }
    
    if (a.dtype() != b.dtype()) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                          "CPU-BASIC MSE_Kernel: 张量数据类型不一致");
    }
    
    // 实现MSE损失函数
    size_t count = a.numel();
    const float* CT_RESTRICT data_a = a.data_read<float>();
    const float* CT_RESTRICT data_b = b.data_read<float>();
    
    float sum_squared_error = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        float diff = data_a[i] - data_b[i];
        sum_squared_error += diff * diff;
    }
    
    float mse = sum_squared_error / count;
    
    // 创建标量结果张量
    Tensor result(mse);
    return result;
}