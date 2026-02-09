/**
 * @file CrossEntropy_BASIC_kernel.cpp
 * @brief CPU-BASIC CrossEntropy（交叉熵）算子
 * @author GhostFace
 * @date 2026/02/09
 */

#include "./../kernels.h"
#include "./../../Ctorch_Error.h"
#include "./../../Tensor.h"

Tensor CrossEntropy_BASIC_kernel(const Tensor& a, const Tensor& b) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU || b.device() != DeviceType::kCPU) {
        Ctorch_Error::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-BASIC CrossEntropy_Kernel: 仅在CPU支持");
    }
    
    // 校验形状和数据类型
    if (a.sizes() != b.sizes()) {
        Ctorch_Error::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                          "CPU-BASIC CrossEntropy_Kernel: 张量形状不一致");
    }
    
    if (a.dtype() != b.dtype()) {
        Ctorch_Error::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                          "CPU-BASIC CrossEntropy_Kernel: 张量数据类型不一致");
    }
    
    // 实现CrossEntropy损失函数
    size_t count = a.numel();
    const float *data_a = a.data<float>();
    const float *data_b = b.data<float>();
    
    float cross_entropy = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        // 确保预测值不为0，避免log(0)的情况
        float pred = std::max(data_a[i], 1e-10f);
        cross_entropy -= data_b[i] * std::log(pred);
    }
    
    // 创建标量结果张量
    Tensor result(cross_entropy);
    return result;
}