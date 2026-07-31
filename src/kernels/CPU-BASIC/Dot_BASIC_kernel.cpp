/**
 * @file Dot_BASIC_kernel.cpp
 * @brief CPU-BASIC Dot算子
 * @author GhostFace
 * @date 2026/2/2
 */

#include "./../kernels.h"
#include "./../../../include/CtorchError.h"
#include "./../../../include/Tensor.h"
#include "./../../../include/CoreDefs.h"

CT_HOT Tensor Dot_BASIC_kernel(const Tensor& a, const Tensor& b) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU || b.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                         ErrorType::DEVICE_COMPAT, "CPU-BASIC Dot_Kernel: 仅在CPU支持");
    }

    // 校验元素数量：必须相同
    if (a.numel() != b.numel()) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kCPU,
                         ErrorType::DIMENSION, "CPU-BASIC Dot_Kernel: Tensor元素数量不匹配");
    }

    // 校验数据类型
    if (a.dtype() != b.dtype()) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kCPU,
                         ErrorType::DATATYPE, "CPU-BASIC Dot_Kernel: Tensor数据类型不匹配");
    }

    size_t elem_count = a.numel();

    // 获取Tensor数据指针
    const float* CT_RESTRICT a_data = a.data_read<float>();
    const float* CT_RESTRICT b_data = b.data_read<float>();

    // 计算点乘：对应元素相乘后求和
    float dot_result = 0.0f;
    for (size_t i = 0; i < elem_count; ++i) {
        dot_result += a_data[i] * b_data[i];
    }

    // 创建标量结果Tensor
    Tensor result(dot_result);
    return result;
}