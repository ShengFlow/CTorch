/**
 * @file Sigmoid_BASIC_kernel.cpp
 * @brief CPU-BASIC Sigmoid算子
 * @author GhostFace
 * @date 2026/02/09
 */

#include <cmath>
#include "./../kernels.h"
#include "./../../../include/CtorchError.h"
#include "./../../../include/Tensor.h"
#include "./../../../include/CoreDefs.h"

CT_HOT Tensor Sigmoid_BASIC_kernel(const Tensor &a) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                          ErrorType::DEVICE_COMPAT, "CPU-BASIC Sigmoid_Kernel: 仅在CPU支持");
    }
    // 实现Sigmoid激活函数
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());

    size_t count = a.numel();
    const float* CT_RESTRICT a_data = a.data<float>();
    float* CT_RESTRICT result_data = result.data<float>();
    for (size_t i = 0; i < count; ++i) {
        result_data[i] = 1.0f / (1.0f + std::exp(-a_data[i]));
    }

    return result;
}
