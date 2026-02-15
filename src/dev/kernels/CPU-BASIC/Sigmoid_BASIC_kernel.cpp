/**
 * @file Sigmoid_BASIC_kernel.cpp
 * @brief CPU-BASIC Sigmoid算子
 * @author GhostFace
 * @date 2026/02/09
 */

#include <cmath>
#include "./../kernels.h"
#include "./../../Ctorch_Error.h"
#include "./../../Tensor.h"

Tensor Sigmoid_BASIC_kernel(const Tensor &a) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU) {
        Ctorch_Error::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                          ErrorType::DEVICE_COMPAT, "CPU-BASIC Sigmoid_Kernel: 仅在CPU支持");
    }
    // 实现Sigmoid激活函数
    Tensor result(a);

    size_t count = a.numel();
    float *data  = result.data<float>();
    for (size_t i = 0; i < count; ++i) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }

    return result;
}
