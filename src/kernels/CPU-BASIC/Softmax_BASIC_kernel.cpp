/**
 * @file Softmax_BASIC_kernel.cpp
 * @brief CPU-BASIC Softmax算子
 * @author GhostFace
 * @date 2026/02/09
 */

#include <cmath>
#include "./../kernels.h"
#include "./../../../include/Ctorch_Error.h"
#include "./../../../include/Tensor.h"

Tensor Softmax_BASIC_kernel(const Tensor &a) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU) {
        Ctorch_Error::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                          ErrorType::DEVICE_COMPAT, "CPU-BASIC Softmax_Kernel: 仅在CPU支持");
    }
    // 实现Softmax激活函数
    Tensor result(a);

    size_t count = a.numel();
    float *data  = result.data<float>();

    // 计算指数和
    float sum_exp = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        data[i] = std::exp(data[i]);
        sum_exp += data[i];
    }

    // 归一化
    for (size_t i = 0; i < count; ++i) {
        data[i] /= sum_exp;
    }

    return result;
}
