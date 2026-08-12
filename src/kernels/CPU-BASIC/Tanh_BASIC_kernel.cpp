/**
 * @file Tanh_BASIC_kernel.cpp
 * @brief CPU-BASIC Tanh算子
 * @author GhostFace
 * @date 2026/02/09
 */

#include <cmath>
#include "./../kernels.h"
#include "./../../../include/CtorchError.h"
#include "./../../../include/Tensor.h"
#include "./../../../include/CoreDefs.h"

CT_HOT Tensor Tanh_BASIC_kernel(const Tensor &a) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                          ErrorType::DEVICE_COMPAT, "CPU-BASIC Tanh_Kernel: 仅在CPU支持");
    }
    // 实现Tanh激活函数
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device(), false);

    size_t count = a.numel();
    const float* CT_RESTRICT a_data = a.data_read<float>();
    float* CT_RESTRICT result_data = result.data_write<float>();
    for (size_t i = 0; i < count; ++i) {
        float exp_x     = std::exp(a_data[i]);
        float exp_neg_x = std::exp(-a_data[i]);
        result_data[i]  = (exp_x - exp_neg_x) / (exp_x + exp_neg_x);
    }

    return result;
}
