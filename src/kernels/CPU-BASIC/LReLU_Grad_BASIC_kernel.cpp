/**
 * @file LReLU_Grad_BASIC_kernel.cpp
 * @brief CPU-BASIC Leaky ReLU 反向梯度算子
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include "../../../include/CoreDefs.h"

CT_HOT Tensor LReLU_Grad_BASIC_kernel(const Tensor& x, const Tensor& grad_out) {
    if (x.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(x.device()),
                         ErrorType::DEVICE_COMPAT,
                         "CPU-BASIC LReLU_Grad_Kernel: 仅在CPU支持");
    }
    if (x.device() != grad_out.device() || x.sizes() != grad_out.sizes()) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(x.device()),
                         ErrorType::DIMENSION,
                         "CPU-BASIC LReLU_Grad_Kernel: 输入设备或形状不匹配");
    }

    constexpr float negative_slope = 0.01f;
    Tensor grad_x(ShapeTag{}, x.sizes(), x.dtype(), x.device());
    size_t count = x.numel();
    const float* CT_RESTRICT x_data = x.data<float>();
    const float* CT_RESTRICT grad_out_data = grad_out.data<float>();
    float* CT_RESTRICT grad_x_data = grad_x.data<float>();

    for (size_t i = 0; i < count; ++i) {
        grad_x_data[i] = x_data[i] > 0.0f ? grad_out_data[i] : grad_out_data[i] * negative_slope;
    }

    return grad_x;
}
