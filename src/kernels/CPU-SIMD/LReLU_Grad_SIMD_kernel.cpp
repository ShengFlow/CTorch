/**
 * @file LReLU_Grad_SIMD_kernel.cpp
 * @brief CPU-SIMD Leaky ReLU 反向梯度算子
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include "../../../include/CoreDefs.h"

CT_HOT Tensor LReLU_Grad_SIMD_kernel(const Tensor& x, const Tensor& grad_out) {
    if (x.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(x.device()),
                         ErrorType::DEVICE_COMPAT,
                         "CPU-SIMD LReLU_Grad_Kernel: 仅在CPU支持");
    }
    if (x.device() != grad_out.device() || x.sizes() != grad_out.sizes()) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(x.device()),
                         ErrorType::DIMENSION,
                         "CPU-SIMD LReLU_Grad_Kernel: 输入设备或形状不匹配");
    }

    constexpr float negative_slope = 0.01f;
    Tensor grad_x(ShapeTag{}, x.sizes(), x.dtype(), x.device());
    size_t n = x.numel();
    const float* CT_RESTRICT x_p = x.data_read<float>();
    const float* CT_RESTRICT gout_p = grad_out.data_read<float>();
    float* CT_RESTRICT gx_p = grad_x.data_write<float>();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        gx_p[i] = x_p[i] > 0.0f ? gout_p[i] : gout_p[i] * negative_slope;
    }

    return grad_x;
}
