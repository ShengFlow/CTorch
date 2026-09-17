/**
 * @file Sqrt_BASIC_kernel.cpp
 * @author 苏璃珞
 * @brief CPU-BASIC 平方根算子
 * @date 2026/9/17
 */

#include "../../../include/Tensor.h"
#include "../../../include/Ctools.h"
#include "../../../include/CtorchError.h"
#include "../../../include/CoreDefs.h"
#include <cmath>

CT_HOT Tensor Sqrt_BASIC_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT, "CPU-BASIC Sqrt_Kernel: 仅在CPU支持");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device(), false);
    const float* CT_RESTRICT in = a.data_read<float>();
    float* CT_RESTRICT out = result.data_write<float>();

    const size_t count = a.numel();
    for (size_t i = 0; i < count; ++i) {
        out[i] = std::sqrt(in[i]);
    }

    return result;
}
