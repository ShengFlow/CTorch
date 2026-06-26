/**
 * @file Sin_BASIC_kernel.h
 * @brief CPU-BASIC Sin算子
 * @author GhostFace
 * @date 2026/2/4
 */

#include <cmath>
#include "./../kernels.h"
#include "./../../../include/CtorchError.h"
#include "./../../../include/Tensor.h"

Tensor Sin_BASIC_kernel(const Tensor &a) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU) {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                          ErrorType::DEVICE_COMPAT, "CPU-BASIC Sin_Kernel: 仅在CPU支持");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());

    size_t count = a.numel();
    const float *a_data = a.data<float>();
    float *result_data = result.data<float>();
    for (size_t i = 0; i < count; ++i) {
        result_data[i] = std::sin(a_data[i]);
    }
    return result;
}
