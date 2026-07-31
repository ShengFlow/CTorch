/**
 * @file LReLU_BASIC_kernel.cpp
 * @brief CPU-BASIC Leaky ReLU 算子
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include "../../../include/CoreDefs.h"
#include <algorithm>

CT_HOT Tensor LReLU_BASIC_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                         ErrorType::DEVICE_COMPAT,
                         "CPU-BASIC LReLU_Kernel: 仅在CPU支持");
    }

    constexpr float negative_slope = 0.01f;
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
    size_t count = a.numel();
    const float* CT_RESTRICT a_data = a.data_read<float>();
    float* CT_RESTRICT result_data = result.data_write<float>();

    for (size_t i = 0; i < count; ++i) {
        result_data[i] = a_data[i] > 0.0f ? a_data[i] : a_data[i] * negative_slope;
    }

    return result;
}
