/**
 * @file GELU_BASIC_kernel.cpp
 * @brief CPU-BASIC GELU算子
 * @author GhostFace
 * @date 2026/07/28
 */

#include <cmath>
#include "./../kernels.h"
#include "./../../../include/CtorchError.h"
#include "./../../../include/Tensor.h"
#include "./../../../include/CoreDefs.h"

namespace {
constexpr float kSqrt2OverPi = 0.7978845608f;
constexpr float kGeluCoeff = 0.044715f;

inline float gelu_scalar(float x) {
    float v = kSqrt2OverPi * (x + kGeluCoeff * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(v));
}
}

CT_HOT Tensor GELU_BASIC_kernel(const Tensor &a) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                          ErrorType::DEVICE_COMPAT, "CPU-BASIC GELU_Kernel: 仅在CPU支持");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());

    size_t count = a.numel();
    const float* CT_RESTRICT a_data = a.data_read<float>();
    float* CT_RESTRICT result_data = result.data_write<float>();
    for (size_t i = 0; i < count; ++i) {
        result_data[i] = gelu_scalar(a_data[i]);
    }

    return result;
}
