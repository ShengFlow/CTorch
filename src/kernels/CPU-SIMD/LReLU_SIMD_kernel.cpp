/**
 * @file LReLU_SIMD_kernel.cpp
 * @brief CPU-SIMD Leaky ReLU 算子
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include "../../../include/CoreDefs.h"
#include <algorithm>

Tensor LReLU_SIMD_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                         ErrorType::DEVICE_COMPAT,
                         "CPU-SIMD LReLU_Kernel: 仅在CPU支持");
    }

    constexpr float negative_slope = 0.01f;
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
    size_t n = a.numel();
    const float* CT_RESTRICT src = a.data<float>();
    float* CT_RESTRICT dst = result.data<float>();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        dst[i] = src[i] > 0.0f ? src[i] : src[i] * negative_slope;
    }

    return result;
}
