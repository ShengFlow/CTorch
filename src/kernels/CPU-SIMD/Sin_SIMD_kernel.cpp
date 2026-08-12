/**
 * @file Sin_SIMD_kernel.cpp
 * @brief CPU-SIMD 正弦算子
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include "../../../include/CoreDefs.h"
#include <cmath>

Tensor Sin_SIMD_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                         ErrorType::DEVICE_COMPAT,
                         "CPU-SIMD Sin_Kernel: 仅在CPU支持");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device(), false);
    size_t n = a.numel();
    const float* CT_RESTRICT src = a.data_read<float>();
    float* CT_RESTRICT dst = result.data_write<float>();

    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        dst[i] = std::sin(src[i]);
    }

    return result;
}
