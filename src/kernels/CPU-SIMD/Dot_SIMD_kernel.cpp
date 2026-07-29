/**
 * @file Dot_SIMD_kernel.cpp
 * @brief CPU-SIMD 点乘算子
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include "../../../include/CoreDefs.h"

Tensor Dot_SIMD_kernel(const Tensor& a, const Tensor& b) {
    if (a.device() != DeviceType::kCPU || b.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                         ErrorType::DEVICE_COMPAT,
                         "CPU-SIMD Dot_Kernel: 仅在CPU支持");
    }
    if (a.dtype() != b.dtype()) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                         "CPU-SIMD Dot_Kernel: 张量数据类型不一致");
    }
    if (a.numel() != b.numel()) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                         "CPU-SIMD Dot_Kernel: 张量元素数量不一致");
        return Tensor(0.0f);
    }

    size_t n = a.numel();
    const float* CT_RESTRICT a_data = a.data<float>();
    const float* CT_RESTRICT b_data = b.data<float>();
    float sum = 0.0f;

    #pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < n; ++i) {
        sum += a_data[i] * b_data[i];
    }

    return Tensor(sum);
}
