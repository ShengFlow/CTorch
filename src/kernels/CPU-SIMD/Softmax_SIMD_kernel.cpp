/**
 * @file Softmax_SIMD_kernel.cpp
 * @brief CPU-SIMD Softmax 算子
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include "../../../include/CoreDefs.h"
#include <cmath>
#include <algorithm>

Tensor Softmax_SIMD_kernel(const Tensor& a, int dim) {
    if (a.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                         ErrorType::DEVICE_COMPAT,
                         "CPU-SIMD Softmax_Kernel: 仅在CPU支持");
    }

    int actual_dim = dim;
    if (actual_dim < 0) {
        actual_dim = static_cast<int>(a.sizes().size()) + actual_dim;
    }
    if (actual_dim < 0 || actual_dim >= static_cast<int>(a.sizes().size())) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                         "CPU-SIMD Softmax_Kernel: dim 超出范围");
        return Tensor();
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
    const float* src = a.data<float>();
    float* dst = result.data<float>();
    const auto& shape = a.sizes();

    size_t outer = 1;
    for (int i = 0; i < actual_dim; ++i) outer *= shape[i];
    size_t inner = 1;
    for (size_t i = actual_dim + 1; i < shape.size(); ++i) inner *= shape[i];
    size_t axis_size = shape[actual_dim];
    size_t axis_stride = inner;

    #pragma omp parallel for
    for (size_t o = 0; o < outer; ++o) {
        for (size_t i = 0; i < inner; ++i) {
            size_t base = o * axis_size * axis_stride + i;
            float max_val = src[base];
            for (size_t j = 1; j < axis_size; ++j) {
                max_val = std::max(max_val, src[base + j * axis_stride]);
            }
            float exp_sum = 0.0f;
            #pragma omp simd reduction(+:exp_sum)
            for (size_t j = 0; j < axis_size; ++j) {
                float e = std::exp(src[base + j * axis_stride] - max_val);
                dst[base + j * axis_stride] = e;
                exp_sum += e;
            }
            float inv_sum = 1.0f / exp_sum;
            #pragma omp simd
            for (size_t j = 0; j < axis_size; ++j) {
                dst[base + j * axis_stride] *= inv_sum;
            }
        }
    }

    return result;
}
