/**
 * @file SwiGLU_AMX_kernel.cpp
 * @brief CPU-AMX SwiGLU 算子 (PEL25 Stage 3.2, 双输入, AMX 降级)
 * @details swiglu(x, gate) = silu(x) * gate
 *          AMX 槽位对 binary/elementwise 操作降级 (PEL23 §13 NR-3)
 * @author Mavis (PEL25 §6 SwiGLU 算子开发协议)
 * @date 2026-09-06
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include <cmath>

namespace {
inline float swiglu_scalar(float x, float gate) {
    float silu_x = x / (1.0f + std::exp(-x));
    return silu_x * gate;
}
}

// AMX 降级: SwiGLU 不是 AMX 原生, 标量 fallback
Tensor SwiGLU_AMX_kernel(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kAutoDiff, ErrorType::DIMENSION,
                          "CPU-AMX SwiGLU_Kernel: a 和 b shape 必须一致");
    }
    if (a.device() != DeviceType::kCPU) {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-AMX SwiGLU_Kernel: 仅在CPU支持");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device(), false);

    size_t count = a.numel();
    const float* a_data = a.data_read<float>();
    const float* b_data = b.data_read<float>();
    float* result_data = result.data_write<float>();
    for (size_t i = 0; i < count; ++i) {
        result_data[i] = swiglu_scalar(a_data[i], b_data[i]);
    }
    return result;
}
