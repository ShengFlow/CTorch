/**
 * @file SwiGLU_BASIC_kernel.cpp
 * @brief CPU-BASIC SwiGLU 算子 (PEL25 Stage 3.2, 双输入)
 * @details swiglu(x, gate) = silu(x) * gate = (x / (1 + exp(-x))) * gate
 *          标量版本, 通用 CPU + 测试对照
 * @author Mavis (PEL25 §6 SwiGLU 算子开发协议)
 * @date 2026-09-06
 */

#include <cmath>
#include "./../kernels.h"
#include "./../../../include/CtorchError.h"
#include "./../../../include/Tensor.h"
#include "./../../../include/CoreDefs.h"

namespace {
inline float swiglu_scalar(float x, float gate) {
    float silu_x = x / (1.0f + std::exp(-x));  // silu(x)
    return silu_x * gate;                       // silu(x) * gate
}
}

CT_HOT Tensor SwiGLU_BASIC_kernel(const Tensor &a, const Tensor &b) {
    // 校验: a/b 同 shape
    if (a.shape() != b.shape()) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kAutoDiff, ErrorType::DIMENSION,
                          "CPU-BASIC SwiGLU_Kernel: a 和 b shape 必须一致");
    }
    if (a.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                          ErrorType::DEVICE_COMPAT, "CPU-BASIC SwiGLU_Kernel: 仅在CPU支持");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device(), false);

    size_t count = a.numel();
    const float* CT_RESTRICT a_data = a.data_read<float>();
    const float* CT_RESTRICT b_data = b.data_read<float>();
    float* CT_RESTRICT result_data = result.data_write<float>();
    for (size_t i = 0; i < count; ++i) {
        result_data[i] = swiglu_scalar(a_data[i], b_data[i]);
    }

    return result;
}
