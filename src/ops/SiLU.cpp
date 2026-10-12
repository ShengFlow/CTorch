/**
 * @file SiLU.cpp
 * @brief SiLU 算子 Eager CPU 实现 (BASIC + SIMD AVX2)
 * @details 标量 fallback 处理 tail (<8 elements), SIMD 主路径走 AVX2 8-wide
 * @author Mavis (PEL25 §6 SwiGLU 算子开发协议 Stage 1)
 * @date 2026-09-05
 *
 * Stage 1 实施: 不接 C3 dispatch, 走纯 Eager CPU + Autograd
 * Stage 2 (PEL25 §7 HITL 决策点): op 枚举扩展 + C3 Kernel Registry 接入
 */

#include "ops/SiLU.h"
#include "CtorchError.h"
#include "kernels/SIMDMath.h"

namespace ct::ops {

// ============================================================
// Public API
// ============================================================

Tensor silu_forward(const Tensor& input) {
    if (input.dtype() != DType::kFloat) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "SiLU forward: only float32 supported (Stage 1)");
    }
    if (input.device() != DeviceType::kCPU) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT, "SiLU forward: only CPU device supported (Stage 1)");
    }

    const auto& shape = input.shape();
    size_t n = 1;
    for (auto dim : shape) n *= dim;

    Tensor output(ShapeTag{}, shape, DType::kFloat, DeviceType::kCPU);

    const float* in_ptr = input.data_read<float>();
    float* out_ptr = output.data_write<float>();

    ct::kernels::simd::vsilu(in_ptr, out_ptr, n);

    return output;
}

Tensor silu_backward(const Tensor& grad_output, const Tensor& input) {
    if (grad_output.dtype() != DType::kFloat || input.dtype() != DType::kFloat) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "SiLU backward: only float32 supported (Stage 1)");
    }
    if (grad_output.device() != DeviceType::kCPU || input.device() != DeviceType::kCPU) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT, "SiLU backward: only CPU device supported (Stage 1)");
    }
    if (grad_output.shape() != input.shape()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION, "SiLU backward: grad_output and input shape mismatch");
    }

    const auto& shape = input.shape();
    size_t n = 1;
    for (auto dim : shape) n *= dim;

    Tensor grad_input(ShapeTag{}, shape, DType::kFloat, DeviceType::kCPU);

    const float* go_ptr = grad_output.data_read<float>();
    const float* in_ptr = input.data_read<float>();
    float* gi_ptr = grad_input.data_write<float>();

    ct::kernels::simd::vsilu_backward(go_ptr, in_ptr, gi_ptr, n);

    return grad_input;
}

}  // namespace ct::ops
