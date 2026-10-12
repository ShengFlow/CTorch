/**
 * @file SwiGLU.cpp
 * @brief SwiGLU 算子 Eager CPU 实现 (BASIC + SIMD AVX2)
 * @details swiglu(x, gate) = silu(x) * gate, 双输入 elementwise
 * @author Mavis (PEL25 §6 SwiGLU 算子开发协议 Stage 1)
 * @date 2026-09-05
 */

#include "ops/SwiGLU.h"
#include "CtorchError.h"
#include "kernels/SIMDMath.h"

namespace ct::ops {

Tensor swiglu_forward(const Tensor& x, const Tensor& gate) {
    if (x.dtype() != DType::kFloat || gate.dtype() != DType::kFloat) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "SwiGLU forward: only float32 supported (Stage 1)");
    }
    if (x.device() != DeviceType::kCPU || gate.device() != DeviceType::kCPU) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT, "SwiGLU forward: only CPU device supported (Stage 1)");
    }
    if (x.shape() != gate.shape()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION, "SwiGLU forward: x and gate shape mismatch (Stage 1 不做 broadcasting)");
    }

    const auto& shape = x.shape();
    size_t n = 1;
    for (auto dim : shape) n *= dim;

    Tensor output(ShapeTag{}, shape, DType::kFloat, DeviceType::kCPU);

    const float* x_ptr = x.data_read<float>();
    const float* g_ptr = gate.data_read<float>();
    float* y_ptr = output.data_write<float>();

    ct::kernels::simd::vswiglu(x_ptr, g_ptr, y_ptr, n);

    return output;
}

std::pair<Tensor, Tensor> swiglu_backward(const Tensor& grad_output, const Tensor& x, const Tensor& gate) {
    if (grad_output.dtype() != DType::kFloat || x.dtype() != DType::kFloat || gate.dtype() != DType::kFloat) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "SwiGLU backward: only float32 supported (Stage 1)");
    }
    if (grad_output.shape() != x.shape() || x.shape() != gate.shape()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION, "SwiGLU backward: shape mismatch");
    }

    const auto& shape = x.shape();
    size_t n = 1;
    for (auto dim : shape) n *= dim;

    Tensor grad_x(ShapeTag{}, shape, DType::kFloat, DeviceType::kCPU);
    Tensor grad_gate(ShapeTag{}, shape, DType::kFloat, DeviceType::kCPU);

    const float* go_ptr = grad_output.data_read<float>();
    const float* x_ptr = x.data_read<float>();
    const float* g_ptr = gate.data_read<float>();
    float* gx_ptr = grad_x.data_write<float>();
    float* gg_ptr = grad_gate.data_write<float>();

    ct::kernels::simd::vswiglu_backward(go_ptr, x_ptr, g_ptr, gx_ptr, gg_ptr, n);

    return {grad_x, grad_gate};
}

}  // namespace ct::ops
