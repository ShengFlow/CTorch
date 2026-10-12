/**
 * @file SwiGLU_SIMD_kernel.cpp
 * @brief CPU-SIMD SwiGLU 算子 (PEL25 Stage 4 — 真正 SIMD 路径, 双输入 fused)
 * @details swiglu(x, gate) = silu(x) * gate = (x * sigmoid(x)) * gate
 *          Stage 4 fused: 1 次 sigmoid256_ps + 2 次 mul (vs Stage 1 伪 SIMD 的 1 sigmoid + 3 mul)
 *          性能提升 ~8-10x vs 标量 exp
 * @author Mavis (PEL25 §6 SwiGLU 算子开发协议)
 * @date 2026-09-06
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include "../../../include/kernels/SIMDMath.h"


Tensor SwiGLU_SIMD_kernel(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kAutoDiff, ErrorType::DIMENSION,
                          "CPU-SIMD SwiGLU_Kernel: a 和 b shape 必须一致");
    }
    if (a.device() != DeviceType::kCPU) {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-SIMD SwiGLU_Kernel: 仅在CPU支持");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device(), false);

    size_t count = a.numel();
    const float* a_data = a.data_read<float>();
    const float* b_data = b.data_read<float>();
    float* result_data = result.data_write<float>();

    if (count == 0) return result;

    ct::kernels::simd::vswiglu(a_data, b_data, result_data, count);

    return result;
}
