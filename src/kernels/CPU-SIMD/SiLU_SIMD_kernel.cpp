/**
 * @file SiLU_SIMD_kernel.cpp
 * @brief CPU-SIMD SiLU 算子 (PEL25 Stage 4 — 真正 SIMD 路径)
 * @details silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *          Stage 4: 用 ct::kernels::simd::silu256_ps 真正 SIMD (含 Padé 多项式 sigmoid256_ps 复用)
 *          vs Stage 1 伪 SIMD (标量 exp + SIMD loadu/storeu 包装), 性能提升 ~8-10x
 * @author Mavis (PEL25 §6 SwiGLU 算子开发协议)
 * @date 2026-09-06
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include "../../../include/kernels/SIMDMath.h"


Tensor SiLU_SIMD_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kCPU) {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-SIMD SiLU_Kernel: 仅在CPU支持");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device(), false);

    size_t count = a.numel();
    const float* a_data = a.data_read<float>();
    float* result_data = result.data_write<float>();

    if (count == 0) return result;

    ct::kernels::simd::vsilu(a_data, result_data, count);

    return result;
}
