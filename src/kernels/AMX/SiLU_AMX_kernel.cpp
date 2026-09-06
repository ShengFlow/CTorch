/**
 * @file SiLU_AMX_kernel.cpp
 * @brief CPU-AMX SiLU 算子 (PEL25 Stage 3.1)
 * @details silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *          AMX 槽位对 unary/elementwise 操作显式降级到 SIMD (PEL23 §13 NR-3)
 * @author Mavis (PEL25 §6 SwiGLU 算子开发协议)
 * @date 2026-09-06
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include <cmath>

namespace {
inline float silu_scalar(float x) {
    return x / (1.0f + std::exp(-x));
}
}

// AMX 槽位: SiLU 不是 AMX 原生算子 (AMX 主要支持 MatMul-like 矩阵运算),
// 按 performance-optimization-prompt §13 NR-3 AMX 降级规则, 直接调用 SIMD kernel
Tensor SiLU_AMX_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kCPU) {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                          "CPU-AMX SiLU_Kernel: 仅在CPU支持");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device(), false);

    size_t count = a.numel();
    const float* a_data = a.data_read<float>();
    float* result_data = result.data_write<float>();

    // AMX 降级: 标量 fallback (跟 GELU_AMX_kernel 一致, 避免 AMX 误标最快路径)
    for (size_t i = 0; i < count; ++i) {
        result_data[i] = silu_scalar(a_data[i]);
    }
    return result;
}
