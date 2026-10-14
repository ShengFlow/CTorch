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
        CtorchError::throwException(ErrorPlatform::kAutoDiff, ErrorType::DIMENSION,
                                     "CPU-AMX SwiGLU_Kernel: a 和 b shape 必须一致");
    }
    if (a.device() != DeviceType::kCPU) {
        CtorchError::throwException(DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT,
                                     "CPU-AMX SwiGLU_Kernel: 仅在CPU支持");
    }

    static std::once_flag warned;
    std::call_once(warned, []() {
        CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kAMX, ErrorType::DEVICE_COMPAT,
                          "AMX SwiGLU_Kernel: 无专用实现，降级到 SIMD");
    });
    return SwiGLU_SIMD_kernel(a, b);
}
