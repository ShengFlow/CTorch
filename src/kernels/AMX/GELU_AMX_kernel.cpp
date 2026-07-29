/**
 * @file GELU_AMX_kernel.cpp
 * @brief AMX GELU算子占位
 * @author GhostFace
 * @date 2026/07/28
 * @details 当前 AMX 仅对 MatMul 有硬件加速，unary 激活函数无 AMX 实现。
 *          Scheduler 中对 GELU 的 AMX 槽注册 nullptr，自动降级到 SIMD/BASIC。
 *          本文件仅保留符号，防止未来直接引用时缺失定义。
 */

#include "./../../../include/CtorchError.h"
#include "./../../../include/Tensor.h"
#include "./../kernels.h"

Tensor GELU_AMX_kernel(const Tensor& a) {
    CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kAMX, ErrorType::DEVICE_COMPAT,
                      "AMX GELU_Kernel: 无专用实现，降级到 SIMD");
    return GELU_SIMD_kernel(a);
}
