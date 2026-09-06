/**
 * @file SiLU_BASIC_kernel.cpp
 * @brief CPU-BASIC SiLU 算子 (PEL25 Stage 3.1)
 * @details silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *          标量版本, 用于通用 CPU + 测试对照
 * @author Mavis (PEL25 §6 SwiGLU 算子开发协议)
 * @date 2026-09-06
 */

#include <cmath>
#include "./../kernels.h"
#include "./../../../include/CtorchError.h"
#include "./../../../include/Tensor.h"
#include "./../../../include/CoreDefs.h"

namespace {
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
inline float silu_scalar(float x) {
    return x / (1.0f + std::exp(-x));
}
}

CT_HOT Tensor SiLU_BASIC_kernel(const Tensor &a) {
    // 校验设备: 仅支持CPU张量
    if (a.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                          ErrorType::DEVICE_COMPAT, "CPU-BASIC SiLU_Kernel: 仅在CPU支持");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device(), false);

    size_t count = a.numel();
    const float* CT_RESTRICT a_data = a.data_read<float>();
    float* CT_RESTRICT result_data = result.data_write<float>();
    for (size_t i = 0; i < count; ++i) {
        result_data[i] = silu_scalar(a_data[i]);
    }

    return result;
}

// SiLU inplace 版本在 src/kernels/CPU-BASIC/unary_inplace_BASIC_kernels.cpp 单独定义
// 避免重复定义 (PEL25 Stage 3.4 dispatch 注册引用那个版本)
