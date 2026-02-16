/**
 * @file Neg_BASIC_kernel.h
 * @brief CPU-BASIC 负号算子
 * @author GhostFace
 * @date 2025/12/20
 */
#include "./../kernels.h"
#include "./../../../include/Ctorch_Error.h"
#include "./../../../include/Tensor.h"

Tensor Neg_BASIC_kernel(const Tensor& a) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU) {
        Ctorch_Error::log(ErrorLevel::ERROR,DeviceTypeToErrorPlatform(a.device()),ErrorType::DEVICE_COMPAT,"CPU-BASIC Neg_Kernel: 仅在CPU支持");
    }

    int elem_count = a.numel();

    // 获取Tensor数据指针
    const float* a_data = a.data<float>();

    // 创建结果Tensor
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
    float* result_data = result.data<float>();

    //  朴素逐元素加法
    for (int i = 0; i < elem_count; ++i) {
        result_data[i] = -a_data[i];
    }
    return result;
}