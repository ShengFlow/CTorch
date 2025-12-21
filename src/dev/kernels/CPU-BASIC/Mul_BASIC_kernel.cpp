/**
 * @file Mul_BASIC_kernel.h
 * @brief CPU-BASIC 乘法算子
 * @author GhostFace
 * @date 2025/12/21
 */

// DONE: name fix & git push
#include "./../kernels.h"
#include "./../../Ctorch_Error.h"
#include "./../../Tensor.h"

Tensor Mul_BASIC_kernel(const Tensor& a, const Tensor& b) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU || b.device() != DeviceType::kCPU) {
        Ctorch_Error::log(ErrorLevel::ERROR,DeviceTypeToErrorPlatform(a.device()),ErrorType::DEVICE_COMPAT,"CPU-BASIC Mul_Kernel: 仅在CPU支持");
    }
    // 校验形状：必须一致
    if (a.sizes() != b.sizes()) {
        Ctorch_Error::log(ErrorLevel::ERROR,ErrorPlatform::kCPU,ErrorType::DIMENSION,"CPU-BASIC Mul_Kernel: Tensor形状不匹配");
    }
    // 校验数据类型
    if (a.dtype() != b.dtype()) {
        Ctorch_Error::log(ErrorLevel::ERROR,ErrorPlatform::kCPU,ErrorType::DIMENSION,"CPU-BASIC Mul_Kernel: Tensor数据类型不匹配");
    }

    int elem_count = a.numel();

    // 获取Tensor数据指针
    const float* a_data = a.data<float>();
    const float* b_data = b.data<float>();

    // 创建结果Tensor
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
    float* result_data = result.data<float>();

    //  朴素逐元素加法
    for (int i = 0; i < elem_count; ++i) {
        result_data[i] = a_data[i] * b_data[i];
    }
    return result;
}