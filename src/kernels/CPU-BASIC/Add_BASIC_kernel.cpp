/**
 * @file Add_BASIC_kernel.h
 * @brief CPU-BASIC 加法算子
 * @author GhostFace
 * @date 2025/12/20
 */

#include "./../kernels.h"
#include "./../../../include/Ctorch_Error.h"
#include "./../../../include/Tensor.h"
Tensor Add_BASIC_kernel(const Tensor& a, const Tensor& b) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU || b.device() != DeviceType::kCPU) {
        Ctorch_Error::log(ErrorLevel::ERROR,DeviceTypeToErrorPlatform(a.device()),ErrorType::DEVICE_COMPAT,"CPU-BASIC Add_Kernel: 仅在CPU支持");
    }
    // 校验数据类型
    if (a.dtype() != b.dtype()) {
        Ctorch_Error::log(ErrorLevel::ERROR,ErrorPlatform::kCPU,ErrorType::DIMENSION,"CPU-BASIC Add_Kernel: Tensor数据类型不匹配");
    }

    // 检查形状兼容性
    bool shapes_match = (a.sizes() == b.sizes());
    bool b_is_scalar = (b.numel() == 1);
    bool can_broadcast = false;
    
    // 检查是否可以广播
    if (b.sizes().size() == 1 && a.sizes().size() == 2 && b.sizes()[0] == a.sizes()[1]) {
        can_broadcast = true;
    }
    
    // 如果形状不匹配且不能广播，抛出异常
    if (!shapes_match && !b_is_scalar && !can_broadcast) {
        Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DIMENSION, "CPU-BASIC Add_Kernel: Tensor形状不匹配，且无法广播");
    }
    
    // 支持广播：使用a的形状作为结果形状
    int elem_count = a.numel();

    // 获取Tensor数据指针
    const float* a_data = a.data<float>();
    const float* b_data = b.data<float>();

    // 创建结果Tensor
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
    float* result_data = result.data<float>();

    // 支持广播的逐元素加法
    if (b_is_scalar) {
        // b是标量，广播到a的所有元素
        float b_val = b_data[0];
        for (int i = 0; i < elem_count; ++i) {
            result_data[i] = a_data[i] + b_val;
        }
    } else if (can_broadcast) {
        // b是一维张量，a是二维张量，且b的长度等于a的列数
        int batch_size = a.sizes()[0];
        int hidden_size = a.sizes()[1];
        for (int i = 0; i < batch_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                int idx = i * hidden_size + j;
                result_data[idx] = a_data[idx] + b_data[j];
            }
        }
    } else {
        // 形状相同的情况
        for (int i = 0; i < elem_count; ++i) {
            result_data[i] = a_data[i] + b_data[i];
        }
    }
    return result;
}