/**
 * @file Mul_BASIC_kernel.h
 * @brief CPU-BASIC 乘法算子
 * @author GhostFace
 * @date 2025/12/21
 */

// DONE: name fix & git push
#include "./../kernels.h"
#include "./../../../include/CtorchError.h"
#include "./../../../include/Tensor.h"
#include "./../../../include/CoreDefs.h"

CT_HOT Tensor Mul_BASIC_kernel(const Tensor& a, const Tensor& b) {
    // 校验设备：仅支持CPU张量
    if (a.device() != DeviceType::kCPU || b.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR,DeviceTypeToErrorPlatform(a.device()),ErrorType::DEVICE_COMPAT,"CPU-BASIC Mul_Kernel: 仅在CPU支持");
    }
    // 校验数据类型
    if (a.dtype() != b.dtype()) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR,ErrorPlatform::kCPU,ErrorType::DATATYPE,"CPU-BASIC Mul_Kernel: Tensor数据类型不匹配");
    }

    // 检查是否需要广播
    if (a.sizes() != b.sizes()) {
        // 处理0D张量的情况
        if (a.dim() == 0) {
            // a是标量，广播到b的形状
            Tensor a_broadcasted = a.broadcast_to(b.sizes());
            return Mul_BASIC_kernel(a_broadcasted, b);
        } else if (b.dim() == 0) {
            // b是标量，广播到a的形状
            Tensor b_broadcasted = b.broadcast_to(a.sizes());
            return Mul_BASIC_kernel(a, b_broadcasted);
        } else {
            // 计算广播后形状
            std::vector<size_t> broadcast_shape;
            size_t max_dims = std::max(a.sizes().size(), b.sizes().size());
            broadcast_shape.reserve(max_dims);
            
            // 从后往前计算广播形状
            for (size_t i = 0; i < max_dims; ++i) {
                size_t a_dim = i < a.sizes().size() ? a.sizes()[a.sizes().size() - 1 - i] : 1;
                size_t b_dim = i < b.sizes().size() ? b.sizes()[b.sizes().size() - 1 - i] : 1;
                if (a_dim != 1 && b_dim != 1 && a_dim != b_dim) {
                    CtorchError::log(ErrorLevel::ERROR,ErrorPlatform::kCPU,ErrorType::DIMENSION,"CPU-BASIC Mul_Kernel: Tensor形状不兼容，无法广播");
                    return Tensor();
                }
                broadcast_shape.push_back(std::max(a_dim, b_dim));
            }
            
            // 反转形状以恢复正确顺序
            std::reverse(broadcast_shape.begin(), broadcast_shape.end());
            
            // 广播两个张量
            Tensor a_broadcasted = a.broadcast_to(broadcast_shape);
            Tensor b_broadcasted = b.broadcast_to(broadcast_shape);
            
            return Mul_BASIC_kernel(a_broadcasted, b_broadcasted);
        }
    }

    int elem_count = a.numel();

    // 获取Tensor数据指针
    const float* CT_RESTRICT a_data = a.data<float>();
    const float* CT_RESTRICT b_data = b.data<float>();

    // 创建结果Tensor
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
    float* CT_RESTRICT result_data = result.data<float>();

    // 朴素逐元素乘法
    for (int i = 0; i < elem_count; ++i) {
        result_data[i] = a_data[i] * b_data[i];
    }
    return result;
}