/**
 * @file Sub_BASIC_kernel.h
 * @brief CPU-BASIC 减法算子
 * @author GhostFace
 * @date 2025/12/20
 */

#include "./../../../include/Ctorch_Error.h"
#include "./../../../include/Tensor.h"
#include "./../kernels.h"

Tensor Sub_BASIC_kernel(const Tensor &a, const Tensor &b) {
  // 校验设备：仅支持CPU张量
  if (a.device() != DeviceType::kCPU || b.device() != DeviceType::kCPU) {
    Ctorch_Error::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                      ErrorType::DEVICE_COMPAT,
                      "CPU-BASIC Sub_Kernel: 仅在CPU支持");
  }
  // 校验数据类型
  if (a.dtype() != b.dtype()) {
    Ctorch_Error::log(ErrorLevel::ERROR, ErrorPlatform::kCPU,
                      ErrorType::DATATYPE,
                      "CPU-BASIC Sub_Kernel: Tensor数据类型不匹配");
  }

  // 检查是否需要广播
  if (a.sizes() != b.sizes()) {
    // 处理0D张量的情况
    if (a.dim() == 0) {
      // a是标量，广播到b的形状
      Tensor a_broadcasted = a.broadcast_to(b.sizes());
      return Sub_BASIC_kernel(a_broadcasted, b);
    } else if (b.dim() == 0) {
      // b是标量，广播到a的形状
      Tensor b_broadcasted = b.broadcast_to(a.sizes());
      return Sub_BASIC_kernel(a, b_broadcasted);
    } else {
      // 两个都是非0D张量，尝试广播到相同形状
      Tensor a_broadcasted = a.broadcast_to(b.sizes());
      Tensor b_broadcasted = b.broadcast_to(a.sizes());

      // 检查广播是否成功
      if (a_broadcasted.sizes() == b_broadcasted.sizes()) {
        return Sub_BASIC_kernel(a_broadcasted, b_broadcasted);
      } else {
        Ctorch_Error::log(ErrorLevel::ERROR, ErrorPlatform::kCPU,
                          ErrorType::DIMENSION,
                          "CPU-BASIC Sub_Kernel: Tensor形状不兼容，无法广播");
      }
    }
  }

  int elem_count = a.numel();

  // 获取Tensor数据指针
  const float *a_data = a.data<float>();
  const float *b_data = b.data<float>();

  // 创建结果Tensor
  Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
  float *result_data = result.data<float>();

  // 朴素逐元素减法
  for (int i = 0; i < elem_count; ++i) {
    result_data[i] = a_data[i] - b_data[i];
  }
  return result;
}