/**
 * @file MAE_BASIC_kernel.cpp
 * @brief CPU-BASIC MAE（平均绝对误差）算子
 * @author GhostFace
 * @date 2026/02/09
 */

#include "./../../../include/Ctorch_Error.h"
#include "./../../../include/Tensor.h"
#include "./../kernels.h"

Tensor MAE_BASIC_kernel(const Tensor &a, const Tensor &b) {
  // 校验设备：仅支持CPU张量
  if (a.device() != DeviceType::kCPU || b.device() != DeviceType::kCPU) {
    Ctorch_Error::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                      ErrorType::DEVICE_COMPAT,
                      "CPU-BASIC MAE_Kernel: 仅在CPU支持");
  }

  // 校验形状和数据类型
  if (a.sizes() != b.sizes()) {
    Ctorch_Error::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL,
                      ErrorType::DIMENSION,
                      "CPU-BASIC MAE_Kernel: 张量形状不一致");
  }

  if (a.dtype() != b.dtype()) {
    Ctorch_Error::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL,
                      ErrorType::DATATYPE,
                      "CPU-BASIC MAE_Kernel: 张量数据类型不一致");
  }

  // 实现MAE损失函数
  size_t count = a.numel();
  const float *data_a = a.data<float>();
  const float *data_b = b.data<float>();

  float sum_absolute_error = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    float diff = data_a[i] - data_b[i];
    sum_absolute_error += std::abs(diff);
  }

  float mae = sum_absolute_error / count;

  // 创建标量结果张量
  Tensor result(mae);
  return result;
}