/**
 * @file Cos_BASIC_kernel.h
 * @brief CPU-BASIC Cos算子
 * @author GhostFace
 * @date 2026/2/4
 */

#include "./../../../include/Ctorch_Error.h"
#include "./../../../include/Tensor.h"
#include "./../kernels.h"
#include <cmath>

Tensor Cos_BASIC_kernel(const Tensor &a) {
  // 校验设备：仅支持CPU张量
  if (a.device() != DeviceType::kCPU) {
    Ctorch_Error::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                      ErrorType::DEVICE_COMPAT,
                      "CPU-BASIC Cos_Kernel: 仅在CPU支持");
  }
  Tensor result(a);

  size_t count = a.numel();
  float *data = result.data<float>();
  for (size_t i = 0; i < count; ++i) {
    data[i] = std::cos(data[i]);
  }
  return result;
}
