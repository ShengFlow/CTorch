/**
 * @file Softmax_BASIC_kernel.cpp
 * @brief CPU-BASIC Softmax算子
 * @author GhostFace
 * @date 2026/02/09
 */

#include "./../../../include/Ctorch_Error.h"
#include "./../../../include/Tensor.h"
#include "./../kernels.h"
#include <cmath>

// 说明：此 BASIC 版本 softmax 的实现与 Tensor::softmax
// 的数值稳定手写实现保持一致：
//  - 对每一行/向量先减去该行最大值，再做 exp，再按行归一化；
//  - 当前仅支持 float，且只处理 1D / 2D（2D 默认按最后一维做
//  softmax，典型形状为 [batch, num_classes]）。
Tensor Softmax_BASIC_kernel(const Tensor &a) {
  // 校验设备：仅支持CPU张量
  if (a.device() != DeviceType::kCPU) {
    Ctorch_Error::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                      ErrorType::DEVICE_COMPAT,
                      "CPU-BASIC Softmax_Kernel: 仅在CPU支持");
  }

  if (a.dtype() != DType::kFloat) {
    Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DATATYPE,
                                 "CPU-BASIC Softmax_Kernel: 目前仅支持 float");
  }

  const auto &shape = a.sizes();
  size_t dim = shape.size();
  if (dim == 0) {
    Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DIMENSION,
                                 "CPU-BASIC Softmax_Kernel: 不支持标量");
  }

  // 构造输出张量
  Tensor result(ShapeTag{}, shape, a.dtype(), a.device());
  const float *in = a.data<float>();
  float *out = result.data<float>();

  if (dim == 1) {
    // 1D: 对整个向量做 softmax
    size_t n = shape[0];
    if (n == 0)
      return result;

    // 先找最大值
    float max_val = in[0];
    for (size_t i = 1; i < n; ++i) {
      if (in[i] > max_val)
        max_val = in[i];
    }

    // 计算 exp(x - max) 并累加
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
      float e = std::exp(in[i] - max_val);
      out[i] = e;
      sum += e;
    }

    // 归一化
    if (sum > 0.0f) {
      for (size_t i = 0; i < n; ++i)
        out[i] /= sum;
    }
  } else if (dim == 2) {
    // 2D: 默认按照最后一维（列）做 softmax，典型用于 [batch, num_classes]
    size_t rows = shape[0];
    size_t cols = shape[1];
    if (rows == 0 || cols == 0)
      return result;

    for (size_t i = 0; i < rows; ++i) {
      // 每一行先找最大值
      float max_val = in[i * cols];
      for (size_t j = 1; j < cols; ++j) {
        float v = in[i * cols + j];
        if (v > max_val)
          max_val = v;
      }

      // 计算 exp(x - max) 并累加
      float sum = 0.0f;
      for (size_t j = 0; j < cols; ++j) {
        float e = std::exp(in[i * cols + j] - max_val);
        out[i * cols + j] = e;
        sum += e;
      }

      // 归一化
      if (sum > 0.0f) {
        for (size_t j = 0; j < cols; ++j) {
          out[i * cols + j] /= sum;
        }
      }
    }
  } else {
    Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DIMENSION,
                                 "CPU-BASIC Softmax_Kernel: 暂仅支持 1D / "
                                 "2D（如 [N] 或 [batch, classes]）");
  }

  return result;
}
