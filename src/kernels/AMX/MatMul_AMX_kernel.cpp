/**
 * @file MatMul_AMX_kernel.cpp
 * @brief Apple AMX Acceleration MatMul算子 (使用 Accelerate BLAS)
 * @author GhostFace
 * @date 2026/04/04
 */

// 先包含 CTorch 头文件，避免与 Accelerate 的命名冲突
#include "./../../../include/Ctorch_Error.h"
#include "./../../../include/Tensor.h"
#include "./../kernels.h"

// 包含 Accelerate 时禁用可能冲突的宏
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

// 全局的matMul函数
Tensor MatMul_AMX_kernel(const Tensor &a, const Tensor &b) {

  // 校验设备：仅支持CPU张量
  if (a.device() != DeviceType::kCPU || b.device() != DeviceType::kCPU) {
    Ctorch_Error::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                      ErrorType::DEVICE_COMPAT,
                      "AMX MatMul_Kernel: 仅在CPU支持");
  }

  // 仅支持2D张量
  if (a.shape().size() != 2 || b.shape().size() != 2) {
    Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DIMENSION,
                                 "AMX MatMul仅支持2D张量");
  }
  if (a.shape()[1] != b.shape()[0]) {
    Ctorch_Error::throwException(ErrorPlatform::kCPU, ErrorType::DIMENSION,
                                 "矩阵维度不匹配");
  }

  size_t m = a.shape()[0];
  size_t k = a.shape()[1];
  size_t n = b.shape()[1];

  Tensor result(ShapeTag{}, {m, n}, a.dtype(), a.device());

  // 检查是否是连续内存的张量 (stride为默认值)
  const bool a_is_contiguous = (a.strides()[0] == k && a.strides()[1] == 1);
  const bool b_is_contiguous = (b.strides()[0] == n && b.strides()[1] == 1);
  
  if (a_is_contiguous && b_is_contiguous) {
    // 使用 cblas_sgemm 进行加速
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        (int)m,
        (int)n,
        (int)k,
        alpha,
        a.data<float>(),
        (int)k,
        b.data<float>(),
        (int)n,
        beta,
        result.data<float>(),
        (int)n
    );
  } else {
    // 如果不连续，回退到通用实现（非连续内存无法直接使用BLAS）
    const std::vector<size_t> &a_strides = a.strides();
    const std::vector<size_t> &b_strides = b.strides();
    const std::vector<size_t> &result_strides = result.strides();

    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < n; ++j) {
        float sum = 0.0f;
        for (size_t l = 0; l < k; ++l) {
          size_t a_idx = i * a_strides[0] + l * a_strides[1];
          size_t b_idx = l * b_strides[0] + j * b_strides[1];
          sum += a.data<float>()[a_idx] * b.data<float>()[b_idx];
        }
        size_t result_idx = i * result_strides[0] + j * result_strides[1];
        result.data<float>()[result_idx] = sum;
      }
    }
  }

  return result;
}
