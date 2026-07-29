/**
 * @file MatMul_AMX_kernel.cpp
 * @brief Apple AMX Acceleration MatMul算子 (使用 Accelerate BLAS)
 * @author GhostFace
 * @date 2026/04/04
 */

// 先包含 CTorch 头文件，避免与 Accelerate 的命名冲突
#include "./../../../include/CtorchError.h"
#include "./../../../include/Tensor.h"
#include "./../kernels.h"

// 包含 Accelerate 时禁用可能冲突的宏
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

// 全局的matMul函数
Tensor MatMul_AMX_kernel(const Tensor &a, const Tensor &b) {
  // 校验设备：支持CPU和MPS张量（MPS内存是共享的）
  if (a.device() != DeviceType::kCPU && a.device() != DeviceType::kMPS) {
    CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                      ErrorType::DEVICE_COMPAT,
                      "AMX MatMul_Kernel: 仅在CPU/MPS支持");
  }
  if (a.device() != b.device()) {
    CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                      ErrorType::DEVICE_COMPAT,
                      "AMX MatMul_Kernel: 设备类型不匹配");
  }

  // 仅支持2D张量
  if (a.shape().size() != 2 || b.shape().size() != 2) {
    CtorchError::throwException(ErrorPlatform::kCPU, ErrorType::DIMENSION,
                                 "AMX MatMul仅支持2D张量");
  }
  if (a.shape()[1] != b.shape()[0]) {
    CtorchError::throwException(ErrorPlatform::kCPU, ErrorType::DIMENSION,
                                 "矩阵维度不匹配");
  }

  size_t m = a.shape()[0];
  size_t k = a.shape()[1];
  size_t n = b.shape()[1];

  Tensor result(ShapeTag{}, {m, n}, a.dtype(), a.device());

  // 检查是否是连续内存的张量 (stride为默认值)
  CTORCH_TRACE(ErrorPlatform::kAMX, "MatMul_AMX_kernel - a shape: [" + std::to_string(m) + ", " + std::to_string(k) + "]");
  CTORCH_TRACE(ErrorPlatform::kAMX, "MatMul_AMX_kernel - a strides: [" + std::to_string(a.strides()[0]) + ", " + std::to_string(a.strides()[1]) + "]");
  CTORCH_TRACE(ErrorPlatform::kAMX, "MatMul_AMX_kernel - b shape: [" + std::to_string(k) + ", " + std::to_string(n) + "]");
  CTORCH_TRACE(ErrorPlatform::kAMX, "MatMul_AMX_kernel - b strides: [" + std::to_string(b.strides()[0]) + ", " + std::to_string(b.strides()[1]) + "]");

  const bool a_is_contiguous = (a.strides()[0] == k && a.strides()[1] == 1);
  const bool b_is_contiguous = (b.strides()[0] == n && b.strides()[1] == 1);

  CTORCH_TRACE(ErrorPlatform::kAMX, "MatMul_AMX_kernel - a_is_contiguous: " + std::to_string(a_is_contiguous));
  CTORCH_TRACE(ErrorPlatform::kAMX, "MatMul_AMX_kernel - b_is_contiguous: " + std::to_string(b_is_contiguous));

  Tensor a_contig;
  Tensor b_contig;
  const Tensor* a_ptr = &a;
  const Tensor* b_ptr = &b;

  if (!a_is_contiguous) {
    CTORCH_TRACE(ErrorPlatform::kAMX, "MatMul_AMX_kernel - Creating contiguous copy for a");
    a_contig = Tensor(ShapeTag{}, {m, k}, a.dtype(), a.device());
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < k; ++j) {
        size_t idx = i * a.strides()[0] + j * a.strides()[1];
        a_contig.data<float>()[i * k + j] = a.data<float>()[idx];
      }
    }
    a_ptr = &a_contig;
  }

  if (!b_is_contiguous) {
    CTORCH_TRACE(ErrorPlatform::kAMX, "MatMul_AMX_kernel - Creating contiguous copy for b");
    b_contig = Tensor(ShapeTag{}, {k, n}, b.dtype(), b.device());
    for (size_t i = 0; i < k; ++i) {
      for (size_t j = 0; j < n; ++j) {
        size_t idx = i * b.strides()[0] + j * b.strides()[1];
        b_contig.data<float>()[i * n + j] = b.data<float>()[idx];
      }
    }
    b_ptr = &b_contig;
  }

  CTORCH_TRACE(ErrorPlatform::kAMX, "MatMul_AMX_kernel - Using cblas_sgemm for acceleration");
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
      a_ptr->data<float>(),
      (int)k,
      b_ptr->data<float>(),
      (int)n,
      beta,
      result.data<float>(),
      (int)n
  );

  return result;
}
