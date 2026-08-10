/**
 * @file ApplyHk_AMX_kernel.cpp
 * @brief Apple AMX Acceleration Householder 反射应用算子（cblas_sgemv + cblas_sger）
 * @details 原地更新矩阵 M 的 [k:, :p] 子块：
 *            M[k:, :p] -= tau * v * (v^T M[k:, :p])
 *          用于 HouseholderQR 的反射器链式 apply
 * @author GhostFace
 * @date 2026/08/10
 */

#include "./../../../include/CtorchError.h"
#include "./../../../include/Tensor.h"
#include "./../kernels.h"

// 包含 Accelerate 时禁用可能冲突的宏
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <vector>

/**
 * @brief Householder 反射应用算子（AMX 加速，使用 Accelerate cblas_sgemv + cblas_sger）
 *
 * @param M 目标矩阵（m × p 矩阵，会被原地修改）
 * @param v Householder 反射向量（m 维，v[0..k_offset-1] 应为 0）
 * @param tau 反射系数
 * @param k_offset 起始行（从哪一行开始 apply）
 * @param p_cols 要更新的列数（默认 = M.shape()[1]，M 必须是 row-major 连续）
 *
 * @details 数学定义：
 *          H = I - tau * v * v^T
 *          M[k_offset:, :p_cols] := H @ M[k_offset:, :p_cols]
 *
 *          展开成两步 BLAS：
 *            w = v[k_offset:]^T @ M[k_offset:, :p_cols]      (1 × p_cols)
 *            M[k_offset:, :p_cols] -= tau * v[k_offset:] * w^T
 *
 * @warning M 必须是 row-major 连续 2D 矩阵（stride[0] = p_cols, stride[1] = 1）
 *          v 必须是 1D 或 row-major 连续（stride = 1）
 *          不满足会回退到朴素循环
 */
void ApplyHk_AMX_kernel(Tensor& M, const Tensor& v, float tau,
                        std::size_t k_offset, std::size_t p_cols) {
    // 校验设备
    if (M.device() != DeviceType::kCPU && M.device() != DeviceType::kMPS) {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(M.device()),
                         ErrorType::DEVICE_COMPAT,
                         "AMX ApplyHk_Kernel: 仅在 CPU/MPS 支持");
    }
    if (M.dtype() != DType::kFloat || v.dtype() != DType::kFloat) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                         "AMX ApplyHk_Kernel: 仅支持 float32");
    }
    if (M.shape().size() != 2) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                     "AMX ApplyHk_Kernel: M 必须是 2D 矩阵");
    }
    if (v.shape().size() != 1) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                     "AMX ApplyHk_Kernel: v 必须是 1D 向量");
    }

    const std::size_t m = M.shape()[0];
    const std::size_t p = M.shape()[1];
    if (p_cols == 0) p_cols = p;
    if (k_offset >= m || p_cols == 0) return;
    if (v.numel() < m) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                     "AMX ApplyHk_Kernel: v 长度不足（必须 >= M 行数）");
    }
    const std::size_t m_sub = m - k_offset;
    if (v.numel() < m) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                                     "AMX ApplyHk_Kernel: v 长度不足");
    }

    float* M_data = M.data_write<float>();
    const float* v_data = v.data<float>();

#ifdef __APPLE__
    // MPS 路径：host 写入 shared buffer 后需 mark
    if (M.device() == DeviceType::kMPS) {
        MPS_markBufferModified(M_data, m * p * sizeof(float));
    }
#endif

    // 检查 M 是否 row-major 连续（stride[0] == p, stride[1] == 1）
    const bool m_contig = (M.strides()[0] == p && M.strides()[1] == 1);
    // v 是否 1D 连续
    const bool v_contig = (v.strides().empty() || v.strides()[0] == 1);

    if (m_contig && v_contig) {
        // === BLAS 快速路径 ===
        // 分配 w (p_cols 维临时向量)
        std::vector<float> w(p_cols, 0.0f);

        // 1) w = v[k_offset:]^T @ M[k_offset:, :p_cols]
        //    cblas_sgemv: y = alpha * A * x + beta * y
        //    A = M[k_offset:, :p_cols] 形状 m_sub × p_cols
        //    x = v[k_offset:] 形状 m_sub
        //    y = w 形状 p_cols
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    (int)m_sub, (int)p_cols,
                    1.0f,
                    M_data + k_offset * p, (int)p,        // A 起点
                    v_data + k_offset, 1,                  // x 起点
                    0.0f,                                   // beta = 0
                    w.data(), 1);

        // 2) M[k_offset:, :p_cols] -= tau * v[k_offset:] * w^T
        //    cblas_sger: A := alpha * x * y^T + A
        cblas_sger(CblasRowMajor,
                   (int)m_sub, (int)p_cols,
                   -tau,                                   // alpha = -tau
                   v_data + k_offset, 1,                  // x = v[k_offset:]
                   w.data(), 1,                            // y = w
                   M_data + k_offset * p, (int)p);         // A = M[k_offset:, :]
    } else {
        // === 朴素回退路径 ===
        // w[j] = sum_{i=k..m-1} v[i] * M[i, j]
        std::vector<float> w(p_cols, 0.0f);
        for (std::size_t j = 0; j < p_cols; ++j) {
            float sum = 0.0f;
            for (std::size_t i = k_offset; i < m; ++i) {
                sum += v_data[i] * M_data[i * p + j];
            }
            w[j] = sum;
        }
        // M[i, j] -= tau * v[i] * w[j]
        for (std::size_t i = k_offset; i < m; ++i) {
            const float v_i = v_data[i];
            for (std::size_t j = 0; j < p_cols; ++j) {
                M_data[i * p + j] -= tau * v_i * w[j];
            }
        }
    }
}
