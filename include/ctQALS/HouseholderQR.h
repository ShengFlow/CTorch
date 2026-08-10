//
// HouseholderQR.h
// Householder 反射 QR 分解（RSVD 用）
// 2026-08-10
//
// 设计要点：
//   * 输入 A：m×n row-major float，m ≥ n
//   * 反射器约定：Q = H_0 * H_1 * ... * H_{n-1}
//                 H_k = I - tau_k * v_k * v_k^T
//   * v_k 存到 V_ 的第 k 列（V_ 是 m × n row-major）：
//     V_[i, k] = v_k[i - k]  for i ≥ k
//     V_[i, k] = 0            for i < k
//     也就是 H_k 矩阵中"v 向量"按 m 维展开，前 k 个分量为 0
//   * 不调用 LAPACK
//
// 这是 RSVD 的 Layer 0-b 子模块，无前置依赖。
//

#ifndef CTORCH_HOUSEHOLDER_QR_H
#define CTORCH_HOUSEHOLDER_QR_H
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Apple Accelerate BLAS（cblas_sgemv + cblas_sger 用于 Householder apply）
// 其他平台用朴素标量循环
#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

namespace ctQALS {
namespace linalg {

// ============================================================
// 经典 Householder QR 分解
// ============================================================
class HouseholderQR {
public:
    HouseholderQR(const float* A, std::size_t m_rows, std::size_t n_cols)
        : m_(m_rows), n_(n_cols) {
        if (m_rows == 0 || n_cols == 0) {
            throw std::invalid_argument("HouseholderQR: dims must be > 0");
        }
        if (m_rows < n_cols) {
            throw std::invalid_argument("HouseholderQR: requires m >= n");
        }
        // V_ 存反射器矩阵（m × n row-major，V_[i, k] = v_k[i-k] for i >= k）
        V_.assign(m_rows * n_cols, 0.0f);
        // work_ 存 R 的非对角上三角（m × n row-major）
        work_.assign(A, A + m_rows * n_cols);
        tau_.assign(n_cols, 0.0f);
        R_diag_.assign(n_cols, 0.0f);
        compute_inplace();
    }

    std::size_t m() const noexcept { return m_; }
    std::size_t n() const noexcept { return n_; }

    // 提取 R（n×n 上三角，row-major）
    void get_R(float* R) const {
        for (std::size_t i = 0; i < n_; ++i) {
            for (std::size_t j = 0; j < n_; ++j) {
                if (i == j) {
                    R[i * n_ + j] = R_diag_[i];
                } else if (j > i) {
                    R[i * n_ + j] = work_[i * n_ + j];
                } else {
                    R[i * n_ + j] = 0.0f;
                }
            }
        }
    }

    // 瘦 Q（m×n，列正交，row-major）: Q = H_0 * H_1 * ... * H_{n-1}
    // 实现：Q := H_k * Q（左乘），k = n-1, n-2, ..., 0
    // 这样 Q 累积为 H_0 H_1 ... H_{n-1} 的前 n 列
    void get_Q_thin(float* Q) const {
        for (std::size_t i = 0; i < m_; ++i) {
            for (std::size_t j = 0; j < n_; ++j) {
                Q[i * n_ + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        for (std::size_t kk = n_; kk > 0; --kk) {
            const std::size_t k = kk - 1;
            apply_Hk_to_matrix_left(Q, m_, n_, /*ld=*/n_, k);
        }
    }

    // 全 Q（m×m，正交矩阵）
    void get_Q_full(float* Q) const {
        for (std::size_t i = 0; i < m_; ++i) {
            for (size_t j = 0; j < m_; ++j) {
                Q[i * m_ + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        for (std::size_t kk = n_; kk > 0; --kk) {
            const std::size_t k = kk - 1;
            apply_Hk_to_matrix_left(Q, m_, m_, /*ld=*/m_, k);
        }
    }

    // 应用 Q^T x
    void apply_Qt(float* x) const {
        for (std::size_t k = 0; k < n_; ++k) {
            apply_Hk_to_vector(x, /*stride=*/1, k);
        }
    }

    // 应用 Q x。Q x = H_0 * H_1 * ... * H_{n-1} * x
    // 顺序：k = 0, 1, ..., n-1（左乘）
    void apply_Q(float* x) const {
        for (std::size_t k = 0; k < n_; ++k) {
            apply_Hk_to_vector(x, /*stride=*/1, k);
        }
    }

    // 应用 Q^T 给一个 m×p 矩阵 X in-place
    // Q^T = H_{n-1} * ... * H_0，X := Q^T * X = H_{n-1} * ... * H_0 * X（左乘）
    void apply_Qt_matrix_inplace(float* X, std::size_t p_cols) const {
        for (std::size_t kk = n_; kk > 0; --kk) {
            const std::size_t k = kk - 1;
            apply_Hk_to_matrix_left(X, m_, p_cols, /*ld=*/p_cols, k);
        }
    }

    // 应用 Q 给一个 m×p 矩阵 X in-place：X := Q * X
    // Q = H_0 * ... * H_{n-1}，X := H_0 * H_1 * ... * H_{n-1} * X（左乘）
    void apply_Q_matrix_inplace(float* X, std::size_t p_cols) const {
        for (std::size_t k = 0; k < n_; ++k) {
            apply_Hk_to_matrix_left(X, m_, p_cols, /*ld=*/p_cols, k);
        }
    }

    // 重建 A = Q * R（验证用），输出 m_ × n_ 矩阵
    void reconstruct(float* A_rec_m_by_n) const {
        std::vector<float> Q_slice(m_ * n_, 0.0f);
        for (std::size_t i = 0; i < m_; ++i) {
            for (std::size_t j = 0; j < n_; ++j) {
                Q_slice[i * n_ + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        for (std::size_t kk = n_; kk > 0; --kk) {
            const std::size_t k = kk - 1;
            apply_Hk_to_matrix_left(Q_slice.data(), m_, n_, /*ld=*/n_, k);
        }

        std::vector<float> R_n(n_ * n_);
        get_R(R_n.data());

        for (std::size_t i = 0; i < m_; ++i) {
            for (std::size_t j = 0; j < n_; ++j) {
                float sum = 0.0f;
                for (std::size_t r = 0; r < n_; ++r) {
                    sum += Q_slice[i * n_ + r] * R_n[r * n_ + j];
                }
                A_rec_m_by_n[i * n_ + j] = sum;
            }
        }
    }

    float reconstruction_error(const float* A_orig) const {
        std::vector<float> Arec(m_ * n_);
        reconstruct(Arec.data());

        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < m_ * n_; ++i) {
            const double d = static_cast<double>(A_orig[i] - Arec[i]);
            num += d * d;
            den += static_cast<double>(A_orig[i]) * A_orig[i];
        }
        return static_cast<float>(std::sqrt(num / std::max(den, 1e-30)));
    }

    const std::vector<float>& tau() const noexcept { return tau_; }

private:
    std::size_t           m_;
    std::size_t           n_;
    std::vector<float>    V_;        // m × n row-major：V_[i, k] = v_k[i-k]（H 矩阵展开）
    std::vector<float>    work_;     // m × n：R 的非对角上三角
    std::vector<float>    tau_;      // n
    std::vector<float>    R_diag_;   // n：R 对角线元素

    void compute_inplace() {
        for (std::size_t k = 0; k < n_; ++k) {
            // work_ 中 A 的第 k 列从 work_[k * n_ + k] 开始
            float* col_k = work_.data() + k * n_ + k;
            const std::size_t len = m_ - k;

            float xnorm_sq = 0.0f;
            for (std::size_t i = 0; i < len; ++i) {
                xnorm_sq += col_k[i * n_] * col_k[i * n_];
            }
            const float xnorm = std::sqrt(xnorm_sq);

            if (xnorm == 0.0f) {
                tau_[k] = 0.0f;
                R_diag_[k] = 0.0f;
                continue;
            }

            const float x0 = col_k[0];
            const float sign = (x0 >= 0.0f) ? 1.0f : -1.0f;
            const float v0 = x0 + sign * xnorm;

            float vnorm2 = v0 * v0;
            for (std::size_t i = 1; i < len; ++i) {
                vnorm2 += col_k[i * n_] * col_k[i * n_];
            }
            float tau_k;
            if (vnorm2 < 1e-30f) {
                tau_k = 0.0f;
            } else {
                tau_k = 2.0f / vnorm2;
            }
            tau_[k] = tau_k;
            R_diag_[k] = -sign * xnorm;

            // 把 v 存到 V_ 第 k 列（m 维展开，H_k 矩阵形式）：
            // V_[i, k] = v_k[i - k] for i >= k
            // 也就是 V_[k, k] = v0, V_[k+1, k] = x[1], V_[k+2, k] = x[2], ...
            V_[k * n_ + k] = v0;  // V_[k, k]
            for (std::size_t i = 1; i < len; ++i) {
                V_[(k + i) * n_ + k] = col_k[i * n_];  // V_[k+i, k]
            }

            // 应用 H_k 到 A[k:m, k+1:n]
            for (std::size_t j = k + 1; j < n_; ++j) {
                float* col_j = work_.data() + k * n_ + j;  // 行 k，列 j
                // dot = v_k^T * A[k:m, j]
                float dot = v0 * col_j[0];
                for (std::size_t i = 1; i < len; ++i) {
                    dot += col_k[i * n_] * col_j[i * n_];
                }
                if (tau_k != 0.0f && dot != 0.0f) {
                    const float scale = tau_k * dot;
                    col_j[0] -= scale * v0;
                    for (std::size_t i = 1; i < len; ++i) {
                        col_j[i * n_] -= scale * col_k[i * n_];
                    }
                }
            }
        }
    }

    // 内部：把 H_k 作用到一个 m 维向量（任意 stride）
    // v 来自 V_ 的第 k 列（V_[0, k]..V_[m-1, k]，前 k 个分量为 0）
    void apply_Hk_to_vector(float* x, std::size_t stride, std::size_t k) const {
        const float* V_col = V_.data() + k;  // V_ 第 k 列起始
        const float tau = tau_[k];
        if (tau == 0.0f) return;

        float dot = 0.0f;
        for (std::size_t r = 0; r < m_; ++r) {
            dot += V_col[r * n_] * x[r * stride];
        }
        const float scale = tau * dot;
        for (std::size_t r = 0; r < m_; ++r) {
            x[r * stride] -= scale * V_col[r * n_];
        }
    }

    // 内部：把 H_k 左乘到 m×p 矩阵 X in-place：X := H_k * X
    // 等价于逐列应用 H_k：X[:, c] := H_k * X[:, c]
    // 2026-08-10：Apple Silicon 走 cblas_sgemv + cblas_sger（BLAS 加速）
    void apply_Hk_to_matrix_left(float* X, std::size_t /*rows*/,
                                  std::size_t p, std::size_t ld,
                                  std::size_t k) const {
        const float* V_col = V_.data() + k;  // V_ 第 k 列起始
        const float tau = tau_[k];
        if (tau == 0.0f) return;

#ifdef __APPLE__
        // BLAS 路径：
        //   w = v_k^T X[k:, :p]      ← cblas_sgemv
        //   X[k:, :p] -= tau * v_k * w^T  ← cblas_sger (rank-1 update)
        // v_k 在 V_ 中按 stride=n_ 访问，V_ 第 k 列从 V_[k*ld + k] 开始
        //   w[j] = sum_{i=k..m-1} v_k[i] * X[i, j]
        //   = (M_sub^T * v_sub)[j]   (M_sub = X[k:, :p], v_sub = v_k[k:])
        //   所以用 cblas_sgemv CblasTrans：A=M_sub, x=v_sub, y=w
        const std::size_t m_sub = m_ - k;
        std::vector<float> w(p, 0.0f);
        cblas_sgemv(CblasRowMajor, CblasTrans,
                    (int)m_sub, (int)p,
                    1.0f,
                    X + k * ld, (int)ld,                       // A = X[k:, :p] (m_sub × p)
                    V_col + k * n_, (int)n_,                   // x = v_k[k:] (m_sub 维，stride=n_)
                    0.0f, w.data(), 1);                        // y = w (p 维)
        // X[k:, :p] -= tau * v_k * w^T
        cblas_sger(CblasRowMajor,
                   (int)m_sub, (int)p,
                   -tau,                                       // alpha = -tau
                   V_col + k * n_, (int)n_,                   // x = v_k[k:]
                   w.data(), 1,                                // y = w
                   X + k * ld, (int)ld);                       // A = X[k:, :p]
#else
        for (std::size_t c = 0; c < p; ++c) {
            float dot = 0.0f;
            for (std::size_t r = k; r < m_; ++r) {
                dot += V_col[r * n_] * X[r * ld + c];
            }
            const float scale = tau * dot;
            for (std::size_t r = k; r < m_; ++r) {
                X[r * ld + c] -= scale * V_col[r * n_];
            }
        }
#endif
    }

    // 内部：把 H_k 右乘到 m×p 矩阵 X in-place：X := X * H_k
    // (X H_k)[i, j] = sum_l X[i, l] * H_k[l, j]
    // H_k 是 m × m，X 是 m × p（p <= m），所以 X H_k 也是 m × p
    // 2026-08-10：Apple Silicon 走 cblas_sgemv + cblas_sger（BLAS 加速）
    void apply_Hk_to_matrix_right(float* X, std::size_t /*rows*/,
                                   std::size_t p, std::size_t ld,
                                   std::size_t k) const {
        const float* V_col = V_.data() + k;  // V_ 第 k 列起始
        const float tau = tau_[k];
        if (tau == 0.0f) return;

#ifdef __APPLE__
        // BLAS 路径：
        //   w = X[:, k:p] * v_k[k:p]    ← cblas_sgemv (CblasTrans)
        //   X[:, k:p] -= tau * w * v_k^T ← cblas_sger
        const std::size_t p_sub = p - k;
        std::vector<float> w(m_, 0.0f);
        // w = X[:, k:p]^T * v_k[k:p_sub]  (m × p_sub 矩阵 × p_sub 向量 = m 向量)
        // CblasTrans: y = alpha * A^T * x + beta * y
        //   A = X[:, k:p] (m × p_sub),  A^T = p_sub × m
        //   x = v_k[0..p_sub-1] (stride=n_), y = w (m 维)
        cblas_sgemv(CblasRowMajor, CblasTrans,
                    (int)m_, (int)p_sub,
                    1.0f,
                    X + k, (int)ld,                            // A = X[0, k] 起点（m × p_sub）
                    V_col + k * n_, (int)n_,                   // x = v_k[0..p_sub-1] (stride=n_)
                    0.0f, w.data(), 1);                        // y = w
        // X[:, k:p] -= tau * w * v_k^T
        cblas_sger(CblasRowMajor,
                   (int)m_, (int)p_sub,
                   -tau,                                       // alpha = -tau
                   w.data(), 1,                                // x = w (m 维)
                   V_col + k * n_, (int)n_,                   // y = v_k (stride=n_)
                   X + k, (int)ld);                            // A = X[0, k] (m × p_sub)
#else
        // 计算 w[i] = sum_{l=k..p-1} X[i, l] * V_col[l * n_]
        // X 是 m × p 矩阵（p 通常 = n_），所以 l 限制到 p - 1
        std::vector<float> w(m_, 0.0f);
        for (std::size_t i = 0; i < m_; ++i) {
            float s = 0.0f;
            for (std::size_t l = k; l < p; ++l) {
                s += X[i * ld + l] * V_col[l * n_];
            }
            w[i] = s;
        }

        // 更新 X[i, j] for j >= k, j < p
        for (std::size_t i = 0; i < m_; ++i) {
            for (std::size_t j = k; j < p; ++j) {
                X[i * ld + j] -= tau * w[i] * V_col[j * n_];
            }
        }
#endif
    }
};

} // namespace linalg
} // namespace ctQALS
#endif // CTORCH_HOUSEHOLDER_QR_H
