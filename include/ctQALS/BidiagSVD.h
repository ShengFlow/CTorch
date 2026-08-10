//
// BidiagSVD.h
// Householder bidiagonal 化（Golub-Kahan bidiagonalization）
// 2026-08-10
//
// 设计要点：
//   * 把 m × n (m ≤ n) 矩阵 A 分解为 A = U * B * V^T
//     - U: m × m 正交
//     - B: m × n 上 bidiagonal（对角 B[i,i] + 上次对角 B[i,i+1]）
//     - V: n × n 正交
//   * 替换 one-sided Jacobi（O(m²n × sweeps)）
//   * bidiagonal 化是 O(m²n) 一次性成本，后续 SVD 迭代 O(m² × iters)
//   * 5-10× 加速
//   * 不调用 LAPACK
//
// 这是 RSVD 的 Layer 0-d 子模块，无前置依赖。
//

#ifndef CTORCH_BIDIAG_SVD_H
#define CTORCH_BIDIAG_SVD_H
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

namespace ctQALS {
namespace linalg {

// ============================================================
// Householder bidiagonal 化
// 输入 A (m × n, m <= n)
// 输出 B (m × n bidiagonal) + U (m × m 正交) + V (n × n 正交)
//
// 满足 A = U * B * V^T
// B 的非零元素：
//   - 对角 B[i, i]       for i = 0..m-1
//   - 上次对角 B[i, i+1] for i = 0..m-2
// ============================================================
inline void householder_bidiagonalize(const float* A, std::size_t m, std::size_t n,
                                      float* B, float* U, float* V) {
    if (m == 0 || n == 0) {
        throw std::invalid_argument("bidiagonalize: dims must be > 0");
    }
    if (m > n) {
        throw std::invalid_argument("bidiagonalize: requires m <= n");
    }

    // B = A
    std::copy(A, A + m * n, B);
    // U = I_m
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            U[i * m + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    // V = I_n
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            V[i * n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // 交替用 H_a（左乘）和 H_b（右乘）消元
    //   关键：bidiag 化只需要 min(m, n) 步，对 m < n 矩阵，k 超过 m 之后
    //   B[k, :] 是 OOB（数组只有 m 行），必须 break
    //   H_a 应用 m 次（k = 0..m-1）
    //   H_b 应用 m-1 次（k = 0..m-2）
    const std::size_t K_end = std::min(m, n);
    for (std::size_t k = 0; k < K_end; ++k) {
#ifdef CT_DEBUG
        std::fprintf(stderr, "[DBG bidiag] k=%zu m=%zu n=%zu\n", k, m, n);
#endif
        // ---- 1) 左乘 H_a：把 B[k:, k] 化成 [alpha, 0, ..., 0]^T ----
        if (k < m) {
            const std::size_t m_sub = m - k;
            if (m_sub > 1) {
                // 算 v_a：长度 m_sub
                std::vector<float> v_a(m_sub);
                float x0 = B[k * n + k];
                float norm_x_sq = x0 * x0;
                for (std::size_t i = 1; i < m_sub; ++i) {
                    norm_x_sq += B[(k + i) * n + k] * B[(k + i) * n + k];
                }
                float norm_x = std::sqrt(norm_x_sq);
                if (norm_x >= 1e-30f) {
                    const float sign = (x0 >= 0.0f) ? 1.0f : -1.0f;
                    v_a[0] = x0 + sign * norm_x;
                    for (std::size_t i = 1; i < m_sub; ++i) {
                        v_a[i] = B[(k + i) * n + k];
                    }
                    float v_norm_sq = 0.0f;
                    for (std::size_t i = 0; i < m_sub; ++i) v_norm_sq += v_a[i] * v_a[i];
                    const float tau = (v_norm_sq < 1e-30f) ? 0.0f : (2.0f / v_norm_sq);

                    // B[k:, k:n] -= tau * v_a * v_a^T * B[k:, k:n]
                    //   w = v_a^T @ B[k:, k:n]  (1 × (n-k))
                    //   B[k:, k:n] -= tau * v_a * w
#ifdef __APPLE__
                    std::vector<float> w(n - k);
                    cblas_sgemv(CblasRowMajor, CblasTrans,
                                (int)m_sub, (int)(n - k),
                                1.0f,
                                B + k * n + k, (int)n,
                                v_a.data(), 1,
                                0.0f,
                                w.data(), 1);
                    cblas_sger(CblasRowMajor,
                               (int)m_sub, (int)(n - k),
                               -tau,
                               v_a.data(), 1,
                               w.data(), 1,
                               B + k * n + k, (int)n);
#else
                    std::vector<float> w(n - k);
                    for (std::size_t j = k; j < n; ++j) {
                        float sum = 0.0f;
                        for (std::size_t i = 0; i < m_sub; ++i) {
                            sum += v_a[i] * B[(k + i) * n + j];
                        }
                        w[j - k] = sum;
                    }
                    for (std::size_t i = 0; i < m_sub; ++i) {
                        const float vi = v_a[i];
                        for (std::size_t j = k; j < n; ++j) {
                            B[(k + i) * n + j] -= tau * vi * w[j - k];
                        }
                    }
#endif

                    // 累积 U：U[:, k:m] -= tau * U[:, k:m] * v_a * v_a^T
                    //   w = U[:, k:m] @ v_a  (m 维)
                    //   U[:, k:m] -= tau * w * v_a^T
#ifdef __APPLE__
                    std::vector<float> w_u(m);
                    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                                (int)m, (int)m_sub,
                                1.0f,
                                U + k, (int)m,
                                v_a.data(), 1,
                                0.0f,
                                w_u.data(), 1);
                    cblas_sger(CblasRowMajor,
                               (int)m, (int)m_sub,
                               -tau,
                               w_u.data(), 1,
                               v_a.data(), 1,
                               U + k, (int)m);
#else
                    std::vector<float> w_u(m);
                    for (std::size_t i = 0; i < m; ++i) {
                        float sum = 0.0f;
                        for (std::size_t j = 0; j < m_sub; ++j) {
                            sum += U[i * m + (k + j)] * v_a[j];
                        }
                        w_u[i] = sum;
                    }
                    for (std::size_t i = 0; i < m; ++i) {
                        for (std::size_t j = 0; j < m_sub; ++j) {
                            U[i * m + (k + j)] -= tau * w_u[i] * v_a[j];
                        }
                    }
#endif
                }
            }
        }

        // ---- 2) 右乘 H_b：把 B[k, k+1:n] 化成 [0, beta, 0, ..., 0] ----
        if (k >= n - 1) continue;
        const std::size_t n_sub = n - k - 1;
        if (n_sub <= 1) continue;

        // 算 v_b：长度 n_sub
        std::vector<float> v_b(n_sub);
        float y0 = B[k * n + k + 1];
        float norm_y_sq = y0 * y0;
        for (std::size_t i = 1; i < n_sub; ++i) {
            norm_y_sq += B[k * n + k + 1 + i] * B[k * n + k + 1 + i];
        }
        float norm_y = std::sqrt(norm_y_sq);
        if (norm_y < 1e-30f) {
            // 行已经基本是 0，跳过 H_b
            continue;
        }
        const float sign = (y0 >= 0.0f) ? 1.0f : -1.0f;
        v_b[0] = y0 + sign * norm_y;
        for (std::size_t i = 1; i < n_sub; ++i) {
            v_b[i] = B[k * n + k + 1 + i];
        }
        float v_norm_sq = 0.0f;
        for (std::size_t i = 0; i < n_sub; ++i) v_norm_sq += v_b[i] * v_b[i];
        const float tau = (v_norm_sq < 1e-30f) ? 0.0f : (2.0f / v_norm_sq);

        // B[k:m, k+1:n] -= tau * B[k:m, k+1:n] * v_b * v_b^T
        //   w = B[k:m, k+1:n] @ v_b  ((m-k) 维)
        //   B[k:m, k+1:n] -= tau * w * v_b^T
#ifdef __APPLE__
        std::vector<float> w(m - k);
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    (int)(m - k), (int)n_sub,
                    1.0f,
                    B + k * n + (k + 1), (int)n,
                    v_b.data(), 1,
                    0.0f,
                    w.data(), 1);
        cblas_sger(CblasRowMajor,
                   (int)(m - k), (int)n_sub,
                   -tau,
                   w.data(), 1,
                   v_b.data(), 1,
                   B + k * n + (k + 1), (int)n);
#else
        std::vector<float> w(m - k);
        for (std::size_t i = k; i < m; ++i) {
            float sum = 0.0f;
            for (std::size_t j = 0; j < n_sub; ++j) {
                sum += B[i * n + (k + 1 + j)] * v_b[j];
            }
            w[i - k] = sum;
        }
        for (std::size_t i = k; i < m; ++i) {
            const float wi = w[i - k];
            for (std::size_t j = 0; j < n_sub; ++j) {
                B[i * n + (k + 1 + j)] -= tau * wi * v_b[j];
            }
        }
#endif

        // 累积 V：V[:, k+1:n] -= tau * V[:, k+1:n] * v_b * v_b^T
#ifdef __APPLE__
        std::vector<float> w_v(n);
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    (int)n, (int)n_sub,
                    1.0f,
                    V + (k + 1), (int)n,
                    v_b.data(), 1,
                    0.0f,
                    w_v.data(), 1);
        cblas_sger(CblasRowMajor,
                   (int)n, (int)n_sub,
                   -tau,
                   w_v.data(), 1,
                   v_b.data(), 1,
                   V + (k + 1), (int)n);
#else
        std::vector<float> w_v(n);
        for (std::size_t i = 0; i < n; ++i) {
            float sum = 0.0f;
            for (std::size_t j = 0; j < n_sub; ++j) {
                sum += V[i * n + (k + 1 + j)] * v_b[j];
            }
            w_v[i] = sum;
        }
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n_sub; ++j) {
                V[i * n + (k + 1 + j)] -= tau * w_v[i] * v_b[j];
            }
        }
#endif
    }
}

} // namespace linalg
} // namespace ctQALS
#endif // CTORCH_BIDIAG_SVD_H
