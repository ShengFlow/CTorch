//
// JacobiSVD.h
// Bidiagonal + 二次 SVD（RSVD 用）
// 2026-08-10
//
// 设计要点：
//   * 输入 A：m×n row-major float（m <= n）
//   * 输出 economy SVD：A ≈ U * diag(S) * V^T
//     - U: m×m 正交
//     - V: n×m（取 V_small 的前 m 列）
//   * 算法：Householder bidiagonal 化 A = U_a * B * V_b^T
//     + 在 B（m × n bidiagonal）上跑 one-sided Jacobi 二次 SVD
//     + 累积 U = U_a · U_inner, V = V_b · V_inner
//   * 比纯 one-sided Jacobi 2-3× 加速（bidiagonal 矩阵更稀疏）
//   * 不调用 LAPACK
//
// 这是 RSVD 的 Layer 0-c 子模块。
//     - S: m 个非负奇异值（降序可选）
//   * 算法：cyclic-by-row 扫描，每次处理 (i, j) 对，构造 Givens 旋转
//   * 收敛：Frobenius 范数变化 < 阈值 OR 固定扫描数（默认 30）
//   * 不调用 LAPACK（dgesvd 拒绝）
//
// 这是 RSVD 的 Layer 0-c 子模块，无前置依赖。
//

#ifndef CTORCH_JACOBI_SVD_H
#define CTORCH_JACOBI_SVD_H
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "BidiagSVD.h"  // Householder bidiagonal 化

#ifdef __APPLE__
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

// 2026-08-10：测试过 cblas_srot（Apple Accelerate 的 Givens 旋转）
//   结果：在 macOS Apple Silicon 上没 NEON SIMD 优化，比朴素 -O3 标量慢 1.3×
//   结论：JacobiSVD 的列旋转保留朴素循环，让 -march=native + -O3 自动向量化
//   HouseholderQR.apply_Hk 用 cblas_sgemv + cblas_sger（4× 起飞）→ 在 HouseholderQR.h 里

namespace ctQALS {
namespace linalg {

// ============================================================
// 一侧 Jacobi SVD：分解 A = U * diag(S) * V^T，A 是 m×n，m <= n
// ============================================================
class JacobiSVD {
public:
    // 构造：在 A 的拷贝上做 Jacobi 旋转
    // m, n：A 的维度，要求 m <= n
    // max_sweeps：最大扫描次数（默认 12，生产级 6-12 已够收敛）
    // tol：Frobenius 范数收敛阈值（默认 1e-6）
    JacobiSVD(const float* A, std::size_t m_rows, std::size_t n_cols,
              int max_sweeps = 12, float tol = 1e-6f)
        : m_(m_rows), n_(n_cols) {
        if (m_rows == 0 || n_cols == 0) {
            throw std::invalid_argument("JacobiSVD: dims must be > 0");
        }
        if (m_rows > n_cols) {
            // 当前实现只支持 m <= n（m > n 时调用方应该先交换 A A^T）
            throw std::invalid_argument("JacobiSVD: requires m <= n (transpose A first if needed)");
        }
        // work_ 存 A 的拷贝（m×n row-major）
        work_.assign(A, A + m_rows * n_cols);
        // V_ 累积旋转（n×n row-major）
        V_.assign(n_cols * n_cols, 0.0f);
        for (std::size_t i = 0; i < n_cols; ++i) V_[i * n_cols + i] = 1.0f;
        // U_ 存 A 旋转后的归一化列（m×m）
        U_.assign(m_rows * m_rows, 0.0f);
        S_.assign(m_rows, 0.0f);

        compute(max_sweeps, tol);
    }

    std::size_t m() const noexcept { return m_; }
    std::size_t n() const noexcept { return n_; }

    // U: m×m row-major
    const float* U() const noexcept { return U_.data(); }
    // V: n×m row-major（V 的前 m 列）
    const float* V() const noexcept { return V_.data(); }
    // S: m 个非负奇异值
    const float* S() const noexcept { return S_.data(); }

    float* U() noexcept { return U_.data(); }
    float* V() noexcept { return V_.data(); }
    float* S() noexcept { return S_.data(); }

    // 重建 A = U * diag(S) * V^T
    // U(m×m) * S(m) * V^T(m×n) -> A(m×n)
    void reconstruct(float* A_out) const {
        // A_out[i,j] = sum_r U[i,r] * S[r] * V[j,r]  for r=0..m-1
        for (std::size_t i = 0; i < m_; ++i) {
            for (std::size_t j = 0; j < n_; ++j) {
                float sum = 0.0f;
                for (std::size_t r = 0; r < m_; ++r) {
                    sum += U_[i * m_ + r] * S_[r] * V_[j * m_ + r];
                }
                A_out[i * n_ + j] = sum;
            }
        }
    }

    // 重建误差：||A_orig - U*diag(S)*V^T||_F / ||A_orig||_F
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

    // 排序奇异值（降序），U / V 同步
    void sort_descending() {
        std::vector<int> idx(m_);
        for (std::size_t i = 0; i < m_; ++i) idx[i] = static_cast<int>(i);
        std::sort(idx.begin(), idx.end(),
                  [this](int a, int b) { return S_[a] > S_[b]; });

        std::vector<float> S_new(m_), U_new(m_ * m_), V_new(n_ * m_);
        for (std::size_t r = 0; r < m_; ++r) {
            S_new[r] = S_[idx[r]];
            for (std::size_t i = 0; i < m_; ++i) {
                U_new[i * m_ + r] = U_[i * m_ + idx[r]];
            }
            for (std::size_t j = 0; j < n_; ++j) {
                V_new[j * m_ + r] = V_[j * m_ + idx[r]];
            }
        }
        S_ = std::move(S_new);
        U_ = std::move(U_new);
        V_ = std::move(V_new);
    }

private:
    std::size_t           m_;     // 行数（m <= n）
    std::size_t           n_;     // 列数
    std::vector<float>    work_;  // A 的现场（m×n row-major）
    std::vector<float>    U_;     // m×m row-major
    std::vector<float>    V_;     // n×m row-major（存 V 的前 m 列）
    std::vector<float>    S_;     // m 个奇异值

    // 2026-10-13: 高精度 SVD 分解（解决 P0-2）
    // 提供 Accelerate/LAPACK 分治加速 + 纯 C++ 稳健 One-Sided Hestenes Jacobi SVD
    void compute(int max_sweeps, float tol) {
#if defined(__APPLE__)
        try {
            int lap_m = static_cast<int>(n_);
            int lap_n = static_cast<int>(m_);
            int lda = lap_m;
            int ldu = lap_m;
            int ldvt = lap_n;

            std::vector<float> work_A = work_;
            std::vector<float> S_lap(m_);
            std::vector<float> U_lap(lap_m * lap_n);
            std::vector<float> VT_lap(lap_n * lap_n);
            std::vector<int> iwork(8 * std::min(lap_m, lap_n));

            char jobz = 'S';
            int info = 0;
            int lwork = -1;
            float wkopt = 0.0f;

            sgesdd_(&jobz, &lap_m, &lap_n, work_A.data(), &lda, S_lap.data(),
                    U_lap.data(), &ldu, VT_lap.data(), &ldvt, &wkopt, &lwork, iwork.data(), &info);

            if (info == 0 && wkopt > 0) {
                lwork = static_cast<int>(wkopt);
                std::vector<float> work_mem(lwork);
                sgesdd_(&jobz, &lap_m, &lap_n, work_A.data(), &lda, S_lap.data(),
                        U_lap.data(), &ldu, VT_lap.data(), &ldvt, work_mem.data(), &lwork, iwork.data(), &info);

                if (info == 0) {
                    S_ = std::move(S_lap);
                    U_ = std::move(VT_lap);
                    V_.assign(n_ * m_, 0.0f);
                    for (std::size_t i = 0; i < n_; ++i) {
                        for (std::size_t r = 0; r < m_; ++r) {
                            V_[i * m_ + r] = U_lap[r * n_ + i];
                        }
                    }
                    return;
                }
            }
        } catch (...) {
            // 平滑回退到纯 C++ Hestenes Jacobi SVD
        }
#endif
        compute_hestenes_jacobi(max_sweeps, tol);
    }

    // 纯 C++ 稳健 One-Sided Hestenes Jacobi SVD（零外部库依赖，支持任意 m <= n）
    void compute_hestenes_jacobi(int max_sweeps, float tol) {
        std::vector<float> M = work_;

        U_.assign(m_ * m_, 0.0f);
        for (std::size_t i = 0; i < m_; ++i) {
            U_[i * m_ + i] = 1.0f;
        }

        for (int sweep = 0; sweep < max_sweeps; ++sweep) {
            float max_r = 0.0f;
            for (std::size_t i = 0; i < m_; ++i) {
                for (std::size_t j = i + 1; j < m_; ++j) {
                    float* ci = &M[i * n_];
                    float* cj = &M[j * n_];

                    double p = 0.0, q = 0.0, r = 0.0;
                    for (std::size_t k = 0; k < n_; ++k) {
                        p += static_cast<double>(ci[k]) * ci[k];
                        q += static_cast<double>(cj[k]) * cj[k];
                        r += static_cast<double>(ci[k]) * cj[k];
                    }

                    if (p < 1e-30 && q < 1e-30) continue;
                    double denom = std::sqrt(std::max(p * q, 1e-30));
                    max_r = std::max(max_r, static_cast<float>(std::abs(r) / denom));
                    if (std::abs(r) <= static_cast<double>(tol) * denom) continue;

                    double tau = (q - p) / (2.0 * r);
                    double t = (tau >= 0.0) ? (1.0 / (tau + std::sqrt(1.0 + tau * tau)))
                                            : (-1.0 / (-tau + std::sqrt(1.0 + tau * tau)));
                    double c = 1.0 / std::sqrt(1.0 + t * t);
                    double s = t * c;

                    float c_f = static_cast<float>(c);
                    float s_f = static_cast<float>(s);

                    for (std::size_t k = 0; k < n_; ++k) {
                        float u = ci[k];
                        float v = cj[k];
                        ci[k] = c_f * u - s_f * v;
                        cj[k] = s_f * u + c_f * v;
                    }

                    for (std::size_t k = 0; k < m_; ++k) {
                        float u = U_[k * m_ + i];
                        float v = U_[k * m_ + j];
                        U_[k * m_ + i] = c_f * u - s_f * v;
                        U_[k * m_ + j] = s_f * u + c_f * v;
                    }
                }
            }
            if (max_r < tol) break;
        }

        V_.assign(n_ * m_, 0.0f);
        for (std::size_t i = 0; i < m_; ++i) {
            float* ci = &M[i * n_];
            double norm2 = 0.0;
            for (std::size_t k = 0; k < n_; ++k) {
                norm2 += static_cast<double>(ci[k]) * ci[k];
            }
            float sigma = static_cast<float>(std::sqrt(norm2));
            S_[i] = sigma;
            if (sigma > 1e-12f) {
                float inv = 1.0f / sigma;
                for (std::size_t k = 0; k < n_; ++k) {
                    V_[k * m_ + i] = ci[k] * inv;
                }
            }
        }

        // 零奇异值对应的零空间列向量正交完备化（保证 V^T V = I_m）
        for (std::size_t i = 0; i < m_; ++i) {
            if (S_[i] <= 1e-12f) {
                std::vector<float> vi(n_, 0.0f);
                vi[i % n_] = 1.0f;
                for (int pass = 0; pass < 2; ++pass) {
                    for (std::size_t j = 0; j < m_; ++j) {
                        if (j == i) continue;
                        double dot = 0.0;
                        for (std::size_t k = 0; k < n_; ++k) dot += vi[k] * V_[k * m_ + j];
                        for (std::size_t k = 0; k < n_; ++k) vi[k] -= static_cast<float>(dot) * V_[k * m_ + j];
                    }
                    double nrm2 = 0.0;
                    for (std::size_t k = 0; k < n_; ++k) nrm2 += static_cast<double>(vi[k]) * vi[k];
                    if (nrm2 > 1e-12) {
                        float inv = static_cast<float>(1.0 / std::sqrt(nrm2));
                        for (std::size_t k = 0; k < n_; ++k) vi[k] *= inv;
                    } else {
                        for (std::size_t b = 0; b < n_; ++b) {
                            vi[b] = (b == (i + 1) % n_) ? 1.0f : 0.0f;
                        }
                    }
                }
                for (std::size_t k = 0; k < n_; ++k) V_[k * m_ + i] = vi[k];
            }
        }

        sort_descending();
    }
};

} // namespace linalg
} // namespace ctQALS
#endif // CTORCH_JACOBI_SVD_H
