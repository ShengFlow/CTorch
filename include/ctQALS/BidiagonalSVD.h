//
// BidiagonalSVD.h
// Bidiagonal SVD（Golub-Kahan-Reinsch bidiagonal 化 + 相邻 Jacobi sweep）
// 2026-08-10
//
// 算法：
//   1) Bidiagonal 化：A → U_b^T A V_b = B (m × n bidiagonal)
//      用双侧 Householder 反射累积 U_b (m×m) 和 V_b (n×n)
//   2) SVD of bidiagonal B：B = U_q Σ V_q^T
//      用 adjacent Jacobi sweep（只对 (i, i+1) 对做 Givens 旋转）
//      累积 U_q (m×m) 和 V_q (n×m)
//   3) 输出：U = U_b U_q (m×m), V = V_b V_q (n×m), S = Σ (m)
//   4) S 降序排序，U / V 同步重排
//
// 加速原理：
//   * bidiagonal 化 1 次：O(mn²) flops，n 步
//   * adjacent Jacobi on bidiagonal：每对 O(m) flops（不是 O(m²)）
//   * 对 m=42, n=1024：bidiagonal 化 ~22 Gflops，adjacent Jacobi ~2M flops
//   * 总 ~22 Gflops，Apple M1 NEON ~22 Gflops/s ≈ 1s
//   * 对比 full m×n Jacobi: ~25 Gflops，100s → 100× 加速
//
#ifndef CTORCH_BIDIAGONAL_SVD_H
#define CTORCH_BIDIAGONAL_SVD_H
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

namespace ctQALS {
namespace linalg {

// ============================================================
// Householder 反射
//   x (n 维, stride s) → v (n 维) + tau
//   H = I - tau * v * v^T
//   H @ x = -sign(x0) * ||x|| * e_1
//   约定 v[0] = 1（不存 v[0]），但这里我们存 full v（v[0] = v0）
// ============================================================
inline void householder_compute(const float* x, std::size_t stride, std::size_t n,
                                 std::vector<float>* v_out, float* tau_out) {
    float xnorm_sq = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        const float xi = x[i * stride];
        xnorm_sq += xi * xi;
    }
    const float xnorm = std::sqrt(xnorm_sq);
    if (xnorm == 0.0f) {
        *tau_out = 0.0f;
        v_out->assign(n, 0.0f);
        return;
    }
    const float x0 = x[0];
    const float sign = (x0 >= 0.0f) ? 1.0f : -1.0f;
    const float v0 = x0 + sign * xnorm;

    v_out->resize(n);
    (*v_out)[0] = v0;
    for (std::size_t i = 1; i < n; ++i) {
        (*v_out)[i] = x[i * stride];
    }

    // tau = 2 / (v^T v)
    float vtv = 0.0f;
    for (std::size_t i = 0; i < n; ++i) vtv += (*v_out)[i] * (*v_out)[i];
    if (vtv < 1e-30f) {
        *tau_out = 0.0f;
    } else {
        *tau_out = 2.0f / vtv;
    }
}

// 应用 H = I - tau * v v^T 到 X 的前 m_sub 行子矩阵（X 是 m_sub × p_cols, ld=ld_X）
// 2026-08-10: 朴素版（cblas_sger 对子矩阵起点 + 非对齐 lda 在 Apple Silicon 上 segfault）
inline void apply_h_left_inplace(float* X, std::size_t ld_X, std::size_t m_sub, std::size_t p_cols,
                                  const std::vector<float>& v, float tau) {
    if (tau == 0.0f || m_sub == 0) return;
    std::vector<float> w(p_cols, 0.0f);
    for (std::size_t j = 0; j < p_cols; ++j) {
        float sum = 0.0f;
        for (std::size_t i = 0; i < m_sub; ++i) sum += v[i] * X[i * ld_X + j];
        w[j] = sum;
    }
    for (std::size_t i = 0; i < m_sub; ++i) {
        const float vi = v[i];
        for (std::size_t j = 0; j < p_cols; ++j) {
            X[i * ld_X + j] -= tau * vi * w[j];
        }
    }
}

// 应用 H = I - tau * v v^T 到 X 的列子矩阵（X 是 p_rows × n_sub, ld=ld_X）
inline void apply_h_right_inplace(float* X, std::size_t ld_X, std::size_t p_rows, std::size_t n_sub,
                                   const std::vector<float>& v, float tau) {
    if (tau == 0.0f || n_sub == 0) return;
    std::fprintf(stderr, "[DBG apply_right] X=%p ld=%zu p_rows=%zu n_sub=%zu tau=%f v.size=%zu\n",
                 (void*)X, ld_X, p_rows, n_sub, tau, v.size());
    // [DBG 2026-08-10] 临时用朴素版排查 cblas 崩溃问题
    std::vector<float> w(p_rows, 0.0f);
    for (std::size_t i = 0; i < p_rows; ++i) {
        float sum = 0.0f;
        for (std::size_t j = 0; j < n_sub; ++j) sum += X[i * ld_X + j] * v[j];
        w[i] = sum;
    }
    std::fprintf(stderr, "[DBG apply_right] after w computed\n");
    for (std::size_t i = 0; i < p_rows; ++i) {
        const float wi = w[i];
        for (std::size_t j = 0; j < n_sub; ++j) {
            X[i * ld_X + j] -= tau * wi * v[j];
        }
    }
    std::fprintf(stderr, "[DBG apply_right] after X update, return\n");
#if 0
#ifdef __APPLE__
    // w = X v (p_rows 维)
    std::vector<float> w(p_rows, 0.0f);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, (int)p_rows, (int)n_sub,
                1.0f, X, (int)ld_X, v.data(), 1, 0.0f, w.data(), 1);
    // X -= tau * w * v^T
    cblas_sger(CblasRowMajor, (int)p_rows, (int)n_sub, -tau,
               w.data(), 1, v.data(), 1, X, (int)ld_X);
#else
    std::vector<float> w(p_rows, 0.0f);
    for (std::size_t i = 0; i < p_rows; ++i) {
        float sum = 0.0f;
        for (std::size_t j = 0; j < n_sub; ++j) sum += X[i * ld_X + j] * v[j];
        w[i] = sum;
    }
    for (std::size_t i = 0; i < p_rows; ++i) {
        const float wi = w[i];
        for (std::size_t j = 0; j < n_sub; ++j) {
            X[i * ld_X + j] -= tau * wi * v[j];
        }
    }
#endif
#endif
}

// ============================================================
// Bidiagonal SVD
// ============================================================
class BidiagonalSVD {
public:
    BidiagonalSVD(const float* A, std::size_t m_rows, std::size_t n_cols) {
        if (m_rows == 0 || n_cols == 0) {
            throw std::invalid_argument("BidiagonalSVD: dims must be > 0");
        }
        if (m_rows > n_cols) {
            throw std::invalid_argument("BidiagonalSVD: requires m <= n (transpose first)");
        }
        m_ = m_rows;
        n_ = n_cols;

        // 1) 拷贝 A 到 B
        B_.assign(A, A + m_ * n_);

        // 2) 初始化 U_b = I_m, V_b = I_n
        U_b_.assign(m_ * m_, 0.0f);
        for (std::size_t i = 0; i < m_; ++i) U_b_[i * m_ + i] = 1.0f;
        V_b_.assign(n_ * n_, 0.0f);
        for (std::size_t i = 0; i < n_; ++i) V_b_[i * n_ + i] = 1.0f;

        // 3) 步 1：Bidiagonal 化
        bidiagonalize();

        // 4) 步 2：SVD of bidiagonal B
        U_q_.assign(m_ * m_, 0.0f);
        for (std::size_t i = 0; i < m_; ++i) U_q_[i * m_ + i] = 1.0f;
        V_q_.assign(n_ * m_, 0.0f);
        for (std::size_t i = 0; i < m_; ++i) V_q_[i * m_ + i] = 1.0f;  // V_q 前 m 行 = I_m
        svd_of_bidiagonal();

        // 5) 计算 U = U_b U_q (m×m), V = V_b V_q (n×m)
        compute_final_UV();

        // 6) 排序 S 降序
        sort_descending();
    }

    std::size_t m() const noexcept { return m_; }
    std::size_t n() const noexcept { return n_; }

    const float* U() const noexcept { return U_.data(); }
    const float* V() const noexcept { return V_.data(); }
    const float* S() const noexcept { return S_.data(); }

    void reconstruct(float* A_out) const {
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

private:
    std::size_t          m_, n_;
    std::vector<float>   B_;     // m × n (bidiagonal after step 1, diagonal after step 2)
    std::vector<float>   U_b_;   // m × m
    std::vector<float>   V_b_;   // n × n
    std::vector<float>   U_q_;   // m × m
    std::vector<float>   V_q_;   // n × m
    std::vector<float>   U_;     // m × m = U_b * U_q
    std::vector<float>   V_;     // n × m = V_b * V_q
    std::vector<float>   S_;     // m

    // ============================================================
    // 步骤 1：Bidiagonal 化
    //   A → U_b^T A V_b = B (bidiagonal, m × n)
    //   B in-place 修改，U_b / V_b 累积
    // ============================================================
    void bidiagonalize() {
        const std::size_t K = std::min(m_, n_);  // 步数
        std::vector<float> v;

        for (std::size_t k = 0; k < K; ++k) {
            std::fprintf(stderr, "[DBG bidiag] === k=%zu (m=%zu n=%zu) ===\n", k, m_, n_);
            // (a) 左 Householder：清 B[k+1:m, k]
            {
                const std::size_t m_sub = m_ - k;
                v.clear();
                v.resize(m_sub);
                // 反射向量基于 B[k, k] (stride n_)，长度 m_sub
                householder_compute(&B_[k * n_ + k], n_, m_sub, &v, &v_tau_l_);
                std::fprintf(stderr, "[DBG bidiag] left tau=%.4e, v[0]=%.3e, m_sub=%zu\n",
                             v_tau_l_, v[0], m_sub);
                if (v_tau_l_ != 0.0f) {
                    std::fprintf(stderr, "[DBG bidiag]   apply left to B[k:, :] (X=&B[%zu], ld=%zu, m_sub=%zu, p=%zu)\n",
                                 k * n_, n_, m_sub, n_ - k);
                    apply_h_left_inplace(&B_[k * n_], n_, m_sub, n_ - k, v, v_tau_l_);
                    std::fprintf(stderr, "[DBG bidiag]   apply left to U_b[:,k:] (X=&U_b[%zu], ld=%zu, m_sub=%zu, p=%zu)\n",
                                 k * m_ + k, m_, m_sub, m_ - k);
                    apply_h_left_inplace(&U_b_[k * m_ + k], m_, m_sub, m_ - k, v, v_tau_l_);
                }
            }
            // [DBG] 验证 B[k+1:m, k] 应该是 0
            {
                double max_below = 0;
                for (std::size_t r = k + 1; r < m_; ++r) {
                    max_below = std::max(max_below, (double)std::abs(B_[r * n_ + k]));
                }
                std::fprintf(stderr, "[DBG bidiag] k=%zu  max|B[r,k]| for r>k = %.3e  (期望 0)\n", k, max_below);
            }
            // (b) 右 Householder：清 B[k, k+2:n]
            if (k + 1 < n_) {
                const std::size_t n_sub = n_ - k - 1;
                v.clear();
                v.resize(n_sub);
                std::fprintf(stderr, "[DBG bidiag] before right, k=%zu, n_sub=%zu, &B[%zu] stride 1\n",
                             k, n_sub, k * n_ + k + 1);
                // 反射向量基于 B[k, k+1] (stride 1)，长度 n_sub
                householder_compute(&B_[k * n_ + k + 1], 1, n_sub, &v, &v_tau_r_);
                std::fprintf(stderr, "[DBG bidiag] right tau=%.4e v[0]=%.3e\n", v_tau_r_, v[0]);
                if (v_tau_r_ != 0.0f) {
                    std::fprintf(stderr, "[DBG bidiag]   apply right to B[:,k+1:] (X=&B[%zu], ld=%zu, p=%zu, n_sub=%zu)\n",
                                 k + 1, n_, m_, n_sub);
                    apply_h_right_inplace(&B_[k + 1], n_, m_, n_sub, v, v_tau_r_);
                    std::fprintf(stderr, "[DBG bidiag]   apply right to V_b[:,k+1:] (X=&V_b[%zu], ld=%zu, p=%zu, n_sub=%zu)\n",
                                 (k + 1) * n_ + (k + 1), n_, n_, n_sub);
                    apply_h_right_inplace(&V_b_[(k + 1) * n_ + (k + 1)], n_, n_, n_sub, v, v_tau_r_);
                }
            }
            // [DBG] 验证 B[k, k+2:n] 应该是 0
            std::fprintf(stderr, "[DBG bidiag] about to enter max_right loop, k=%zu, n_=%zu\n", k, n_);
            {
                double max_right = 0;
                for (std::size_t j = k + 2; j < n_; ++j) {
                    max_right = std::max(max_right, (double)std::abs(B_[k * n_ + j]));
                }
                std::fprintf(stderr, "[DBG bidiag] k=%zu  max|B[k,j]| for j>k+1 = %.3e  (期望 0)\n", k, max_right);
            }
            std::fprintf(stderr, "[DBG bidiag] k=%zu end of iter, k++\n", k);
        }
        std::fprintf(stderr, "[DBG bidiag] end of bidiagonalize loop, all k done\n");
    }

    // ============================================================
    // 步骤 2：SVD of bidiagonal B
    //   B = U_q Σ V_q^T
    //   B 已经 bidiagonal（B[i+1, i] = 0 for i, 主对角 + 副对角非零）
    //   用 adjacent Jacobi sweep：只处理 (i, i+1) 对
    // ============================================================
    void svd_of_bidiagonal() {
        std::fprintf(stderr, "[DBG bidiag] === svd_of_bidiagonal start ===\n");
        const int max_sweeps = 30;
        double prev_off = std::numeric_limits<double>::infinity();

        for (int sweep = 0; sweep < max_sweeps; ++sweep) {
            double off = 0.0;
            for (std::size_t i = 0; i + 1 < m_; ++i) {
                std::size_t j = i + 1;
                // alpha = ||B[:, i]||^2, beta = ||B[:, j]||^2, gamma = B[:, i] . B[:, j]
                float alpha = 0.0f, beta = 0.0f, gamma = 0.0f;
                for (std::size_t r = 0; r < m_; ++r) {
                    const float bi = B_[r * n_ + i];
                    const float bj = B_[r * n_ + j];
                    alpha += bi * bi;
                    beta  += bj * bj;
                    gamma += bi * bj;
                }
                off += 2.0f * gamma * gamma;
                if (std::abs(gamma) < 1e-30f) continue;

                // 计算 Givens 旋转
                const float diff = beta - alpha;
                float t;
                if (std::abs(gamma) < 1e-12f * std::max(alpha, beta)) {
                    t = (gamma > 0.0f) ? 1e-7f : -1e-7f;
                } else {
                    const float zeta = diff / (2.0f * gamma);
                    const float z_abs = std::abs(zeta);
                    t = ((zeta >= 0.0f) ? 1.0f : -1.0f) /
                        (z_abs + std::sqrt(1.0f + zeta * zeta));
                }
                const float c = 1.0f / std::sqrt(1.0f + t * t);
                const float s = t * c;

                // 更新 B[:, i] 和 B[:, j]
                for (std::size_t r = 0; r < m_; ++r) {
                    const float bi = B_[r * n_ + i];
                    const float bj = B_[r * n_ + j];
                    B_[r * n_ + i] = c * bi - s * bj;
                    B_[r * n_ + j] = s * bi + c * bj;
                }
                // 更新 U_q[:, i] 和 U_q[:, j]（m × m 矩阵）
                for (std::size_t r = 0; r < m_; ++r) {
                    const float ui = U_q_[r * m_ + i];
                    const float uj = U_q_[r * m_ + j];
                    U_q_[r * m_ + i] = c * ui - s * uj;
                    U_q_[r * m_ + j] = s * ui + c * uj;
                }
                // 更新 V_q[:, i] 和 V_q[:, j]（n × m 矩阵）
                for (std::size_t r = 0; r < n_; ++r) {
                    const float vi = V_q_[r * m_ + i];
                    const float vj = V_q_[r * m_ + j];
                    V_q_[r * m_ + i] = c * vi - s * vj;
                    V_q_[r * m_ + j] = s * vi + c * vj;
                }
            }
            off = std::sqrt(off);
            if (sweep > 0) {
                const double rel = std::abs(off - prev_off) / std::max(off, 1e-30);
                if (rel < 1e-6) break;
            }
            prev_off = off;
        }

        // S = B 的列范数
        S_.assign(m_, 0.0f);
        for (std::size_t j = 0; j < m_; ++j) {
            float cn = 0.0f;
            for (std::size_t r = 0; r < m_; ++r) cn += B_[r * n_ + j] * B_[r * n_ + j];
            S_[j] = std::sqrt(cn);
        }
        // 清零 B（用不到了）
        B_.clear();
        B_.shrink_to_fit();
    }

    // ============================================================
    // 步骤 5：U = U_b U_q (m×m), V = V_b V_q (n×m)
    // ============================================================
    void compute_final_UV() {
        std::fprintf(stderr, "[DBG bidiag] === compute_final_UV start ===\n");
        // U = U_b U_q
        U_.assign(m_ * m_, 0.0f);
        for (std::size_t i = 0; i < m_; ++i) {
            for (std::size_t j = 0; j < m_; ++j) {
                float sum = 0.0f;
                for (std::size_t k = 0; k < m_; ++k) {
                    sum += U_b_[i * m_ + k] * U_q_[k * m_ + j];
                }
                U_[i * m_ + j] = sum;
            }
        }
        // V = V_b V_q
        V_.assign(n_ * m_, 0.0f);
        for (std::size_t i = 0; i < n_; ++i) {
            for (std::size_t j = 0; j < m_; ++j) {
                float sum = 0.0f;
                for (std::size_t k = 0; k < n_; ++k) {
                    sum += V_b_[i * n_ + k] * V_q_[k * m_ + j];
                }
                V_[i * m_ + j] = sum;
            }
        }
        // 清理
        U_b_.clear(); U_b_.shrink_to_fit();
        V_b_.clear(); V_b_.shrink_to_fit();
        U_q_.clear(); U_q_.shrink_to_fit();
        V_q_.clear(); V_q_.shrink_to_fit();
    }

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

    // 临时（仅 bidiagonalize 使用）
    float v_tau_l_ = 0.0f;
    float v_tau_r_ = 0.0f;
};

} // namespace linalg
} // namespace ctQALS
#endif // CTORCH_BIDIAGONAL_SVD_H
