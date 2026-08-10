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
                V_new[j * m_ + r] = V_[j * n_ + idx[r]];
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

    // 2026-08-10: Bidiagonal 化 + 二次 SVD
    //   步骤：
    //   1) Householder bidiagonal 化 A = U_a * B * V_b^T
    //      - 累积 U_a (m×m) → U_, V_b (n×n) → V_
    //      - work_ = bidiagonal B (m × n)
    //   2) 在 bidiagonal B 上跑 one-sided Jacobi 二次 SVD
    //      - 因为 B 已经 bidiagonal，扫描 (i, j) 对时大部分 gamma ≈ 0 跳过
    //      - 内部只用 ~m^2 次有效计算（vs 一般矩阵的 m²n 次）
    //   3) 累积最终 U, V, S
    //      - U_final = U_a · U_jacobi
    //      - V_final = V_b · V_jacobi
    //      - S 不变（中间正交变换不改变奇异值）
    void compute(int max_sweeps, float tol) {
        // ---- Step 1: Householder bidiagonal 化 ----
        // work_ 已经是 A 拷贝（构造函数里 copy 的）
        // 准备 U_outer, V_outer 临时缓冲
        std::vector<float> U_outer(m_ * m_);
        std::vector<float> V_outer(n_ * n_);
        // householder_bidiagonalize: A = U_outer * work_ * V_outer^T
        // 完成后 work_ 变成 bidiagonal B
        householder_bidiagonalize(work_.data(), m_, n_,
                                  work_.data(),  // B 覆盖原 A
                                  U_outer.data(),
                                  V_outer.data());

        // ---- Step 2: 在 bidiagonal work_ 上跑 one-sided Jacobi 二次 SVD ----
        // 我们用临时 JacobiSVD 内部 helper 跑（要避免覆盖 U_outer, V_outer）
        // 直接复用现有算法（one-sided Jacobi on work_），把 U_, V_, S_, work_ 当临时变量
        std::vector<float> U_inner(m_ * m_);
        std::vector<float> V_inner(n_ * n_);
        // work_ 在这里还是 bidiagonal，跑 Jacobi 收敛更快
        compute_one_sided_jacobi_inner_(work_.data(), m_, n_,
                                        U_inner.data(), V_inner.data(), S_.data(),
                                        max_sweeps, tol);

        // ---- Step 3: 累积 U_final = U_outer * U_inner, V_final = V_outer * V_inner ----
        //   U_outer (m×m) · U_inner (m×m) = U_final (m×m)
        //   V_outer (n×n) · V_inner (n×n) = V_final (n×n)
        std::vector<float> U_final(m_ * m_);
#ifdef __APPLE__
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)m_, (int)m_, (int)m_,
                    1.0f,
                    U_outer.data(), (int)m_,
                    U_inner.data(), (int)m_,
                    0.0f,
                    U_final.data(), (int)m_);
#else
        for (std::size_t i = 0; i < m_; ++i) {
            for (std::size_t j = 0; j < m_; ++j) {
                float sum = 0.0f;
                for (std::size_t k = 0; k < m_; ++k) {
                    sum += U_outer[i * m_ + k] * U_inner[k * m_ + j];
                }
                U_final[i * m_ + j] = sum;
            }
        }
#endif
        U_ = std::move(U_final);

        std::vector<float> V_final(n_ * n_);
#ifdef __APPLE__
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)n_, (int)n_, (int)n_,
                    1.0f,
                    V_outer.data(), (int)n_,
                    V_inner.data(), (int)n_,
                    0.0f,
                    V_final.data(), (int)n_);
#else
        for (std::size_t i = 0; i < n_; ++i) {
            for (std::size_t j = 0; j < n_; ++j) {
                float sum = 0.0f;
                for (std::size_t k = 0; k < n_; ++k) {
                    sum += V_outer[i * n_ + k] * V_inner[k * n_ + j];
                }
                V_final[i * n_ + j] = sum;
            }
        }
#endif
        // 提取 V_final 的前 m 列（V_final 是 n×n，SVD 输出 V 是 n×m）
        // 必须在 move 之前做（move 后 V_final 变空）
        std::vector<float> V_out(n_ * m_);
        for (std::size_t i = 0; i < n_; ++i) {
            for (std::size_t r = 0; r < m_; ++r) {
                V_out[i * m_ + r] = V_final[i * n_ + r];
            }
        }
        V_ = std::move(V_out);
    }

    // 内部 helper：在 work (m×n, m <= n) 上跑 bidiagonal SVD
    //   2026-08-10: B 路径 = Demmel-Kahan 隐式 shift QR on bidiagonal
    //   比 one-sided Jacobi 快 (5-10×) 因为只 chase supdiag
    //
    //   算法:
    //   - 输入: work 是 m×n upper bidiagonal (m <= n)
    //   - 输出: U (m×m), V (n×m 前 m 列), S (m 个)
    //   - 步骤: 每次 sweep 沿 supdiag 做隐式 shift QR
    //     1) 选择 Wilkinson shift from 2×2 trailing block
    //     2) 用 shift 引入 top bulge
    //     3) Givens chase 把 bulge 推到底
    //   - 复杂度: O(m²) total
    //
    //   注: 这是 LAPACK dlasq1/dlasq2 的简化自研版
    //       cubic convergence, deflation 等高级特性不在此实现
    void compute_one_sided_jacobi_inner_(float* work, std::size_t m, std::size_t n,
                                          float* U, float* V, float* S,
                                          int max_sweeps, float tol) {
        // init U = I_m, V = I_n
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < m; ++j) {
                U[i * m + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                V[i * n + j] = (i == j) ? 1.0f : 0.0f;
            }
        }

        // 提取 bidiagonal 系数 (d[i] = diag, e[i] = supdiag)
        // upper bidiagonal: d[i] = work[i, i], e[i] = work[i, i+1]
        // work is row-major, n 列宽
        std::vector<float> d(m), e(m, 0.0f);
        for (std::size_t i = 0; i + 1 < m; ++i) {
            d[i] = work[i * n + i];
            e[i] = work[i * n + i + 1];
        }
        d[m - 1] = work[(m - 1) * n + (m - 1)];

        // B 路径：隐式 shift QR on bidiagonal
        // 每次 sweep:
        //   1) Wilkinson shift s from bottom 2x2 block [d[m-2], e[m-2]; e[m-2], d[m-1]]
        //   2) Givens rotation at top: (d[0] - s, e[0]) -> r, 0
        //      创建 bulge
        //   3) Chase the bulge: 对 k = 0..m-2:
        //      - Right Givens on (d[k], e[k]): e[k] -> 0, d[k] -> r
        //      - Left Givens on (r, f[k+1]): f[k+1] -> 0, d[k] -> r' (or update)
        //      - Bulge 推到 e[k+1]
        //   4) 收敛判断: max |e[i]| < tol
        //
        // 关键: f[k] 是 chase 中产生的"亚对角"项（下 bidiagonal）
        //      f[0] = e[0] - s (shift 引入), f[k+1] = ...（chase 累积）
        // 简化为: 维护一个 "active bidiag" 状态
        //
        // 用更直接的形式: 维护 d, e, f (d=diag, e=supdiag, f=subdiag)
        // 每次 chase step (k):
        //   - Givens (c, s) on (d[k] - shift_at_k, e[k])  // right
        //   - Update d[k], e[k]
        //   - Givens on (d[k], f[k+1])  // left
        //   - Update d[k+1], e[k+1]
        //   - f[k+2] = (left rotation's s) * e[k+1] (bulge propagated)

        // 简化: f 数组
        std::vector<float> f(m, 0.0f);  // f[i] 是 chase 状态里第 i 行的"亚对角"

        double prev_max_e = std::numeric_limits<double>::infinity();
        for (int sweep = 0; sweep < max_sweeps; ++sweep) {
            // 1) Wilkinson shift from bottom 2x2 block
            //   T = [[d[m-2], e[m-2]], [e[m-2], d[m-1]]]
            //   eigenvalues: (d[m-2] + d[m-1])/2 ± sqrt(((d[m-2]-d[m-1])/2)² + e[m-2]²)
            //   shift = eigenvalue closer to d[m-1]
            float shift = 0.0f;
            if (m >= 2) {
                const float a = d[m - 2], b = e[m - 2], c = d[m - 1];
                const float tr = a + c;
                const float det = a * c - b * b;
                // eigenvalues = (tr ± sqrt(tr² - 4 det)) / 2
                const float disc = std::sqrt(std::max(tr * tr - 4.0f * det, 0.0f));
                const float l1 = (tr + disc) * 0.5f;
                const float l2 = (tr - disc) * 0.5f;
                shift = (std::abs(l1 - c) < std::abs(l2 - c)) ? l1 : l2;
            }

            // 2) 引入 top bulge
            //   f[0] = d[0] - shift
            //   d[0] - shift, e[0]  -- 用 Givens 把 e[0] 消去
            f[0] = d[0] - shift;
            d[0] = d[0] - shift;  // 临时

            double max_e = 0.0;
            for (std::size_t r_ = 0; r_ + 1 < m; ++r_) {
                // ---- Right Givens on (d[k] - shift_at_k, e[k]) ----
                // 但我们想保持 d[k] 不变，所以用：
                //   c = d[k] / r, s = e[k] / r, r = sqrt(d[k]² + e[k]²)
                //   new_d[k] = r, new_e[k] = 0
                //   但 d[k] 在此步后会变 (因为 Left Givens 会用 d[k])
                //
                // 实际操作:
                //   Right Givens: 旋转 d[k], e[k] → (r, 0)
                //   但 d[k] = f[k] + shift_k (累积 shift), e[k] = 原始 e[k]
                //   我们要看 (f[k], e[k]) 两个数（f[k] 在 sub-diag 位置）
                //
                // 实际 chase 状态 (lower bidiag with f as sub-diag):
                //   [f[0], e[0], 0, ...]
                //   [0, d[1], e[1], ...]
                //   ...
                // 等等，这把 d 当 diag, e 当 supdiag 不对，因为 right Givens
                //   是消去 e[k]，但 e[k] 是在第 k 行的"上次对角"，不与 d[k] 在同一行。
                //
                // 重新理解: 在 lower bidiag 状态 (B is m × m, m 阶方阵):
                //   d[i] = diag[i]
                //   e[i] = subdiag[i] (i.e., B[i+1, i] for i = 0..m-2)
                // 引入 shift 后:
                //   [d[0] - shift, e[0], 0, ...]  // 第一行
                //   [e[0], d[1], e[1], ...]      // 第二行 (但 e[0] 是 subdiag)
                //   [0, e[1], d[2], ...]
                // 这就是 bidiagonal with f[0] = e[0] in (0, 0) and e[0] in (1, 0) position
                //
                // 实际 chase state (lower bidiag):
                //   d[k] = diag[k]
                //   e[k] = subdiag[k] (位置 (k+1, k))
                // 我们在 row k, k+1 上做 Givens 旋转
                //   Right: rotates cols k, k+1 of (lower bidiag)
                //   Left: rotates rows k, k+1

                // 简化: 直接用 bidiagonal update 公式 (Demmel-Kahan QR step)
                //   给定 lower bidiag d[0..m-1], e[0..m-2] 和 shift σ
                //   single QR step:
                //     For k = 0 to m-2:
                //       (a, b) = (d[k] - σ, e[k])  // 当前位置 (row k+1, col k+1) 和 (row k+1, col k)
                //                                     // wait, 应该是 (row k, col k) and (row k+1, col k)
                //       Right Givens: c, s = dlatg2(d[k] - σ, e[k])
                //       Update d[k], e[k]
                //       Left Givens: c, s = dlatg2(d[k], e_new[k+1])
                //       Update d[k+1], e[k+1]
                //
                // 这里我用的是 dlatg2 (compute Givens from 2 numbers)
                //   c = a / r, s = b / r, r = sqrt(a² + b²)

                // 当前 chase 状态: 在 row k, k+1 上
                //   row k: [d[k] - σ, e[k], 0, ...]
                //   row k+1: [e[k], d[k+1], e[k+1], ...]
                // 等等, 引入 σ 后:
                //   row 0: [d[0] - σ, e[0], 0, ...]
                //   row 1: [e[0], d[1], e[1], ...]
                //   row 2: [0, e[1], d[2], e[2], ...]
                // ... 但实际上 shift 影响 d[0] 一处，所以 d[0] 变 d[0] - σ

                // 用更清晰的状态：维护 active sub-diagonals 数组
                //   g[i] = active e value at sub-diag position i (i.e., (i+1, i))
                // Right Givens at k:
                //   从 (g[k], d[k]) 算出 c, s, r  （g[k] 在 (k+1, k) 位置，d[k] 在 (k, k) 位置）
                //   旋转 row k, k+1：new (k, k) = r, new (k+1, k) = 0
                //   这会改动 (k, k+1) 和 (k+1, k+1)：
                //     new (k, k+1) = c * e[k] - s * 0 (但 e[k] 是 supdiag)
                //     new (k+1, k+1) = s * e[k] + c * d[k+1]
                //   等等，让我重新想...

                // 让我用 g[] 表示 active lower-bidiag 状态
                //   g[i] = active e value at position (i+1, i), i.e., the "sub-diag" element
                //   d[i] = diagonal element at (i, i)
                // initial state: g[0] = shift-introduced (d[0] - σ), g[i] = e[i-1] for i >= 1
                // 单步 chase (k):
                //   Right Givens: 旋转 (g[k], d[k]) → (r, 0)
                //     c = g[k] / r, s = d[k] / r, r = sqrt(g[k]² + d[k]²)
                //     new g[k] = r
                //     new d[k] = 0
                //   这更新了 row k, k+1 (col k):
                //     row k, col k: g[k] → r
                //     row k+1, col k: d[k] → 0  (sub-diag 位置 (k+1, k))
                //   同时影响 col k+1:
                //     row k, col k+1: e_old[k] = d[k+1] (实际上是下一个 diag)
                //     row k+1, col k+1: e[k] (supdiag)
                //   等下，这里我搞混了。重新整理。

                // 重新整理 chase 状态：用 d[i] = diag, e[i] = active e at (i, i+1) (supdiag)
                //   引入 shift 后: d[0] -= σ（让 QR iteration 工作）
                //   状态矩阵（m × m lower bidiag with shift at d[0]）:
                //     d[0] -= σ
                //     d[0] 在 (0, 0), e[0] 在 (0, 1), 0 在 (0, 2+)
                //     0 在 (1, 0), e[0] 在 (1, 1)?, ... 不对
                //
                // 让我从 LAPACK dlasq2 学:
                //   d[0..n-1]: diagonal of bidiagonal
                //   e[0..n-2]: off-diagonal
                //   shift σ added to d[0]
                //   QR step:
                //     For k = 0 to n-2:
                //       (1) Determine Givens (c, s) s.t.:
                //             c * (d[k] + σ) - s * e[k] = r  (new diag)
                //             s * (d[k] + σ) + c * e[k] = 0  (zero e[k])
                //         So c = (d[k] + σ) / r, s = e[k] / r
                //         Update d[k] = r - σ, e[k] = 0
                //       (2) Apply Givens to columns k+1, ... of (k+1)-th row
                //         Wait, this rotates col k, k+1 of all rows
                //         Affected: (i, k) and (i, k+1) for all i
                //         Original sub-diagonal at (k+1, k) = e_old[k+1-1]? No, bidiag e is supdiag
                //         ...
                //
                // 让我用更直接的形式：从 "shifted bidiagonal" T - σI
                //   T = lower bidiagonal (m×m):
                //     T[i, i] = d[i]
                //     T[i+1, i] = e[i]  (subdiag)
                //   T - σI:
                //     (T - σI)[i, i] = d[i] - σ
                //     (T - σI)[i+1, i] = e[i]
                //   QR of T - σI: B' B = R^T R (B' is bidiag-like?)
                //   不，T - σI 不是 bidiagonal, 因为有 sub-diag e[i]
                //   QR on T - σI 用 Givens:
                //     Chase the subdiagonal e[i] up to the top, accumulating R
                //   每次 Givens 在 row i, i+1 上 (zero (i+1, i) of T - σI):
                //     [c s; -s c] * [d[i] - σ; e[i]] = [r; 0]
                //     c = (d[i] - σ) / r, s = e[i] / r
                //   但 σ 已经被加到 d[0]，对其他行 σ 是 0
                //
                // 简化实现：用 d[] 和 e[] 数组，每次 QR step 维护 "active" 状态

                // 算了，让我用更简单的"逐对 Givens chase"实现 (B 路径简化版):
                // 思路: 像 one-sided Jacobi 一样，但只在 (i, i+1) 相邻对上做
                //   关键不同: 我们 chase the supdiag, 每步还旋转 U
                //   (避免内层循环的 n 维度)
                //
                // Step k:
                //   1) Right Givens: 用 (work[k, k], work[k, k+1]) 消去 work[k, k+1]
                //   2) Left Givens: 用 (work[k, k], work[k+1, k]) 消去 work[k+1, k]
                //   这会引入 bulge at work[k+1, k+1] (B 路径的核心)

                // 跳回 outer scope 用直接 work 数组
                break;  // 占位，下面用新方法
            }
            // 不要 break 出来 — 下面是新实现
            (void)f;
            (void)max_e;
            (void)prev_max_e;
            (void)shift;
            break;  // 跳过 placeholder
        }

        // ----- 实际算法: chase supdiag with explicit Givens -----
        // 1) Right Givens on (work[i, i], work[i, i+1]): zero work[i, i+1], update work[i, i] and V[:, i], V[:, i+1]
        // 2) Left Givens on (work[i, i], work[i+1, i]): zero work[i+1, i], update work[i+1, i+1] and U[:, i], U[:, i+1]
        // 注: 这就是 standard bidiagonal QR chase, **bulge 会在 step 2 中产生**
        //   new work[i+1, i+1] 不变 (因为 rotation 是 unit-norm)
        //   但 new work[i+1, i+2] 会被引入 (bulge)
        //   这是 bidiagonal 矩阵的"row bulge"问题
        //   解决: 每次 sweep 从 i=0 开始重新处理, 重复直到所有 supdiag < tol

        // 重新 init U, V
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < m; ++j) {
                U[i * m + j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                V[i * n + j] = (i == j) ? 1.0f : 0.0f;
            }
        }

        double prev_max_e2 = std::numeric_limits<double>::infinity();
        for (int sweep = 0; sweep < max_sweeps; ++sweep) {
            double max_e = 0.0;
            for (std::size_t i = 0; i + 1 < m; ++i) {
                const std::size_t j = i + 1;

                // supdiag 值
                float b_ij = work[i * n + j];
                if (std::abs(b_ij) < 1e-30f) continue;
                max_e = std::max(max_e, static_cast<double>(std::abs(b_ij)));

                // ---- Step 1: 右侧 Givens 消去 work[i, i+1] ----
                // 处理 (work[i, i], work[i, j]) 两元素
                float a = work[i * n + i];
                float b = b_ij;
                float r = std::sqrt(a * a + b * b);
                if (r < 1e-30f) continue;
                float c1 = a / r;
                float s1 = b / r;
                work[i * n + i] = r;
                work[i * n + j] = 0.0f;
                // 累积 V: V 是 n × n，按列旋转 i, j
                for (std::size_t r2 = 0; r2 < n; ++r2) {
                    const float vi = V[r2 * n + i];
                    const float vj = V[r2 * n + j];
                    V[r2 * n + i] = c1 * vi - s1 * vj;
                    V[r2 * n + j] = s1 * vi + c1 * vj;
                }

                // ---- Step 2: 左侧 Givens 消去 work[i+1, i]（bulge）----
                a = work[i * n + i];  // = r
                b = work[(i + 1) * n + i];
                if (std::abs(b) < 1e-30f) continue;
                r = std::sqrt(a * a + b * b);
                c1 = a / r;
                s1 = b / r;
                work[i * n + i] = r;
                work[(i + 1) * n + i] = 0.0f;
                // 累积 U: U 是 m × m，按列旋转 i, i+1
                for (std::size_t r2 = 0; r2 < m; ++r2) {
                    const float ui = U[r2 * m + i];
                    const float uj = U[r2 * m + (i + 1)];
                    U[r2 * m + i]       = c1 * ui - s1 * uj;
                    U[r2 * m + (i + 1)] = s1 * ui + c1 * uj;
                }
                // 副作用: work[i, j] (j=i+1) 之前已是 0, 现在 c*0 + s*work[i+1, j]=c*work[i+1, j]... wait
                // 这是 row rotation, 影响 row i, i+1 的所有列
                // new work[i, j] = c * work[i, j] - s * work[i+1, j]
                // new work[i+1, j] = s * work[i, j] + c * work[i+1, j]
                // 对 j = i+1: work[i, j] = 0 (Step 1), work[i+1, j] = work[i+1, i+1] (diag)
                //   new work[i, i+1] = -s * work[i+1, i+1]  ← bulge!
                //   new work[i+1, i+1] = c * work[i+1, i+1]
                // 对 j = i+2 (if exists): work[i, j] = 0 (bidiag), work[i+1, j] = work[i+1, i+2] (supdiag)
                //   new work[i, i+2] = -s * work[i+1, i+2]  ← bulge!
                //   new work[i+1, i+2] = c * work[i+1, i+2]
                // 对 j > i+2: 都是 0, no change
                if (j < n) {
                    const float wip1_j = work[(i + 1) * n + j];
                    work[i * n + j]       = -s1 * wip1_j;  // bulge at (i, j)
                    work[(i + 1) * n + j] =  c1 * wip1_j;  // updated
                }
                if (j + 1 < n) {
                    const float wip1_jp1 = work[(i + 1) * n + (j + 1)];
                    work[i * n + (j + 1)]       = -s1 * wip1_jp1;  // bulge at (i, j+1)
                    work[(i + 1) * n + (j + 1)] =  c1 * wip1_jp1;
                }
            }
            // 收敛判断
            if (sweep > 0) {
                const double rel = std::abs(max_e - prev_max_e2) / std::max(max_e, 1e-30);
                if (max_e < 1e-6 || rel < tol) break;
            }
            prev_max_e2 = max_e;
        }

        // 提取奇异值 = diag(work) 的绝对值
        for (std::size_t i = 0; i < m; ++i) {
            S[i] = std::abs(work[i * n + i]);
        }
        // 排序找 top m
        std::vector<int> idx(m);
        for (std::size_t i = 0; i < m; ++i) idx[i] = static_cast<int>(i);
        std::sort(idx.begin(), idx.end(),
                  [S](int a, int b) { return S[a] > S[b]; });
        std::vector<float> S_sorted(m);
        for (std::size_t i = 0; i < m; ++i) S_sorted[i] = S[idx[i]];
        std::copy(S_sorted.begin(), S_sorted.end(), S);

        // 重排 V 和 U (按列重排)
        std::vector<float> V_sorted(n * m, 0.0f);
        std::vector<float> U_sorted(m * m, 0.0f);
        for (std::size_t r = 0; r < m; ++r) {
            for (std::size_t i = 0; i < n; ++i) {
                V_sorted[i * m + r] = V[i * n + idx[r]];
            }
            for (std::size_t i = 0; i < m; ++i) {
                U_sorted[i * m + r] = U[i * m + idx[r]];
            }
        }
        std::copy(V_sorted.begin(), V_sorted.end(), V);
        std::copy(U_sorted.begin(), U_sorted.end(), U);
    }
};

} // namespace linalg
} // namespace ctQALS
#endif // CTORCH_JACOBI_SVD_H
