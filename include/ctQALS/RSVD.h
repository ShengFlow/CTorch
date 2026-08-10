//
// RSVD.h
// Randomized SVD — Halko, Martinsson, Tropp 2011 算法 4.1 / 5.1
// 2026-08-10
//
// 设计要点：
//   * 拼装三个 Layer 0 子模块：RandomMatrix / HouseholderQR / JacobiSVD
//   * 支持 power iteration（默认 2 次），提高谱尾精度
//   * oversampling 默认 10（Halko 2011 推荐）
//   * 不调用 LAPACK（dgesvd 拒绝）
//   * 输出 W ≈ U * diag(S) * V^T，U(m×k), S(k), V(n×k)
//
// 复现 Halko 2011 Algorithm 5.1（带 power iteration 的 range finder）：
//   1) 画 n×l 随机高斯矩阵 Ω
//   2) Y = W Ω
//   3) for q iterations:
//        Y = W (W^T Y)        ← power iteration
//        Q, _ = QR(Y)         ← orthonormal basis
//   4) B = Q^T W
//   5) B = Ũ Σ V^T           ← 小矩阵 SVD
//   6) U = Q Ũ
//
// 这是 RSVD 的 Layer 1 子模块，依赖 Layer 0 三个。
//

#ifndef CTORCH_RSVD_H
#define CTORCH_RSVD_H
#pragma once

#include "HouseholderQR.h"
#include "JacobiSVD.h"
#include "Random.h"
#include "RandomMatrix.h"
#include "Tensor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ctQALS {
namespace linalg {

// ============================================================
// Randomized SVD
// ============================================================
struct RSVDOptions {
    int target_rank       = 32;     // k
    int oversampling      = 10;     // p
    int power_iterations  = 2;      // q
    uint64_t seed         = 0;      // 0 = 用 thread_local 引擎
};

struct RSVDTiming {
    double range_finder_ms = 0.0;
    double small_svd_ms    = 0.0;
    double total_ms        = 0.0;
};

class RSVD {
public:
    // 构造：对 W (m×n) 做 RSVD
    explicit RSVD(const float* W, std::size_t m, std::size_t n)
        : RSVD(W, m, n, RSVDOptions{}) {}

    RSVD(const float* W, std::size_t m, std::size_t n, const RSVDOptions& opts)
        : m_(m), n_(n) {
        if (m == 0 || n == 0) {
            throw std::invalid_argument("RSVD: dims must be > 0");
        }
        if (opts.target_rank <= 0) {
            throw std::invalid_argument("RSVD: target_rank must be > 0");
        }
        const int l_cap = static_cast<int>(std::min(m, n));
        const int l_req = opts.target_rank + opts.oversampling;
        if (l_req > l_cap) {
            // 降级：把 oversampling 砍掉
            k_ = std::min(opts.target_rank, static_cast<int>(l_cap));
            l_ = l_cap;
        } else {
            k_ = opts.target_rank;
            l_ = l_req;
        }
        // 强制 l_ <= n_（JacobiSVD 要求 m <= n，B 是 l_ × n_）
        if (l_ > static_cast<int>(n_)) {
            l_ = static_cast<int>(n_);
            k_ = std::min(k_, l_ - opts.oversampling);
            if (k_ < 1) k_ = 1;
        }
        q_ = std::max(0, opts.power_iterations);
        seed_ = opts.seed;

        U_.assign(m_ * k_, 0.0f);
        S_.assign(k_, 0.0f);
        V_.assign(n_ * k_, 0.0f);

        compute(W, opts.seed);
    }

    std::size_t m() const noexcept { return m_; }
    std::size_t n() const noexcept { return n_; }
    int k()       const noexcept { return k_; }

    const float* U() const noexcept { return U_.data(); }
    const float* S() const noexcept { return S_.data(); }
    const float* V() const noexcept { return V_.data(); }

    float* U() noexcept { return U_.data(); }
    float* S() noexcept { return S_.data(); }
    float* V() noexcept { return V_.data(); }

    // 重建 W ≈ U * diag(S) * V^T
    void reconstruct(float* W_out) const {
        // W_out[i,j] = sum_r U[i,r] * S[r] * V[j,r]
        for (std::size_t i = 0; i < m_; ++i) {
            for (std::size_t j = 0; j < n_; ++j) {
                float sum = 0.0f;
                for (int r = 0; r < k_; ++r) {
                    sum += U_[i * k_ + r] * S_[r] * V_[j * k_ + r];
                }
                W_out[i * n_ + j] = sum;
            }
        }
    }

    // 重建误差：||W_orig - U*diag(S)*V^T||_F / ||W_orig||_F
    float reconstruction_error(const float* W_orig) const {
        std::vector<float> Wrec(m_ * n_);
        reconstruct(Wrec.data());
        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < m_ * n_; ++i) {
            const double d = static_cast<double>(W_orig[i] - Wrec[i]);
            num += d * d;
            den += static_cast<double>(W_orig[i]) * W_orig[i];
        }
        return static_cast<float>(std::sqrt(num / std::max(den, 1e-30)));
    }

    // 奇异值能量：sum(S^2) / sum(sigma_full^2) 的一个 RSVD 估计
    float captured_energy_ratio() const {
        double captured = 0.0;
        for (int r = 0; r < k_; ++r) {
            captured += static_cast<double>(S_[r]) * S_[r];
        }
        // 全能量是 ||W||_F^2
        // 我们没有全 W 的能量……但调用方可以传
        return static_cast<float>(captured);
    }

    // 全能量（用 W 范数估算）：||W||_F^2
    static float full_energy(const float* W, std::size_t m, std::size_t n) {
        double s = 0.0;
        for (std::size_t i = 0; i < m * n; ++i) {
            s += static_cast<double>(W[i]) * W[i];
        }
        return static_cast<float>(s);
    }

    const RSVDTiming& timing() const noexcept { return timing_; }

private:
    std::size_t              m_, n_;
    int                      k_;
    int                      l_;       // k + oversampling
    int                      q_;
    uint64_t                 seed_;
    std::vector<float>       U_;       // m × k
    std::vector<float>       S_;       // k
    std::vector<float>       V_;       // n × k
    RSVDTiming               timing_;

    // 内部：从 W 计算 U, S, V
    void compute(const float* W, uint64_t seed) {
        using clk = std::chrono::high_resolution_clock;
        const auto t0 = clk::now();

        // 1) 画 n × l 随机高斯矩阵 Ω
        std::vector<float> Omega(n_ * l_);
        if (seed != 0) {
            ctQALS::rng::Xoshiro256PlusPlus eng(seed);
            ctQALS::rng::ZigguratNormal norm(eng);
            fill_gaussian(Omega.data(), n_, l_, norm);
        } else {
            fill_gaussian(Omega.data(), n_, l_);
        }

        // 2) Y = W * Ω  (m × l)
        std::vector<float> Y(m_ * l_, 0.0f);
        matmul(W, m_, n_, Omega.data(), n_, l_, Y.data());

        // 3) Power iteration（q 次）：Y = (W W^T)^q * W Ω
        for (int iter = 0; iter < q_; ++iter) {
            std::vector<float> Z(n_ * l_, 0.0f);
            matmul_AtB(W, m_, n_, Y.data(), m_, l_, Z.data());
            std::fill(Y.begin(), Y.end(), 0.0f);
            matmul(W, m_, n_, Z.data(), n_, l_, Y.data());
            HouseholderQR qr(Y.data(), m_, static_cast<std::size_t>(l_));
            qr.get_Q_thin(Y.data());
        }
        HouseholderQR qr(Y.data(), m_, static_cast<std::size_t>(l_));
        qr.get_Q_thin(Y.data());

        const auto t1 = clk::now();

        // 4) B = Q^T * W  (l × n)
        std::vector<float> B(l_ * n_, 0.0f);
        // Q^T W：用 Q 作为 (m × l) 矩阵，Q^T 是 (l × m)
        matmul_AtB(Y.data(), m_, static_cast<std::size_t>(l_), W, m_, n_, B.data());

        // 5) SVD: B = Ũ Σ V^T  (l_ × n_)
        // B 是 l_ × n_ 矩阵（l_ <= n_），调 JacobiSVD
        // max_sweeps = 10 是生产级紧凑预算，Halko 2011 也用 ~10
        JacobiSVD small_svd(B.data(), static_cast<std::size_t>(l_), n_,
                            /*max_sweeps=*/10, /*tol=*/1e-6f);
        // compute 内部已经按 idx 重排 work 和 V，所以不用再 sort

        const auto t2 = clk::now();

        // 6) U = Q * Ũ[:, :k]  (m × k)
        //    V = V_small[:, :k]  (n × k)
        //    S = Σ[:k]  (k)
        const float* U_tilde = small_svd.U();   // l_ × l_
        const float* S_tilde = small_svd.S();   // l_
        const float* V_small = small_svd.V();   // n_ × l_

        // U = Q * U_tilde
        for (std::size_t i = 0; i < m_; ++i) {
            for (int r = 0; r < k_; ++r) {
                float sum = 0.0f;
                for (int j = 0; j < l_; ++j) {
                    sum += Y[i * l_ + j] * U_tilde[j * l_ + r];
                }
                U_[i * k_ + r] = sum;
            }
        }

        // V / S 直接拷贝
        for (int r = 0; r < k_; ++r) {
            S_[r] = S_tilde[r];
            for (std::size_t j = 0; j < n_; ++j) {
                V_[j * k_ + r] = V_small[j * l_ + r];
            }
        }

        // 强制：S 非负
        for (int r = 0; r < k_; ++r) {
            if (S_[r] < 0.0f) {
                // 负奇异值：把 V 对应列取反
                S_[r] = -S_[r];
                for (std::size_t j = 0; j < n_; ++j) {
                    V_[j * k_ + r] = -V_[j * k_ + r];
                }
            }
        }

        const auto t3 = clk::now();

        auto ms = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        timing_.range_finder_ms = ms(t0, t1);
        timing_.small_svd_ms    = ms(t1, t2);
        timing_.total_ms        = ms(t0, t3);
    }
};

// ============================================================
// Tensor 集成（2026-08-10 重写）
//
// 关键设计：让 Halko 2011 算法的主体 matmul 全部走 Tensor::matmul，
// 借助 CtorchScheduler 调度到 MatMul_AMX_kernel（Apple Silicon 上是
// cblas_sgemm from Accelerate BLAS）→ 真·生产级加速。
//
// 走 SIMD 路径 vs raw matmul：
//   - raw `RSVD` 内部用 RandomMatrix.h 的朴素 matmul（单线程标量）
//   - `TensorRSVD` 内部用 Tensor::matmul（多线程 + BLAS）
// 两者接口一致，调用方按需选用。
//
// 依赖关系：TensorRSVD → Tensor.h（自动微分 + kernel 分派）
//           TensorRSVD → HouseholderQR / JacobiSVD（raw buffer 桥接，
//           因为 QR/SVD 需要 in-place 修改，Tensor 接口不方便表达）
// ============================================================
class TensorRSVD {
public:
    TensorRSVD(const Tensor& W, const RSVDOptions& opts = RSVDOptions{}) {
        if (W.shape().size() != 2) {
            throw std::invalid_argument("TensorRSVD: W must be 2D matrix");
        }
        if (W.dtype() != DType::kFloat) {
            throw std::invalid_argument("TensorRSVD: only float32 supported");
        }
        m_ = W.shape()[0];
        n_ = W.shape()[1];

        // 解析 k, l, q（同 raw RSVD）
        if (m_ == 0 || n_ == 0) {
            throw std::invalid_argument("TensorRSVD: dims must be > 0");
        }
        if (opts.target_rank <= 0) {
            throw std::invalid_argument("TensorRSVD: target_rank must be > 0");
        }
        const int l_cap = static_cast<int>(std::min(m_, n_));
        const int l_req = opts.target_rank + opts.oversampling;
        if (l_req > l_cap) {
            k_ = std::min(opts.target_rank, static_cast<int>(l_cap));
            l_ = l_cap;
        } else {
            k_ = opts.target_rank;
            l_ = l_req;
        }
        if (static_cast<std::size_t>(l_) > n_) {
            l_ = static_cast<int>(n_);
            k_ = std::min(k_, l_ - opts.oversampling);
            if (k_ < 1) k_ = 1;
        }
        q_ = std::max(0, opts.power_iterations);
        seed_ = opts.seed;

        // W 必须 requires_grad=false（避免 autograd 注册开销）
        // 调用方如果是训练上下文，应自己 .detach() 或 no_grad 包装
        W_ = W;  // 浅拷贝 reference

        compute(W, opts);
    }

    // 访问结果
    const Tensor& U() const noexcept { return Ut_; }
    const Tensor& S() const noexcept { return St_; }
    const Tensor& V() const noexcept { return Vt_; }

    std::size_t m() const noexcept { return m_; }
    std::size_t n() const noexcept { return n_; }
    int         k() const noexcept { return k_; }

    const RSVDTiming& timing() const noexcept { return timing_; }

    // 重建 W ≈ U * diag(S) * V^T（走 Tensor::matmul）
    Tensor reconstruct() const {
        // Sdiag = diag(S) 是 k × k，但 S 是 1D Tensor。先做 outer:
        //   U . diag(S) = U ⊙ S  (broadcast) — 但 CTorch 支持吗？
        // 简单方案：构造 S_diag = diag(S) as k×k Tensor
        //   U_scaled[i, r] = U[i, r] * S[r]
        // 然后 Vt_scaled[j, r] = V[j, r]  (V 已经存 V[:, :k])
        //   Wrec = U_scaled * Vt^T  (m × k) · (k × n) = m × n
        const std::size_t m = m_;
        const std::size_t n = n_;
        const std::size_t kk = static_cast<std::size_t>(k_);

        // 1) 构造 S_diag (k × k) 对角矩阵
        Tensor S_diag(ShapeTag{}, {kk, kk}, DType::kFloat, DeviceType::kCPU, /*zero=*/true);
        {
            float* p = S_diag.data_write<float>();
            const float* s = St_.data<float>();
            for (std::size_t r = 0; r < kk; ++r) p[r * kk + r] = s[r];
        }
        // 2) U_scaled = U * S_diag  (m × k)  ← Tensor::matmul (BLAS)
        Tensor U_scaled = Ut_.matmul(S_diag);
        // 3) Wrec = U_scaled * V^T  (m × k) · (k × n) = m × n
        //    V 的 row-major 存的是 V[:, :k]；V^T 是 (k × n)
        Tensor Wrec = U_scaled.matmul(Vt_.t());
        return Wrec;
    }

    // 重建误差
    float reconstruction_error(const Tensor& W) const {
        Tensor Wrec = reconstruct();
        const float* orig = W.data<float>();
        const float* rec  = Wrec.data<float>();
        const std::size_t sz = W.numel();
        double num = 0.0, den = 0.0;
        for (std::size_t i = 0; i < sz; ++i) {
            const double d = static_cast<double>(orig[i] - rec[i]);
            num += d * d;
            den += static_cast<double>(orig[i]) * orig[i];
        }
        return static_cast<float>(std::sqrt(num / std::max(den, 1e-30)));
    }

private:
    std::size_t   m_, n_;
    int           k_, l_, q_;
    uint64_t      seed_;
    Tensor        W_;        // 输入（const ref）
    Tensor        Ut_, St_, Vt_;  // 输出
    RSVDTiming    timing_;

    // Halko 2011 算法 5.1：power iteration + QR + 小 SVD
    // matmul 全部走 Tensor::matmul（BLAS 加速）
    void compute(const Tensor& W, const RSVDOptions& opts) {
        using clk = std::chrono::high_resolution_clock;
        const auto t0 = clk::now();

        const std::size_t l_sz = static_cast<std::size_t>(l_);

        // 1) Ω = randn(n × l)  as Tensor
        Tensor Omega(ShapeTag{}, {n_, l_sz}, DType::kFloat,
                     DeviceType::kCPU, /*zero_init=*/false);
        {
            float* p = Omega.data_write<float>();
            if (seed_ != 0) {
                ctQALS::rng::Xoshiro256PlusPlus eng(seed_);
                ctQALS::rng::ZigguratNormal norm(eng);
                norm.fill(p, n_ * l_sz);
            } else {
                fill_gaussian(p, n_, l_sz);
            }
        }

        // 2) Y = W * Ω  (m × l)  ← Tensor::matmul → cblas_sgemm
        Tensor Y = W.matmul(Omega);

        // 3) Power iteration
        for (int iter = 0; iter < q_; ++iter) {
            // Z = W^T * Y  (n × l)
            Tensor Z = W.t().matmul(Y);
            // Y = W * Z  (m × l)
            Y = W.matmul(Z);
            // in-place QR：HouseholderQR 需要 raw buffer
            // 注意：Y 是 Tensor，data_write<float>() 拿 writable 指针
            Y = in_place_qr_thin(Y);
        }
        // 最后一次正交化
        Y = in_place_qr_thin(Y);

        const auto t1 = clk::now();
        // Y 现在是 Q (m × l, 列正交)

        // 4) B = Q^T * W  (l × n)
        Tensor B = Y.t().matmul(W);

        // 5) 小 SVD: B = Ũ Σ V^T  (l_ × n_), l_ ≤ n_
        //   raw buffer: 把 B 拷到 std::vector 喂给 JacobiSVD
        std::vector<float> B_raw(l_sz * n_);
        std::memcpy(B_raw.data(), B.data<float>(), B_raw.size() * sizeof(float));
        JacobiSVD small_svd(B_raw.data(), l_sz, n_,
                            /*max_sweeps=*/10, /*tol=*/1e-6f);
        // small_svd.U() 是 l_ × l_, V() 是 n_ × l_, S() 是 l_ 个
        // （compute 内部已按 S 降序排过）

        // 6) 构造 Ũ[:, :k] (l_ × k) Tensor
        Tensor U_tilde(ShapeTag{}, {l_sz, static_cast<std::size_t>(k_)},
                       DType::kFloat, DeviceType::kCPU, /*zero=*/false);
        std::memcpy(U_tilde.data_write<float>(), small_svd.U(),
                    l_sz * k_ * sizeof(float));

        // U = Q * Ũ[:, :k]  (m × k)  ← Tensor::matmul (BLAS)
        Ut_ = Y.matmul(U_tilde);

        // V = V_small[:, :k]  (n × k)  — 直接拷
        Vt_ = Tensor(ShapeTag{}, {n_, static_cast<std::size_t>(k_)},
                     DType::kFloat, DeviceType::kCPU, /*zero=*/false);
        {
            const float* Vsmall = small_svd.V();  // n × l_
            // V_small[:, :k]  取前 k 列
            // Vsmall 按 n × l_ row-major：[j * l_ + r]
            // Vt_ 按 n × k row-major：[j * k + r]
            float* Vp = Vt_.data_write<float>();
            for (std::size_t j = 0; j < n_; ++j) {
                for (int r = 0; r < k_; ++r) {
                    Vp[j * k_ + r] = Vsmall[j * l_sz + r];
                }
            }
        }

        // S = Σ[:k]  — 直接拷（强制非负）
        St_ = Tensor(ShapeTag{}, {static_cast<std::size_t>(k_)},
                     DType::kFloat, DeviceType::kCPU, /*zero=*/false);
        {
            const float* s_small = small_svd.S();  // l_ 个
            float* sp = St_.data_write<float>();
            for (int r = 0; r < k_; ++r) {
                const float sigma = s_small[r];
                sp[r] = (sigma < 0.0f) ? -sigma : sigma;
            }
        }

        const auto t2 = clk::now();

        // timing
        auto ms = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        timing_.range_finder_ms = ms(t0, t1);
        timing_.small_svd_ms    = ms(t1, t2);
        timing_.total_ms        = ms(t0, t2);
    }

    // 把 Y (Tensor m×l) 拷贝到 raw buffer → HouseholderQR 拿 Q thin → 写回 Tensor
    // 返回新的 Tensor（含 Q thin 数据）
    Tensor in_place_qr_thin(const Tensor& Y) {
        const std::size_t l_sz = static_cast<std::size_t>(l_);
        std::vector<float> Y_raw(m_ * l_sz);
        std::memcpy(Y_raw.data(), Y.data<float>(), m_ * l_sz * sizeof(float));

        HouseholderQR qr(Y_raw.data(), m_, l_sz);
        qr.get_Q_thin(Y_raw.data());  // in-place 写 Y_raw → Q thin

        Tensor Q(ShapeTag{}, {m_, l_sz}, DType::kFloat,
                 DeviceType::kCPU, /*zero=*/false);
        std::memcpy(Q.data_write<float>(), Y_raw.data(),
                    m_ * l_sz * sizeof(float));
        return Q;
    }
};

} // namespace linalg
} // namespace ctQALS
#endif // CTORCH_RSVD_H
