// mini test: 验证 B 路径（bidiagonal SVD）精度
// 1) B 路径 SVD 重建误差 ||A - U·S·V^T||_F / ||A||_F (期望 ~1e-6)
// 2) RSVD 端到端低秩恢复 (true rank 8, k=8, err < 0.01)
// 3) 1024×1024 单独 SVD 性能

#include "ctQALS/JacobiSVD.h"
#include "ctQALS/RandomMatrix.h"
#include "ctQALS/Random.h"
#include "ctQALS/RSVD.h"

#include <Accelerate/Accelerate.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <chrono>

using namespace ctQALS::linalg;
using clk = std::chrono::high_resolution_clock;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static float relerr_F(const float* a, const float* b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i) {
        num += (a[i] - b[i]) * (a[i] - b[i]);
        den += a[i] * a[i];
    }
    return static_cast<float>(std::sqrt(num / std::max(den, 1e-30)));
}

int main() {
    // 测试 1: 128×256 B 路径 SVD 重建 + 数值检查
    {
        std::printf("\n=== Test 1: JacobiSVD (B 路径) 128x256 精度 (m <= n) ===\n");
        const int M = 128, N = 256;
        MatrixXf A = randn(M, N);
        std::vector<float> A_buf(A.data(), A.data() + M * N);

        auto t0 = now_ms();
        JacobiSVD svd(A_buf.data(), M, N, 12, 1e-6f);
        auto t1 = now_ms();

        // 1a) SVD 重建 ||A - U·S·V^T||_F / ||A||_F (SVD 算法应保 ~1e-6)
        const float* U = svd.U();   // M×M
        const float* V = svd.V();   // N×M (前 M 列)
        const float* S = svd.S();   // M
        std::vector<float> A_rec(M * N, 0.0f);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int r = 0; r < M; ++r) {
                    sum += U[i * M + r] * S[r] * V[j * M + r];
                }
                A_rec[i * N + j] = sum;
            }
        }
        const float svd_recon_err = relerr_F(A_buf.data(), A_rec.data(), M * N);
        std::printf("  SVD 重建误差 ||A - U·S·V^T||_F/||A||_F = %.3e  (期望 ~1e-6)\n", svd_recon_err);

        // 1b) U 正交性: ||U^T U - I_M||_F
        std::vector<float> UtU(M * M, 0.0f);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < M; ++k) sum += U[k * M + i] * U[k * M + j];
                UtU[i * M + j] = sum;
            }
        }
        double uerr = 0;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) {
                double t = (i == j) ? 1.0 : 0.0;
                double d = UtU[i * M + j] - t;
                uerr += d * d;
            }
        }
        uerr = std::sqrt(uerr);
        std::printf("  ||U^T U - I_M||_F = %.3e  (期望 < 1e-5)\n", uerr);

        // 1c) V 正交性: ||V^T V - I_M||_F (V 是 N×M, 我们只看前 M 列)
        std::vector<float> VtV(M * M, 0.0f);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < N; ++k) sum += V[k * M + i] * V[k * M + j];
                VtV[i * M + j] = sum;
            }
        }
        double verr = 0;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) {
                double t = (i == j) ? 1.0 : 0.0;
                double d = VtV[i * M + j] - t;
                verr += d * d;
            }
        }
        verr = std::sqrt(verr);
        std::printf("  ||V^T V - I_M||_F = %.3e  (期望 < 1e-5)\n", verr);

        std::printf("  S[0..7] = %.3e %.3e %.3e %.3e %.3e %.3e %.3e %.3e\n",
                    S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7]);
        // 1d) 测试不同 max_sweeps 看收敛
        std::printf("\n  [Sweep 收敛测试 128x256]\n");
        for (int sw : {5, 10, 20, 50, 100, 200}) {
            std::vector<float> A_copy = A_buf;
            auto t_a = now_ms();
            JacobiSVD svd_test(A_copy.data(), M, N, sw, 1e-9f);
            auto t_b = now_ms();

            // 重建误差
            const float* Ut = svd_test.U();
            const float* Vt2 = svd_test.V();
            const float* St = svd_test.S();
            std::vector<float> A_re(M * N, 0.0f);
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0.0f;
                    for (int r = 0; r < M; ++r) sum += Ut[i * M + r] * St[r] * Vt2[j * M + r];
                    A_re[i * N + j] = sum;
                }
            }
            float err = relerr_F(A_buf.data(), A_re.data(), M * N);
            std::printf("    sweep=%3d  time=%6.1fms  recon_err=%.3e\n", sw, t_b - t_a, err);
        }
    }

    // 测试 2: 1024×1024 B 路径 SVD 重建（性能 + 精度）
    {
        std::printf("\n=== Test 2: JacobiSVD 1024x1024 ===\n");
        const int M = 1024, N = 1024; // OK m == n
        MatrixXf A = randn(M, N);
        std::vector<float> A_buf(A.data(), A.data() + M * N);

        auto t0 = now_ms();
        JacobiSVD svd(A_buf.data(), M, N, 100, 1e-8f);  // 大 sweep 数 + 更严 tol
        auto t1 = now_ms();
        const float* U = svd.U();
        const float* V = svd.V();
        const float* S = svd.S();
        std::vector<float> A_rec(M * N, 0.0f);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int r = 0; r < M; ++r) {
                    sum += U[i * M + r] * S[r] * V[j * M + r];
                }
                A_rec[i * N + j] = sum;
            }
        }
        const float svd_recon_err = relerr_F(A_buf.data(), A_rec.data(), M * N);
        std::printf("  SVD 重建误差: %.3e  (期望 ~1e-6)\n", svd_recon_err);
        std::printf("  S[0..4] = %.3e %.3e %.3e %.3e %.3e\n", S[0], S[1], S[2], S[3], S[4]);
        std::printf("  S[%d..%d] (尾部) = %.3e %.3e\n", M-2, M-1, S[M-2], S[M-1]);
        std::printf("  JacobiSVD 1024² time: %.1f ms\n", t1 - t0);
    }

    // 测试 3: RSVD 低秩恢复 1024×1024 true rank 8
    {
        std::printf("\n=== Test 3: RSVD low-rank recovery 1024x1024 true rank 8 ===\n");
        const int M = 1024, N = 1024;
        const int true_rank = 8;

        // W = U_real * V_real^T, U_real M×k, V_real N×k
        MatrixXf U_real = randn(M, true_rank);
        MatrixXf V_real = randn(N, true_rank);
        std::vector<float> W_buf(M * N, 0.0f);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float s = 0.0f;
                for (int r = 0; r < true_rank; ++r) s += U_real(i, r) * V_real(j, r);
                W_buf[i * N + j] = s;
            }
        }

        // RSVD 截断到 k = true_rank
        RSVDOptions opts;
        opts.target_rank      = true_rank;
        opts.oversampling     = 10;
        opts.power_iterations = 2;
        opts.seed             = 42;

        auto t0 = now_ms();
        RSVD rsvd(W_buf.data(), M, N, opts);
        auto t1 = now_ms();

        const float err = rsvd.reconstruction_error(W_buf.data());
        std::printf("  RSVD 重建误差 ||W - U·S·V^T||_F/||W||_F = %.3e  (期望 < 0.01)\n", err);
        std::printf("  S[0..2] = %.3e %.3e %.3e  S[7] = %.3e\n",
                    rsvd.S()[0], rsvd.S()[1], rsvd.S()[2], rsvd.S()[7]);
        std::printf("  RSVD 1024x1024 time: %.1f ms\n", t1 - t0);

        if (err < 1e-3f) {
            std::printf("  [PASS] 精度足够\n");
        } else if (err < 0.1f) {
            std::printf("  [WARN] 精度可用（err < 0.1）\n");
        } else {
            std::printf("  [FAIL] 精度不足（err >= 0.1）\n");
        }
    }

    // 测试 4: RSVD 低秩恢复 1024×1024 true rank 32
    {
        std::printf("\n=== Test 4: RSVD low-rank recovery 1024x1024 true rank 32 ===\n");
        const int M = 1024, N = 1024;
        const int true_rank = 32;

        MatrixXf U_real = randn(M, true_rank);
        MatrixXf V_real = randn(N, true_rank);
        std::vector<float> W_buf(M * N, 0.0f);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float s = 0.0f;
                for (int r = 0; r < true_rank; ++r) s += U_real(i, r) * V_real(j, r);
                W_buf[i * N + j] = s;
            }
        }

        RSVDOptions opts;
        opts.target_rank      = true_rank;
        opts.oversampling     = 10;
        opts.power_iterations = 2;
        opts.seed             = 42;

        auto t0 = now_ms();
        RSVD rsvd(W_buf.data(), M, N, opts);
        auto t1 = now_ms();

        const float err = rsvd.reconstruction_error(W_buf.data());
        std::printf("  RSVD 重建误差 = %.3e  (期望 < 0.01)\n", err);
        std::printf("  RSVD 1024x1024 time: %.1f ms\n", t1 - t0);
    }

    return 0;
}
