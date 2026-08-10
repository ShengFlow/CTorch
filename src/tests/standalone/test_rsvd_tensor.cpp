//
// test_rsvd_tensor.cpp
// RSVD Tensor 集成测试（链接 CTorch → Tensor::matmul 走 cblas_sgemm）
// 2026-08-10
//
// 目标：验证 TensorRSVD 数值正确性 + 对比 raw 路径性能
//
// 编译（CMake 已注册）：
//   cd build-simd-integration && cmake --build . --target test_rsvd_tensor
//   ./test_rsvd_tensor
//
// 关键性能数据点：
//   - 1024×1024 k=32：raw 40s（手写朴素 matmul）vs Tensor ?s（BLAS）
//   - 256×256 k=16：   raw 175ms  vs Tensor ?ms
//   - 1024² rank 8：   raw <1s    vs Tensor ?ms
//

#include "ctQALS/RSVD.h"
#include "ctQALS/RandomMatrix.h"
#include "ctQALS/HouseholderQR.h"
#include "ctQALS/JacobiSVD.h"

#include "Tensor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace ctQALS::linalg;
using clk = std::chrono::high_resolution_clock;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        clk::now().time_since_epoch()).count();
}

#define CHECK(cond, msg) do {                                              \
    if (!(cond)) {                                                         \
        std::fprintf(stderr, "[FAIL] %s  (line %d)\n", msg, __LINE__);     \
        return 1;                                                          \
    }                                                                      \
} while (0)

// ============================================================
// 准备：标准正态 W (m × n) as Tensor
// ============================================================
static Tensor make_W_tensor(std::size_t m, std::size_t n, uint64_t seed = 42) {
    Tensor W(ShapeTag{}, {m, n}, DType::kFloat, DeviceType::kCPU, /*zero=*/false);
    ctQALS::rng::Xoshiro256PlusPlus eng(seed);
    ctQALS::rng::ZigguratNormal norm(eng);
    norm.fill(W.data_write<float>(), m * n);
    return W;
}

// ============================================================
// 准备：标准正态 W (m × n) as raw buffer (用 MatrixXf)
// ============================================================
static std::vector<float> make_W_raw(std::size_t m, std::size_t n, uint64_t seed = 42) {
    MatrixXf Mmat = randn(m, n);
    std::vector<float> W(Mmat.data(), Mmat.data() + m * n);
    return W;
}

// ============================================================
// 重建误差辅助
// ============================================================
static float relerr(const float* a, const float* b, std::size_t n) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(a[i] - b[i]);
        num += d * d;
        den += static_cast<double>(a[i]) * a[i];
    }
    return static_cast<float>(std::sqrt(num / std::max(den, 1e-30)));
}

// ============================================================
// 拷贝 raw → Tensor
// ============================================================
static Tensor raw_to_tensor(const float* p, std::size_t m, std::size_t n) {
    Tensor T(ShapeTag{}, {m, n}, DType::kFloat, DeviceType::kCPU, /*zero=*/false);
    std::memcpy(T.data_write<float>(), p, m * n * sizeof(float));
    return T;
}
static Tensor raw_to_tensor_1d(const float* p, std::size_t n) {
    Tensor T(ShapeTag{}, {n}, DType::kFloat, DeviceType::kCPU, /*zero=*/false);
    std::memcpy(T.data_write<float>(), p, n * sizeof(float));
    return T;
}

// ============================================================
// 共享的 RSVDOptions
// ============================================================
static RSVDOptions make_opts(int k, int q = 2) {
    RSVDOptions o;
    o.target_rank      = k;
    o.oversampling     = 10;
    o.power_iterations = q;
    o.seed             = 42;
    return o;
}

// ============================================================
// Test 1: TensorRSVD 数值正确性（小规模 sanity check）
// ============================================================
int test_tensor_rsvd_correctness() {
    std::printf("\n=== Test 1: TensorRSVD correctness (256x128, k=16) ===\n");

    const std::size_t M = 256, N = 128;
    const int k = 16;

    Tensor W = make_W_tensor(M, N, 7);

    // Raw 路径
    std::vector<float> W_raw(W.data<float>(), W.data<float>() + M * N);
    auto opts = make_opts(k);
    RSVD rsvd_raw(W_raw.data(), M, N, opts);
    const float err_raw = rsvd_raw.reconstruction_error(W_raw.data());

    // Tensor 路径
    TensorRSVD trsvd(W, opts);
    Tensor Wrec = trsvd.reconstruct();
    const float err_tensor = trsvd.reconstruction_error(W);

    std::printf("  raw  recon err = %.3e\n", err_raw);
    std::printf("  tens recon err = %.3e\n", err_tensor);

    // 对比 U/S/V（raw 是 m*k, k, n*k）
    Tensor U_raw_t = raw_to_tensor(rsvd_raw.U(), M, k);
    Tensor S_raw_t = raw_to_tensor_1d(rsvd_raw.S(), k);
    Tensor V_raw_t = raw_to_tensor(rsvd_raw.V(), N, k);

    const float* Ut = trsvd.U().data<float>();
    const float* St = trsvd.S().data<float>();
    const float* Vt = trsvd.V().data<float>();
    const float* Ur = U_raw_t.data<float>();
    const float* Sr = S_raw_t.data<float>();
    const float* Vr = V_raw_t.data<float>();

    // U/V 列可能有 sign 差异（奇异向量 ± 不定）——逐列取 min(|A-B|, |A+B|)
    auto relerr_abs_col = [](const float* A, const float* B, std::size_t m,
                             std::size_t n) {
        double num = 0.0, den = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            // 对每列：累加 |a - b|^2 和 |a + b|^2，取小
            double col_pos = 0.0, col_neg = 0.0, col_den = 0.0;
            for (std::size_t i = 0; i < m; ++i) {
                const double a = A[i * n + j], b = B[i * n + j];
                const double dp = a - b, dn = a + b;
                col_pos += dp * dp;
                col_neg += dn * dn;
                col_den += a * a;
            }
            // 每列单独：误差小的那种 sign 作为该列 sign
            // 但我们想的是「整列 sign 一致」的 SVD 比较。
            // 标准做法是：每列 sign 由该列第一对元素决定（argmax |a|），
            // 翻转 B 列，再累加误差。
            // 这里采用更稳健的：每列单独取 min(col_pos, col_neg) 后加到总误差
            num += std::min(col_pos, col_neg);
            den += col_den;
        }
        return static_cast<float>(std::sqrt(num / std::max(den, 1e-30)));
    };

    const float eU = relerr_abs_col(Ur, Ut, M, k);
    const float eS = relerr(Sr, St, k);
    const float eV = relerr_abs_col(Vr, Vt, N, k);

    std::printf("  ||U_raw vs U_tens||_rel (sign-agnostic) = %.3e\n", eU);
    std::printf("  ||S_raw vs S_tens||_rel               = %.3e\n", eS);
    std::printf("  ||V_raw vs V_tens||_rel (sign-agnostic) = %.3e\n", eV);

    CHECK(err_raw < 2.0f,     "raw 重建误差爆炸");
    CHECK(err_tensor < 2.0f,  "tensor 重建误差爆炸");
    CHECK(eU < 0.5f,          "U 数值差异过大");
    CHECK(eS < 1e-2f,         "S 数值差异过大");
    CHECK(eV < 0.5f,          "V 数值差异过大");

    std::printf("  [PASS]\n");
    return 0;
}

// ============================================================
// Test 2: 性能对比：raw vs Tensor 路径
// ============================================================
int test_perf_comparison() {
    std::printf("\n=== Test 2: raw vs Tensor RSVD performance ===\n");
    std::printf("  %6s × %6s | %3s | %3s | %10s | %10s | %8s | %10s\n",
                "m", "n", "k", "q", "raw ms", "tensor ms", "speedup", "err_tens");

    struct Case { std::size_t m, n; int k; int q; };
    const std::vector<Case> cases = {
        { 128,  128,  8,  2},
        { 256,  256,  16, 2},
        { 512,  512,  32, 2},
        { 1024, 1024, 32, 2},   // ← 团队 hot path
    };

    for (const auto& c : cases) {
        std::printf("  [dbg] starting %zu x %zu k=%d ...\n", c.m, c.n, c.k);
        std::fflush(stdout);
        try {
            // 同一 W（seed 固定）跑两个路径
            Tensor W = make_W_tensor(c.m, c.n, 42);
            std::printf("  [dbg] W constructed\n"); std::fflush(stdout);
            std::vector<float> W_raw(W.data<float>(), W.data<float>() + c.m * c.n);
            auto opts = make_opts(c.k, c.q);

            // raw 路径
            const double t0 = now_ms();
            RSVD rsvd_raw(W_raw.data(), c.m, c.n, opts);
            const double t1 = now_ms();
            const double raw_ms = t1 - t0;
            std::printf("  [dbg] raw done: %.1f ms  S[0]=%.3e\n", raw_ms, rsvd_raw.S()[0]);
            std::fflush(stdout);

            // Tensor 路径（构造 + reconstruct）
            const double t2 = now_ms();
            TensorRSVD trsvd(W, opts);
            std::printf("  [dbg] TensorRSVD constructed\n"); std::fflush(stdout);
            const double t3 = now_ms();
            const double tensor_ms = t3 - t2;
            std::printf("  [dbg] TensorRSVD time: %.1f ms  S[0]=%.3e  U[0]=%.3e\n",
                        tensor_ms,
                        trsvd.S().data<float>()[0],
                        trsvd.U().data<float>()[0]);
            std::fflush(stdout);

            const float err_tens = trsvd.reconstruction_error(W);
            std::printf("  [dbg] reconstruct_error done  err=%.3e\n", err_tens);
            std::fflush(stdout);
            const double speedup = raw_ms / std::max(tensor_ms, 1e-9);

            std::printf("  %6zu × %6zu | %3d | %3d | %10.1f | %10.1f | %7.2fx | %.3e\n",
                        c.m, c.n, c.k, c.q, raw_ms, tensor_ms, speedup, err_tens);
            std::fflush(stdout);

            CHECK(err_tens < 2.0f, "Tensor 重建误差爆炸");
        } catch (const std::exception& e) {
            std::fprintf(stderr, "  [EXCEPTION] %s x %s k=%d: %s\n",
                         std::to_string(c.m).c_str(),
                         std::to_string(c.n).c_str(), c.k, e.what());
            std::fflush(stderr);
            return 1;
        } catch (...) {
            std::fprintf(stderr, "  [UNKNOWN EXCEPTION] %s x %s k=%d\n",
                         std::to_string(c.m).c_str(),
                         std::to_string(c.n).c_str(), c.k);
            std::fflush(stderr);
            return 1;
        }
    }
    std::printf("  [PASS]\n");
    return 0;
}

// ============================================================
// Test 3: 低秩矩阵恢复（关键 RSVD 用例，验证收敛质量）
// ============================================================
int test_low_rank_recovery_tensor() {
    std::printf("\n=== Test 3: Low-rank recovery (rank 8 in 512x512) via TensorRSVD ===\n");

    const std::size_t M = 512, N = 512;
    const int true_rank = 8;

    // W = U_real * V_real^T
    MatrixXf U_real = randn(M, true_rank);
    MatrixXf V_real = randn(N, true_rank);
    std::vector<float> W_buf(M * N, 0.0f);
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            float s = 0.0f;
            for (int r = 0; r < true_rank; ++r) s += U_real(i, r) * V_real(j, r);
            W_buf[i * N + j] = s;
        }
    }
    Tensor W(ShapeTag{}, {M, N}, DType::kFloat, DeviceType::kCPU, /*zero=*/false);
    std::memcpy(W.data_write<float>(), W_buf.data(), M * N * sizeof(float));

    RSVDOptions opts;
    opts.target_rank      = true_rank;
    opts.oversampling     = 10;
    opts.power_iterations = 2;
    opts.seed             = 1234;

    const double t0 = now_ms();
    TensorRSVD trsvd(W, opts);
    const double t1 = now_ms();

    const float err = trsvd.reconstruction_error(W);
    std::printf("  recon err (rank-%d) = %.3e\n", true_rank, err);
    std::printf("  TensorRSVD time     = %.1f ms\n", t1 - t0);
    std::printf("  S[0..2] = %.3f %.3f %.3f ... S[%d] = %.3e\n",
                trsvd.S().data<float>()[0],
                trsvd.S().data<float>()[1],
                trsvd.S().data<float>()[2],
                true_rank - 1,
                trsvd.S().data<float>()[true_rank - 1]);

    CHECK(err < 0.1f, "低秩恢复误差爆炸（应该 < 0.01 但留余量）");
    std::printf("  [PASS]\n");
    return 0;
}

// ============================================================
// Test 4: 验证 timing 报告 + SIMD 路径命中
// ============================================================
int test_simd_path_hit() {
    std::printf("\n=== Test 4: confirm Tensor::matmul hits BLAS path ===\n");

    // 独立测一下 Tensor::matmul 1000×1000 走 AMX/BLAS 的速度
    // 朴素 O(n^3) 应该 1-3s；BLAS 应该 < 100ms
    const std::size_t M = 1000, N = 1000, K = 1000;
    Tensor A(ShapeTag{}, {M, K}, DType::kFloat, DeviceType::kCPU, /*zero=*/false);
    Tensor B(ShapeTag{}, {K, N}, DType::kFloat, DeviceType::kCPU, /*zero=*/false);
    ctQALS::rng::Xoshiro256PlusPlus eng(1);
    ctQALS::rng::ZigguratNormal norm(eng);
    norm.fill(A.data_write<float>(), M * K);
    norm.fill(B.data_write<float>(), K * N);

    // 第一次：触发 kernel 选择 + JIT
    Tensor C1 = A.matmul(B);

    // 第二次：测时间
    const double t0 = now_ms();
    Tensor C2 = A.matmul(B);
    const double t1 = now_ms();

    const double gflops = 2.0 * M * N * K / 1e9;
    const double t_s = (t1 - t0) / 1000.0;
    std::printf("  Tensor::matmul 1000^3: %.1f ms  (%.1f GFLOP/s)\n",
                t1 - t0, gflops / t_s);

    // BLAS 期望 10+ GFLOP/s（Apple Accelerate），朴素 1-2 GFLOP/s
    CHECK(t1 - t0 < 500.0, "Tensor::matmul 应该是 BLAS 加速");
    std::printf("  [PASS] (BLAS path confirmed)\n");
    return 0;
}

// ============================================================
// main
// ============================================================
int main() {
    std::printf("====== TensorRSVD Integration Test ======\n");
    std::printf("Build: simd-integration, Apple Silicon BLAS expected\n");
    std::printf("Device: ");
#ifdef CT_ENABLE_MPS
    std::printf("MPS + AMX (BLAS)");
#else
    std::printf("AMX (BLAS) + SIMD");
#endif
    std::printf("\n");

    int rc = 0;
    rc |= test_simd_path_hit();
    rc |= test_tensor_rsvd_correctness();
    rc |= test_low_rank_recovery_tensor();
    rc |= test_perf_comparison();

    std::printf("\n=========================================\n");
    if (rc == 0) {
        std::printf("[ALL PASS] TensorRSVD 集成 OK\n");
    } else {
        std::printf("[SOME FAIL] rc = %d\n", rc);
    }
    return rc;
}
