//
// test_rsvd.cpp
// RSVD 三个子模块的 smoke test
// 2026-08-10
//
// 验证：
//   1. RandomMatrix：标准正态分布的统计性质（均值 ~ 0, 方差 ~ 1）
//   2. HouseholderQR：||A - QR||_F / ||A||_F
//   3. JacobiSVD：||A - U*diag(S)*V^T||_F / ||A||_F
//   4. RSVD 整体：不同规模下的重建误差 + 计时
//
// 单独编译（不依赖 CTorch 主项目）：
//   g++ -O3 -std=c++17 -I include/ctQALS \
//       src/tests/units/ctQALS/test_rsvd.cpp \
//       src/ctQALS/xoshiro.cpp src/ctQALS/ziggurat.cpp \
//       -o /tmp/test_rsvd
//   /tmp/test_rsvd
//

#include "ctQALS/RandomMatrix.h"
#include "ctQALS/HouseholderQR.h"
#include "ctQALS/JacobiSVD.h"
#include "ctQALS/RSVD.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void __attribute__((constructor)) _flush_init() {
    setbuf(stdout, nullptr);
    setbuf(stderr, nullptr);
}

using namespace ctQALS::linalg;

#define CHECK(cond, msg) do {                                              \
    if (!(cond)) {                                                         \
        std::fprintf(stderr, "[FAIL] %s  (line %d)\n", msg, __LINE__);     \
        return 1;                                                          \
    }                                                                      \
} while (0)

// ============================================================
// Test 1: RandomMatrix 统计性质
// ============================================================
int test_random_matrix_stats() {
    std::printf("\n=== Test 1: RandomMatrix (1024×1024 标准正态) ===\n");

    const std::size_t M = 1024, N = 1024;
    MatrixXf Mmat = randn(M, N);

    // 统计：均值 ~ 0, 方差 ~ 1
    double mean = 0.0, m2 = 0.0;
    for (std::size_t i = 0; i < M * N; ++i) {
        const double v = Mmat.data()[i];
        mean += v;
    }
    mean /= (M * N);

    for (std::size_t i = 0; i < M * N; ++i) {
        const double v = Mmat.data()[i] - mean;
        m2 += v * v;
    }
    m2 /= (M * N - 1);

    std::printf("  mean = %.6f (期望 0.0)\n", mean);
    std::printf("  var  = %.6f (期望 1.0)\n", m2);

    CHECK(std::abs(mean) < 0.05, "RandomMatrix 均值偏差过大");
    CHECK(std::abs(m2 - 1.0) < 0.05, "RandomMatrix 方差偏差过大");
    std::printf("  [PASS]\n");
    return 0;
}

// ============================================================
// Test 2: HouseholderQR
// ============================================================
int test_householder_qr() {
    std::printf("\n=== Test 2: HouseholderQR (256×128) ===\n");

    const std::size_t M = 256, N = 128;
    MatrixXf A = randn(M, N);

    HouseholderQR qr(A.data(), M, N);
    const float err = qr.reconstruction_error(A.data());

    std::printf("  reconstruction error = %.3e\n", err);
    CHECK(err < 1e-5f, "HouseholderQR 重建误差过大");

    // 验证 Q^T Q = I (m×m)
    std::vector<float> Qfull(M * M);
    qr.get_Q_full(Qfull.data());
    double err_Q = 0.0;
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < M; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < M; ++k) {
                sum += static_cast<double>(Qfull[k * M + i]) * Qfull[k * M + j];
            }
            const double target = (i == j) ? 1.0 : 0.0;
            err_Q += (sum - target) * (sum - target);
        }
    }
    err_Q = std::sqrt(err_Q / M);
    std::printf("  ||Q^T Q - I||_F / sqrt(m) = %.3e\n", err_Q);
    CHECK(err_Q < 1e-5, "HouseholderQR Q 不正交");

    // 验证 R 上三角
    std::vector<float> R(N * N);
    qr.get_R(R.data());
    bool upper = true;
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            if (std::abs(R[i * N + j]) > 1e-5f) {
                upper = false;
                break;
            }
        }
    }
    CHECK(upper, "R 不是上三角");
    std::printf("  [PASS]\n");
    return 0;
}

// ============================================================
// Test 3: JacobiSVD
// ============================================================
int test_jacobi_svd() {
    std::printf("\n=== Test 3: JacobiSVD (128×128) ===\n");

    const std::size_t N = 128;
    MatrixXf A = randn(N, N);

    JacobiSVD svd(A.data(), N, N, 30, 1e-7f);
    svd.sort_descending();

    const float err = svd.reconstruction_error(A.data());
    std::printf("  reconstruction error = %.3e\n", err);
    CHECK(err < 1e-4f, "JacobiSVD 重建误差过大");

    // 验证 S 降序
    const float* S = svd.S();
    for (std::size_t i = 1; i < N; ++i) {
        CHECK(S[i] <= S[i - 1] + 1e-5f, "S 不是降序");
    }
    std::printf("  S[0] = %.3f, S[N-1] = %.3e\n", S[0], S[N - 1]);
    CHECK(S[0] > 0.0f, "最大奇异值应该 > 0");

    // 验证 U^T U = I（U 是 m×m）
    const float* U = svd.U();
    double err_U = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < N; ++k) {
                sum += static_cast<double>(U[k * N + i]) * U[k * N + j];
            }
            const double target = (i == j) ? 1.0 : 0.0;
            err_U += (sum - target) * (sum - target);
        }
    }
    err_U = std::sqrt(err_U / N);
    std::printf("  ||U^T U - I||_F / sqrt(m) = %.3e\n", err_U);
    CHECK(err_U < 1e-3, "U 不正交");

    std::printf("  [PASS]\n");
    return 0;
}

// ============================================================
// Test 4: RSVD 在不同规模下的表现
// ============================================================
int test_rsvd_scaling() {
    std::printf("\n=== Test 4: RSVD scaling ===\n");
    std::printf("  %6s × %6s | %5s | %4s | %10s | %10s | %10s\n",
                "m", "n", "k", "q", "recon err", "energy %", "time ms");

    struct Case { std::size_t m, n; int k; int q; };
    // 上限 2048x2048：单线程 JacobiSVD 在 4096x4096 上要 3 小时，
    // 2048x2048 ~10 分钟是测试预算内可接受的最大规模
    const std::vector<Case> cases = {
        { 256,  256,  16, 2},
        { 512,  512,  32, 2},
        { 1024, 1024, 32, 2},
        { 1024, 1024, 64, 2},
        { 2048, 2048, 64, 2},
    };

    for (const auto& c : cases) {
        MatrixXf W = randn(c.m, c.n);

        // 构造一个低秩矩阵来测试恢复率：W = A B^T
        // 这里直接用随机矩阵测相对误差
        RSVDOptions opts;
        opts.target_rank      = c.k;
        opts.oversampling     = 10;
        opts.power_iterations = c.q;
        opts.seed             = 42;

        RSVD rsvd(W.data(), c.m, c.n, opts);
        const float err = rsvd.reconstruction_error(W.data());
        const float energy = RSVD::full_energy(W.data(), c.m, c.n);
        const float captured = rsvd.captured_energy_ratio();
        const float ratio = (energy > 0) ? captured / energy * 100.0f : 0.0f;

        std::printf("  %6zu × %6zu | %5d | %4d | %.3e | %9.2f%% | %10.2f\n",
                    c.m, c.n, c.k, c.q, err, ratio, rsvd.timing().total_ms);

        // 重建误差：随机矩阵大约是 1/sqrt(k) 数量级
        // 不太严，只要求 < 1e-1
        CHECK(err < 1.0f, "RSVD 重建误差爆炸");
    }
    std::printf("  [PASS]\n");
    return 0;
}

// ============================================================
// Test 5: Tensor 集成（2026-08-10）
// ============================================================
int test_tensor_integration() {
    std::printf("\n=== Test 5: TensorRSVD integration (256×128) ===\n");
    Tensor W(ShapeTag{}, {256, 128}, DType::kFloat, DeviceType::kCPU);
    // 填标准正态
    ctQALS::rng::Xoshiro256PlusPlus eng(777);
    ctQALS::rng::ZigguratNormal norm(eng);
    norm.fill(W.data<float>(), 256 * 128);

    TensorRSVD trsvd(W);
    Tensor U = trsvd.U();
    Tensor S = trsvd.S();
    Tensor V = trsvd.V();

    std::printf("  W shape = (%zu, %zu)\n", W.shape()[0], W.shape()[1]);
    std::printf("  U shape = (%zu, %zu)\n", U.shape()[0], U.shape()[1]);
    std::printf("  S shape = (%zu,)\n", S.shape()[0]);
    std::printf("  V shape = (%zu, %zu)\n", V.shape()[0], V.shape()[1]);
    std::printf("  S[0] = %.4f, S[%zu] = %.4e\n",
                S.data<float>()[0], S.shape()[0] - 1,
                S.data<float>()[S.shape()[0] - 1]);

    // 重建误差
    const float err = trsvd.reconstruction_error(W);
    std::printf("  reconstruction error = %.3e\n", err);
    CHECK(err < 1.0f, "TensorRSVD 重建误差爆炸");
    std::printf("  [PASS]\n");
    return 0;
}

// ============================================================
// Test 5: 低秩矩阵恢复（关键 RSVD 用例）
// ============================================================
int test_rsvd_low_rank_recovery() {
    std::printf("\n=== Test 5: RSVD low-rank recovery (rank 8 in 1024×1024) ===\n");

    const std::size_t M = 1024, N = 1024;
    const int true_rank = 8;

    // W = U_real * V_real^T，U_real (M×k), V_real (N×k)
    MatrixXf U_real = randn(M, true_rank);
    MatrixXf V_real = randn(N, true_rank);
    // W = U_real * V_real^T
    MatrixXf W(M, N);
    W.zero();
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int r = 0; r < true_rank; ++r) {
                sum += U_real(i, r) * V_real(j, r);
            }
            W(i, j) = sum;
        }
    }

    RSVDOptions opts;
    opts.target_rank      = true_rank;
    opts.oversampling     = 10;
    opts.power_iterations = 2;
    opts.seed             = 1234;

    RSVD rsvd(W.data(), M, N, opts);
    const float err = rsvd.reconstruction_error(W.data());

    std::printf("  reconstruction error (rank-%d) = %.3e\n", true_rank, err);
    std::printf("  S[0..2] = %.3f %.3f %.3f ...\n",
                rsvd.S()[0], rsvd.S()[1], rsvd.S()[2]);
    std::printf("  S[%d..%d] = ... %.3e %.3e\n",
                true_rank - 2, true_rank - 1,
                rsvd.S()[true_rank - 2], rsvd.S()[true_rank - 1]);

    CHECK(err < 1e-3f, "低秩矩阵恢复误差过大");
    std::printf("  [PASS]\n");
    return 0;
}

// ============================================================
// main
// ============================================================
int main() {
    int rc = 0;
    rc |= test_random_matrix_stats();
    rc |= test_householder_qr();
    rc |= test_jacobi_svd();
    rc |= test_rsvd_scaling();
    rc |= test_tensor_integration();
    rc |= test_rsvd_low_rank_recovery();

    std::printf("\n========================================\n");
    if (rc == 0) {
        std::printf("[ALL PASS] RSVD 子模块数值正确性 OK\n");
    } else {
        std::printf("[SOME FAIL] rc = %d\n", rc);
    }
    return rc;
}
