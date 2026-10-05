/**
 * @file test_amx_cluster_mpgemm.cpp
 * @brief GTest Suite for Apple Silicon AMX Cluster-Affinity MpGEMM & INT4 Dequantization Super-Kernel.
 * @details Validates:
 *   1. MLIR ct.amx.cluster_mpgemm Dialect Emission
 *   2. Multi-Cluster Spatial Decomposition Accuracy (1 thread vs 4 threads)
 *   3. High-Concurrency Scaling & Throughput Profiling (500 passes)
 *   4. MLIR ct.amx.dequant_fused Dialect Emission
 *   5. On-the-Fly INT4 Dequantization & SwiGLU Super-Kernel Execution
 */

#include "C3/AmxClusterMpGemmEngine.h"
#include "C3/AmxDequantFusionEngine.h"

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

using namespace ct::c3;

// ==============================================================================
// 1. AMX Cluster MpGEMM Dialect & Multiprocessing Verification
// ==============================================================================

TEST(AmxClusterMpGemmTest, MlirDialectEmission) {
    AmxClusterMpGemmEngine::MpGemmConfig cfg{
        .M = 64,
        .D = 128,
        .D_ffn = 256,
        .num_threads = 4
    };

    std::string mlir = AmxClusterMpGemmEngine::emit_mlir_ir(cfg);
    EXPECT_NE(mlir.find("module @ct_amx_cluster_mpgemm"), std::string::npos);
    EXPECT_NE(mlir.find("amx.cluster_affinity = true"), std::string::npos);
    EXPECT_NE(mlir.find("amx.thread_local_set_state"), std::string::npos);
    EXPECT_NE(mlir.find("arm_neon.fused_silu_mul"), std::string::npos);
}

TEST(AmxClusterMpGemmTest, MultiClusterNumericalAccuracyAndEquivalence) {
    constexpr size_t M = 64;
    constexpr size_t D = 128;
    constexpr size_t D_ffn = 256;

    AmxClusterMpGemmEngine::MpGemmConfig cfg_1{
        .M = M,
        .D = D,
        .D_ffn = D_ffn,
        .num_threads = 1
    };

    AmxClusterMpGemmEngine::MpGemmConfig cfg_4{
        .M = M,
        .D = D,
        .D_ffn = D_ffn,
        .num_threads = 4
    };

    std::vector<float> X(M * D);
    std::vector<float> Wg(D * D_ffn);
    std::vector<float> Wu(D * D_ffn);
    std::vector<float> Wd(D_ffn * D);
    std::vector<float> R(M * D);
    std::vector<float> Y_seq(M * D, 0.0f);
    std::vector<float> Y_mp(M * D, 0.0f);

    std::mt19937 gen(42);
    std::normal_distribution<float> dist_f(0.0f, 0.1f);

    for (auto& v : X) v = dist_f(gen);
    for (auto& v : Wg) v = dist_f(gen);
    for (auto& v : Wu) v = dist_f(gen);
    for (auto& v : Wd) v = dist_f(gen);
    for (auto& v : R) v = dist_f(gen);

    AmxClusterMpGemmEngine::execute_cluster_mpgemm(X, Wg, Wu, Wd, R, Y_seq, cfg_1);
    AmxClusterMpGemmEngine::execute_cluster_mpgemm(X, Wg, Wu, Wd, R, Y_mp, cfg_4);

    float max_diff = 0.0f;
    for (size_t i = 0; i < M * D; ++i) {
        float diff = std::abs(Y_seq[i] - Y_mp[i]);
        if (diff > max_diff) max_diff = diff;
    }

    EXPECT_LT(max_diff, 1e-6f);
}

TEST(AmxClusterMpGemmTest, HighConcurrencyScalingBenchmark) {
    constexpr size_t M = 64;
    constexpr size_t D = 128;
    constexpr size_t D_ffn = 256;

    AmxClusterMpGemmEngine::MpGemmConfig cfg_4{
        .M = M,
        .D = D,
        .D_ffn = D_ffn,
        .num_threads = 4
    };

    std::vector<float> X(M * D, 0.05f);
    std::vector<float> Wg(D * D_ffn, 0.02f);
    std::vector<float> Wu(D * D_ffn, 0.02f);
    std::vector<float> Wd(D_ffn * D, 0.02f);
    std::vector<float> R(M * D, 0.01f);
    std::vector<float> Y(M * D, 0.0f);

    constexpr size_t kBenchmarkPasses = 500;
    auto t0 = std::chrono::steady_clock::now();

    for (size_t p = 0; p < kBenchmarkPasses; ++p) {
        AmxClusterMpGemmEngine::execute_cluster_mpgemm(X, Wg, Wu, Wd, R, Y, cfg_4);
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double total_tokens = static_cast<double>(M * kBenchmarkPasses);
    double tokens_per_sec = (total_tokens / (ms / 1000.0));

    EXPECT_GT(tokens_per_sec, 10000.0);
    for (float v : Y) {
        EXPECT_FALSE(std::isnan(v));
    }
}

// ==============================================================================
// 2. AMX INT4 Dequantization & SwiGLU Super-Kernel Verification
// ==============================================================================

TEST(AmxClusterMpGemmTest, DequantFusionMlirDialectEmission) {
    AmxDequantFusionEngine::DequantConfig cfg{
        .M = 16,
        .D = 64,
        .D_ffn = 128
    };

    std::string mlir = AmxDequantFusionEngine::emit_mlir_ir(cfg);
    EXPECT_NE(mlir.find("module @ct_amx_dequant_fused"), std::string::npos);
    EXPECT_NE(mlir.find("amx.zero_intermediate_spill = true"), std::string::npos);
    EXPECT_NE(mlir.find("amx.quant_type = \"W4A16\""), std::string::npos);
    EXPECT_NE(mlir.find("amx.fused_ops"), std::string::npos);
}

TEST(AmxClusterMpGemmTest, DequantFusionNumericalEquivalence) {
    constexpr size_t M = 16;
    constexpr size_t D = 64;
    constexpr size_t D_ffn = 128;
    constexpr size_t packed_cols = D_ffn / 2;

    AmxDequantFusionEngine::DequantConfig cfg{
        .M = M,
        .D = D,
        .D_ffn = D_ffn
    };

    std::vector<float> X(M * D, 0.1f);
    std::vector<uint8_t> Wg_packed(D * packed_cols, 0x88); // Nibble 8 = zp -> 0.0
    std::vector<float> scale_g(D_ffn, 0.02f);
    std::vector<float> zp_g(D_ffn, 8.0f);

    std::vector<uint8_t> Wu_packed(D * packed_cols, 0x99); // Nibble 9 -> (9-8)*0.02 = 0.02
    std::vector<float> scale_u(D_ffn, 0.02f);
    std::vector<float> zp_u(D_ffn, 8.0f);

    std::vector<float> Wd(D_ffn * D, 0.01f);
    std::vector<float> R(M * D, 0.5f);
    std::vector<float> Y(M * D, 0.0f);

    AmxDequantFusionEngine::execute_amx_dequant_fused(
        X, Wg_packed, scale_g, zp_g, Wu_packed, scale_u, zp_u, Wd, R, Y, cfg);

    // Because Wg nibble is 8, wg_val = (8 - 8) * 0.02 = 0.0
    // acc_g = 0.0 -> sig = 0.5 -> h_vec[k] = (0 * 0.5) * acc_u = 0.0
    // Y should equal R (0.5f)
    for (size_t i = 0; i < M * D; ++i) {
        EXPECT_NEAR(Y[i], 0.5f, 1e-5f);
    }
}

TEST(AmxClusterMpGemmTest, NonMultipleBatchDimensions) {
    constexpr size_t M = 35; // Unevenly split across 4 threads: 9, 9, 9, 8
    constexpr size_t D = 64;
    constexpr size_t D_ffn = 128;

    AmxClusterMpGemmEngine::MpGemmConfig cfg_1{
        .M = M,
        .D = D,
        .D_ffn = D_ffn,
        .num_threads = 1
    };

    AmxClusterMpGemmEngine::MpGemmConfig cfg_4{
        .M = M,
        .D = D,
        .D_ffn = D_ffn,
        .num_threads = 4
    };

    std::vector<float> X(M * D);
    std::vector<float> Wg(D * D_ffn);
    std::vector<float> Wu(D * D_ffn);
    std::vector<float> Wd(D_ffn * D);
    std::vector<float> R(M * D);
    std::vector<float> Y_seq(M * D, 0.0f);
    std::vector<float> Y_mp(M * D, 0.0f);

    std::mt19937 gen(35);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    for (auto& v : X) v = dist(gen);
    for (auto& v : Wg) v = dist(gen);
    for (auto& v : Wu) v = dist(gen);
    for (auto& v : Wd) v = dist(gen);
    for (auto& v : R) v = dist(gen);

    AmxClusterMpGemmEngine::execute_cluster_mpgemm(X, Wg, Wu, Wd, R, Y_seq, cfg_1);
    AmxClusterMpGemmEngine::execute_cluster_mpgemm(X, Wg, Wu, Wd, R, Y_mp, cfg_4);

    float max_diff = 0.0f;
    for (size_t i = 0; i < M * D; ++i) {
        float diff = std::abs(Y_seq[i] - Y_mp[i]);
        if (diff > max_diff) max_diff = diff;
    }

    EXPECT_LT(max_diff, 1e-6f);
}

TEST(AmxClusterMpGemmTest, EmptyInputsHandling) {
    AmxClusterMpGemmEngine::MpGemmConfig cfg{
        .M = 0,
        .D = 64,
        .D_ffn = 128,
        .num_threads = 4
    };
    std::vector<float> empty_buf;
    EXPECT_NO_THROW(AmxClusterMpGemmEngine::execute_cluster_mpgemm(empty_buf, empty_buf, empty_buf, empty_buf, empty_buf, empty_buf, cfg));

    AmxDequantFusionEngine::DequantConfig dcfg{
        .M = 0,
        .D = 64,
        .D_ffn = 128
    };
    std::vector<uint8_t> empty_u8;
    EXPECT_NO_THROW(AmxDequantFusionEngine::execute_amx_dequant_fused(empty_buf, empty_u8, empty_buf, empty_buf, empty_u8, empty_buf, empty_buf, empty_buf, empty_buf, empty_buf, dcfg));
}
