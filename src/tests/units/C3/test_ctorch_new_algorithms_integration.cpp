/**
 * @file test_ctorch_new_algorithms_integration.cpp
 * @brief GTest Suite for CTorch C3 New Algorithms Hub & Comprehensive Integration.
 * @details Validates:
 *   1. Unified Algorithm Registry & Reflection Capabilities
 *   2. RAM-AD Ragged Sequence Engine (Theorem 20)
 *   3. RAM-AD Curvature K-FAC In-Register Engine (Theorem 19)
 *   4. NUMA-SMP Multi-Domain Static Memory Planner Integration
 *   5. End-to-End MNIST Training Pipeline with IA-IGC Static Arena & RAM-AD
 */

#include "C3/CTorchNewAlgorithmsRegistry.h"

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <array>
#include <thread>
#include <atomic>
#include <iomanip>
#include <chrono>

using namespace ct;
using namespace ct::c3;

// ==============================================================================
// 1. Algorithm Registry and Connectivity
// ==============================================================================

TEST(CTorchNewAlgorithmsTest, AlgorithmRegistryAndConnectivity) {
    auto algos = CTorchNewAlgorithmsHub::get_registered_algorithms();
    EXPECT_GE(algos.size(), 9);

    bool has_curvature = false;
    bool has_ragged = false;
    bool has_numa = false;
    bool has_tro = false;
    bool has_amx_fa = false;
    bool has_amx_gemm = false;
    bool has_amx_dequant = false;
    bool has_fa3_fp8 = false;

    for (const auto& algo : algos) {
        if (algo.find("CurvatureKFAC") != std::string::npos) has_curvature = true;
        if (algo.find("DynamicRagged") != std::string::npos) has_ragged = true;
        if (algo.find("NUMA-SMP") != std::string::npos) has_numa = true;
        if (algo.find("TRO-SMP") != std::string::npos) has_tro = true;
        if (algo.find("AMX::FlashAttention") != std::string::npos) has_amx_fa = true;
        if (algo.find("AMX::ClusterMpGEMM") != std::string::npos) has_amx_gemm = true;
        if (algo.find("AMX::DequantFusion") != std::string::npos) has_amx_dequant = true;
        if (algo.find("FlashAttention3::FP8") != std::string::npos) has_fa3_fp8 = true;
    }

    EXPECT_TRUE(has_curvature);
    EXPECT_TRUE(has_ragged);
    EXPECT_TRUE(has_numa);
    EXPECT_TRUE(has_tro);
    EXPECT_TRUE(has_amx_fa);
    EXPECT_TRUE(has_amx_gemm);
    EXPECT_TRUE(has_amx_dequant);
    EXPECT_TRUE(has_fa3_fp8);
}

// ==============================================================================
// 2. RAM-AD Variable-Length Ragged Sequence Execution (Theorem 20)
// ==============================================================================

TEST(CTorchNewAlgorithmsTest, RaggedSequenceExecution) {
    constexpr size_t Din = 16, Dmid = 32, Dout = 16, tile_size = 16;
    RamAdDynamicRaggedEngine<double>::RaggedConfig cfg{Din, Dmid, Dout, tile_size};

    std::vector<size_t> seq_lens = {16, 32, 8, 24};
    std::vector<size_t> offsets = {0};
    for (size_t s : seq_lens) offsets.push_back(offsets.back() + s);
    const size_t T_total = offsets.back();

    std::vector<double> X(T_total * Din, 0.5);
    std::vector<double> W1(Din * Dmid, 0.1);
    std::vector<double> W2(Dmid * Dout, 0.1);
    std::vector<double> dY(T_total * Dout, 0.2);

    std::vector<double> out_Y(T_total * Dout);
    std::vector<double> out_dX(T_total * Din);
    std::vector<double> out_dW1(Din * Dmid);
    std::vector<double> out_dW2(Dmid * Dout);

    CTorchNewAlgorithmsHub::execute_ragged_sequence<double>(
        X, W1, W2, dY, offsets,
        out_Y, out_dX, out_dW1, out_dW2, cfg);

    EXPECT_NE(out_Y[0], 0.0);
    EXPECT_NE(out_dX[0], 0.0);
    EXPECT_NE(out_dW1[0], 0.0);
    EXPECT_NE(out_dW2[0], 0.0);

    for (double y : out_Y) EXPECT_FALSE(std::isnan(y));
    for (double dx : out_dX) EXPECT_FALSE(std::isnan(dx));
    for (double dw1 : out_dW1) EXPECT_FALSE(std::isnan(dw1));
    for (double dw2 : out_dW2) EXPECT_FALSE(std::isnan(dw2));

    // Verify MLIR generation
    std::string mlir = RamAdDynamicRaggedEngine<double>::emit_mlir_ir(cfg, T_total, seq_lens.size());
    EXPECT_NE(mlir.find("module @ct_ram_ad_dynamic_ragged"), std::string::npos);
    EXPECT_NE(mlir.find("ram_ad.zero_ragged_activation_tape = true"), std::string::npos);
}

// ==============================================================================
// 3. RAM-AD Second-Order Curvature & K-FAC In-Register Execution (Theorem 19)
// ==============================================================================

TEST(CTorchNewAlgorithmsTest, CurvatureKFacExecution) {
    constexpr size_t B = 16, Din = 32, Dout = 16;
    RamAdCurvatureEngine<double>::CurvatureConfig cfg{B, Din, Dout};

    std::vector<double> X(B * Din, 0.5);
    std::vector<double> W(Din * Dout, 0.1);
    std::vector<double> dY(B * Dout, 0.2);
    std::vector<double> V(Din * Dout, 0.05);

    std::vector<double> out_Y(B * Dout);
    std::vector<double> out_dW(Din * Dout);
    std::vector<double> out_A(Din * Din);
    std::vector<double> out_S(Dout * Dout);
    std::vector<double> out_FVP(Din * Dout);

    CTorchNewAlgorithmsHub::execute_kfac_curvature<double>(
        X, W, dY, V,
        out_Y, out_dW, out_A, out_S, out_FVP, cfg);

    EXPECT_NE(out_A[0], 0.0);
    EXPECT_NE(out_S[0], 0.0);
    EXPECT_NE(out_FVP[0], 0.0);
    EXPECT_NE(out_dW[0], 0.0);
    EXPECT_NE(out_Y[0], 0.0);

    for (double a : out_A) EXPECT_FALSE(std::isnan(a));
    for (double s : out_S) EXPECT_FALSE(std::isnan(s));
    for (double fvp : out_FVP) EXPECT_FALSE(std::isnan(fvp));

    // Verify MLIR generation
    std::string mlir = RamAdCurvatureEngine<double>::emit_mlir_ir(cfg);
    EXPECT_NE(mlir.find("module @ct_ram_ad_curvature"), std::string::npos);
    EXPECT_NE(mlir.find("ram_ad.zero_sample_tape_materialization = true"), std::string::npos);
}

// ==============================================================================
// 4. NUMA-SMP Multi-Domain Static Memory Planner Execution
// ==============================================================================

TEST(CTorchNewAlgorithmsTest, NumaSmpWorkloadPlanning) {
    constexpr size_t MB = 1024 * 1024;
    std::vector<NumaTensorDescriptor> workload = {
        {1, "shared_emb", 16 * MB, 0, 4, -1},
        {2, "n0_attn", 32 * MB, 1, 3, 0},
        {3, "n1_attn", 32 * MB, 1, 3, 1},
        {4, "allreduce", 16 * MB, 3, 5, -1},
        {5, "n0_ffn", 32 * MB, 5, 8, 0},
        {6, "n1_ffn", 32 * MB, 5, 8, 1},
        {7, "n0_bwd", 32 * MB, 8, 10, 0},
        {8, "n1_bwd", 32 * MB, 8, 10, 1},
    };

    auto res = CTorchNewAlgorithmsHub::plan_numa_workload(workload, 2);

    EXPECT_GT(res.node_arenas[0], 0);
    EXPECT_GT(res.node_arenas[1], 0);
    EXPECT_GT(res.shared_arena, 0);
    EXPECT_GT(res.total_arena, 0);
    EXPECT_LT(res.total_arena, 224 * MB); // Must be strictly smaller than unoptimized sum
}

// ==============================================================================
// 5. CTorch MNIST End-to-End Execution & Speed Analysis
// ==============================================================================

TEST(CTorchNewAlgorithmsTest, MnistFullSpeedComparison) {
    constexpr size_t BATCH_SIZE = 128;
    constexpr size_t NUM_BATCHES = 30;
    constexpr int EPOCHS = 2;
    constexpr float LR = 0.05f;

    TransparentMNISTNet<float> net(BATCH_SIZE, LR);

    std::vector<GenericTensor<float>> batches_x;
    std::vector<GenericTensor<float>> batches_y;
    batches_x.reserve(NUM_BATCHES);
    batches_y.reserve(NUM_BATCHES);

    for (size_t b = 0; b < NUM_BATCHES; ++b) {
        GenericTensor<float> bx({BATCH_SIZE, 784});
        GenericTensor<float> by({BATCH_SIZE, 10});
        for (size_t i = 0; i < BATCH_SIZE; ++i) {
            int lbl = b % 10;
            for (size_t j = 0; j < 784; ++j) {
                bx.data()[i * 784 + j] = 0.01f;
            }
            by.data()[i * 10 + lbl] = 1.0f;
        }
        batches_x.push_back(std::move(bx));
        batches_y.push_back(std::move(by));
    }

    float epoch_1_loss = 0.0f;
    float epoch_last_loss = 0.0f;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        float epoch_loss = 0.0f;
        for (size_t b = 0; b < NUM_BATCHES; ++b) {
            float loss = net.train_step(batches_x[b], batches_y[b]);
            epoch_loss += loss;
        }
        if (epoch == 1) epoch_1_loss = epoch_loss / static_cast<float>(NUM_BATCHES);
        if (epoch == EPOCHS) epoch_last_loss = epoch_loss / static_cast<float>(NUM_BATCHES);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    EXPECT_GT(total_ms, 0.0);
    EXPECT_LT(epoch_last_loss, epoch_1_loss);
}

// ==============================================================================
// 6. Phase 4 Micro-Kernel Hub Dispatchers Execution Verification
// ==============================================================================

TEST(CTorchNewAlgorithmsTest, Phase4HubDispatcherExecution) {
    // 1. Amx FlashAttention Dispatcher
    {
        AmxFlashAttentionEngine::FlashAttentionConfig cfg{
            .seq_len = 32,
            .head_dim = 16,
            .tile_br = 16,
            .tile_bc = 16,
            .is_causal = true,
            .scale_factor = 0.25f
        };
        std::vector<float> Q(32 * 16, 0.1f);
        std::vector<float> K(32 * 16, 0.1f);
        std::vector<float> V(32 * 16, 0.1f);
        std::vector<float> O(32 * 16, 0.0f);

        CTorchNewAlgorithmsHub::execute_amx_flash_attention(Q, K, V, O, cfg);
        for (float v : O) {
            EXPECT_FALSE(std::isnan(v));
            EXPECT_GT(v, 0.0f);
        }
    }

    // 2. Amx Cluster MpGEMM Dispatcher
    {
        AmxClusterMpGemmEngine::MpGemmConfig cfg{
            .M = 16,
            .D = 32,
            .D_ffn = 64,
            .num_threads = 2
        };
        std::vector<float> X(16 * 32, 0.05f);
        std::vector<float> Wg(32 * 64, 0.02f);
        std::vector<float> Wu(32 * 64, 0.02f);
        std::vector<float> Wd(64 * 32, 0.02f);
        std::vector<float> R(16 * 32, 0.1f);
        std::vector<float> Y(16 * 32, 0.0f);

        CTorchNewAlgorithmsHub::execute_amx_cluster_mpgemm(X, Wg, Wu, Wd, R, Y, cfg);
        for (float v : Y) {
            EXPECT_FALSE(std::isnan(v));
        }
    }

    // 3. Amx Dequant Fusion Dispatcher
    {
        AmxDequantFusionEngine::DequantConfig cfg{
            .M = 8,
            .D = 16,
            .D_ffn = 32
        };
        std::vector<float> X(8 * 16, 0.1f);
        std::vector<uint8_t> Wg_packed(16 * 16, 0x88);
        std::vector<float> scale_g(32, 0.02f);
        std::vector<float> zp_g(32, 8.0f);
        std::vector<uint8_t> Wu_packed(16 * 16, 0x99);
        std::vector<float> scale_u(32, 0.02f);
        std::vector<float> zp_u(32, 8.0f);
        std::vector<float> Wd(32 * 16, 0.01f);
        std::vector<float> R(8 * 16, 0.5f);
        std::vector<float> Y(8 * 16, 0.0f);

        CTorchNewAlgorithmsHub::execute_amx_dequant_fused(
            X, Wg_packed, scale_g, zp_g, Wu_packed, scale_u, zp_u, Wd, R, Y, cfg);
        for (float v : Y) {
            EXPECT_NEAR(v, 0.5f, 1e-5f);
        }
    }

    // 4. FlashAttention-3 FP8 Dispatcher
    {
        FlashAttention3Config cfg{
            .seq_len = 64,
            .head_dim = 32,
            .tile_br = 32,
            .tile_bc = 32,
            .is_causal = true,
            .scale_factor = 1.0f / std::sqrt(32.0f)
        };
        std::vector<float> Q(64 * 32, 0.05f);
        std::vector<float> K(64 * 32, 0.05f);
        std::vector<float> V(64 * 32, 0.05f);
        std::vector<float> O(64 * 32, 0.0f);

        CTorchNewAlgorithmsHub::execute_flash_attention_3_fp8(Q, K, V, O, cfg);
        for (float v : O) {
            EXPECT_FALSE(std::isnan(v));
            EXPECT_GT(v, 0.0f);
        }
    }
}

// ==============================================================================
// 6. Hardening Regression Tests: P1/P2 Defenses Verification
// ==============================================================================

TEST(CTorchNewAlgorithmsTest, IntegratedCompilerLockFreeDispatchConcurrentStress) {
    LockFreeDispatchTable<float> table;
    std::string keyA = "kernel_key_alpha";
    std::string keyB = "kernel_key_beta";

    auto dummy_fn = [](const float* const*, float* const*, size_t) noexcept {};
    IntegratedCompiledKernel<float> kA(keyA, 1024, {0}, {0}, {1}, dummy_fn);
    IntegratedCompiledKernel<float> kB(keyB, 1024, {0}, {0}, {1}, dummy_fn);

    std::atomic<bool> stop_flag{false};
    std::atomic<size_t> wrong_kernel_reads{0};
    std::atomic<size_t> valid_reads{0};

    std::thread writer([&]() {
        for (int i = 0; i < 50000; ++i) {
            table.install(keyA, &kA);
            table.install(keyB, &kB);
        }
        stop_flag.store(true, std::memory_order_release);
    });

    std::vector<std::thread> readers;
    for (int r = 0; r < 4; ++r) {
        readers.emplace_back([&]() {
            while (!stop_flag.load(std::memory_order_acquire)) {
                const auto* ptr = table.lookup(keyA);
                if (ptr != nullptr) {
                    if (ptr == &kB || ptr->cacheKey() != keyA) {
                        wrong_kernel_reads.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        valid_reads.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        });
    }

    writer.join();
    for (auto& t : readers) {
        t.join();
    }

    EXPECT_EQ(wrong_kernel_reads.load(), 0u) << "Concurrent reader observed torn read or mismatched kernel pointer!";
    EXPECT_GT(valid_reads.load(), 0u);
}

TEST(CTorchNewAlgorithmsTest, IntegratedCompiledKernelZeroAllocationExecuteInto) {
    auto add_fn = [](const float* const* ins, float* const* outs, size_t count) noexcept {
        for (size_t i = 0; i < count; ++i) {
            outs[0][i] = ins[0][i] + ins[1][i];
        }
    };

    IntegratedCompiledKernel<float> kernel(
        "add_kernel",
        1024,
        {0, 64}, // input offsets
        {128},   // output offsets
        {16},    // output element counts
        add_fn
    );

    std::array<float, 16> in0, in1, out;
    in0.fill(1.5f);
    in1.fill(2.5f);
    out.fill(0.0f);

    const float* in_ptrs[2] = {in0.data(), in1.data()};
    float* out_ptrs[1] = {out.data()};

    EXPECT_NO_THROW(kernel.execute_into(
        std::span<const float* const>(in_ptrs, 2),
        std::span<float*>(out_ptrs, 1)
    ));

    for (size_t i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(out[i], 4.0f);
    }

    // Invalid span size checks
    EXPECT_THROW(kernel.execute_into(
        std::span<const float* const>(in_ptrs, 1),
        std::span<float*>(out_ptrs, 1)
    ), std::invalid_argument);
}

TEST(CTorchNewAlgorithmsTest, RamAdCurvatureCallerWorkspaceAndBounds) {
    RamAdCurvatureEngine<double>::CurvatureConfig cfg{
        .B = 4,
        .Din = 16,
        .Dout = 8
    };

    std::vector<double> X(4 * 16, 0.5);
    std::vector<double> W(16 * 8, 0.2);
    std::vector<double> dY(4 * 8, 0.1);
    std::vector<double> V(16 * 8, 0.3);
    std::vector<double> out_Y(4 * 8, 0.0);
    std::vector<double> out_dW(16 * 8, 0.0);
    std::vector<double> out_A(16 * 16, 0.0);
    std::vector<double> out_S(8 * 8, 0.0);
    std::vector<double> out_FVP(16 * 8, 0.0);

    // 1. Caller-supplied workspace execution
    std::vector<double> caller_ws(16 * 8, 0.0);
    EXPECT_NO_THROW(RamAdCurvatureEngine<double>::execute_curvature_kfac(
        X, W, dY, V, out_Y, out_dW, out_A, out_S, out_FVP, cfg, caller_ws
    ));

    for (double y : out_Y) EXPECT_FALSE(std::isnan(y));
    for (double fvp : out_FVP) EXPECT_FALSE(std::isnan(fvp));

    // 2. Undersized workspace throws
    std::vector<double> small_ws(5, 0.0);
    EXPECT_THROW(RamAdCurvatureEngine<double>::execute_curvature_kfac(
        X, W, dY, V, out_Y, out_dW, out_A, out_S, out_FVP, cfg, small_ws
    ), std::invalid_argument);

    // 3. Invalid config dimensions throw
    RamAdCurvatureEngine<double>::CurvatureConfig bad_cfg{ .B = 0, .Din = 16, .Dout = 8 };
    EXPECT_THROW(RamAdCurvatureEngine<double>::execute_curvature_kfac(
        X, W, dY, V, out_Y, out_dW, out_A, out_S, out_FVP, bad_cfg
    ), std::invalid_argument);
}

TEST(CTorchNewAlgorithmsTest, AmxClusterMpGemmDefensiveBoundsGuards) {
    AmxClusterMpGemmEngine::MpGemmConfig cfg{
        .M = 4,
        .D = 8,
        .D_ffn = 10000, // Exceeds 8192 static scratchpad bound
        .num_threads = 2
    };

    std::vector<float> X(4 * 8, 1.0f);
    std::vector<float> Wg(8 * 10000, 1.0f);
    std::vector<float> Wu(8 * 10000, 1.0f);
    std::vector<float> Wd(10000 * 8, 1.0f);
    std::vector<float> R(4 * 8, 0.0f);
    std::vector<float> Y(4 * 8, 99.0f);

    EXPECT_NO_FATAL_FAILURE(AmxClusterMpGemmEngine::execute_cluster_mpgemm(
        X, Wg, Wu, Wd, R, Y, cfg
    ));
    EXPECT_FLOAT_EQ(Y[0], 99.0f); // Early return preserves output buffer

    // Undersized buffer guard
    cfg.D_ffn = 16;
    std::vector<float> short_X(2);
    EXPECT_NO_FATAL_FAILURE(AmxClusterMpGemmEngine::execute_cluster_mpgemm(
        short_X, Wg, Wu, Wd, R, Y, cfg
    ));
}

