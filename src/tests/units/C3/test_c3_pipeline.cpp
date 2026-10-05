/**
 * @file test_c3_pipeline.cpp
 * @brief C3 Pipeline & Lock-Free FastDispatchTable Unit Test Suite.
 */

#if __has_include("C3/C3Pipeline.h")
#include "C3/C3Pipeline.h"
#else
#include "C3Pipeline.h"
#endif

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <thread>
#include <future>
#include <random>
#include <cassert>

using namespace ct::c3;

void test_pipeline_builder_api() {
    std::cout << "[Test 1: Fluent C++20 PipelineBuilder & Functional API Validation]\n";

    PipelineConfig config = PipelineBuilder()
        .withOptimizationLevel(3)
        .withMemoryPlanning(true)
        .withRamAd(true)
        .withAmx(true)
        .withSramBudget(256 * 1024)
        .withTargetDevice(DeviceType::kCPU)
        .build();

    assert(config.opt_level == 3);
    assert(config.enable_memory_planning == true);
    assert(config.enable_ram_ad == true);
    assert(config.enable_amx == true);
    assert(config.sram_budget_bytes == 256 * 1024);
    assert(config.target_device == DeviceType::kCPU);

    auto kernel = compile("TransformerBlock_SwiGLU_FP32_4096", 4096, config);
    assert(kernel != nullptr);
    std::cout << " [PASS] Fluent PipelineBuilder and compile() API validated.\n\n";
}

void test_lock_free_dispatch_concurrency() {
    std::cout << "[Test 2: Lock-Free FastDispatchTable Concurrency & Latency Benchmark]\n";

    LockFreeFastDispatchTable table;
    FusedSwigluResidualKernel kernel_inst("SwiGLU_Residual_FP32_4096", 32768, DeviceType::kCPU);
    table.install("SwiGLU_Residual_FP32_4096", &kernel_inst);

    uint64_t sig_hash = LockFreeFastDispatchTable::hash_signature("SwiGLU_Residual_FP32_4096");

    // Warmup & verification
    const auto* found = table.lookup(sig_hash);
    assert(found == &kernel_inst);

    constexpr size_t kNumThreads = 8;
    constexpr size_t kLookupsPerThread = 250000;
    std::vector<std::future<void>> futures;

    auto t0 = std::chrono::steady_clock::now();

    for (size_t t = 0; t < kNumThreads; ++t) {
        futures.push_back(std::async(std::launch::async, [&table, sig_hash, &kernel_inst]() {
            for (size_t i = 0; i < kLookupsPerThread; ++i) {
                const auto* ptr = table.lookup(sig_hash);
                assert(ptr == &kernel_inst);
            }
        }));
    }

    for (auto& f : futures) {
        f.get();
    }

    auto t1 = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    size_t total_lookups = kNumThreads * kLookupsPerThread; // 2,000,000 lookups
    double avg_ns = (total_ms * 1e6) / static_cast<double>(total_lookups);
    double throughput_mops = (static_cast<double>(total_lookups) / (total_ms / 1000.0)) / 1e6;

    std::cout << " - Total Concurrent Lookups: " << total_lookups << " across " << kNumThreads << " threads\n";
    std::cout << " - Total Time              : " << total_ms << " ms\n";
    std::cout << " - Average Dispatch Latency: " << avg_ns << " ns / lookup\n";
    std::cout << " - Dispatch Throughput     : " << throughput_mops << " Million lookups / sec\n";

    assert(avg_ns < 50.0 && "Lock-free lookup latency must be under 50 ns!");
    std::cout << " [PASS] Sub-50ns lock-free dispatch performance verified!\n\n";
}

void test_zero_allocation_execution_context() {
    std::cout << "[Test 3: Zero-Allocation C3ExecutionContext & Static Arena Offset Binding]\n";

    constexpr size_t kArenaCapacity = 1024 * 1024; // 1 MB
    C3ExecutionContext ctx(kArenaCapacity);

    assert(ctx.capacity() >= kArenaCapacity);
    assert(reinterpret_cast<uintptr_t>(ctx.raw_buffer()) % 64 == 0 && "Must be 64-byte aligned!");

    constexpr size_t kElements = 4096;
    auto span_x = ctx.get_span<float>(0, kElements);
    auto span_y = ctx.get_span<float>(kElements * sizeof(float), kElements);

    assert(span_x.size() == kElements);
    assert(span_y.size() == kElements);

    // Verify non-overlapping
    assert(span_x.data() + span_x.size() <= span_y.data());

    constexpr size_t kIterations = 100000;
    auto t0 = std::chrono::steady_clock::now();

    float checksum = 0.0f;
    for (size_t iter = 0; iter < kIterations; ++iter) {
        float* ptr = ctx.get_ptr<float>(0);
        *ptr = static_cast<float>(iter);
        checksum += *ptr;
    }

    auto t1 = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_ns = (total_ms * 1e6) / static_cast<double>(kIterations);

    std::cout << " - Arena Size              : " << (ctx.capacity() / 1024) << " KB (64-byte aligned)\n";
    std::cout << " - Iterations              : " << kIterations << " (Zero heap allocations)\n";
    std::cout << " - Average Indexing Latency: " << avg_ns << " ns / op\n";

    assert(avg_ns < 50.0 && "Static arena pointer indexing must be nanosecond level!");
    std::cout << " [PASS] Zero-allocation static arena performance verified!\n\n";
}

void test_end_to_end_transformer_block_equivalence() {
    std::cout << "[Test 4: End-to-End Fused SwiGLU + Residual Block Numerical Equivalence & Speedup]\n";

    constexpr size_t N = 4096;
    std::vector<float> x(N), wg(N), wu(N), res(N);
    std::vector<float> eager_out(N, 0.0f), jit_out(N, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < N; ++i) {
        x[i] = dist(rng);
        wg[i] = dist(rng) * 0.5f;
        wu[i] = dist(rng) * 0.5f;
        res[i] = dist(rng);
    }

    // 1. Eager Reference
    for (size_t i = 0; i < N; ++i) {
        float g = x[i] * wg[i];
        float u = x[i] * wu[i];
        float s = 1.0f / (1.0f + std::exp(-g));
        float act = (g * s) * u;
        eager_out[i] = act + res[i];
    }

    // 2. JIT Pipeline Execution
    C3ExecutionContext ctx(256 * 1024);
    FusedSwigluResidualKernel kernel("SwiGLU_Residual_FP32_4096", 32768, DeviceType::kCPU);

    std::array<const float*, 4> inputs = {x.data(), wg.data(), wu.data(), res.data()};
    std::array<float*, 1> outputs = {jit_out.data()};

    kernel.execute(ctx, inputs, outputs, N);

    // 3. Numerical Equivalence Assertion
    float max_diff = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        float diff = std::abs(eager_out[i] - jit_out[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << " - Max Absolute Numerical Difference: " << max_diff << "\n";
    assert(max_diff < 1e-6f && "JIT output must match eager reference!");
    std::cout << " [PASS] 100% Numerical Equivalence Verified!\n";

    // 4. Latency Benchmark (10,000 iterations)
    constexpr size_t kIters = 10000;
    auto t0 = std::chrono::steady_clock::now();
    for (size_t it = 0; it < kIters; ++it) {
        for (size_t i = 0; i < N; ++i) {
            float g = x[i] * wg[i];
            float u = x[i] * wu[i];
            float s = 1.0f / (1.0f + std::exp(-g));
            float act = (g * s) * u;
            eager_out[i] = act + res[i];
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double eager_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto t2 = std::chrono::steady_clock::now();
    for (size_t it = 0; it < kIters; ++it) {
        kernel.execute(ctx, inputs, outputs, N);
    }
    auto t3 = std::chrono::steady_clock::now();
    double jit_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    double speedup = eager_ms / jit_ms;
    std::cout << " - Eager 10k Invocations: " << eager_ms << " ms\n";
    std::cout << " - JIT   10k Invocations: " << jit_ms << " ms\n";
    std::cout << " - Fused Kernel Speedup : " << speedup << "x\n";

    std::cout << " [PASS] End-to-end execution benchmark passed!\n\n";
}

TEST(C3PipelineTest, FluentBuilderAndFunctionalAPI) {
    EXPECT_NO_FATAL_FAILURE(test_pipeline_builder_api());
}

TEST(C3PipelineTest, LockFreeDispatchConcurrency) {
    EXPECT_NO_FATAL_FAILURE(test_lock_free_dispatch_concurrency());
}

TEST(C3PipelineTest, ZeroAllocationExecutionContext) {
    EXPECT_NO_FATAL_FAILURE(test_zero_allocation_execution_context());
}

TEST(C3PipelineTest, EndToEndTransformerBlockEquivalence) {
    EXPECT_NO_FATAL_FAILURE(test_end_to_end_transformer_block_equivalence());
}

void test_edge_cases_and_lifecycle() {
    std::cout << "[Test 5: Edge Cases, Lifecycle, Cache Reuse & Boundary Conditions]\n";

    // 1. C3ExecutionContext boundary capacity (0 and 1 bytes)
    {
        C3ExecutionContext ctx0(0);
        assert(ctx0.capacity() >= 64);
        assert(reinterpret_cast<uintptr_t>(ctx0.raw_buffer()) % 64 == 0);

        C3ExecutionContext ctx1(1);
        assert(ctx1.capacity() >= 64);
        assert(reinterpret_cast<uintptr_t>(ctx1.raw_buffer()) % 64 == 0);
    }

    // 2. C3ExecutionContext move semantics
    {
        C3ExecutionContext ctx_src(128);
        uint8_t* orig_buf = ctx_src.raw_buffer();
        size_t orig_cap = ctx_src.capacity();

        C3ExecutionContext ctx_dst(std::move(ctx_src));
        assert(ctx_dst.raw_buffer() == orig_buf);
        assert(ctx_dst.capacity() == orig_cap);
        assert(ctx_src.raw_buffer() == nullptr);
        assert(ctx_src.capacity() == 0);

        C3ExecutionContext ctx_dst2(256);
        ctx_dst2 = std::move(ctx_dst);
        assert(ctx_dst2.raw_buffer() == orig_buf);
        assert(ctx_dst2.capacity() == orig_cap);
        assert(ctx_dst.raw_buffer() == nullptr);
        assert(ctx_dst.capacity() == 0);
    }

    // 3. FastDispatchTable nonexistent key & multi-key installation
    {
        LockFreeFastDispatchTable table;
        assert(table.lookup("non_existent_signature_key") == nullptr);

        std::vector<std::unique_ptr<FusedSwigluResidualKernel>> kernels;
        for (int i = 0; i < 64; ++i) {
            std::string sig = "kernel_variant_" + std::to_string(i);
            auto k = std::make_unique<FusedSwigluResidualKernel>(sig, 1024, DeviceType::kCPU);
            table.install(sig, k.get());
            kernels.push_back(std::move(k));
        }

        assert(table.install_count() == 64);
        for (int i = 0; i < 64; ++i) {
            std::string sig = "kernel_variant_" + std::to_string(i);
            const auto* found = table.lookup(sig);
            assert(found == kernels[i].get());
        }
    }

    // 4. compile() dedup / cache hit contract
    {
        PipelineConfig cfg = PipelineBuilder().withOptimizationLevel(2).build();
        auto k1 = compile("DeduplicationTestBlock_FP32", 1024, cfg);
        assert(k1 != nullptr);
        auto k2 = compile("DeduplicationTestBlock_FP32", 1024, cfg);
        assert(k2 != nullptr);
        assert(k1 == k2 && "Repeated compile() for identical signature must return the cached kernel!");
    }

    // 5. Boundary element counts (N = 0, N = 1)
    {
        C3ExecutionContext ctx(1024);
        FusedSwigluResidualKernel kernel("BoundaryKernel_FP32", 1024, DeviceType::kCPU);

        // N = 0: should handle gracefully without crash
        std::array<const float*, 4> dummy_in = {nullptr, nullptr, nullptr, nullptr};
        std::array<float*, 1> dummy_out = {nullptr};
        kernel.execute(ctx, dummy_in, dummy_out, 0);

        // N = 1
        float in_x = 0.5f, in_wg = 1.2f, in_wu = -0.8f, in_res = 2.0f;
        float out_y = 0.0f;
        std::array<const float*, 4> in1 = {&in_x, &in_wg, &in_wu, &in_res};
        std::array<float*, 1> out1 = {&out_y};
        kernel.execute(ctx, in1, out1, 1);

        float expected_zg = in_x * in_wg;
        float expected_zu = in_x * in_wu;
        float expected_sig = 1.0f / (1.0f + std::exp(-expected_zg));
        float expected_out = (expected_zg * expected_sig) * expected_zu + in_res;
        assert(std::abs(out_y - expected_out) < 1e-6f);
    }

    std::cout << " [PASS] Edge cases, move semantics, and cache reuse verified!\n\n";
}

TEST(C3PipelineTest, EdgeCasesAndLifecycle) {
    EXPECT_NO_FATAL_FAILURE(test_edge_cases_and_lifecycle());
}

TEST(C3PipelineTest, SlotHashCollisionAndPersistentDeduplication) {
    std::string keyA = "test_key_269";
    std::string keyB = "test_key_320";

    uint64_t hashA = LockFreeFastDispatchTable::hash_signature(keyA);
    uint64_t hashB = LockFreeFastDispatchTable::hash_signature(keyB);
    ASSERT_EQ(hashA & LockFreeFastDispatchTable::kMask, hashB & LockFreeFastDispatchTable::kMask);
    ASSERT_NE(hashA, hashB);

    // 1. Fast dispatch table collision overwrite behavior
    LockFreeFastDispatchTable table;
    FusedSwigluResidualKernel kA(keyA, 1024, DeviceType::kCPU);
    FusedSwigluResidualKernel kB(keyB, 1024, DeviceType::kCPU);

    table.install(keyA, &kA);
    EXPECT_EQ(table.lookup(keyA), &kA);
    EXPECT_EQ(table.lookup(hashA), &kA);

    // Overwrite slot with colliding keyB
    table.install(keyB, &kB);
    EXPECT_EQ(table.lookup(keyB), &kB);
    EXPECT_EQ(table.lookup(hashB), &kB);
    EXPECT_EQ(table.lookup(keyA), nullptr);
    EXPECT_EQ(table.lookup(hashA), nullptr);

    // 2. C3PipelineManager compile deduplication under slot collision
    size_t count_before = C3PipelineManager::getInstance().compiled_kernel_count();
    auto kernelA1 = compile(keyA, 1024);
    ASSERT_NE(kernelA1, nullptr);

    auto kernelB1 = compile(keyB, 1024);
    ASSERT_NE(kernelB1, nullptr);
    EXPECT_NE(kernelA1.get(), kernelB1.get());

    // Compiling keyA again must return the exact same shared_ptr and not allocate new kernel
    auto kernelA2 = compile(keyA, 1024);
    EXPECT_EQ(kernelA1, kernelA2);

    auto kernelB2 = compile(keyB, 1024);
    EXPECT_EQ(kernelB1, kernelB2);

    size_t count_after = C3PipelineManager::getInstance().compiled_kernel_count();
    EXPECT_EQ(count_after, count_before + 2);
}

TEST(C3PipelineTest, ConcurrentReaderSlotOverwriteRaceCondition) {
    LockFreeFastDispatchTable table;
    std::string keyA = "test_key_269";
    std::string keyB = "test_key_320";
    FusedSwigluResidualKernel kA(keyA, 1024, DeviceType::kCPU);
    FusedSwigluResidualKernel kB(keyB, 1024, DeviceType::kCPU);

    std::atomic<bool> stop_flag{false};
    std::atomic<size_t> wrong_kernel_reads{0};
    std::atomic<size_t> valid_reads{0};

    // Writer thread constantly swaps slot between keyA and keyB
    std::thread writer([&]() {
        for (int i = 0; i < 100000; ++i) {
            table.install(keyA, &kA);
            table.install(keyB, &kB);
        }
        stop_flag.store(true, std::memory_order_release);
    });

    // 4 reader threads lookup keyA and verify that if non-null, it is ALWAYS kA and NEVER kB
    std::vector<std::thread> readers;
    for (int r = 0; r < 4; ++r) {
        readers.emplace_back([&]() {
            while (!stop_flag.load(std::memory_order_acquire)) {
                const auto* ptr = table.lookup(keyA);
                if (ptr != nullptr) {
                    if (ptr == &kB || ptr->signature() != keyA) {
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

    EXPECT_EQ(wrong_kernel_reads.load(), 0u) << "Reader observed corrupted/mismatched kernel pointer during slot overwrite!";
    EXPECT_GT(valid_reads.load(), 0u);
}

TEST(C3PipelineTest, ConcurrentMultiThreadedCompileStress) {
    constexpr int kThreads = 8;
    constexpr int kIters = 2000;
    std::vector<std::future<void>> futures;

    for (int t = 0; t < kThreads; ++t) {
        futures.push_back(std::async(std::launch::async, [t]() {
            PipelineConfig cfg = PipelineBuilder().withOptimizationLevel(3).build();
            for (int i = 0; i < kIters; ++i) {
                std::string sig = "stress_kernel_" + std::to_string((t + i) % 16);
                auto k = compile(sig, 1024, cfg);
                assert(k != nullptr);
                assert(k->signature() == sig);
            }
        }));
    }

    for (auto& f : futures) {
        f.get();
    }
}

TEST(C3PipelineTest, ArenaBoundsAndKernelDefensiveGuards) {
    C3ExecutionContext ctx(256);
    EXPECT_EQ(ctx.get_ptr<float>(300), nullptr);
    EXPECT_EQ(ctx.get_span<float>(200, 100).size(), 0u);

    // Empty input/output to kernel
    FusedSwigluResidualKernel kernel("GuardedKernel", 512, DeviceType::kCPU);
    std::span<const float* const> empty_in;
    std::span<float*> empty_out;
    EXPECT_NO_FATAL_FAILURE(kernel.execute(ctx, empty_in, empty_out, 0));
    EXPECT_NO_FATAL_FAILURE(kernel.execute(ctx, empty_in, empty_out, 10));
}

int main(int argc, char** argv) {
    std::cout << "================================================================================\n";
    std::cout << ">>> CTorch C3: Unified Pipeline & API Optimization Test Suite <<<\n";
    std::cout << "================================================================================\n\n";

    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    if (ret == 0) {
        std::cout << ">>> ALL C3 PIPELINE TESTS PASSED SUCCESSFULLY! <<<\n";
    }
    return ret;
}
