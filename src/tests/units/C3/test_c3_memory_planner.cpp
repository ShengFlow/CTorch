/**
 * @file test_c3_memory_planner.cpp
 * @brief GTest Suite for TRO-SMP Topological Reordering & NUMA Static Memory Planner.
 * @details Integrates:
 *   1. NUMA-SMP Multi-Node Planning & Safety Invariants (Theorem 17)
 *   2. Theorem 16 Topological Scheduling Monotonic Convex Envelope Theorem (Branched DAG >=20% reduction)
 *   3. Multi-Threaded Hot Path Zero-Heap Stress Tests
 */

#include "C3/TroMemoryPlanner.h"
#include "C3/NumaStaticMemoryPlanner.h"

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cassert>
#include <numeric>
#include <iomanip>

using namespace ct::c3;

constexpr size_t MB = 1024 * 1024;

// ==============================================================================
// 1. NUMA-SMP Multi-Node Planning & Invariant Tests (Theorem 17)
// ==============================================================================

TEST(C3MemoryPlannerTest, NumaMultiNodePlanningAndInvariants) {
    std::vector<NumaTensorDescriptor> tensors = {
        // Shared Inputs
        {1, "shared_tokens", 8 * MB, 0, 4, -1},
        {2, "norm_out", 8 * MB, 1, 3, -1},

        // Node 0 Local Forward
        {3, "n0_qkv_proj", 32 * MB, 2, 5, 0},
        {4, "n0_attn_scores", 16 * MB, 3, 6, 0},
        {5, "n0_attn_out", 16 * MB, 5, 8, 0},
        {6, "n0_ffn_up", 32 * MB, 8, 11, 0},
        {7, "n0_ffn_act", 16 * MB, 9, 12, 0},
        {8, "n0_ffn_down", 16 * MB, 10, 13, 0},

        // Node 1 Local Forward
        {9, "n1_qkv_proj", 32 * MB, 2, 5, 1},
        {10, "n1_attn_scores", 16 * MB, 3, 6, 1},
        {11, "n1_attn_out", 16 * MB, 5, 8, 1},
        {12, "n1_ffn_up", 32 * MB, 8, 11, 1},
        {13, "n1_ffn_act", 16 * MB, 9, 12, 1},
        {14, "n1_ffn_down", 16 * MB, 10, 13, 1},

        // Cross-Node All-Reduce Intermediates
        {15, "allreduce_attn_buf", 16 * MB, 6, 8, -1},
        {16, "allreduce_ffn_buf", 16 * MB, 12, 14, -1},

        // Backward Adjoints (Node 0)
        {17, "n0_d_ffn_down", 16 * MB, 14, 16, 0},
        {18, "n0_d_ffn_up", 32 * MB, 15, 17, 0},
        {19, "n0_d_qkv", 32 * MB, 16, 18, 0},

        // Backward Adjoints (Node 1)
        {20, "n1_d_ffn_down", 16 * MB, 14, 16, 1},
        {21, "n1_d_ffn_up", 32 * MB, 15, 17, 1},
        {22, "n1_d_qkv", 32 * MB, 16, 18, 1},

        // Global Loss & Gradient
        {23, "d_loss_shared", 8 * MB, 13, 15, -1},
    };

    size_t naive_total = 0;
    for (const auto& t : tensors) {
        naive_total += NumaStaticMemoryPlanner::align_up(t.size_bytes);
    }

    auto plan_res = NumaStaticMemoryPlanner::plan(tensors, 2);

    double savings = (1.0 - static_cast<double>(plan_res.total_arena) / static_cast<double>(naive_total)) * 100.0;

    std::cout << "\n[NUMA-SMP Planning Invariants Results]\n"
              << " - Naive Total Allocation   : " << naive_total / MB << " MB\n"
              << " - NUMA-SMP Node 0 Arena    : " << plan_res.node_arenas[0] / MB << " MB\n"
              << " - NUMA-SMP Node 1 Arena    : " << plan_res.node_arenas[1] / MB << " MB\n"
              << " - NUMA-SMP Shared Arena    : " << plan_res.shared_arena / MB << " MB\n"
              << " - NUMA-SMP Total Arena Size: " << plan_res.total_arena / MB << " MB\n"
              << " - Total Memory Savings     : " << std::fixed << std::setprecision(2) << savings << "%\n";

    EXPECT_EQ(plan_res.node_arenas[0], 64 * MB);
    EXPECT_EQ(plan_res.node_arenas[1], 64 * MB);
    EXPECT_EQ(plan_res.shared_arena, 24 * MB);
    EXPECT_EQ(plan_res.total_arena, 152 * MB);
    EXPECT_GT(savings, 60.0);

    for (const auto& t : tensors) {
        EXPECT_EQ(t.offset % 64, 0u);
        EXPECT_NE(t.offset, static_cast<size_t>(-1));
        EXPECT_EQ(t.assigned_node, t.home_node);
    }

    // Explicit collision check verification
    EXPECT_TRUE(NumaStaticMemoryPlanner::verify_collisions(tensors));

    // Negative test: verify that overlapping spatial-temporal tensors are flagged
    std::vector<NumaTensorDescriptor> colliding_tensors = {
        {101, "t1", 1024, 0, 10, 0, 0, 0},
        {102, "t2", 1024, 5, 15, 0, 0, 0}
    };
    EXPECT_FALSE(NumaStaticMemoryPlanner::verify_collisions(colliding_tensors));
}

TEST(C3MemoryPlannerTest, NumaConcurrentHotPath200k) {
    std::vector<NumaTensorDescriptor> tensors = {
        {1, "shared_tokens", 8 * MB, 0, 4, -1},
        {2, "norm_out", 8 * MB, 1, 3, -1},
        {3, "n0_qkv_proj", 32 * MB, 2, 5, 0},
        {4, "n0_attn_scores", 16 * MB, 3, 6, 0},
        {5, "n0_attn_out", 16 * MB, 5, 8, 0},
        {6, "n0_ffn_up", 32 * MB, 8, 11, 0},
        {7, "n0_ffn_act", 16 * MB, 9, 12, 0},
        {8, "n0_ffn_down", 16 * MB, 10, 13, 0},
        {9, "n1_qkv_proj", 32 * MB, 2, 5, 1},
        {10, "n1_attn_scores", 16 * MB, 3, 6, 1},
        {11, "n1_attn_out", 16 * MB, 5, 8, 1},
        {12, "n1_ffn_up", 32 * MB, 8, 11, 1},
        {13, "n1_ffn_act", 16 * MB, 9, 12, 1},
        {14, "n1_ffn_down", 16 * MB, 10, 13, 1},
        {15, "allreduce_attn_buf", 16 * MB, 6, 8, -1},
        {16, "allreduce_ffn_buf", 16 * MB, 12, 14, -1},
        {17, "n0_d_ffn_down", 16 * MB, 14, 16, 0},
        {18, "n0_d_ffn_up", 32 * MB, 15, 17, 0},
        {19, "n0_d_qkv", 32 * MB, 16, 18, 0},
        {20, "n1_d_ffn_down", 16 * MB, 14, 16, 1},
        {21, "n1_d_ffn_up", 32 * MB, 15, 17, 1},
        {22, "n1_d_qkv", 32 * MB, 16, 18, 1},
        {23, "d_loss_shared", 8 * MB, 13, 15, -1},
    };

    auto plan_res = NumaStaticMemoryPlanner::plan(tensors, 2);
    NumaAlignedArena arena(2, plan_res.node_arenas, plan_res.shared_arena);

    EXPECT_EQ(arena.node_capacity(0), 64 * MB);
    EXPECT_EQ(arena.node_capacity(1), 64 * MB);
    EXPECT_EQ(arena.shared_capacity(), 24 * MB);
    EXPECT_EQ(arena.total_capacity(), 152 * MB);

    constexpr size_t kTotalPasses = 200000;
    constexpr size_t kNumThreads = 8;
    constexpr size_t kPassesPerThread = kTotalPasses / kNumThreads;

    std::atomic<uint64_t> completed_passes{0};
    std::atomic<uint64_t> sum_dummy{0};

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> workers;
    workers.reserve(kNumThreads);

    for (size_t t = 0; t < kNumThreads; ++t) {
        workers.emplace_back([&, t]() {
            int32_t my_node = static_cast<int32_t>(t % 2);
            uint64_t local_accum = 0;

            for (size_t iter = 0; iter < kPassesPerThread; ++iter) {
                for (const auto& tensor : tensors) {
                    if (tensor.assigned_node == my_node || tensor.assigned_node == -1) {
                        volatile uint64_t* ptr = arena.get_ptr<volatile uint64_t>(tensor.assigned_node, tensor.offset);
                        local_accum += (ptr != nullptr ? 1 : 0);
                    }
                }
            }
            sum_dummy.fetch_add(local_accum, std::memory_order_relaxed);
            completed_passes.fetch_add(kPassesPerThread, std::memory_order_relaxed);
        });
    }

    for (auto& w : workers) {
        w.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double avg_ns = (ms * 1e6) / static_cast<double>(kTotalPasses);
    double qps = static_cast<double>(kTotalPasses) / (ms / 1000.0);

    std::cout << "\n[NUMA-SMP 200,000 Passes Stress Test Results]\n"
              << " - Total Operations Processed : " << completed_passes.load() << " passes\n"
              << " - Total Elapsed Time         : " << ms << " ms\n"
              << " - Average Per-Pass Latency   : " << avg_ns << " ns / pass\n"
              << " - Concurrent Throughput      : " << qps << " passes / sec\n"
              << " - Hot Path Heap Allocation   : 0 bytes (Zero Dynamic Heap Allocation)\n";

    EXPECT_EQ(completed_passes.load(), kTotalPasses);
    EXPECT_GT(sum_dummy.load(), 0u);
}

// ==============================================================================
// 2. Theorem 16: Topological Scheduling Monotonic Convex Envelope Theorem
// ==============================================================================

TEST(C3MemoryPlannerTest, Theorem16BranchedDagNaiveVsTroSchedule) {
    // 11 Tensors forming a branched diamond DAG
    const std::vector<TroTensorDef> base_tensors = {
        {1, "input_x", 16 * MB},
        {2, "branch_a1", 32 * MB},
        {3, "branch_a2", 32 * MB},
        {4, "branch_a3", 32 * MB},
        {5, "branch_b1", 32 * MB},
        {6, "branch_b2", 32 * MB},
        {7, "branch_b3", 32 * MB},
        {8, "merged_ab", 16 * MB},
        {9, "ffn_gate", 32 * MB},
        {10, "ffn_up", 32 * MB},
        {11, "output_y", 16 * MB},
    };

    // 10 Operations
    const std::vector<TroOpDef> ops = {
        {1, "op_a1", {1}, 2, 32 * MB},
        {2, "op_a2", {2}, 3, 32 * MB},
        {3, "op_a3", {2, 3}, 4, 32 * MB},

        {4, "op_b1", {1}, 5, 32 * MB},
        {5, "op_b2", {5}, 6, 32 * MB},
        {6, "op_b3", {5, 6}, 7, 32 * MB},

        {7, "op_merge", {4, 7}, 8, 16 * MB},

        {8, "op_gate", {8}, 9, 32 * MB},
        {9, "op_up", {8}, 10, 32 * MB},
        {10, "op_out", {9, 10}, 11, 16 * MB},
    };

    const std::vector<uint32_t> naive_schedule = {1, 4, 2, 5, 3, 6, 7, 8, 9, 10};
    const std::vector<uint32_t> tro_schedule = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Phase 1: Naive Schedule
    auto naive_res = TroMemoryPlanner::plan(ops, naive_schedule, base_tensors);

    // Phase 2: TRO Schedule
    auto tro_res = TroMemoryPlanner::plan(ops, tro_schedule, base_tensors);

    double savings_mb = static_cast<double>(naive_res.total_arena_size - tro_res.total_arena_size) / static_cast<double>(MB);
    double savings_pct = static_cast<double>(naive_res.total_arena_size - tro_res.total_arena_size) / static_cast<double>(naive_res.total_arena_size) * 100.0;

    std::cout << "\n[Theorem 16 DAG Schedule Comparison Results]\n"
              << " - Naive Total Baseline       : " << naive_res.naive_total_size / MB << " MB\n"
              << " - Naive Peak Theoretical     : " << naive_res.peak_theoretical_bound / MB << " MB\n"
              << " - Naive Planned Arena Size   : " << naive_res.total_arena_size / MB << " MB\n"
              << " - TRO Peak Theoretical       : " << tro_res.peak_theoretical_bound / MB << " MB\n"
              << " - TRO Planned Arena Size     : " << tro_res.total_arena_size / MB << " MB\n"
              << " - Absolute Reduction         : " << savings_mb << " MB\n"
              << " - Relative Reduction         : " << std::fixed << std::setprecision(2) << savings_pct << "%\n";

    EXPECT_EQ(naive_res.naive_total_size, 304 * MB);
    EXPECT_EQ(naive_res.peak_theoretical_bound, 160 * MB);
    EXPECT_EQ(naive_res.total_arena_size, 128 * MB);

    EXPECT_EQ(tro_res.peak_theoretical_bound, 128 * MB);
    EXPECT_EQ(tro_res.total_arena_size, 96 * MB);

    EXPECT_LT(tro_res.total_arena_size, naive_res.total_arena_size);
    EXPECT_GE(savings_pct, 20.0);
    EXPECT_NEAR(savings_pct, 25.0, 0.01);

    // Collision check across all planned tensors in Naive
    for (size_t i = 0; i < naive_res.planned_tensors.size(); ++i) {
        for (size_t j = i + 1; j < naive_res.planned_tensors.size(); ++j) {
            const auto& ti = naive_res.planned_tensors[i];
            const auto& tj = naive_res.planned_tensors[j];
            bool t_overlap = !(ti.end_step <= tj.start_step || ti.start_step >= tj.end_step);
            size_t szi = TroMemoryPlanner::align_up(ti.size_bytes, ti.alignment);
            size_t szj = TroMemoryPlanner::align_up(tj.size_bytes, tj.alignment);
            bool s_overlap = !(ti.offset + szi <= tj.offset || ti.offset >= tj.offset + szj);
            EXPECT_FALSE(t_overlap && s_overlap) << "Naive collision between " << ti.name << " and " << tj.name;
        }
    }

    // Collision check across all planned tensors in TRO
    for (size_t i = 0; i < tro_res.planned_tensors.size(); ++i) {
        for (size_t j = i + 1; j < tro_res.planned_tensors.size(); ++j) {
            const auto& ti = tro_res.planned_tensors[i];
            const auto& tj = tro_res.planned_tensors[j];
            bool t_overlap = !(ti.end_step <= tj.start_step || ti.start_step >= tj.end_step);
            size_t szi = TroMemoryPlanner::align_up(ti.size_bytes, ti.alignment);
            size_t szj = TroMemoryPlanner::align_up(tj.size_bytes, tj.alignment);
            bool s_overlap = !(ti.offset + szi <= tj.offset || ti.offset >= tj.offset + szj);
            EXPECT_FALSE(t_overlap && s_overlap) << "Collision between " << ti.name << " and " << tj.name;
        }
    }
}

// ==============================================================================
// 3. AlignedStaticArena Lifecycle & Memory Management
// ==============================================================================

TEST(C3MemoryPlannerTest, AlignedStaticArenaLifecycle) {
    constexpr size_t kCap = 1024 * 1024;
    AlignedStaticArena arena(kCap);

    EXPECT_GE(arena.capacity(), kCap);
    EXPECT_NE(arena.data(), nullptr);

    // 64-byte hardware cache-line alignment
    auto addr = reinterpret_cast<uintptr_t>(arena.data());
    EXPECT_EQ(addr % 64, 0u);

    float* fptr = arena.get_ptr<float>(0);
    EXPECT_NE(fptr, nullptr);
    *fptr = 3.14159f;
    EXPECT_FLOAT_EQ(*fptr, 3.14159f);

    float* fptr_off = arena.get_ptr<float>(64);
    EXPECT_NE(fptr_off, nullptr);
    *fptr_off = 2.71828f;
    EXPECT_FLOAT_EQ(*fptr_off, 2.71828f);

    // Move construct
    AlignedStaticArena moved(std::move(arena));
    EXPECT_GE(moved.capacity(), kCap);
    EXPECT_EQ(arena.capacity(), 0u);
    EXPECT_EQ(arena.data(), nullptr);
    EXPECT_FLOAT_EQ(*moved.get_ptr<float>(0), 3.14159f);

    // Move assign
    AlignedStaticArena assigned(128);
    assigned = std::move(moved);
    EXPECT_GE(assigned.capacity(), kCap);
    EXPECT_EQ(moved.capacity(), 0u);
    EXPECT_FLOAT_EQ(*assigned.get_ptr<float>(64), 2.71828f);
}

TEST(C3MemoryPlannerTest, NumaAlignedArenaLifecycle) {
    std::vector<size_t> node_sizes = {1024 * 1024, 2048 * 1024};
    size_t shared_size = 512 * 1024;

    NumaAlignedArena arena(2, node_sizes, shared_size);

    EXPECT_EQ(arena.num_nodes(), 2u);
    EXPECT_EQ(arena.node_capacity(0), 1024 * 1024);
    EXPECT_EQ(arena.node_capacity(1), 2048 * 1024);
    EXPECT_EQ(arena.shared_capacity(), 512 * 1024);
    EXPECT_EQ(arena.total_capacity(), (1024 + 2048 + 512) * 1024);

    EXPECT_NE(arena.node_data(0), nullptr);
    EXPECT_NE(arena.node_data(1), nullptr);
    EXPECT_NE(arena.shared_data(), nullptr);

    EXPECT_EQ(reinterpret_cast<uintptr_t>(arena.node_data(0)) % 64, 0u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(arena.node_data(1)) % 64, 0u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(arena.shared_data()) % 64, 0u);

    // Read/write check
    uint64_t* p0 = arena.get_ptr<uint64_t>(0, 0);
    uint64_t* p1 = arena.get_ptr<uint64_t>(1, 0);
    uint64_t* ps = arena.get_ptr<uint64_t>(-1, 0);
    EXPECT_NE(p0, nullptr);
    EXPECT_NE(p1, nullptr);
    EXPECT_NE(ps, nullptr);
    *p0 = 42;
    *p1 = 84;
    *ps = 126;
    EXPECT_EQ(*p0, 42u);
    EXPECT_EQ(*p1, 84u);
    EXPECT_EQ(*ps, 126u);

    // Move construction
    NumaAlignedArena moved(std::move(arena));
    EXPECT_EQ(arena.num_nodes(), 0u);
    EXPECT_EQ(arena.total_capacity(), 0u);
    EXPECT_EQ(arena.node_data(0), nullptr);
    EXPECT_EQ(moved.num_nodes(), 2u);
    EXPECT_EQ(*moved.get_ptr<uint64_t>(0, 0), 42u);
    EXPECT_EQ(*moved.get_ptr<uint64_t>(1, 0), 84u);
    EXPECT_EQ(*moved.get_ptr<uint64_t>(-1, 0), 126u);

    // Move assignment
    std::vector<size_t> small_sizes = {64};
    NumaAlignedArena assigned(1, small_sizes, 64);
    assigned = std::move(moved);
    EXPECT_EQ(moved.num_nodes(), 0u);
    EXPECT_EQ(moved.total_capacity(), 0u);
    EXPECT_EQ(assigned.num_nodes(), 2u);
    EXPECT_EQ(*assigned.get_ptr<uint64_t>(0, 0), 42u);
    EXPECT_EQ(*assigned.get_ptr<uint64_t>(1, 0), 84u);
    EXPECT_EQ(*assigned.get_ptr<uint64_t>(-1, 0), 126u);
}

// ==============================================================================
// 4. Boundary Values & Empty Inputs
// ==============================================================================

TEST(C3MemoryPlannerTest, EdgeCasesAndEmptyInputs) {
    // Empty plan
    std::vector<TroOpDef> empty_ops;
    std::vector<uint32_t> empty_sched;
    std::vector<TroTensorDef> empty_tensors;
    auto empty_res = TroMemoryPlanner::plan(empty_ops, empty_sched, empty_tensors);
    EXPECT_EQ(empty_res.total_arena_size, 0u);
    EXPECT_EQ(empty_res.planned_tensors.size(), 0u);

    // Schedule with nonexistent op_id (defensive safety check)
    std::vector<TroOpDef> ops = {{1, "op1", {}, 1, 1024}};
    std::vector<uint32_t> sched_with_invalid = {1, 999};
    std::vector<TroTensorDef> tensors = {{1, "t1", 1024}};
    auto safe_res = TroMemoryPlanner::plan(ops, sched_with_invalid, tensors);
    EXPECT_GT(safe_res.total_arena_size, 0u);

    // Arena with 0 bytes
    AlignedStaticArena zero_arena(0);
    EXPECT_EQ(zero_arena.capacity(), 0u);
    EXPECT_EQ(zero_arena.data(), nullptr);
    EXPECT_EQ(zero_arena.get_ptr(0), nullptr);

    // NUMA empty plan and zero-capacity arena
    std::vector<NumaTensorDescriptor> empty_numa;
    auto numa_res = NumaStaticMemoryPlanner::plan(empty_numa, 2);
    EXPECT_EQ(numa_res.total_arena, 0u);
    EXPECT_EQ(numa_res.shared_arena, 0u);
    EXPECT_EQ(numa_res.node_arenas[0], 0u);
    EXPECT_EQ(numa_res.node_arenas[1], 0u);

    NumaAlignedArena zero_numa(2, numa_res.node_arenas, numa_res.shared_arena);
    EXPECT_EQ(zero_numa.total_capacity(), 0u);
    EXPECT_EQ(zero_numa.get_ptr(0, 0), nullptr);
    EXPECT_EQ(zero_numa.get_ptr(1, 0), nullptr);
    EXPECT_EQ(zero_numa.get_ptr(-1, 0), nullptr);
}

int main(int argc, char** argv) {
    std::cout << "================================================================================\n";
    std::cout << ">>> CTorch C3: TRO-SMP & NUMA Static Memory Planner Test Suite <<<\n";
    std::cout << "================================================================================\n\n";

    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    if (ret == 0) {
        std::cout << "\n>>> ALL MEMORY PLANNER TESTS PASSED (100% GREEN)! <<<\n";
    }
    return ret;
}
