/**
 * @file test_c3_graph_fusion_clusterer.cpp
 * @brief GTest Suite for C3 Dynamic Operator Auto-Fusion Engine & Greedy Subgraph Clusterer.
 * @details Covers:
 *   1. Transformer SwiGLU FFN Auto-Fusion Benchmark (>50% DRAM bandwidth savings)
 *   2. Complex Diamond Graph DAG Invariant & Acyclicity Verification
 *   3. High-Concurrency Multi-Threaded Stress Test (100,000 Graph Partitions)
 *   4. Edge cases: Single node, Barrier op preservation, Empty graphs
 */

#include "C3/C3GraphFusionClusterer.h"

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <iomanip>
#include <random>

using namespace ct::c3;

// ==============================================================================
// 1. SwiGLU FFN Auto-Fusion Benchmark
// ==============================================================================

TEST(C3GraphFusionClustererTest, SwigluFfnAutoFusion) {
    ComputationGraph g;
    // 0: Input_X
    g.add_node(0, "Input_X", OpKind::ELEMENTWISE, {}, 4096);
    // 1: Gate_GEMM
    g.add_node(1, "Gate_GEMM", OpKind::GEMM, {0}, 16384);
    // 2: Up_GEMM
    g.add_node(2, "Up_GEMM", OpKind::GEMM, {0}, 16384);
    // 3: SiLU_Act
    g.add_node(3, "SiLU_Act", OpKind::ELEMENTWISE, {1}, 16384);
    // 4: SwiGLU_Mul
    g.add_node(4, "SwiGLU_Mul", OpKind::ELEMENTWISE, {3, 2}, 16384);
    // 5: Residual_Add
    g.add_node(5, "Residual_Add", OpKind::ELEMENTWISE, {4, 0}, 4096);
    // 6: LayerNorm (Reduction)
    g.add_node(6, "LayerNorm", OpKind::REDUCTION, {5}, 4096);

    C3GraphFusionClusterer clusterer(g);
    clusterer.run_greedy_clustering();

    auto stats = clusterer.evaluate_dram_traffic();
    std::cout << "\n[SwiGLU FFN Auto-Fusion Results]\n"
              << " - Initial Nodes Count   : " << g.nodes().size() << " nodes\n"
              << " - Fused Clusters Count  : " << clusterer.clusters().size() << " clusters\n"
              << " - Unfused DRAM Traffic  : " << stats.unfused_dram_bytes / 1024.0 << " KB\n"
              << " - Fused DRAM Traffic    : " << stats.fused_dram_bytes / 1024.0 << " KB\n"
              << " - DRAM Traffic Saved    : " << std::fixed << std::setprecision(2) << stats.savings_ratio * 100.0 << "%\n";

    for (const auto& [cid, cluster] : clusterer.clusters()) {
        std::cout << "   Cluster " << cid << ": ";
        for (uint32_t nid : cluster.node_ids) {
            std::cout << g.nodes().at(nid).name << " ";
        }
        std::cout << "\n";
    }

    EXPECT_GT(stats.savings_ratio, 0.50);
    EXPECT_LE(clusterer.clusters().size(), 3u);
}

// ==============================================================================
// 2. Complex Diamond Graph DAG Invariant & Acyclicity Verification
// ==============================================================================

TEST(C3GraphFusionClustererTest, DagAcyclicityInvariant) {
    ComputationGraph g;
    // Diamond with bypass edge:
    // 0 -> 1 -> 3
    // 0 -> 2 -> 3
    // 0 -> 3 (direct bypass)
    g.add_node(0, "Source", OpKind::ELEMENTWISE, {}, 1024);
    g.add_node(1, "BranchA", OpKind::ELEMENTWISE, {0}, 1024);
    g.add_node(2, "BranchB", OpKind::ELEMENTWISE, {0}, 1024);
    g.add_node(3, "JoinNode", OpKind::ELEMENTWISE, {1, 2, 0}, 1024);

    C3GraphFusionClusterer clusterer(g);
    clusterer.run_greedy_clustering();

    // Verify topological sorting on cluster DAG
    std::unordered_map<uint32_t, size_t> in_degrees;
    for (const auto& [cid, _] : clusterer.clusters()) {
        in_degrees[cid] = 0;
    }

    // Build cluster-level edges
    std::unordered_map<uint32_t, std::unordered_set<uint32_t>> cluster_edges;
    for (const auto& [u, node] : g.nodes()) {
        uint32_t cu = 0;
        for (const auto& [cid, cl] : clusterer.clusters()) {
            if (cl.node_ids.contains(u)) { cu = cid; break; }
        }
        auto it = g.edges().find(u);
        if (it != g.edges().end()) {
            for (uint32_t v : it->second) {
                uint32_t cv = 0;
                for (const auto& [cid, cl] : clusterer.clusters()) {
                    if (cl.node_ids.contains(v)) { cv = cid; break; }
                }
                if (cu != cv) {
                    if (!cluster_edges[cu].contains(cv)) {
                        cluster_edges[cu].insert(cv);
                        in_degrees[cv]++;
                    }
                }
            }
        }
    }

    std::queue<uint32_t> q;
    for (const auto& [cid, deg] : in_degrees) {
        if (deg == 0) q.push(cid);
    }

    size_t visited = 0;
    while (!q.empty()) {
        uint32_t curr = q.front();
        q.pop();
        visited++;
        for (uint32_t nxt : cluster_edges[curr]) {
            in_degrees[nxt]--;
            if (in_degrees[nxt] == 0) q.push(nxt);
        }
    }

    std::cout << "\n[DAG Acyclicity Invariant Results]\n"
              << " - Visited Clusters Count : " << visited << " / " << clusterer.clusters().size() << "\n";

    EXPECT_EQ(visited, clusterer.clusters().size());
}

TEST(C3GraphFusionClustererTest, DiamondBypassCyclePrevention) {
    ComputationGraph g;
    // Diamond with bypass edge and barrier on branch A:
    // 0 (Source) -> 1 (Barrier) -> 3 (Join)
    // 0 -> 2 (BranchB) -> 3
    // 0 -> 3 (direct bypass)
    g.add_node(0, "Source", OpKind::ELEMENTWISE, {}, 1024);
    g.add_node(1, "BranchA_Barrier", OpKind::BARRIER, {0}, 1024);
    g.add_node(2, "BranchB", OpKind::ELEMENTWISE, {0}, 1024);
    g.add_node(3, "JoinNode", OpKind::ELEMENTWISE, {1, 2, 0}, 1024);

    C3GraphFusionClusterer clusterer(g);
    clusterer.run_greedy_clustering();

    // Barrier cannot fuse.
    // Source (0) cannot fuse with JoinNode (3) because of the indirect path 0 -> 1 -> 3.
    // Therefore, clusters count must be >= 3 and 0 and 3 cannot be in the same cluster.
    EXPECT_GE(clusterer.clusters().size(), 3u);

    for (const auto& [cid, cl] : clusterer.clusters()) {
        bool has_0 = cl.node_ids.contains(0);
        bool has_3 = cl.node_ids.contains(3);
        EXPECT_FALSE(has_0 && has_3) << "Source and JoinNode must NOT be fused across barrier indirect path!";
    }

    // Verify cluster DAG topological sort
    const auto& c_adj = clusterer.cluster_adj();
    std::unordered_map<uint32_t, size_t> in_deg;
    for (const auto& [cid, _] : clusterer.clusters()) {
        in_deg[cid] = 0;
    }
    for (const auto& [cu, targets] : c_adj) {
        for (uint32_t cv : targets) {
            in_deg[cv]++;
        }
    }
    std::queue<uint32_t> q;
    for (const auto& [cid, deg] : in_deg) {
        if (deg == 0) q.push(cid);
    }
    size_t visited = 0;
    while (!q.empty()) {
        uint32_t curr = q.front();
        q.pop();
        visited++;
        auto it = c_adj.find(curr);
        if (it != c_adj.end()) {
            for (uint32_t nxt : it->second) {
                in_deg[nxt]--;
                if (in_deg[nxt] == 0) q.push(nxt);
            }
        }
    }
    EXPECT_EQ(visited, clusterer.clusters().size());
}

TEST(C3GraphFusionClustererTest, MonteCarlo500RandomDagVerification) {
    std::mt19937 rng(42);
    constexpr size_t kTotalGraphs = 500;
    double total_savings = 0.0;

    for (size_t g_idx = 0; g_idx < kTotalGraphs; ++g_idx) {
        size_t num_nodes = 10 + (rng() % 21); // 10 to 30 nodes
        ComputationGraph g;

        for (size_t i = 0; i < num_nodes; ++i) {
            size_t num_inputs = (i > 0) ? (rng() % std::min<size_t>(4, i + 1)) : 0;
            std::vector<uint32_t> inputs;
            if (num_inputs > 0) {
                std::vector<uint32_t> candidates(i);
                for (size_t c = 0; c < i; ++c) candidates[c] = static_cast<uint32_t>(c);
                std::shuffle(candidates.begin(), candidates.end(), rng);
                inputs.assign(candidates.begin(), candidates.begin() + num_inputs);
            }
            int kind_roll = rng() % 4;
            OpKind kind = (kind_roll == 0) ? OpKind::GEMM :
                          (kind_roll == 1) ? OpKind::REDUCTION :
                          (kind_roll == 2) ? OpKind::BARRIER : OpKind::ELEMENTWISE;
            size_t out_bytes = (1 + (rng() % 8)) * 1024;
            g.add_node(static_cast<uint32_t>(i), "Node_" + std::to_string(i), kind, inputs, out_bytes);
        }

        C3GraphFusionClusterer clusterer(g);
        clusterer.run_greedy_clustering();

        // Topological sort on cluster DAG
        const auto& c_adj = clusterer.cluster_adj();
        std::unordered_map<uint32_t, size_t> in_deg;
        for (const auto& [cid, _] : clusterer.clusters()) {
            in_deg[cid] = 0;
        }
        for (const auto& [cu, targets] : c_adj) {
            for (uint32_t cv : targets) {
                in_deg[cv]++;
            }
        }
        std::queue<uint32_t> q;
        for (const auto& [cid, deg] : in_deg) {
            if (deg == 0) q.push(cid);
        }
        size_t visited = 0;
        while (!q.empty()) {
            uint32_t curr = q.front();
            q.pop();
            visited++;
            auto it = c_adj.find(curr);
            if (it != c_adj.end()) {
                for (uint32_t nxt : it->second) {
                    in_deg[nxt]--;
                    if (in_deg[nxt] == 0) q.push(nxt);
                }
            }
        }

        ASSERT_EQ(visited, clusterer.clusters().size())
            << "Cluster DAG cycle detected in random graph index " << g_idx;
        auto stats = clusterer.evaluate_dram_traffic();
        total_savings += stats.savings_ratio;
    }

    double avg_savings = total_savings / static_cast<double>(kTotalGraphs);
    std::cout << "\n[Monte Carlo 500 Random Dynamic DAGs Results]\n"
              << " - Total DAGs Evaluated   : " << kTotalGraphs << "\n"
              << " - Acyclicity Assertion   : 100% Passed (Zero cycles across all 500 graphs)\n"
              << " - Average DRAM Savings   : " << std::fixed << std::setprecision(2) << avg_savings * 100.0 << "%\n";

    // Following P1-05 hardening, sequential GEMM->GEMM vertical merging is disallowed,
    // resulting in ~17.16% average DRAM savings across pseudo-random DAGs with 25% Barrier nodes.
    EXPECT_GT(avg_savings, 0.15);
}

TEST(C3GraphFusionClustererTest, DisallowSequentialGemmVerticalFusion) {
    ComputationGraph g;
    g.add_node(0, "GEMM_1", OpKind::GEMM, {});
    g.add_node(1, "GEMM_2", OpKind::GEMM, {0}); // Direct edge GEMM_1 -> GEMM_2

    C3GraphFusionClusterer clusterer(g);
    clusterer.run_greedy_clustering();

    // Sequential producer-consumer GEMMs must remain 2 separate clusters
    EXPECT_EQ(clusterer.clusters().size(), 2);
}

// ==============================================================================
// 3. High-Concurrency Multi-Threaded Stress Test (100,000 Graph Partitions)
// ==============================================================================

TEST(C3GraphFusionClustererTest, HighConcurrencyClustererStress100k) {
    constexpr size_t kNumThreads = 4;
    constexpr size_t kPassesPerThread = 25000;
    constexpr size_t kTotalPasses = kNumThreads * kPassesPerThread;

    std::vector<std::thread> workers;
    workers.reserve(kNumThreads);

    auto t0 = std::chrono::steady_clock::now();

    for (size_t tid = 0; tid < kNumThreads; ++tid) {
        workers.emplace_back([tid]() {
            for (size_t p = 0; p < kPassesPerThread; ++p) {
                ComputationGraph g;
                g.add_node(0, "X", OpKind::ELEMENTWISE, {}, 1024);
                g.add_node(1, "W1", OpKind::GEMM, {0}, 2048);
                g.add_node(2, "Act", OpKind::ELEMENTWISE, {1}, 2048);
                g.add_node(3, "Norm", OpKind::REDUCTION, {2}, 1024);

                C3GraphFusionClusterer clusterer(g);
                clusterer.run_greedy_clustering();

                auto stats = clusterer.evaluate_dram_traffic();
                EXPECT_GT(stats.savings_ratio, 0.0);
            }
        });
    }

    for (auto& w : workers) {
        w.join();
    }

    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_us = (ms * 1000.0) / static_cast<double>(kTotalPasses);
    double qps = (static_cast<double>(kTotalPasses) / (ms / 1000.0));

    std::cout << "\n[100,000 Graphs Partitioning Concurrency Stress Results]\n"
              << " - Total Graphs Partitioned : " << kTotalPasses << " graphs\n"
              << " - Total Elapsed Time       : " << ms << " ms\n"
              << " - Average Per-Graph Latency: " << avg_us << " us / graph\n"
              << " - Graph Clustering QPS     : " << qps << " graphs / sec\n";

    EXPECT_GT(qps, 10000.0);
}

// ==============================================================================
// 4. Barrier Op & Edge Cases
// ==============================================================================

TEST(C3GraphFusionClustererTest, BarrierPreventsFusion) {
    ComputationGraph g;
    // 0 -> Barrier -> 2
    g.add_node(0, "A", OpKind::ELEMENTWISE, {}, 1024);
    g.add_node(1, "B_Barrier", OpKind::BARRIER, {0}, 1024);
    g.add_node(2, "C", OpKind::ELEMENTWISE, {1}, 1024);

    C3GraphFusionClusterer clusterer(g);
    clusterer.run_greedy_clustering();

    // Barrier cannot be merged into any cluster; 3 nodes must stay in separate clusters
    EXPECT_EQ(clusterer.clusters().size(), 3u);
}

TEST(C3GraphFusionClustererTest, EmptyAndSingleNodeGraph) {
    // Empty graph
    ComputationGraph empty_g;
    C3GraphFusionClusterer empty_c(empty_g);
    empty_c.run_greedy_clustering();
    EXPECT_EQ(empty_c.clusters().size(), 0u);
    auto empty_stats = empty_c.evaluate_dram_traffic();
    EXPECT_EQ(empty_stats.unfused_dram_bytes, 0u);
    EXPECT_EQ(empty_stats.fused_dram_bytes, 0u);
    EXPECT_DOUBLE_EQ(empty_stats.savings_ratio, 0.0);

    // Single node graph
    ComputationGraph single_g;
    single_g.add_node(0, "Solo", OpKind::ELEMENTWISE, {}, 512);
    C3GraphFusionClusterer single_c(single_g);
    single_c.run_greedy_clustering();
    EXPECT_EQ(single_c.clusters().size(), 1u);
}

int main(int argc, char** argv) {
    std::cout << "================================================================================\n";
    std::cout << ">>> CTorch C3 JIT: Dynamic Operator Auto-Fusion Engine Test Suite <<<\n";
    std::cout << "================================================================================\n\n";

    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    if (ret == 0) {
        std::cout << "\n>>> ALL C3 AUTO-FUSION CLUSTERER TESTS PASSED (100% GREEN)! <<<\n";
    }
    return ret;
}
