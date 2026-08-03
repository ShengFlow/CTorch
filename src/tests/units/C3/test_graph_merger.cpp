/**
 * @file test_graph_merger.cpp
 * @brief GraphMerger 单元测试
 * @date 2026/8/2
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "C3/Graph.h"
#include "C3/GraphMerger.h"

using namespace ct::c3;

// ======================= 辅助：构建一个简单子图 =======================

/// 构建子图：in0, in1 → MatMul → Add(bias) → ReLU → out
static Graph buildFCReLUSubGraph(const std::vector<size_t>& in_shape,
                                   const std::vector<size_t>& w_shape,
                                   const std::vector<size_t>& b_shape,
                                   const std::vector<size_t>& out_shape,
                                   bool add_relu = true) {
    Graph g;
    auto in_desc = TensorDesc::fromShape(in_shape);
    auto w_desc = TensorDesc::fromShape(w_shape);
    auto b_desc = TensorDesc::fromShape(b_shape);
    auto out_desc = TensorDesc::fromShape(out_shape);

    size_t in = g.addInput(in_desc);
    size_t w = g.addInput(w_desc);
    size_t b = g.addInput(b_desc);

    // MatMul(w, in)
    auto mm_desc = TensorDesc::fromShape(out_shape);
    size_t mm = g.addNode(MatMulNode{w_desc, in_desc}, {w, in}, mm_desc);
    // Add(mm, b)
    size_t add = g.addNode(AddNode{mm_desc, b_desc}, {mm, b}, out_desc);
    size_t out_id = add;
    if (add_relu) {
        out_id = g.addNode(ReLUNode{out_desc}, {add}, out_desc);
    }
    g.markOutput(out_id);
    return g;
}

// ======================= 测试 =======================

TEST(GraphMerger, EmptyInput) {
    std::vector<Graph> subs;
    MergeSpec spec;
    EXPECT_THROW(GraphMerger::merge(subs, spec), std::invalid_argument);
}

TEST(GraphMerger, SingleSubgraph) {
    // 单子图应原样返回
    Graph g = buildFCReLUSubGraph({4}, {4, 4}, {4}, {4});
    std::vector<Graph> subs = {g};
    MergeSpec spec;

    MergedGraphInfo info = GraphMerger::merge(subs, spec);
    EXPECT_EQ(info.graph.nodeCount(), g.nodeCount());
    EXPECT_EQ(info.graph.inputCount(), g.inputCount());
    EXPECT_EQ(info.graph.outputCount(), g.outputCount());
}

TEST(GraphMerger, SequentialMerge_TwoLayers) {
    // 子图 1: in=[M, K], w1=[N, K], b1=[N] → out=[M, N]
    // 子图 2: in=[M, N], w2=[P, N], b2=[P] → out=[M, P]
    Graph g1 = buildFCReLUSubGraph({4, 8}, {16, 8}, {16}, {4, 16}, true);
    Graph g2 = buildFCReLUSubGraph({4, 16}, {32, 16}, {32}, {4, 32}, false);

    std::vector<Graph> subs = {g1, g2};
    MergedGraphInfo info = GraphMerger::mergeSequential(subs);

    // 期望融合图有 4 个外部输入（w1, b1, w2, b2），1 个最终输出
    // 链式：x (g1 in[0]) → g1 → out1 → g2.in[0] → g2 → out
    // 实际上：g1.in[0] = x 是外部输入，g1.in[1] = w1, g1.in[2] = b1
    //         g2.in[0] = 由 g1 提供（内部），g2.in[1] = w2, g2.in[2] = b2
    // 所以外部输入：x, w1, b1, w2, b2 = 5 个
    EXPECT_EQ(info.graph.inputCount(), 5u);
    // 输出：1 个
    EXPECT_EQ(info.graph.outputCount(), 1u);

    // external_input_ids 应包含 5 个唯一 ID
    EXPECT_EQ(info.external_input_ids.size(), 5u);

    // 拓扑检查：图有效
    EXPECT_TRUE(info.graph.isValid());
}

TEST(GraphMerger, SequentialMerge_ThreeLayers) {
    // 3 层 MLP
    Graph g1 = buildFCReLUSubGraph({8, 16}, {32, 16}, {32}, {8, 32}, true);
    Graph g2 = buildFCReLUSubGraph({8, 32}, {16, 32}, {16}, {8, 16}, true);
    Graph g3 = buildFCReLUSubGraph({8, 16}, {4, 16},  {4},  {8, 4},  false);

    std::vector<Graph> subs = {g1, g2, g3};
    MergedGraphInfo info = GraphMerger::mergeSequential(subs);

    // 外部输入：x, w1, b1, w2, b2, w3, b3 = 7 个
    EXPECT_EQ(info.graph.inputCount(), 7u);
    EXPECT_EQ(info.graph.outputCount(), 1u);
    EXPECT_EQ(info.external_input_ids.size(), 7u);
    EXPECT_TRUE(info.graph.isValid());
}

TEST(GraphMerger, ShapeMismatch) {
    // 子图 1 输出 [4, 8]，子图 2 输入期望 [4, 8]，但实际是 [4, 7] → 应失败
    Graph g1 = buildFCReLUSubGraph({4, 8}, {16, 8}, {16}, {4, 16}, true);
    Graph g2 = buildFCReLUSubGraph({4, 7}, {32, 7}, {32}, {4, 32}, false);

    std::vector<Graph> subs = {g1, g2};
    MergeSpec spec;
    spec.links.push_back({0, 0});

    EXPECT_THROW(GraphMerger::merge(subs, spec), std::invalid_argument);
}

TEST(GraphMerger, LinkCountMismatch) {
    // 2 个子图需要 1 个链接，但提供 2 个 → 应失败
    Graph g1 = buildFCReLUSubGraph({4, 8}, {16, 8}, {16}, {4, 16}, true);
    Graph g2 = buildFCReLUSubGraph({4, 16}, {32, 16}, {32}, {4, 32}, false);

    std::vector<Graph> subs = {g1, g2};
    MergeSpec spec;
    spec.links.push_back({0, 0});
    spec.links.push_back({0, 0});  // 多余的链接

    EXPECT_THROW(GraphMerger::merge(subs, spec), std::invalid_argument);
}

TEST(GraphMerger, OutOfRangeLink) {
    // 链接目标输入索引越界
    Graph g1 = buildFCReLUSubGraph({4, 8}, {16, 8}, {16}, {4, 16}, true);
    Graph g2 = buildFCReLUSubGraph({4, 16}, {32, 16}, {32}, {4, 32}, false);

    std::vector<Graph> subs = {g1, g2};
    MergeSpec spec;
    spec.links.push_back({0, 999});  // 越界

    EXPECT_THROW(GraphMerger::merge(subs, spec), std::invalid_argument);
}

TEST(GraphMerger, SequentialCanFuseAndOptimize) {
    // 融合图应能继续做 canonicalize + fuse
    Graph g1 = buildFCReLUSubGraph({4, 8}, {16, 8}, {16}, {4, 16}, true);
    Graph g2 = buildFCReLUSubGraph({4, 16}, {32, 16}, {32}, {4, 32}, false);

    std::vector<Graph> subs = {g1, g2};
    MergedGraphInfo info = GraphMerger::mergeSequential(subs);

    // canonicalize 应能处理（无变化，因为无常量折叠机会）
    Graph canonical = info.graph.canonicalize();
    EXPECT_TRUE(canonical.isValid());

    // fuse 应能处理
    Graph fused = canonical.fuse();
    EXPECT_TRUE(fused.isValid());
}
