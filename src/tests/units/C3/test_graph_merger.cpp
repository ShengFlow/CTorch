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

// ======================= P1-1 边界 case 测试 =======================
// 覆盖：多输出子图 / fan-out / 链接索引越界

/// 构造带 2 个输出的子图：in0, in1, in2 → MatMul → out0, in0 + out0 + in2 → Add → out1
/// （每个 input 都被使用，避免 DCE）
static Graph buildTwoOutputSubGraph(const std::vector<size_t>& in_shape,
                                      const std::vector<size_t>& w_shape,
                                      const std::vector<size_t>& b_shape,
                                      const std::vector<size_t>& out_shape) {
    Graph g;
    auto in_desc = TensorDesc::fromShape(in_shape);
    auto w_desc = TensorDesc::fromShape(w_shape);
    auto b_desc = TensorDesc::fromShape(b_shape);
    auto out_desc = TensorDesc::fromShape(out_shape);

    size_t in = g.addInput(in_desc);
    size_t w = g.addInput(w_desc);
    size_t b = g.addInput(b_desc);

    // MatMul: w @ in
    size_t mm = g.addNode(MatMulNode{w_desc, in_desc}, {w, in}, out_desc);
    // Add: mm + b
    size_t add = g.addNode(AddNode{out_desc, b_desc}, {mm, b}, out_desc);

    // 标记两个输出：mm 和 add（都是 markOutput）
    g.markOutput(mm);
    g.markOutput(add);
    return g;
}

TEST(GraphMergerEdgeCase, LastSubgraphMultiOutput) {
    // 子图 0：1 个输出（最后子图）
    // 子图 1：2 个输出（最后子图，多输出场景）
    // 验证"最后子图多输出"被正确保留
    Graph g0 = buildFCReLUSubGraph({4, 8}, {16, 8}, {16}, {4, 16}, true);
    Graph g1 = buildTwoOutputSubGraph({4, 16}, {32, 16}, {32}, {4, 32});

    std::vector<Graph> subs = {g0, g1};

    // link 0: g0.output[0] → g1.input[0]
    MergeSpec spec;
    MergeLink link;
    link.from_output = 0;
    link.to_input = 0;
    spec.links.push_back(link);

    MergedGraphInfo info = GraphMerger::merge(subs, spec);

    // 外部输入：g0 的 in, w, b + g1 的 w, b = 5 个
    EXPECT_EQ(info.graph.inputCount(), 5u);
    // g1 是最后子图，2 个输出都被保留
    EXPECT_EQ(info.graph.outputCount(), 2u);
    EXPECT_TRUE(info.graph.isValid());

    // 应能继续 canonicalize + fuse
    Graph fused = info.graph.canonicalize().fuse();
    EXPECT_TRUE(fused.isValid());
}

TEST(GraphMergerEdgeCase, LinkFromOutputOutOfRange) {
    // link.from_output 超过子图 0 的 outputCount
    Graph g0 = buildFCReLUSubGraph({4, 8}, {16, 8}, {16}, {4, 16}, true);
    Graph g1 = buildFCReLUSubGraph({4, 16}, {32, 16}, {32}, {4, 32}, false);

    std::vector<Graph> subs = {g0, g1};

    MergeSpec spec;
    MergeLink link;
    link.from_output = 999;  // 越界
    link.to_input = 0;
    spec.links.push_back(link);

    EXPECT_THROW(GraphMerger::merge(subs, spec), std::invalid_argument);
}

TEST(GraphMergerEdgeCase, LinkToInputOutOfRange) {
    // link.to_input 超过子图 1 的 inputCount
    Graph g0 = buildFCReLUSubGraph({4, 8}, {16, 8}, {16}, {4, 16}, true);
    Graph g1 = buildFCReLUSubGraph({4, 16}, {32, 16}, {32}, {4, 32}, false);

    std::vector<Graph> subs = {g0, g1};

    MergeSpec spec;
    MergeLink link;
    link.from_output = 0;
    link.to_input = 999;  // 越界
    spec.links.push_back(link);

    EXPECT_THROW(GraphMerger::merge(subs, spec), std::invalid_argument);
}

TEST(GraphMergerEdgeCase, SequentialFanOut) {
    // 构造子图 0: 1 个输出
    // 构造子图 1: 3 个 input（in0, in1, in2），其中 in0 来自子图 0
    //   子图逻辑：Add(in0, in1) → Add(., in2) → ReLU → out
    //   （确保 3 个 input 都被使用）
    Graph g0 = buildFCReLUSubGraph({4, 8}, {16, 8}, {16}, {4, 16}, true);

    Graph g1;
    auto in0_desc = TensorDesc::fromShape({4, 16});
    auto out_desc = TensorDesc::fromShape({4, 16});
    size_t i0 = g1.addInput(in0_desc);
    size_t i1 = g1.addInput(in0_desc);  // 同 shape
    size_t i2 = g1.addInput(in0_desc);  // 同 shape
    // Add(in0, in1) → out1
    size_t add1 = g1.addNode(AddNode{in0_desc, in0_desc}, {i0, i1}, out_desc);
    // Add(out1, in2) → out2
    size_t add2 = g1.addNode(AddNode{out_desc, in0_desc}, {add1, i2}, out_desc);
    // ReLU(out2) → out
    size_t out_id = g1.addNode(ReLUNode{out_desc}, {add2}, out_desc);
    g1.markOutput(out_id);

    // spec：link g0.output[0] → g1.input[0]
    MergeSpec spec;
    MergeLink link;
    link.from_output = 0;
    link.to_input = 0;
    spec.links.push_back(link);

    MergedGraphInfo info = GraphMerger::merge({g0, g1}, spec);
    // 外部输入：g0 的 in, w, b + g1 的 i1, i2 = 5 个（3 个 input 都用上）
    EXPECT_EQ(info.graph.inputCount(), 5u);
    EXPECT_TRUE(info.graph.isValid());
}
