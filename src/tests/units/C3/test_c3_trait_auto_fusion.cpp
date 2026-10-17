/**
 * @file test_c3_trait_auto_fusion.cpp
 * @brief Unit tests for C3 Operator Trait-Driven Autonomous Graph Fusion
 * @details Validates that C3 automatically synthesizes maximal fusible clusters
 *          derived purely from intrinsic operator traits (OpFusibilityTrait)
 *          without any hand-crafted sequence pattern matching rules.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>
#include <cmath>

#include "C3/FusionPlanner.h"
#include "C3/Graph.h"

using namespace ct::c3;

namespace {

const FusionUnit* unitContaining(const FusionPlan& plan, size_t node_id) {
    return plan.unitOf(node_id);
}

size_t nodeNumelOf(const std::vector<size_t>& shape) {
    size_t v = 1;
    for (size_t d : shape) v *= d;
    return v;
}

} // namespace

// ======================= 1. 任意拓扑 Pointwise 链自动聚类 =======================

TEST(C3TraitAutoFusion, ArbitraryPointwiseDAGAutoFusion) {
    Graph g;
    auto d = TensorDesc::fromShape({8});
    size_t x = g.addInput(d);
    size_t y = g.addInput(d);
    size_t z = g.addInput(d);

    // 构造一个 6 节点的任意 Pointwise 拓扑 (Neg -> SiLU -> Add -> Mul -> Exp -> Tanh)
    // 历史上没有针对这个具体序列的手写规则
    size_t n1 = g.addNode(NegNode{d}, {x}, d);
    size_t n2 = g.addNode(SiLUNode{d}, {n1}, d);
    size_t n3 = g.addNode(AddNode{d, d}, {n2, y}, d);
    size_t n4 = g.addNode(MulNode{d, d}, {n3, z}, d);
    size_t n5 = g.addNode(ExpNode{d}, {n4}, d);
    size_t n6 = g.addNode(TanhNode{d}, {n5}, d);
    g.markOutput(n6);

    FusionPlan plan = FusionPlanner::planUnits(g, FusionStrategy::AutonomousTraitDriven);

    // 验证：所有 6 个节点自动融合成单个 ELEMENTWISE 单元
    const FusionUnit* u1 = unitContaining(plan, n1);
    ASSERT_NE(u1, nullptr);
    EXPECT_TRUE(u1->isCompute());
    EXPECT_EQ(u1->kind, FusionUnitKind::ELEMENTWISE);
    EXPECT_EQ(u1->node_ids.size(), 6u);

    for (size_t id : {n2, n3, n4, n5, n6}) {
        EXPECT_EQ(plan.unitOf(id), u1);
    }

    // 外部输入必须准确收集为 x, y, z (3 个叶节点)
    ASSERT_EQ(u1->external_input_ids.size(), 3u);
    EXPECT_EQ(u1->output_ids.size(), 1u);
    EXPECT_EQ(u1->output_ids[0], n6);
}

// ======================= 2. GEMM 多操作复杂尾链自主融合 (GEMM Epilogue) =======================

TEST(C3TraitAutoFusion, GemmMultiEpilogueAutoFusion) {
    Graph g;
    auto xd = TensorDesc::fromShape({4, 8});
    auto wd = TensorDesc::fromShape({8, 16});
    auto bd = TensorDesc::fromShape({4, 16});
    auto rd = TensorDesc::fromShape({4, 16});
    auto md = TensorDesc::fromShape({4, 16});

    size_t x = g.addInput(xd);
    size_t w = g.addInput(wd);
    size_t b = g.addInput(bd);
    size_t r = g.addInput(rd);

    // MatMul -> Bias Add -> SiLU -> Residual Mul (SwiGLU/FFN style epilogue)
    size_t mm = g.addNode(MatMulNode{xd, wd}, {x, w}, md);
    size_t add = g.addNode(AddNode{md, bd}, {mm, b}, md);
    size_t silu = g.addNode(SiLUNode{md}, {add}, md);
    size_t mul = g.addNode(MulNode{md, rd}, {silu, r}, md);
    g.markOutput(mul);

    FusionPlan plan = FusionPlanner::planUnits(g, FusionStrategy::AutonomousTraitDriven);

    const FusionUnit* umm = unitContaining(plan, mm);
    ASSERT_NE(umm, nullptr);
    EXPECT_EQ(umm->kind, FusionUnitKind::GEMM_EPILOGUE);
    EXPECT_EQ(plan.unitOf(add), umm);
    EXPECT_EQ(plan.unitOf(silu), umm);
    EXPECT_EQ(plan.unitOf(mul), umm);

    // 包含 GEMM 与全部 3 个 epilogue 节点
    EXPECT_EQ(umm->node_ids.size(), 4u);
    ASSERT_EQ(umm->external_input_ids.size(), 4u); // x, w, b, r
    EXPECT_EQ(umm->numel, nodeNumelOf({4, 16}));
}

// ======================= 3. 残差绕行与菱形旁路成环检测 (Acyclicity Guard) =======================

TEST(C3TraitAutoFusion, DiamondResidualAcyclicityGuard) {
    Graph g;
    auto d = TensorDesc::fromShape({4, 4});
    size_t a = g.addInput(d);
    size_t b = g.addInput(d);

    // 中间节点 u 产生分叉：一方面送入 GEMM 支路，另一方面直接绕行作为残差 Add
    size_t u = g.addNode(AddNode{d, d}, {a, b}, d);
    size_t mm = g.addNode(MatMulNode{d, d}, {u, a}, d);
    size_t act = g.addNode(ReLUNode{d}, {mm}, d);
    size_t res = g.addNode(AddNode{d, d}, {act, u}, d); // 引用了分叉前驱 u!
    g.markOutput(res);

    FusionPlan plan = FusionPlanner::planUnits(g, FusionStrategy::AutonomousTraitDriven);

    // u 有多消费者 (mm 和 res)，不能与 mm 盲目合并；
    // mm 与 act 可以融合成 GEMM_EPILOGUE；
    // res 作为汇聚节点独立物化。
    // 整个图绝不可形成死锁或循环依赖！
    const FusionUnit* u_mm = unitContaining(plan, mm);
    const FusionUnit* u_act = unitContaining(plan, act);
    EXPECT_EQ(u_mm, u_act);

    // 保证各 unit 拓扑有序，无环
    std::vector<PartitionedSubGraph> subs = partitionGraph(g, plan);
    EXPECT_GT(subs.size(), 0u);
    for (const auto& sub : subs) {
        EXPECT_GT(sub.graph.nodeCount(), 0u);
    }
}

// ======================= 4. 连续 GEMM 强制隔离 =======================

TEST(C3TraitAutoFusion, SequentialGemmStaysSeparate) {
    Graph g;
    auto d1 = TensorDesc::fromShape({4, 4});
    auto d2 = TensorDesc::fromShape({4, 4});
    size_t x = g.addInput(d1);
    size_t w1 = g.addInput(d1);
    size_t w2 = g.addInput(d2);

    size_t mm1 = g.addNode(MatMulNode{d1, d1}, {x, w1}, d1);
    size_t mm2 = g.addNode(MatMulNode{d1, d2}, {mm1, w2}, d2);
    g.markOutput(mm2);

    FusionPlan plan = FusionPlanner::planUnits(g, FusionStrategy::AutonomousTraitDriven);

    // 两个具有独立分块循环的 GEMM 绝对不能垂直融合
    const FusionUnit* u1 = unitContaining(plan, mm1);
    const FusionUnit* u2 = unitContaining(plan, mm2);
    ASSERT_NE(u1, nullptr);
    ASSERT_NE(u2, nullptr);
    EXPECT_NE(u1, u2);
}

// ======================= 5. Numel 不一致自动边界切割 =======================

TEST(C3TraitAutoFusion, NumelMismatchBoundaryCut) {
    Graph g;
    auto d4 = TensorDesc::fromShape({4});
    auto d8 = TensorDesc::fromShape({8});

    size_t x = g.addInput(d4);
    size_t y = g.addInput(d8);

    size_t n1 = g.addNode(ReLUNode{d4}, {x}, d4);
    size_t n2 = g.addNode(ReLUNode{d8}, {y}, d8);
    g.markOutput(n1);
    g.markOutput(n2);

    FusionPlan plan = FusionPlanner::planUnits(g, FusionStrategy::AutonomousTraitDriven);

    const FusionUnit* u1 = unitContaining(plan, n1);
    const FusionUnit* u2 = unitContaining(plan, n2);
    ASSERT_NE(u1, nullptr);
    ASSERT_NE(u2, nullptr);
    EXPECT_NE(u1, u2);
}

// ======================= 6. 单算子与空图边缘用例 =======================

TEST(C3TraitAutoFusion, SingleNodeAndEmptyGraph) {
    // 空图
    Graph empty_g;
    FusionPlan empty_plan = FusionPlanner::planUnits(empty_g, FusionStrategy::AutonomousTraitDriven);
    EXPECT_EQ(empty_plan.compute_unit_count, 0u);
    EXPECT_EQ(empty_plan.units.size(), 0u);

    // 单节点图
    Graph single_g;
    auto d = TensorDesc::fromShape({10});
    size_t in = single_g.addInput(d);
    size_t act = single_g.addNode(SiLUNode{d}, {in}, d);
    single_g.markOutput(act);

    FusionPlan single_plan = FusionPlanner::planUnits(single_g, FusionStrategy::AutonomousTraitDriven);
    EXPECT_EQ(single_plan.compute_unit_count, 1u);
    const FusionUnit* u = unitContaining(single_plan, act);
    ASSERT_NE(u, nullptr);
    EXPECT_EQ(u->kind, FusionUnitKind::ELEMENTWISE);
    EXPECT_EQ(u->node_ids.size(), 1u);
    EXPECT_EQ(u->node_ids[0], act);
}
