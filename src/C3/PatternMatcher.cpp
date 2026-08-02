/**
 * @file PatternMatcher.cpp
 * @brief C3 子图模式匹配引擎实现
 * @details 实现贪心自顶向下匹配策略，从 Graph 的输出节点出发，
 *          尝试匹配 FullyConnected、Activation、BiasAdd 等常见模式。
 * @date 2026/8/2
 */

#include "../../include/C3/PatternMatcher.h"

#include <algorithm>
#include <sstream>
#include <unordered_set>

namespace ct {
namespace c3 {

// 从 NodeVariant 获取节点名称（与 Graph.cpp 中的实现一致）
static const char* nodeName(const NodeVariant& op) {
    return std::visit([](const auto& n) -> const char* {
        return std::remove_reference_t<decltype(n)>::name;
    }, op);
}

// 将 shape 向量转为字符串（用于描述输出）
static std::string shapeToString(const std::vector<size_t>& shape) {
    if (shape.empty()) return "scalar";
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) ss << "x";
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
}

// ======================= 匹配实现 =======================

std::vector<PatternMatch> PatternMatcher::matchAll(const Graph& graph) const {
    std::vector<PatternMatch> results;
    std::unordered_set<size_t> matched_nodes;

    // 优先匹配最具体的模式：FCWithActivation > FC > Activation > BiasAdd
    // 每个节点只参与一个模式

    // 第一轮：匹配 FCWithActivation
    auto fc_act = matchFCWithActivation(graph);
    for (auto& m : fc_act) {
        for (size_t id : m.node_ids) matched_nodes.insert(id);
        results.push_back(std::move(m));
    }

    // 第二轮：匹配 FC（排除已匹配节点）
    auto fc = matchFullyConnected(graph);
    for (auto& m : fc) {
        // 检查是否已被 FCWithActivation 覆盖
        bool already_matched = false;
        for (size_t id : m.node_ids) {
            if (matched_nodes.count(id)) { already_matched = true; break; }
        }
        if (!already_matched) {
            for (size_t id : m.node_ids) matched_nodes.insert(id);
            results.push_back(std::move(m));
        }
    }

    // 第三轮：匹配 Activation（排除已匹配节点）
    auto act = matchActivation(graph);
    for (auto& m : act) {
        bool already_matched = false;
        for (size_t id : m.node_ids) {
            if (matched_nodes.count(id)) { already_matched = true; break; }
        }
        if (!already_matched) {
            for (size_t id : m.node_ids) matched_nodes.insert(id);
            results.push_back(std::move(m));
        }
    }

    // 第四轮：匹配 BiasAdd（排除已匹配节点）
    auto bias = matchBiasAdd(graph);
    for (auto& m : bias) {
        bool already_matched = false;
        for (size_t id : m.node_ids) {
            if (matched_nodes.count(id)) { already_matched = true; break; }
        }
        if (!already_matched) {
            for (size_t id : m.node_ids) matched_nodes.insert(id);
            results.push_back(std::move(m));
        }
    }

    return results;
}

std::vector<PatternMatch> PatternMatcher::matchFullyConnected(const Graph& graph) const {
    std::vector<PatternMatch> results;
    std::unordered_set<size_t> matched_nodes;

    for (const auto& node : graph.nodes()) {
        if (matched_nodes.count(node.id)) continue;
        if (!isMatMulNode(node)) continue;

        // MatMul 后接 Add(bias) → FullyConnected
        size_t add_id = getAddOutput(graph, node);
        if (add_id == SIZE_MAX) continue;
        if (matched_nodes.count(add_id)) continue;

        const Node& add_node = graph.node(add_id);
        if (!isBiasAdd(graph, add_node)) continue;

        // 构建匹配
        PatternMatch match;
        match.type = GraphPatternType::FullyConnected;
        match.node_ids = {node.id, add_id};

        std::ostringstream desc;
        desc << "FullyConnected("
             << "MatMul[" << node.id << "] -> "
             << "BiasAdd[" << add_id << "])";
        // 添加形状信息
        if (std::holds_alternative<MatMulNode>(node.op)) {
            const auto& mm = std::get<MatMulNode>(node.op);
            desc << " [" << shapeToString(mm.lhs_desc.shape) << " x " << shapeToString(mm.rhs_desc.shape) << "]";
        }
        match.description = desc.str();

        matched_nodes.insert(node.id);
        matched_nodes.insert(add_id);
        results.push_back(std::move(match));
    }

    return results;
}

std::vector<PatternMatch> PatternMatcher::matchActivation(const Graph& graph) const {
    std::vector<PatternMatch> results;
    std::unordered_set<size_t> matched_nodes;

    // 从输出节点反向匹配：找到 Activation 节点，检查其前驱
    for (const auto& node : graph.nodes()) {
        if (matched_nodes.count(node.id)) continue;
        if (!isActivationNode(node)) continue;

        // 检查前驱节点：应该是一个线性层（MatMul）或 MatMul+Add
        // 但这里我们只匹配 Activation 本身（MatMul → Activation 或 MatMul+Add → Activation）
        for (size_t in_id : node.inputs) {
            if (!graph.validNodeId(in_id)) continue;
            const Node& in_node = graph.node(in_id);
            if (matched_nodes.count(in_id)) continue;

            // 前驱可以是 MatMul 或 Add（Add 本身可能是 FC 的一部分）
            if (isMatMulNode(in_node) || isAddNode(in_node)) {
                PatternMatch match;
                match.type = GraphPatternType::Activation;

                // 根据前驱类型决定包含的节点
                if (isMatMulNode(in_node)) {
                    match.node_ids = {in_id, node.id};
                } else {
                    // Add 前驱：包含 Add 和 Activation
                    match.node_ids = {in_id, node.id};
                }

                std::ostringstream desc;
                desc << "Activation("
                     << "pred[" << in_id << "] -> "
                     << shapeToString(graph.node(in_id).out_desc.shape) << " -> "
                     << nodeName(node.op) << "[" << node.id << "])";
                match.description = desc.str();

                matched_nodes.insert(in_id);
                matched_nodes.insert(node.id);
                results.push_back(std::move(match));
                break;
            }
        }
    }

    return results;
}

std::vector<PatternMatch> PatternMatcher::matchBiasAdd(const Graph& graph) const {
    std::vector<PatternMatch> results;
    std::unordered_set<size_t> matched_nodes;

    for (const auto& node : graph.nodes()) {
        if (matched_nodes.count(node.id)) continue;
        if (!isAddNode(node)) continue;
        if (!isBiasAdd(graph, node)) continue;

        // 排除 MatMul 后的 Add（那是 FC 模式的一部分）
        bool has_matmul_input = false;
        for (size_t in_id : node.inputs) {
            if (graph.validNodeId(in_id) && isMatMulNode(graph.node(in_id))) {
                has_matmul_input = true;
                break;
            }
        }
        if (has_matmul_input) continue;

        // 独立的 BiasAdd
        PatternMatch match;
        match.type = GraphPatternType::BiasAdd;
        match.node_ids = {node.id};

        std::ostringstream desc;
        desc << "BiasAdd[" << node.id << "]";
        // 找出偏置输入
        for (size_t in_id : node.inputs) {
            if (graph.validNodeId(in_id)) {
                const Node& in_node = graph.node(in_id);
                if (isBiasDesc(in_node.out_desc, node.out_desc)) {
                    desc << " bias=" << shapeToString(in_node.out_desc.shape);
                }
            }
        }
        match.description = desc.str();

        matched_nodes.insert(node.id);
        results.push_back(std::move(match));
    }

    return results;
}

std::vector<PatternMatch> PatternMatcher::matchFCWithActivation(const Graph& graph) const {
    std::vector<PatternMatch> results;
    std::unordered_set<size_t> matched_nodes;

    for (const auto& node : graph.nodes()) {
        if (matched_nodes.count(node.id)) continue;
        if (!isMatMulNode(node)) continue;

        // MatMul → Add(bias) → Activation
        size_t add_id = getAddOutput(graph, node);
        if (add_id == SIZE_MAX) continue;
        if (matched_nodes.count(add_id)) continue;

        const Node& add_node = graph.node(add_id);
        if (!isBiasAdd(graph, add_node)) continue;

        size_t act_id = getActivationOutput(graph, add_node);
        if (act_id == SIZE_MAX) continue;
        if (matched_nodes.count(act_id)) continue;

        // 构建匹配
        PatternMatch match;
        match.type = GraphPatternType::FCWithActivation;
        match.node_ids = {node.id, add_id, act_id};

        std::ostringstream desc;
        desc << "FCWithActivation("
             << "MatMul[" << node.id << "] -> "
             << "BiasAdd[" << add_id << "] -> "
             << nodeName(graph.node(act_id).op) << "[" << act_id << "])";
        // 添加形状信息
        if (std::holds_alternative<MatMulNode>(node.op)) {
            const auto& mm = std::get<MatMulNode>(node.op);
            desc << " [" << shapeToString(mm.lhs_desc.shape) << " x " << shapeToString(mm.rhs_desc.shape) << "]";
        }
        match.description = desc.str();

        matched_nodes.insert(node.id);
        matched_nodes.insert(add_id);
        matched_nodes.insert(act_id);
        results.push_back(std::move(match));
    }

    return results;
}

std::vector<std::pair<GraphPatternType, size_t>> PatternMatcher::getStats(const Graph& graph) const {
    auto matches = matchAll(graph);

    std::unordered_map<GraphPatternType, size_t> counts;
    for (const auto& m : matches) {
        counts[m.type]++;
    }

    std::vector<std::pair<GraphPatternType, size_t>> result;
    result.reserve(counts.size());
    for (const auto& [type, count] : counts) {
        result.emplace_back(type, count);
    }

    // 按类型排序，确保确定性输出
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) {
                  return static_cast<uint8_t>(a.first) < static_cast<uint8_t>(b.first);
              });

    return result;
}

} // namespace c3
} // namespace ct