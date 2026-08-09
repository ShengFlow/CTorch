/**
 * @file PatternMatcher.h
 * @brief C3 子图模式匹配引擎
 * @details 自动识别计算图中的常见子图模式（FullyConnected、Activation 等），
 *          为后续的模式特定优化（如算子融合、替换为专用 kernel）提供基础。
 * @date 2026/8/2
 */

#ifndef CTORCH_C3_PATTERN_MATCHER_H
#define CTORCH_C3_PATTERN_MATCHER_H

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include "Graph.h"

namespace ct {
namespace c3 {

// ======================= 模式类型枚举 =======================

/**
 * @enum GraphPatternType
 * @brief 可识别的子图模式类型
 */
enum class GraphPatternType : uint8_t {
    FullyConnected,      ///< MatMul + Add(bias)
    Activation,          ///< 线性层 + 激活函数（ReLU/Sigmoid/Tanh）
    BiasAdd,             ///< 偏置加法（Add 的一侧为偏置张量）
    FCWithActivation,    ///< FullyConnected + Activation
    ElementWiseChain,    ///< 连续逐元素操作链（≥2 个纯逐元素 op）
    Unknown              ///< 未识别模式
};

/// 模式类型名称（调试用）
inline const char* patternTypeName(GraphPatternType type) {
    switch (type) {
        case GraphPatternType::FullyConnected:   return "FullyConnected";
        case GraphPatternType::Activation:       return "Activation";
        case GraphPatternType::BiasAdd:          return "BiasAdd";
        case GraphPatternType::FCWithActivation: return "FCWithActivation";
        case GraphPatternType::ElementWiseChain: return "ElementWiseChain";
        default:                                 return "Unknown";
    }
}

// ======================= 匹配结果 =======================

/**
 * @struct PatternMatch
 * @brief 一次子图模式匹配的结果
 */
struct PatternMatch {
    GraphPatternType type;          ///< 匹配到的模式类型
    std::vector<size_t> node_ids;   ///< 模式包含的节点 ID（按拓扑顺序）
    std::string description;        ///< 人类可读的描述

    /// 输出节点 ID（模式的最后一个节点）
    [[nodiscard]] size_t outputNode() const {
        return node_ids.empty() ? SIZE_MAX : node_ids.back();
    }

    /// 输入节点 ID（模式的第一个计算节点，通常为 MatMul）
    [[nodiscard]] size_t inputNode() const {
        return node_ids.empty() ? SIZE_MAX : node_ids.front();
    }

    bool operator==(const PatternMatch& other) const {
        return type == other.type && node_ids == other.node_ids;
    }
};

// ======================= 辅助函数 =======================

/**
 * @brief 判断节点是否为激活函数类型
 * @param op 节点算子 variant
 * @return true 如果节点是 ReLU/Sigmoid/Tanh
 */
inline bool isActivationOp(const NodeVariant& op) {
    return std::visit([](auto&& arg) -> bool {
        using T = std::decay_t<decltype(arg)>;
        return std::is_same_v<T, ReLUNode> ||
               std::is_same_v<T, SigmoidNode> ||
               std::is_same_v<T, TanhNode>;
    }, op);
}

/**
 * @brief 判断节点是否为逐元素二元操作
 */
inline bool isElementwiseBinaryOp(const NodeVariant& op) {
    return std::visit([](auto&& arg) -> bool {
        using T = std::decay_t<decltype(arg)>;
        return std::is_same_v<T, AddNode> ||
               std::is_same_v<T, SubNode> ||
               std::is_same_v<T, MulNode> ||
               std::is_same_v<T, DivNode>;
    }, op);
}

/**
 * @brief 判断 TensorDesc 是否为偏置张量（1D 或形状与主张量不同）
 * @details 偏置的启发式判断：shape 维度 < 主张量 或 numel 显著小于主张量。
 *          当两个形状完全相同时不视为偏置。
 */
inline bool isBiasDesc(const TensorDesc& bias_desc, const TensorDesc& main_desc) {
    // 形状完全相同，不是偏置
    if (bias_desc.shape == main_desc.shape) return false;
    if (bias_desc.shape.size() == 1) return true;         // 1D 偏置
    if (bias_desc.numel < main_desc.numel) return true;   // 更小的元素数
    // 最后维度匹配（如 [N, K] 偏置 [K]）
    if (!bias_desc.shape.empty() && !main_desc.shape.empty()) {
        return bias_desc.shape.back() == main_desc.shape.back();
    }
    return false;
}

// ======================= 模式匹配器 =======================

/**
 * @class PatternMatcher
 * @brief 子图模式匹配引擎，从 Graph 中识别常见计算模式
 * @details 使用贪心自顶向下匹配策略：
 *          1. 从输出节点出发，尝试匹配已知模式
 *          2. 优先匹配最具体的模式（FCWithActivation > FC > Activation）
 *          3. 每个节点最多参与一个模式匹配
 */
class PatternMatcher {
public:
    PatternMatcher() = default;

    /**
     * @brief 匹配图中所有已知模式
     * @param graph 输入计算图
     * @return 匹配到的模式列表（按输出节点拓扑顺序排列）
     */
    std::vector<PatternMatch> matchAll(const Graph& graph) const;

    /**
     * @brief 匹配 FullyConnected 模式：MatMul + Add(bias)
     * @param graph 输入计算图
     * @return 匹配到的 FC 模式列表
     */
    std::vector<PatternMatch> matchFullyConnected(const Graph& graph) const;

    /**
     * @brief 匹配 Activation 模式：线性层 -> 激活函数
     * @param graph 输入计算图
     * @return 匹配到的 Activation 模式列表
     */
    std::vector<PatternMatch> matchActivation(const Graph& graph) const;

    /**
     * @brief 匹配 BiasAdd 模式：独立的偏置加法
     * @param graph 输入计算图
     * @return 匹配到的 BiasAdd 模式列表
     */
    std::vector<PatternMatch> matchBiasAdd(const Graph& graph) const;

    /**
     * @brief 匹配 FCWithActivation 模式：MatMul + Add(bias) + Activation
     * @param graph 输入计算图
     * @return 匹配到的 FCWithActivation 模式列表
     */
    std::vector<PatternMatch> matchFCWithActivation(const Graph& graph) const;

    /**
     * @brief 匹配 ElementWiseChain 模式：连续逐元素操作链
     * @param graph 输入计算图
     * @return 匹配到的逐元素链模式列表
     */
    std::vector<PatternMatch> matchElementWiseChain(const Graph& graph) const;

    /**
     * @brief 获取图的模式匹配统计数据
     * @return 每种模式类型的计数映射
     */
    std::vector<std::pair<GraphPatternType, size_t>> getStats(const Graph& graph) const;

private:
    /// 检查节点是否为 MatMul 节点
    static bool isMatMulNode(const Node& node) {
        return std::holds_alternative<MatMulNode>(node.op);
    }

    /// 检查节点是否为 Add 节点
    static bool isAddNode(const Node& node) {
        return std::holds_alternative<AddNode>(node.op);
    }

    /// 检查节点是否为 activation 节点
    static bool isActivationNode(const Node& node) {
        return isActivationOp(node.op);
    }

    /// 检查 Add 节点的一侧输入是否为偏置
    static bool isBiasAdd(const Graph& graph, const Node& add_node) {
        if (!isAddNode(add_node)) return false;
        if (add_node.inputs.size() < 2) return false;

        // 检查两个输入中是否有偏置形状
        for (size_t in_id : add_node.inputs) {
            if (!graph.validNodeId(in_id)) continue;
            const Node& in_node = graph.node(in_id);
            // 跳过 MatMul 输出（另一个输入是 MatMul 时，偏置是另一方）
            if (isMatMulNode(in_node)) continue;
            // 检查是否为偏置形状
            if (isBiasDesc(in_node.out_desc, add_node.out_desc)) {
                return true;
            }
        }
        return false;
    }

    /// 获取 Add 节点的 MatMul 输入 ID（如果存在）
    static size_t getMatMulInput(const Graph& graph, const Node& add_node) {
        if (!isAddNode(add_node)) return SIZE_MAX;
        for (size_t in_id : add_node.inputs) {
            if (graph.validNodeId(in_id) && isMatMulNode(graph.node(in_id))) {
                return in_id;
            }
        }
        return SIZE_MAX;
    }

    /// 获取 MatMul 节点后的 Add 输出 ID（如果存在）
    static size_t getAddOutput(const Graph& graph, const Node& mm_node) {
        if (!isMatMulNode(mm_node)) return SIZE_MAX;
        for (size_t out_id : mm_node.outputs) {
            if (out_id < graph.nodes().size() && isAddNode(graph.node(out_id))) {
                return out_id;
            }
        }
        return SIZE_MAX;
    }

    /// 获取节点后的第一个 activation 输出 ID（如果存在且唯一）
    static size_t getActivationOutput(const Graph& graph, const Node& node) {
        for (size_t out_id : node.outputs) {
            if (out_id < graph.nodes().size() && isActivationNode(graph.node(out_id))) {
                return out_id;
            }
        }
        return SIZE_MAX;
    }
};

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_PATTERN_MATCHER_H