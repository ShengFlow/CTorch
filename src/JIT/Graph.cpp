/**
 * @file Graph.cpp
 * @brief C3 JIT 计算图 IR 实现
 * @details canonicalize 基于 catamorphism（ctfp #24）：自底向上遍历 DAG，
 *          对每个节点应用化简规则，将化简后的子节点交给父节点继续处理。
 * @date 2026/7/31
 */

#include "../../include/JIT/Graph.h"

#include <algorithm>
#include <cassert>
#include <queue>
#include <sstream>
#include <stdexcept>

namespace ct {
namespace jit {

// ======================= 辅助函数 =======================

/// 从 NodeVariant 获取节点名称
static const char* nodeName(const NodeVariant& op) {
    return std::visit([](const auto& n) -> const char* {
        return std::remove_reference_t<decltype(n)>::name;
    }, op);
}

// ======================= CanonicalizeRules =======================

CanonicalizeRules CanonicalizeRules::defaults() {
    CanonicalizeRules rules;

    // 规则 1: Add(x, 0) → x
    rules.addRule("Add(x,0)->x", [](const Node& node) -> std::optional<NodeVariant> {
        if (!std::holds_alternative<AddNode>(node.op)) return std::nullopt;
        // 检查是否有输入是值为 0 的 ConstNode
        for (size_t in_id : node.inputs) {
            // 此检查在 canonicalize 遍历时进行，因为需要访问其他节点
            // 规则本身只检查节点自身
        }
        return std::nullopt; // 实际匹配在 canonicalize 中处理
    });

    // 规则 2: Mul(x, 1) → x
    rules.addRule("Mul(x,1)->x", [](const Node& node) -> std::optional<NodeVariant> {
        if (!std::holds_alternative<MulNode>(node.op)) return std::nullopt;
        return std::nullopt;
    });

    return rules;
}

// ======================= Graph 实现 =======================

size_t Graph::addInput(const TensorDesc& desc) {
    size_t id = nodes_.size();
    // 输入节点使用 ConstNode 作为占位符（标记为输入）
    Node input_node;
    input_node.id = id;
    input_node.op = ConstNode{0.0}; // 占位符
    input_node.out_desc = desc;
    nodes_.push_back(input_node);
    inputs_.push_back(id);
    return id;
}

size_t Graph::addNode(NodeVariant op,
                      const std::vector<size_t>& input_ids,
                      const TensorDesc& out_desc) {
    // 验证所有输入节点 ID 有效
    for (size_t in_id : input_ids) {
        if (!validNodeId(in_id)) {
            throw std::runtime_error(
                "Graph::addNode: invalid input node id " + std::to_string(in_id));
        }
    }

    size_t id = nodes_.size();
    Node node;
    node.id = id;
    node.op = std::move(op);
    node.inputs = input_ids;
    node.out_desc = out_desc;

    // 更新输入节点的 outputs 列表
    for (size_t in_id : input_ids) {
        nodes_[in_id].outputs.push_back(id);
    }

    nodes_.push_back(std::move(node));
    return id;
}

void Graph::markOutput(size_t node_id) {
    if (!validNodeId(node_id)) {
        throw std::runtime_error(
            "Graph::markOutput: invalid node id " + std::to_string(node_id));
    }
    outputs_.push_back(node_id);
}

// ======================= canonicalize（catamorphism — ctfp #24） =======================

Graph Graph::canonicalize(const CanonicalizeRules& rules) const {
    // 步骤 1：拓扑排序（输入节点 → 输出节点方向）
    std::vector<size_t> topo_order;
    {
        std::vector<size_t> in_degree(nodes_.size(), 0);
        for (const auto& node : nodes_) {
            for (size_t out_id : node.outputs) {
                if (out_id < nodes_.size()) {
                    in_degree[out_id]++;
                }
            }
        }
        std::queue<size_t> q;
        for (size_t i = 0; i < nodes_.size(); ++i) {
            if (in_degree[i] == 0) q.push(i);
        }
        while (!q.empty()) {
            size_t cur = q.front(); q.pop();
            topo_order.push_back(cur);
            for (size_t out_id : nodes_[cur].outputs) {
                if (--in_degree[out_id] == 0) {
                    q.push(out_id);
                }
            }
        }
    }

    // 步骤 2：自底向上折叠（catamorphism）
    // node_map[i] = 化简后代表节点 i 的节点 ID（可能被重映射到其他节点）
    std::vector<size_t> node_map(nodes_.size());
    for (size_t i = 0; i < nodes_.size(); ++i) node_map[i] = i;

    // 按拓扑顺序处理（叶子节点先处理）
    for (size_t node_id : topo_order) {
        const Node& node = nodes_[node_id];

        // 跳过已映射到其他节点的（被常量折叠吸收）
        if (node_map[node_id] != node_id) continue;

        // 检查每条规则
        for (size_t r = 0; r < rules.rules.size(); ++r) {
            // 规则 1: Add(x, 0) → x (等价于 Add(x, ConstNode{0}) → x)
            if (rules.rule_names[r] == "Add(x,0)->x" &&
                std::holds_alternative<AddNode>(node.op)) {
                for (size_t in_idx = 0; in_idx < node.inputs.size(); ++in_idx) {
                    size_t in_id = node_map[node.inputs[in_idx]];
                    const Node& in_node = nodes_[in_id];
                    if (std::holds_alternative<ConstNode>(in_node.op)) {
                        auto cn = std::get<ConstNode>(in_node.op);
                        if (cn.value == 0.0) {
                            // 找到 0 常量，映射到另一个输入
                            size_t other_in = node_map[node.inputs[1 - in_idx]];
                            node_map[node_id] = other_in;
                            break;
                        }
                    }
                }
            }

            // 规则 2: Mul(x, 1) → x
            if (rules.rule_names[r] == "Mul(x,1)->x" &&
                std::holds_alternative<MulNode>(node.op)) {
                for (size_t in_idx = 0; in_idx < node.inputs.size(); ++in_idx) {
                    size_t in_id = node_map[node.inputs[in_idx]];
                    const Node& in_node = nodes_[in_id];
                    if (std::holds_alternative<ConstNode>(in_node.op)) {
                        auto cn = std::get<ConstNode>(in_node.op);
                        if (cn.value == 1.0) {
                            size_t other_in = node_map[node.inputs[1 - in_idx]];
                            node_map[node_id] = other_in;
                            break;
                        }
                    }
                }
            }
        }
    }

    // 步骤 3：根据 node_map 重建新图
    Graph result;
    std::vector<size_t> new_id_map(nodes_.size(), SIZE_MAX); // 旧 ID → 新 ID

    // 先添加输入节点
    for (size_t in_id : inputs_) {
        size_t mapped = node_map[in_id];
        if (new_id_map[mapped] == SIZE_MAX) {
            new_id_map[mapped] = result.addInput(nodes_[mapped].out_desc);
        }
    }

    // 添加计算节点（按拓扑顺序）
    for (size_t node_id : topo_order) {
        if (node_map[node_id] != node_id) continue; // 已被折叠

        // 跳过输入节点
        bool is_input = false;
        for (size_t in_id : inputs_) {
            if (node_map[in_id] == node_id) { is_input = true; break; }
        }
        if (is_input) continue;

        const Node& node = nodes_[node_id];

        // 映射输入
        std::vector<size_t> new_inputs;
        for (size_t in_id : node.inputs) {
            size_t mapped = node_map[in_id];
            assert(new_id_map[mapped] != SIZE_MAX);
            new_inputs.push_back(new_id_map[mapped]);
        }

        new_id_map[node_id] = result.addNode(node.op, new_inputs, node.out_desc);
    }

    // 标记输出
    for (size_t out_id : outputs_) {
        size_t mapped = node_map[out_id];
        if (new_id_map[mapped] != SIZE_MAX) {
            result.markOutput(new_id_map[mapped]);
        }
    }

    return result;
}

// ======================= 验证 =======================

bool Graph::isValid() const {
    // 检查所有节点 ID 自洽
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i].id != i) return false;
        for (size_t in_id : nodes_[i].inputs) {
            if (!validNodeId(in_id)) return false;
        }
    }

    // 检查无环（拓扑排序）
    std::vector<size_t> in_degree(nodes_.size(), 0);
    for (const auto& node : nodes_) {
        for (size_t out_id : node.outputs) {
            if (out_id < nodes_.size()) in_degree[out_id]++;
        }
    }
    std::queue<size_t> q;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (in_degree[i] == 0) q.push(i);
    }
    size_t visited = 0;
    while (!q.empty()) {
        size_t cur = q.front(); q.pop();
        visited++;
        for (size_t out_id : nodes_[cur].outputs) {
            if (--in_degree[out_id] == 0) q.push(out_id);
        }
    }
    return visited == nodes_.size();
}

// ======================= 调试输出 =======================

std::string Graph::toString() const {
    std::ostringstream ss;
    ss << "Graph (" << nodes_.size() << " nodes, "
       << inputs_.size() << " inputs, "
       << outputs_.size() << " outputs)\n";

    for (size_t i = 0; i < nodes_.size(); ++i) {
        const auto& node = nodes_[i];

        // 检查是否为输入节点
        bool is_input = (std::find(inputs_.begin(), inputs_.end(), i) != inputs_.end());
        bool is_output = (std::find(outputs_.begin(), outputs_.end(), i) != outputs_.end());

        ss << "  [" << i << "] ";
        if (is_input) ss << "INPUT ";
        ss << nodeName(node.op);

        if (!node.inputs.empty()) {
            ss << " (";
            for (size_t j = 0; j < node.inputs.size(); ++j) {
                if (j > 0) ss << ", ";
                ss << node.inputs[j];
            }
            ss << ")";
        }

        ss << " -> ";
        if (node.out_desc.shape.empty()) {
            ss << "scalar";
        } else {
            ss << "[";
            for (size_t j = 0; j < node.out_desc.shape.size(); ++j) {
                if (j > 0) ss << "x";
                ss << node.out_desc.shape[j];
            }
            ss << "]";
        }

        if (is_output) ss << " *OUTPUT*";
        ss << "\n";
    }
    return ss.str();
}

} // namespace jit
} // namespace ct