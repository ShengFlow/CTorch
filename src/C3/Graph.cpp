/**
 * @file Graph.cpp
 * @brief C3 JIT 计算图 IR 实现
 * @details canonicalize 基于 catamorphism（ctfp #24）：自底向上遍历 DAG，
 *          对每个节点应用化简规则，将化简后的子节点交给父节点继续处理。
 * @date 2026/7/31
 */

#include "../../include/C3/Graph.h"

#include <algorithm>
#include <cassert>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace ct {
namespace c3 {

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

// ======================= fuse（算子融合） =======================

Graph Graph::fuse() const {
    // 辅助函数：判断节点是否为逐元素操作
    auto is_elementwise = [](const NodeVariant& op) -> bool {
        return std::visit([](auto&& arg) -> bool {
            using T = std::decay_t<decltype(arg)>;
            return std::is_same_v<T, AddNode> || std::is_same_v<T, SubNode> ||
                   std::is_same_v<T, MulNode> || std::is_same_v<T, DivNode> ||
                   std::is_same_v<T, NegNode> || std::is_same_v<T, ReLUNode>;
        }, op);
    };

    // 步骤 1: 计算每个节点的消费者数量
    std::vector<size_t> consumer_count(nodes_.size(), 0);
    for (const auto& node : nodes_) {
        for (size_t in_id : node.inputs) {
            if (validNodeId(in_id)) consumer_count[in_id]++;
        }
    }

    // 步骤 2: 从每个输出出发，反向遍历构建融合链
    std::vector<std::vector<size_t>> fusion_chains;
    std::vector<bool> fused(nodes_.size(), false);

    for (size_t out_id : outputs_) {
        if (fused[out_id]) continue;

        std::vector<size_t> chain;
        size_t current = out_id;

        while (true) {
            if (fused[current]) break;
            const Node& node = nodes_[current];
            if (!is_elementwise(node.op)) break;
            // 非输出节点且有多消费者，不能融合
            if (consumer_count[current] > 1 && current != out_id) break;

            chain.push_back(current);

            // 继续向前遍历第一个非 Const 输入
            bool found_next = false;
            for (size_t in_id : node.inputs) {
                if (!validNodeId(in_id)) continue;
                if (std::holds_alternative<ConstNode>(nodes_[in_id].op)) continue;
                if (fused[in_id]) continue;
                // 如果前驱节点有多个消费者，停止向前
                if (consumer_count[in_id] > 1) continue;
                current = in_id;
                found_next = true;
                break;
            }
            if (!found_next) break;
        }

        if (chain.size() >= 2) {
            // 反转链条为输入→输出方向
            std::reverse(chain.begin(), chain.end());
            fusion_chains.push_back(chain);
            for (size_t id : chain) fused[id] = true;
        }
    }

    // 步骤 3: 构建新图
    Graph result;
    std::vector<size_t> new_id_map(nodes_.size(), SIZE_MAX);

    // 先添加输入节点
    for (size_t in_id : inputs_) {
        new_id_map[in_id] = result.addInput(nodes_[in_id].out_desc);
    }

    // 收集融合链起始节点
    std::unordered_set<size_t> chain_start_ids;
    for (const auto& chain : fusion_chains) {
        if (!chain.empty()) chain_start_ids.insert(chain[0]);
    }

    // 拓扑排序
    std::vector<size_t> topo_order;
    {
        std::vector<size_t> in_degree(nodes_.size(), 0);
        for (const auto& node : nodes_) {
            for (size_t o : node.outputs) {
                if (o < nodes_.size()) in_degree[o]++;
            }
        }
        std::queue<size_t> q;
        for (size_t i = 0; i < nodes_.size(); ++i) {
            if (in_degree[i] == 0) q.push(i);
        }
        while (!q.empty()) {
            size_t cur = q.front(); q.pop();
            topo_order.push_back(cur);
            for (size_t o : nodes_[cur].outputs) {
                if (--in_degree[o] == 0) q.push(o);
            }
        }
    }

    // 添加节点
    for (size_t node_id : topo_order) {
        bool is_input = false;
        for (size_t in_id : inputs_) {
            if (in_id == node_id) { is_input = true; break; }
        }
        if (is_input) continue;

        // 跳过已融合节点（非起始节点）
        if (fused[node_id] && chain_start_ids.find(node_id) == chain_start_ids.end()) continue;

        // 找到包含此节点的融合链
        const std::vector<size_t>* chain = nullptr;
        for (auto& c : fusion_chains) {
            if (!c.empty() && c[0] == node_id) { chain = &c; break; }
        }

        if (chain) {
            // 构建 FusedNode
            FusedNode fused_node;
            std::set<size_t> chain_set(chain->begin(), chain->end());
            // 外部输入：使用 vector + sort + unique 确保确定性顺序
            std::vector<size_t> external_inputs;

            for (size_t cid : *chain) {
                const Node& node = nodes_[cid];
                fused_node.ops.push_back(node.op);
                fused_node.op_inputs.push_back(node.inputs);
                for (size_t in_id : node.inputs) {
                    if (chain_set.find(in_id) == chain_set.end()) {
                        external_inputs.push_back(in_id);
                    }
                }
            }

            // 去重并排序，确保跨运行的确定性
            std::sort(external_inputs.begin(), external_inputs.end());
            external_inputs.erase(
                std::unique(external_inputs.begin(), external_inputs.end()),
                external_inputs.end());

            std::vector<size_t> new_inputs;
            for (size_t ext_id : external_inputs) {
                if (new_id_map[ext_id] != SIZE_MAX) {
                    new_inputs.push_back(new_id_map[ext_id]);
                    fused_node.arg_descs.push_back(nodes_[ext_id].out_desc);
                    fused_node.arg_node_ids.push_back(ext_id);
                }
            }
            fused_node.out_desc = nodes_[chain->back()].out_desc;

            new_id_map[node_id] = result.addNode(fused_node, new_inputs, fused_node.out_desc);
        } else {
            // 普通节点
            const Node& node = nodes_[node_id];
            std::vector<size_t> new_inputs;
            for (size_t in_id : node.inputs) {
                if (new_id_map[in_id] != SIZE_MAX) {
                    new_inputs.push_back(new_id_map[in_id]);
                }
            }
            new_id_map[node_id] = result.addNode(node.op, new_inputs, node.out_desc);
        }
    }

    // 标记输出
    for (size_t out_id : outputs_) {
        size_t mapped_id = new_id_map[out_id];
        // 如果输出节点在融合链中，找到链起始节点的映射
        if (mapped_id == SIZE_MAX) {
            for (auto& chain : fusion_chains) {
                auto it = std::find(chain.begin(), chain.end(), out_id);
                if (it != chain.end()) {
                    mapped_id = new_id_map[chain[0]];
                    break;
                }
            }
        }
        if (mapped_id != SIZE_MAX) {
            result.markOutput(mapped_id);
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

        bool is_input = (std::find(inputs_.begin(), inputs_.end(), i) != inputs_.end());
        bool is_output = (std::find(outputs_.begin(), outputs_.end(), i) != outputs_.end());

        ss << "  [" << i << "] ";
        if (is_input) ss << "INPUT ";

        if (std::holds_alternative<FusedNode>(node.op)) {
            const auto& fnode = std::get<FusedNode>(node.op);
            ss << "Fused(";
            for (size_t j = 0; j < fnode.ops.size(); ++j) {
                if (j > 0) ss << " -> ";
                ss << nodeName(fnode.ops[j]);
            }
            ss << ")";
        } else {
            ss << nodeName(node.op);
        }

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

} // namespace c3
} // namespace ct