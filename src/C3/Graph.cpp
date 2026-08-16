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
#include <cmath>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
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
        // 实际匹配在 canonicalize 中自底向上遍历处理
        return std::nullopt;
    });

    // 规则 2: Mul(x, 1) → x
    rules.addRule("Mul(x,1)->x", [](const Node& node) -> std::optional<NodeVariant> {
        if (!std::holds_alternative<MulNode>(node.op)) return std::nullopt;
        return std::nullopt;
    });

    // 规则 3: Mul(x, 0) → 0
    rules.addRule("Mul(x,0)->0", [](const Node& node) -> std::optional<NodeVariant> {
        if (!std::holds_alternative<MulNode>(node.op)) return std::nullopt;
        return std::nullopt;
    });

    // 规则 4: Sub(x, x) → 0
    rules.addRule("Sub(x,x)->0", [](const Node& node) -> std::optional<NodeVariant> {
        if (!std::holds_alternative<SubNode>(node.op)) return std::nullopt;
        return std::nullopt;
    });

    // 规则 5: Div(x, x) → 1
    rules.addRule("Div(x,x)->1", [](const Node& node) -> std::optional<NodeVariant> {
        if (!std::holds_alternative<DivNode>(node.op)) return std::nullopt;
        return std::nullopt;
    });

    // 规则 6: Neg(Neg(x)) → x
    rules.addRule("Neg(Neg(x))->x", [](const Node& node) -> std::optional<NodeVariant> {
        if (!std::holds_alternative<NegNode>(node.op)) return std::nullopt;
        return std::nullopt;
    });

    // 规则 7: Add(x, x) → Mul(x, 2)
    rules.addRule("Add(x,x)->Mul(x,2)", [](const Node& node) -> std::optional<NodeVariant> {
        if (!std::holds_alternative<AddNode>(node.op)) return std::nullopt;
        return std::nullopt;
    });

    // 规则 8: Sub(x, 0) → x
    rules.addRule("Sub(x,0)->x", [](const Node& node) -> std::optional<NodeVariant> {
        if (!std::holds_alternative<SubNode>(node.op)) return std::nullopt;
        return std::nullopt;
    });

    // 规则 9: Div(x, 1) → x
    rules.addRule("Div(x,1)->x", [](const Node& node) -> std::optional<NodeVariant> {
        if (!std::holds_alternative<DivNode>(node.op)) return std::nullopt;
        return std::nullopt;
    });

    // 规则 10: Sub(0, x) → Neg(x)
    rules.addRule("Sub(0,x)->Neg(x)", [](const Node& node) -> std::optional<NodeVariant> {
        if (!std::holds_alternative<SubNode>(node.op)) return std::nullopt;
        return std::nullopt;
    });

    // 规则 11: Mul(x, -1) → Neg(x)
    rules.addRule("Mul(x,-1)->Neg(x)", [](const Node& node) -> std::optional<NodeVariant> {
        if (!std::holds_alternative<MulNode>(node.op)) return std::nullopt;
        return std::nullopt;
    });

    // 规则 12: Div(x, const(y)) -> Mul(x, 1/y)
    rules.addRule("Div(x,const(y))->Mul(x,1/y)", [](const Node& node) -> std::optional<NodeVariant> {
        if (!std::holds_alternative<DivNode>(node.op)) return std::nullopt;
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

size_t Graph::addConstant(double value, const TensorDesc& desc) {
    size_t id = nodes_.size();
    Node node;
    node.id = id;
    node.op = ConstNode{value};
    node.out_desc = desc;
    nodes_.push_back(node);
    return id;
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

    // fold_map[i]：当节点被折叠为常量时的值（SIZE_MAX 表示未折叠为常量）
    std::vector<size_t> fold_map(nodes_.size(), SIZE_MAX);

    // algebraic_folds：记录通过代数化简规则（如 Mul(x,0)→0, Sub(x,x)→0, Div(x,x)→1）
    // 折叠为常量的节点及其常量值，这些节点的输入可能不是 ConstNode，
    // 因此不能通过常规的常量折叠路径处理。
    std::unordered_map<size_t, double> algebraic_folds;

    // 辅助函数：检查节点 ID 是否为图输入
    auto is_graph_input = [&](size_t id) -> bool {
        return std::find(inputs_.begin(), inputs_.end(), id) != inputs_.end();
    };

    // 按拓扑顺序处理（叶子节点先处理）
    for (size_t node_id : topo_order) {
        const Node& node = nodes_[node_id];

        // 跳过已映射到其他节点的（被常量折叠吸收）
        if (node_map[node_id] != node_id) continue;

        // 检查每条规则
        for (size_t r = 0; r < rules.rules.size(); ++r) {
            // 规则 1: Add(x, 0) → x
            if (rules.rule_names[r] == "Add(x,0)->x" &&
                std::holds_alternative<AddNode>(node.op)) {
                for (size_t in_idx = 0; in_idx < node.inputs.size(); ++in_idx) {
                    size_t in_id = node_map[node.inputs[in_idx]];
                    const Node& in_node = nodes_[in_id];
                    if (std::holds_alternative<ConstNode>(in_node.op)) {
                        auto cn = std::get<ConstNode>(in_node.op);
                        if (cn.value == 0.0 && !is_graph_input(in_id)) {
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
                        if (cn.value == 1.0 && !is_graph_input(in_id)) {
                            size_t other_in = node_map[node.inputs[1 - in_idx]];
                            node_map[node_id] = other_in;
                            break;
                        }
                    }
                }
            }

            // 规则 3: Mul(x, 0) → 0
            if (rules.rule_names[r] == "Mul(x,0)->0" &&
                std::holds_alternative<MulNode>(node.op)) {
                for (size_t in_idx = 0; in_idx < node.inputs.size(); ++in_idx) {
                    size_t in_id = node_map[node.inputs[in_idx]];
                    const Node& in_node = nodes_[in_id];
                    if (std::holds_alternative<ConstNode>(in_node.op)) {
                        auto cn = std::get<ConstNode>(in_node.op);
                        if (cn.value == 0.0 && !is_graph_input(in_id)) {
                            algebraic_folds[node_id] = 0.0;
                            fold_map[node_id] = node_id;
                            break;
                        }
                    }
                }
            }

            // 规则 4: Sub(x, x) → 0
            if (rules.rule_names[r] == "Sub(x,x)->0" &&
                std::holds_alternative<SubNode>(node.op) &&
                node.inputs.size() == 2) {
                // 两个输入映射到同一个节点（经过前面的规则化简后）
                if (node_map[node.inputs[0]] == node_map[node.inputs[1]]) {
                    algebraic_folds[node_id] = 0.0;
                    fold_map[node_id] = node_id;
                }
            }

            // 规则 5: Div(x, x) → 1
            if (rules.rule_names[r] == "Div(x,x)->1" &&
                std::holds_alternative<DivNode>(node.op) &&
                node.inputs.size() == 2) {
                if (node_map[node.inputs[0]] == node_map[node.inputs[1]]) {
                    algebraic_folds[node_id] = 1.0;
                    fold_map[node_id] = node_id;
                }
            }

            // 规则 6: Neg(Neg(x)) → x
            if (rules.rule_names[r] == "Neg(Neg(x))->x" &&
                std::holds_alternative<NegNode>(node.op) &&
                node.inputs.size() == 1) {
                size_t inner_id = node_map[node.inputs[0]];
                const Node& inner_node = nodes_[inner_id];
                if (std::holds_alternative<NegNode>(inner_node.op) &&
                    inner_node.inputs.size() == 1) {
                    // 映射到内层 Neg 的输入
                    size_t inner_input = node_map[inner_node.inputs[0]];
                    node_map[node_id] = inner_input;
                }
            }

            // 规则 7: Add(x, x) → Mul(x, 2)
            if (rules.rule_names[r] == "Add(x,x)->Mul(x,2)" &&
                std::holds_alternative<AddNode>(node.op) &&
                node.inputs.size() == 2) {
                if (node_map[node.inputs[0]] == node_map[node.inputs[1]]) {
                    node_map[node_id] = node_id;  // 保持自身，重建时特殊处理
                }
            }

            // 规则 8: Sub(x, 0) → x
            if (rules.rule_names[r] == "Sub(x,0)->x" &&
                std::holds_alternative<SubNode>(node.op) &&
                node.inputs.size() == 2) {
                size_t in1 = node_map[node.inputs[1]];
                const Node& in_node = nodes_[in1];
                if (std::holds_alternative<ConstNode>(in_node.op) &&
                    std::get<ConstNode>(in_node.op).value == 0.0 && !is_graph_input(in1)) {
                    node_map[node_id] = node_map[node.inputs[0]];
                    break;
                }
            }

            // 规则 9: Div(x, 1) → x
            if (rules.rule_names[r] == "Div(x,1)->x" &&
                std::holds_alternative<DivNode>(node.op) &&
                node.inputs.size() == 2) {
                size_t in1 = node_map[node.inputs[1]];
                const Node& in_node = nodes_[in1];
                if (std::holds_alternative<ConstNode>(in_node.op) &&
                    std::get<ConstNode>(in_node.op).value == 1.0 && !is_graph_input(in1)) {
                    node_map[node_id] = node_map[node.inputs[0]];
                    break;
                }
            }
        }

        // 如果规则已折叠此节点，跳过常量折叠
        if (node_map[node_id] != node_id) continue;

        // 常量折叠：检查所有输入是否都是常量（且不是图输入）
        bool all_inputs_are_const = !node.inputs.empty();
        std::vector<double> const_inputs;
        for (size_t in_id : node.inputs) {
            size_t mapped = node_map[in_id];
            // 图输入不能折叠
            if (is_graph_input(mapped)) {
                all_inputs_are_const = false;
                break;
            }
            // 检查是否已被折叠为常量
            if (fold_map[mapped] != SIZE_MAX) {
                // 已被折叠的节点，需要从 folded_constants 获取值
                // 但此时 folded_constants 还未构建，所以先标记为可折叠
                const_inputs.push_back(0.0); // 占位，实际值在重建时计算
                continue;
            }
            const Node& in_node = nodes_[mapped];
            if (!std::holds_alternative<ConstNode>(in_node.op)) {
                all_inputs_are_const = false;
                break;
            }
            const_inputs.push_back(std::get<ConstNode>(in_node.op).value);
        }

        if (all_inputs_are_const) {
            double result = 0.0;
            bool foldable = true;

            std::visit([&](auto&& op) {
                using T = std::decay_t<decltype(op)>;
                if constexpr (std::is_same_v<T, AddNode>) {
                    result = const_inputs[0] + const_inputs[1];
                } else if constexpr (std::is_same_v<T, SubNode>) {
                    result = const_inputs[0] - const_inputs[1];
                } else if constexpr (std::is_same_v<T, MulNode>) {
                    result = const_inputs[0] * const_inputs[1];
                } else if constexpr (std::is_same_v<T, DivNode>) {
                    result = const_inputs[1] != 0.0 ? const_inputs[0] / const_inputs[1] : 0.0;
                } else if constexpr (std::is_same_v<T, NegNode>) {
                    result = -const_inputs[0];
                } else if constexpr (std::is_same_v<T, ReLUNode>) {
                    result = const_inputs[0] > 0.0 ? const_inputs[0] : 0.0;
                } else if constexpr (std::is_same_v<T, SigmoidNode>) {
                    result = 1.0 / (1.0 + std::exp(-const_inputs[0]));
                } else if constexpr (std::is_same_v<T, TanhNode>) {
                    result = std::tanh(const_inputs[0]);
                } else {
                    foldable = false;
                }
            }, node.op);

            if (foldable) {
                // 标记为折叠：node_map 指向自身，fold_map 记录常量值
                fold_map[node_id] = node_id;
            }
        }
    }

    // 步骤 3：根据 node_map 和 fold_map 重建新图
    Graph result;

    // 收集折叠节点及其常量值（按拓扑顺序计算，处理嵌套折叠）
    std::unordered_map<size_t, double> folded_constants;
    for (size_t node_id : topo_order) {
        if (fold_map[node_id] == SIZE_MAX) continue;

        // 优先使用 algebraic_folds 中的值（代数化简规则已计算，输入可能不是 ConstNode）
        if (algebraic_folds.count(node_id)) {
            folded_constants[node_id] = algebraic_folds[node_id];
            continue;
        }

        const Node& node = nodes_[node_id];

        // 获取已映射的输入常量值
        auto get_const_val = [&](size_t input_idx) -> double {
            size_t mapped = node_map[node.inputs[input_idx]];
            if (fold_map[mapped] != SIZE_MAX) {
                return folded_constants.at(mapped);
            }
            return std::get<ConstNode>(nodes_[mapped].op).value;
        };

        std::visit([&](auto&& op) {
            using T = std::decay_t<decltype(op)>;
            if constexpr (std::is_same_v<T, AddNode>) {
                folded_constants[node_id] = get_const_val(0) + get_const_val(1);
            } else if constexpr (std::is_same_v<T, SubNode>) {
                folded_constants[node_id] = get_const_val(0) - get_const_val(1);
            } else if constexpr (std::is_same_v<T, MulNode>) {
                folded_constants[node_id] = get_const_val(0) * get_const_val(1);
            } else if constexpr (std::is_same_v<T, DivNode>) {
                double denom = get_const_val(1);
                folded_constants[node_id] = denom != 0.0 ? get_const_val(0) / denom : 0.0;
            } else if constexpr (std::is_same_v<T, NegNode>) {
                folded_constants[node_id] = -get_const_val(0);
            } else if constexpr (std::is_same_v<T, ReLUNode>) {
                double v = get_const_val(0);
                folded_constants[node_id] = v > 0.0 ? v : 0.0;
            } else if constexpr (std::is_same_v<T, SigmoidNode>) {
                double v = get_const_val(0);
                folded_constants[node_id] = 1.0 / (1.0 + std::exp(-v));
            } else if constexpr (std::is_same_v<T, TanhNode>) {
                double v = get_const_val(0);
                folded_constants[node_id] = std::tanh(v);
            }
        }, node.op);
    }

    std::vector<size_t> new_id_map(nodes_.size(), SIZE_MAX); // 旧 ID → 新 ID

    // 先添加输入节点
    for (size_t in_id : inputs_) {
        size_t mapped = node_map[in_id];
        if (new_id_map[mapped] == SIZE_MAX) {
            new_id_map[mapped] = result.addInput(nodes_[mapped].out_desc);
        }
    }

    // 添加常量节点（来自折叠 + 原始 ConstNode）
    for (size_t node_id : topo_order) {
        // 跳过已折叠的
        if (node_map[node_id] != node_id && fold_map[node_id] == SIZE_MAX) continue;

        // 跳过输入节点
        if (is_graph_input(node_id)) continue;

        if (fold_map[node_id] != SIZE_MAX) {
            // 折叠节点：创建 ConstNode
            if (new_id_map[node_id] == SIZE_MAX) {
                double val = folded_constants[node_id];
                new_id_map[node_id] = result.addConstant(val, nodes_[node_id].out_desc);
            }
            continue;
        }

        const Node& node = nodes_[node_id];

        // 如果是原始 ConstNode（非输入节点），也添加为常量
        if (std::holds_alternative<ConstNode>(node.op)) {
            if (new_id_map[node_id] == SIZE_MAX) {
                auto cn = std::get<ConstNode>(node.op);
                new_id_map[node_id] = result.addConstant(cn.value, node.out_desc);
            }
            continue;
        }

        // 检查是否需要进行代数重建重写
        bool rewritten = false;
        
        // 1. Add(x, x) -> Mul(x, 2)
        if (std::holds_alternative<AddNode>(node.op) && node.inputs.size() == 2) {
            size_t mapped0 = node_map[node.inputs[0]];
            size_t mapped1 = node_map[node.inputs[1]];
            if (mapped0 == mapped1) {
                size_t const_2_id = result.addConstant(2.0, TensorDesc::fromShape({1}, node.out_desc.dtype));
                size_t mapped_new_in = new_id_map[mapped0];
                new_id_map[node_id] = result.addNode(MulNode{}, {mapped_new_in, const_2_id}, node.out_desc);
                rewritten = true;
            }
        }
        // 2. Sub(0, x) -> Neg(x)
        if (!rewritten && std::holds_alternative<SubNode>(node.op) && node.inputs.size() == 2) {
            size_t mapped0 = node_map[node.inputs[0]];
            size_t mapped1 = node_map[node.inputs[1]];
            const Node& in_node0 = nodes_[mapped0];
            if (std::holds_alternative<ConstNode>(in_node0.op) && 
                std::get<ConstNode>(in_node0.op).value == 0.0 && !is_graph_input(mapped0)) {
                size_t mapped_new_in = new_id_map[mapped1];
                new_id_map[node_id] = result.addNode(NegNode{}, {mapped_new_in}, node.out_desc);
                rewritten = true;
            }
        }
        // 3. Mul(x, -1) -> Neg(x)
        if (!rewritten && std::holds_alternative<MulNode>(node.op) && node.inputs.size() == 2) {
            size_t mapped0 = node_map[node.inputs[0]];
            size_t mapped1 = node_map[node.inputs[1]];
            const Node& in_node0 = nodes_[mapped0];
            const Node& in_node1 = nodes_[mapped1];
            if (std::holds_alternative<ConstNode>(in_node0.op) && 
                std::get<ConstNode>(in_node0.op).value == -1.0 && !is_graph_input(mapped0)) {
                size_t mapped_new_in = new_id_map[mapped1];
                new_id_map[node_id] = result.addNode(NegNode{}, {mapped_new_in}, node.out_desc);
                rewritten = true;
            } else if (std::holds_alternative<ConstNode>(in_node1.op) && 
                       std::get<ConstNode>(in_node1.op).value == -1.0 && !is_graph_input(mapped1)) {
                size_t mapped_new_in = new_id_map[mapped0];
                new_id_map[node_id] = result.addNode(NegNode{}, {mapped_new_in}, node.out_desc);
                rewritten = true;
            }
        }

        // 4. Div(x, const_y) -> Mul(x, 1/y)
        if (!rewritten && std::holds_alternative<DivNode>(node.op) && node.inputs.size() == 2) {
            size_t mapped0 = node_map[node.inputs[0]];
            size_t mapped1 = node_map[node.inputs[1]];
            const Node& in_node1 = nodes_[mapped1];
            if (std::holds_alternative<ConstNode>(in_node1.op) && !is_graph_input(mapped1)) {
                double val = std::get<ConstNode>(in_node1.op).value;
                if (val != 0.0) {
                    size_t const_recip_id = result.addConstant(1.0 / val, TensorDesc::fromShape({1}, node.out_desc.dtype));
                    size_t mapped_new_in = new_id_map[mapped0];
                    new_id_map[node_id] = result.addNode(MulNode{}, {mapped_new_in, const_recip_id}, node.out_desc);
                    rewritten = true;
                }
            }
        }

        if (!rewritten) {
            std::vector<size_t> new_inputs;
            for (size_t in_id : node.inputs) {
                size_t mapped = node_map[in_id];
                assert(new_id_map[mapped] != SIZE_MAX);
                new_inputs.push_back(new_id_map[mapped]);
            }
            new_id_map[node_id] = result.addNode(node.op, new_inputs, node.out_desc);
        }
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
                   std::is_same_v<T, NegNode> || std::is_same_v<T, ReLUNode> ||
                   std::is_same_v<T, SigmoidNode> || std::is_same_v<T, TanhNode>;
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
                // 记录每个 op 输出的原始节点 ID（用于 DAG 内部引用 / kernel 生成映射）
                fused_node.op_node_ids.push_back(cid);
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

// ======================= eliminateDeadCode（死代码消除） =======================

Graph Graph::eliminateDeadCode() const {
    return _eliminateDeadCodeForMergedInternal().first;
}

// 内部版本：返回 new graph + old_to_new 映射
// 用于 GraphMerger 等需要跟踪节点 ID 映射的工具类
std::pair<Graph, std::unordered_map<size_t, size_t>>
Graph::_eliminateDeadCodeForMergedInternal() const {
    if (outputs_.empty()) {
        return {Graph(), {}};
    }

    // 步骤 1：从输出节点反向 BFS，收集所有可达节点
    std::unordered_set<size_t> reachable;
    std::queue<size_t> q;
    for (size_t out_id : outputs_) {
        reachable.insert(out_id);
        q.push(out_id);
    }
    while (!q.empty()) {
        size_t cur = q.front(); q.pop();
        for (size_t in_id : nodes_[cur].inputs) {
            if (reachable.insert(in_id).second) {
                q.push(in_id);
            }
        }
    }

    // 步骤 2：拓扑排序（仅可达节点）
    std::vector<size_t> topo_order;
    {
        std::vector<size_t> in_degree(nodes_.size(), 0);
        for (const auto& node : nodes_) {
            if (!reachable.count(node.id)) continue;
            for (size_t out_id : node.outputs) {
                if (reachable.count(out_id)) {
                    in_degree[out_id]++;
                }
            }
        }
        std::queue<size_t> zero_q;
        for (size_t id : reachable) {
            if (in_degree[id] == 0) zero_q.push(id);
        }
        while (!zero_q.empty()) {
            size_t cur = zero_q.front(); zero_q.pop();
            topo_order.push_back(cur);
            for (size_t out_id : nodes_[cur].outputs) {
                if (reachable.count(out_id) && --in_degree[out_id] == 0) {
                    zero_q.push(out_id);
                }
            }
        }
    }

    // 步骤 3：重建新图
    Graph result;
    std::unordered_map<size_t, size_t> old_to_new;

    // 先添加输入节点（仅保留可达的）
    for (size_t in_id : inputs_) {
        if (reachable.count(in_id)) {
            old_to_new[in_id] = result.addInput(nodes_[in_id].out_desc);
        }
    }

    // 添加计算节点
    for (size_t node_id : topo_order) {
        // 跳过输入节点
        if (std::find(inputs_.begin(), inputs_.end(), node_id) != inputs_.end()) continue;

        const Node& node = nodes_[node_id];
        if (std::holds_alternative<ConstNode>(node.op)) {
            auto cn = std::get<ConstNode>(node.op);
            old_to_new[node_id] = result.addConstant(cn.value, node.out_desc);
            continue;
        }

        std::vector<size_t> new_inputs;
        for (size_t in_id : node.inputs) {
            if (old_to_new.count(in_id)) {
                new_inputs.push_back(old_to_new[in_id]);
            }
        }
        old_to_new[node_id] = result.addNode(node.op, new_inputs, node.out_desc);
    }

    // 标记输出
    for (size_t out_id : outputs_) {
        if (old_to_new.count(out_id)) {
            result.markOutput(old_to_new[out_id]);
        }
    }

    return {result, old_to_new};
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
            // 添加 arg_descs 形状到 cache key，确保不同形状的融合图区分
            if (!fnode.arg_descs.empty()) {
                ss << " args:[";
                for (size_t j = 0; j < fnode.arg_descs.size(); ++j) {
                    if (j > 0) ss << ",";
                    for (size_t k = 0; k < fnode.arg_descs[j].shape.size(); ++k) {
                        if (k > 0) ss << "x";
                        ss << fnode.arg_descs[j].shape[k];
                    }
                }
                ss << "]";
            }
        } else if (std::holds_alternative<ConstNode>(node.op)) {
            auto cn = std::get<ConstNode>(node.op);
            ss << "Const(" << cn.value << ")";
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

// ======================= mergeGraph（DEBT-NEW-5 root cause #2 修复）=======================

// 2-arg 重载：默认跳过 source input 占位节点
std::unordered_map<size_t, size_t> Graph::mergeGraph(
    const Graph& other,
    const std::unordered_map<size_t, size_t>& remap_input_ids) {
    return mergeGraph(other, remap_input_ids, /*skip_input_placeholders=*/true);
}

// 3-arg 重载：核心实现
// DEBT-NEW-5: 旧实现把 source input 占位节点重新 push 到本图（用 source id + offset
// 算 id），覆盖了调用方已通过 addInput 分配的节点内容，导致 chain 后续节点的 in_id
// 指向错误的 buffer。修复：若 skip_input_placeholders=true，源图 inputs_ 列表里的
// 节点不复制，仅靠 remap_input_ids 把它们的引用重映射到本图已分配的输入节点。
std::unordered_map<size_t, size_t> Graph::mergeGraph(
    const Graph& other,
    const std::unordered_map<size_t, size_t>& remap_input_ids,
    bool skip_input_placeholders) {
    std::unordered_map<size_t, size_t> old_to_new;

    // 收集源图 input 节点 id 集合（用于 skip 模式）
    std::unordered_set<size_t> source_input_ids;
    if (skip_input_placeholders) {
        source_input_ids.reserve(other.inputs_.size());
        for (size_t in_id : other.inputs_) {
            source_input_ids.insert(in_id);
        }
    }

    // 预填 remap_input_ids（调用方已通过 addInput 在本图分配过这些节点）
    old_to_new.reserve(remap_input_ids.size() + other.nodes_.size());
    for (const auto& kv : remap_input_ids) {
        // 健全性检查：remap 目标必须在本图中已存在
        if (kv.second >= nodes_.size()) {
            throw std::runtime_error(
                "Graph::mergeGraph: remap target id " + std::to_string(kv.second) +
                " not yet allocated in this graph (have " +
                std::to_string(nodes_.size()) + " nodes)");
        }
        old_to_new[kv.first] = kv.second;
    }

    // 复制源图节点：跳过 input 占位（若启用），否则全量复制
    for (const auto& src_node : other.nodes_) {
        if (skip_input_placeholders && source_input_ids.count(src_node.id)) {
            continue;
        }

        // 分配新 id（追加到本图 nodes_ 末尾）
        size_t new_id = nodes_.size();
        Node new_node;
        new_node.id = new_id;
        new_node.op = src_node.op;            // std::variant 拷贝赋值
        new_node.out_desc = src_node.out_desc;
        new_node.inputs.reserve(src_node.inputs.size());

        // 重映射每个输入 id（可能指向 source input → 用 remap_input_ids，
        // 可能指向 source 内部节点 → 已在 old_to_new 中）
        for (size_t old_in_id : src_node.inputs) {
            auto it = old_to_new.find(old_in_id);
            if (it == old_to_new.end()) {
                throw std::runtime_error(
                    "Graph::mergeGraph: unresolved input id " +
                    std::to_string(old_in_id) + " (source node " +
                    std::to_string(src_node.id) +
                    ", skip_input_placeholders=" +
                    std::to_string(skip_input_placeholders) +
                    "). Hint: ensure remap_input_ids covers all source input ids.");
            }
            size_t mapped_id = it->second;
            new_node.inputs.push_back(mapped_id);
            // 更新新输入节点的 outputs 列表（与 addNode 保持一致）
            if (mapped_id < nodes_.size()) {
                nodes_[mapped_id].outputs.push_back(new_id);
            }
        }

        // outputs 字段在本图为空（来源节点的 outputs 是 source graph 内的引用，
        // 在新图里需要重新构建；这里保留空，由后续 topology/use 阶段按需重建）

        old_to_new[src_node.id] = new_id;
        nodes_.push_back(std::move(new_node));
    }

    return old_to_new;
}

} // namespace c3
} // namespace ct