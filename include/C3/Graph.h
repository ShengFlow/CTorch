/**
 * @file Graph.h
 * @brief C3 JIT 计算图 IR（中间表示）
 * @details Graph 遵循自由范畴模型（ctfp #5）：生成元算子（Add/Mul/MatMul/...）为范畴的
 *          生成态射，组合闭包由框架自动推导。Node 使用 std::variant 余积类型（ctfp #7）
 *          建模不同算子类型，避免虚函数表开销，为 MLIR 降层预留干净接口。
 *          canonicalize 基于 catamorphism（ctfp #24）：自底向上折叠，化简规则可插拔。
 * @date 2026/7/31
 */

#ifndef CTORCH_C3_GRAPH_H
#define CTORCH_C3_GRAPH_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

// [Fix] v0.5.2 Linux build: std::function / std::optional 头显式 include
// (DTK clang 17 严格模式, 不 transitive 从 <utility> 拿 <optional> 跟 <functional>)
// macOS clang transitive include 拿到, 但显式 include 不冲突
#include <functional>
#include <optional>

#include "../Ctools.h"

namespace ct {
namespace c3 {

// ======================= 张量描述符 =======================

/**
 * @struct TensorDesc
 * @brief 描述图中一个张量的元数据（shape、dtype、device），不持有数据。
 */
struct TensorDesc {
    std::vector<size_t> shape;  ///< 张量形状
    DType dtype = DType::kFloat; ///< 数据类型
    DeviceType device = DeviceType::kCPU; ///< 设备类型
    size_t numel = 0;           ///< 元素总数（派生字段，shape 各维乘积）

    /// 从 shape 自动计算 numel
    static size_t computeNumel(const std::vector<size_t>& s) {
        size_t n = 1;
        for (size_t d : s) n *= d;
        return n;
    }

    static TensorDesc fromShape(const std::vector<size_t>& s,
                                 DType dt = DType::kFloat,
                                 DeviceType dev = DeviceType::kCPU) {
        return {s, dt, dev, computeNumel(s)};
    }

    bool operator==(const TensorDesc& other) const {
        return shape == other.shape && dtype == other.dtype && device == other.device;
    }
};

// ======================= 图节点类型（余积类型 — ctfp #7） =======================

/**
 * @struct AddNode
 * @brief 逐元素加法节点：out = lhs + rhs
 */
struct AddNode {
    static constexpr const char* name = "Add";
    TensorDesc lhs_desc;
    TensorDesc rhs_desc;
};

/**
 * @struct MulNode
 * @brief 逐元素乘法节点：out = lhs * rhs
 */
struct MulNode {
    static constexpr const char* name = "Mul";
    TensorDesc lhs_desc;
    TensorDesc rhs_desc;
};

/**
 * @struct SubNode
 * @brief 逐元素减法节点：out = lhs - rhs
 */
struct SubNode {
    static constexpr const char* name = "Sub";
    TensorDesc lhs_desc;
    TensorDesc rhs_desc;
};

/**
 * @struct DivNode
 * @brief 逐元素除法节点：out = lhs / rhs
 */
struct DivNode {
    static constexpr const char* name = "Div";
    TensorDesc lhs_desc;
    TensorDesc rhs_desc;
};

/**
 * @struct MatMulNode
 * @brief 矩阵乘法节点：out = lhs @ rhs
 */
struct MatMulNode {
    static constexpr const char* name = "MatMul";
    TensorDesc lhs_desc;
    TensorDesc rhs_desc;
};

/**
 * @struct NegNode
 * @brief 一元取负节点：out = -x
 */
struct NegNode {
    static constexpr const char* name = "Neg";
    TensorDesc in_desc;
};

/**
 * @struct ReLUNode
 * @brief 一元 ReLU 激活节点：out = max(0, x)
 */
struct ReLUNode {
    static constexpr const char* name = "ReLU";
    TensorDesc in_desc;
};

/**
 * @struct SigmoidNode
 * @brief 一元 Sigmoid 激活节点：out = 1 / (1 + exp(-x))
 */
struct SigmoidNode {
    static constexpr const char* name = "Sigmoid";
    TensorDesc in_desc;
};

/**
 * @struct TanhNode
 * @brief 一元 Tanh 激活节点：out = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 */
struct TanhNode {
    static constexpr const char* name = "Tanh";
    TensorDesc in_desc;
};

/**
 * @struct GtNode
 * @brief 大于比较节点：out = (lhs > rhs) ? 1.0f : 0.0f
 * @details 用于 backward 中的 mask 生成（如 ReLU gradient: (x > 0) * grad）。
 *          输出为 0.0 或 1.0，支持广播。
 */
struct GtNode {
    static constexpr const char* name = "Gt";
    TensorDesc lhs_desc;
    TensorDesc rhs_desc;
};

/**
 * @struct SumReduceNode
 * @brief 求和降维节点：out = sum(input, axis) 或全 reduce
 * @details 用于 backward 中广播梯度的收缩（如 AddNode 的 broadcast 反向）。
 *          axis = -1 表示对所有维度求和（降维到标量 1 元素张量）。
 *          输出形状根据 input 形状移除指定 axis 得到。
 */
struct SumReduceNode {
    static constexpr const char* name = "SumReduce";
    TensorDesc in_desc;
    int axis = -1;  ///< 求和维度，-1 表示全 reduce
};

/**
 * @struct TransposeNode
 * @brief 矩阵转置节点：out = transpose(input, dim0, dim1)
 * @details 用于 MatMul backward 中梯度矩阵的转置。
 *          dim0/dim1 指定交换的维度，默认 dim0=0, dim1=1（2D 矩阵转置）。
 */
struct TransposeNode {
    static constexpr const char* name = "Transpose";
    TensorDesc in_desc;
    int dim0 = 0;
    int dim1 = 1;
};

/**
 * @struct ExpNode
 * @brief 指数节点：out = exp(x)
 * @details 用于 Exp backward 和 Sigmoid backward 优化。
 */
struct ExpNode {
    static constexpr const char* name = "Exp";
    TensorDesc in_desc;
};

/**
 * @struct LogNode
 * @brief 对数节点：out = log(x)
 * @details 用于 Log backward。
 */
struct LogNode {
    static constexpr const char* name = "Log";
    TensorDesc in_desc;
};

/**
 * @struct ConstNode
 * @brief 常量节点（用于常量折叠）：value = scalar
 * @details 仅用于 canonicalize 阶段的常量折叠，不参与实际 kernel 执行。
 *          例如 Add(x, 0) → x 化简时会查找 ConstNode。
 */
struct ConstNode {
    static constexpr const char* name = "Const";
    double value = 0.0;
};

// 前向声明：FusedNode 包含 NodeVariant，而 NodeVariant 包含 FusedNode
struct FusedNode;

// ======================= 节点类型 variant =======================

/**
 * @brief 图节点的余积类型（ctfp #7）
 * @details 使用 std::variant 而非虚函数继承：
 *          - 虚函数表在 MLIR 降层时无意义，variant 可直接映射到 MLIR operation
 *          - 图遍历（canonicalize）使用 std::visit 比虚函数模式更高效且显式
 *          - 新增算子类型只需扩展 variant，不破坏现有代码
 */
using NodeVariant = std::variant<AddNode, SubNode, MulNode, DivNode, MatMulNode, NegNode, ReLUNode, SigmoidNode, TanhNode, GtNode, SumReduceNode, TransposeNode, ExpNode, LogNode, ConstNode, FusedNode>;

/**
 * @struct FusedNode
 * @brief 融合节点：将多个逐元素操作合并为单个 kernel
 * @details 融合后，所有 ops 在单次循环中按顺序执行，消除中间张量的内存分配和读写。
 *          例如 (x + y) * z 的融合链：ops[0]=Add(x,y), ops[1]=Mul(tmp,z)。
 *          arg_descs 记录融合节点的外部输入描述符，顺序与 Graph 输入一致。
 */
struct FusedNode {
    static constexpr const char* name = "Fused";
    std::vector<NodeVariant> ops;         ///< 操作序列（按执行顺序，拓扑排序）
    std::vector<std::vector<size_t>> op_inputs; ///< 每个 op 的原始输入节点 ID（用于 kernel 生成时映射）
    std::vector<size_t> op_node_ids;      ///< 每个 op 输出的原始节点 ID（用于 DAG 内部引用）
    std::vector<TensorDesc> arg_descs;    ///< 外部输入张量描述符（去重后）
    std::vector<size_t> arg_node_ids;     ///< 外部输入对应的原始节点 ID
    TensorDesc out_desc;                  ///< 输出张量描述符
};

// ======================= 图节点 =======================

/**
 * @struct Node
 * @brief 计算图中的单个节点，包含余积类型的算子数据 + 拓扑信息。
 */
struct Node {
    size_t id = 0;               ///< 节点唯一 ID（在 Graph 中分配）
    NodeVariant op;              ///< 算子类型（余积类型）
    std::vector<size_t> inputs;  ///< 输入节点的 ID（按拓扑顺序）
    std::vector<size_t> outputs; ///< 输出节点（被哪些节点引用）的 ID
    TensorDesc out_desc;         ///< 输出张量的元数据
};

// ======================= 化简规则（catamorphism — ctfp #24） =======================

/**
 * @struct CanonicalizeRules
 * @brief 可插拔的化简规则集合
 * @details 每个规则是一个函数：给定一个 Node，若能化简则返回替换后的 NodeVariant；
 *          否则返回 std::nullopt。规则按顺序应用，先匹配先生效。
 */
struct CanonicalizeRules {
    std::vector<std::string> rule_names;
    /// 规则函数签名：const Node& -> std::optional<NodeVariant>（化简后节点或空）
    std::vector<std::function<std::optional<NodeVariant>(const Node&)>> rules;

    /// 添加一条规则
    void addRule(const std::string& name,
                 std::function<std::optional<NodeVariant>(const Node&)> fn) {
        rule_names.push_back(name);
        rules.push_back(std::move(fn));
    }

    /// 默认规则集：Add(x, 0) → x, Mul(x, 1) → x
    static CanonicalizeRules defaults();
};

// ======================= 计算图 =======================

/**
 * @class Graph
 * @brief C3 计算图 IR，遵循自由范畴模型（ctfp #5）。
 * @details Graph 是生成元算子（Node）的有向无环图。
 *          每个 addInput / addNode 返回一个 Value 句柄，可作为后续节点的输入，
 *          形成组合闭包。禁止在图中存储"已组合过的中间态射"作为独立生成元。
 */
class Graph {
public:
    Graph() = default;

    /**
     * @brief 添加输入张量
     * @param desc 输入张量的元数据
     * @return 输入节点的 ID（Value 句柄）
     */
    size_t addInput(const TensorDesc& desc);

    /**
     * @brief 添加计算节点
     * @param op 算子类型（余积类型）
     * @param input_ids 输入节点 ID 列表
     * @param out_desc 输出张量元数据
     * @return 新节点的 ID（Value 句柄）
     */
    size_t addNode(NodeVariant op,
                   const std::vector<size_t>& input_ids,
                   const TensorDesc& out_desc);

    /**
     * @brief 标记输出节点
     * @param node_id 输出节点 ID
     */
    void markOutput(size_t node_id);

    /**
     * @brief 对图做规范化（catamorphism：自底向上折叠）
     * @param rules 化简规则集（默认为 Add(x,0)→x, Mul(x,1)→x）
     * @return 规范化后的新图
     * @details 自底向上遍历 DAG：先递归化简子节点，再将化简后的子节点交给父节点
     *          的规则处理。保证结构递归的完备性——不会遗漏深层嵌套的子图。
     */
    Graph canonicalize(const CanonicalizeRules& rules = CanonicalizeRules::defaults()) const;

    /**
     * @brief 算子融合：将相邻的逐元素操作合并为 FusedNode
     * @return 融合后的新图
     * @details 从每个输出节点出发，反向遍历生产者链。若节点是逐元素操作
     *          （Add/Sub/Mul/Div/Neg）且输出仅被一个下游节点消费，则加入融合组。
     *          遇到非逐元素操作（MatMul）或多消费者节点时结束当前融合组。
     *          融合组包含 ≥2 个操作时替换为单个 FusedNode。
     */
    Graph fuse() const;

    /**
     * @brief 死代码消除：移除不可达节点（未被任何输出节点引用的子图）
     * @return 消除死代码后的新图
     * @details 从输出节点出发反向 BFS 收集所有可达节点，仅保留可达节点重建图。
     *          输入节点若不被任何可达节点引用也会被移除。
     */
    Graph eliminateDeadCode() const;

    /**
     * @brief 添加常量节点（用于常量折叠）
     * @param value 常量值
     * @param desc 张量描述符（通常为标量 shape={1}）
     * @return 常量节点的 ID
     */
    size_t addConstant(double value, const TensorDesc& desc);

    // ======================= 访问器 =======================

    [[nodiscard]] const std::vector<size_t>& inputs() const { return inputs_; }
    [[nodiscard]] const std::vector<size_t>& outputs() const { return outputs_; }
    [[nodiscard]] const std::vector<Node>& nodes() const { return nodes_; }
    [[nodiscard]] const Node& node(size_t id) const { return nodes_[id]; }
    [[nodiscard]] size_t nodeCount() const { return nodes_.size(); }
    [[nodiscard]] size_t inputCount() const { return inputs_.size(); }
    [[nodiscard]] size_t outputCount() const { return outputs_.size(); }

    /// 拓扑排序是否有效（无环）
    [[nodiscard]] bool isValid() const;

    /// 将图转为可读字符串（调试用）
    [[nodiscard]] std::string toString() const;

    /// 检查节点 ID 是否有效
    [[nodiscard]] bool validNodeId(size_t id) const { return id < nodes_.size(); }

    /**
     * @brief 合并另一个图到当前图
     * @param other 源图
     * @param remap_input_ids 源图输入节点 ID → 当前图节点 ID 的映射
     * @param remap_output_ids 源图输出节点 ID → 当前图节点 ID 的映射（可选）
     * @return 源图节点 ID → 当前图节点 ID 的映射
     * @details 将 other 的所有节点复制到当前图，并重映射输入/输出引用。
     *          调用方需提供 remap_input_ids 将源图的输入节点映射到当前图的已有节点。
     */
    std::unordered_map<size_t, size_t> mergeGraph(
        const Graph& other,
        const std::unordered_map<size_t, size_t>& remap_input_ids);

    /**
     * @brief 合并图（跳过 source input 占位节点，避免 id 冲突）
     * @details source graph 的 input 占位节点（addInput 创建的 ConstNode{0}）已被
     *          调用方通过 addInput 替换。如果不跳过，mergeGraph 会用 source input id +
     *          offset 把占位节点再次 push 到本图，覆盖已经 addInput 分配的节点内容，
     *          导致 chain 后续节点的 in_id 指向错误的 buffer。
     * @param other source graph
     * @param remap_input_ids 源图输入节点 id → 本图已分配节点 id 的映射
     * @return 旧 id → 新 id 的映射表
     */
    std::unordered_map<size_t, size_t> mergeGraph(
        const Graph& other,
        const std::unordered_map<size_t, size_t>& remap_input_ids,
        bool /*skip_input_placeholders*/);

private:
    std::vector<Node> nodes_;       ///< 所有节点（按添加顺序，ID 即索引）
    std::vector<size_t> inputs_;    ///< 输入节点 ID 列表
    std::vector<size_t> outputs_;   ///< 输出节点 ID 列表

    // 内部修改接口（仅供同命名空间内的 C3 工具类使用）
    friend class GraphMerger;

    /// 内部：重写所有节点中引用 old_id 的 inputs 边为 new_id
    /// 同时同步更新 nodes_[old_id].outputs（移除指向 new 边的反向引用）和
    /// nodes_[new_id].outputs（追加所有原本指向 old_id 的消费者）。
    /// 修复 bug：仅重写 inputs 不更新 outputs 会导致拓扑排序错乱和后续 dead-code elimination 输出残缺图。
    /// 注意：若同一节点的 inputs 中多次引用 old_id（如 inputs=[old_id, old_id]），
    /// 确保 new_id 的 outputs 不会产生重复条目。
    void _rewriteInputRefInternal(size_t old_id, size_t new_id) {
        if (old_id == new_id) return;
        for (auto& node : nodes_) {
            bool already_added = false;
            for (auto& in : node.inputs) {
                if (in == old_id) {
                    in = new_id;
                    // 反向引用：把"old_id 的消费者"也记到 new_id 上
                    // 使用 already_added 避免同一节点多次引用 old_id 时产生重复条目
                    if (new_id < nodes_.size() && !already_added) {
                        nodes_[new_id].outputs.push_back(node.id);
                        already_added = true;
                    }
                }
            }
        }
        // 清空 old_id 的 outputs（所有反向引用都已迁移到 new_id）
        if (old_id < nodes_.size()) {
            nodes_[old_id].outputs.clear();
        }
    }

    /// 内部：执行死代码消除并返回 old_id → new_id 映射
    /// 供 GraphMerger 等需要跟踪节点 ID 映射的工具类使用
    std::pair<Graph, std::unordered_map<size_t, size_t>>
    _eliminateDeadCodeForMergedInternal() const;
};

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_GRAPH_H