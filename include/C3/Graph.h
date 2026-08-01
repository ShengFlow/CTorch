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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

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
using NodeVariant = std::variant<AddNode, SubNode, MulNode, DivNode, MatMulNode, NegNode, ReLUNode, ConstNode, FusedNode>;

/**
 * @struct FusedNode
 * @brief 融合节点：将多个逐元素操作合并为单个 kernel
 * @details 融合后，所有 ops 在单次循环中按顺序执行，消除中间张量的内存分配和读写。
 *          例如 (x + y) * z 的融合链：ops[0]=Add(x,y), ops[1]=Mul(tmp,z)。
 *          arg_descs 记录融合节点的外部输入描述符，顺序与 Graph 输入一致。
 */
struct FusedNode {
    static constexpr const char* name = "Fused";
    std::vector<NodeVariant> ops;         ///< 操作序列（按执行顺序）
    std::vector<std::vector<size_t>> op_inputs; ///< 每个 op 的原始输入节点 ID（用于 kernel 生成时映射）
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

private:
    std::vector<Node> nodes_;       ///< 所有节点（按添加顺序，ID 即索引）
    std::vector<size_t> inputs_;    ///< 输入节点 ID 列表
    std::vector<size_t> outputs_;   ///< 输出节点 ID 列表

    /// 检查节点 ID 是否有效
    bool validNodeId(size_t id) const { return id < nodes_.size(); }
};

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_GRAPH_H