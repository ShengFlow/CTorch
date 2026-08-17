/**
 * @file C3BackwardCapture.h
 * @generation SHARED 跨代共有（JIT-2.0/2.x/3.0 共用后端捕获）
 * @brief 反向图 JIT 捕获与编译引擎
 * @details 在 autograd 的 backward 执行路径中插入 C3 编译/执行尝试。
 *          核心流程：forward 时同步记录 C3 Graph 节点 → backward 时查 C3KernelRegistry
 *          → 命中则执行编译后的 backward kernel → 未命中则回退 eager + 异步编译。
 *
 *          支持的反向映射表：
 *          - ReLUNode: Mul(Gt(x, 0), grad) → 新增 GtNode
 *          - SigmoidNode: Mul(Mul(Sigmoid(x), Sub(1, Sigmoid(x))), grad) → 已有节点
 *          - TanhNode: Mul(Sub(1, Mul(Tanh(x), Tanh(x))), grad) → 已有节点
 *          - AddNode: grad, grad（广播时 SumReduce）→ 新增 SumReduceNode
 *          - MulNode: Mul(grad, B), Mul(A, grad) → 已有节点
 *          - MatMulNode: MatMul(grad, Transpose(B)), MatMul(Transpose(A), grad) → 新增 TransposeNode
 *          - NegNode: Neg(grad) → 已有节点
 *          - SubNode: grad, Neg(grad) → 已有节点
 *          - DivNode: Div(grad, B), Neg(Mul(A, Div(grad, Mul(B, B)))) → 已有节点
 *          - ExpNode: Mul(Exp(x), grad) → 新增 ExpNode
 *          - LogNode: Div(grad, x) → 已有节点
 *
 *          多输出策略（MLIR 后端函数签名仅支持单输出指针）：
 *          多输入节点（Add/Mul/MatMul/Sub/Div）的 backward 需产生多个梯度，
 *          这里不改造 codegen，而是为每个上游梯度编译一个独立单输出 kernel，
 *          注册 key 追加 "|in:<i>" 后缀区分。tryExecuteBackward 遍历输入逐 key
 *          查找并执行，任一缺失则整体回退 eager 以保证正确性。
 *
 *          回退策略：
 *          - 形状不匹配 → 静默回退到 eager（预期行为）
 *          - 未注册节点类型 → 静默回退到 eager
 *          - 任一输入梯度缺少编译 kernel → 整体回退到 eager
 *          - 执行异常 → 回退 + 记录日志（不影响训练正确性）
 * @date 2026/8/4
 */

#ifndef CTORCH_C3_BACKWARD_CAPTURE_H
#define CTORCH_C3_BACKWARD_CAPTURE_H

#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "C3Engine.h"
#include "C3KernelRegistry.h"
#include "Graph.h"
#include "../AutoGrad/Node.h"

namespace ct {
namespace c3 {

/**
 * @class C3BackwardCapture
 * @brief 反向图 JIT 捕获与编译引擎
 * @details 单例类，负责将 autograd 节点的 backward 捕获为 C3 Graph 并编译。
 *          与 C3KernelRegistry 集成以实现热替换。
 */
class C3BackwardCapture {
public:
    /**
     * @brief backward 子图构建结果：{Graph, fwd_input_map}
     * @details fwd_input_map[k] = 该反向图的第 (k+1) 个输入（grad 之后的第 k 个）
     *          对应的 forward_inputs 索引。因最小集 build 只加实际用到的 forward 输入，
     *          图输入顺序 ≠ forward_inputs 顺序，运行时必须按此表喂入（见 C3KernelRegistry）。
     */
    using BackwardGraph = std::pair<Graph, std::vector<size_t>>;

    /** @brief 获取 C3BackwardCapture 单例实例 */
    static C3BackwardCapture& getInstance();

    /**
     * @brief 清除反向融合捕获器中的所有临时状态与缓存。
     * @details 消除测试用例间的状态和残留节点的交叉污染，确保完整的环境隔离。
     */
    void clear();

    /**
     * @brief [优化 2026-08-16] 清理单次 backward 调用范围内的临时状态。
     * @details 在每次反向传播结束时调用，清空未被消费的截获梯度与 miss marker 节点，
     *          防止因为内存地址复用导致 Stale Tensor / UAF 崩溃或错误的梯度匹配。
     */
    void clearCallScopedState();

    /**
     * @brief 尝试执行编译后的 backward kernel
     * @param node 当前 autograd 节点
     * @param grad 下游梯度张量
     * @return 若命中 C3 缓存且执行成功，返回上游梯度列表；否则返回 nullopt
     * @details 首先查 C3KernelRegistry 是否有匹配的 backward kernel。
     *          命中则执行，未命中则返回 nullopt（调用方回退 eager）。
     *          执行失败时静默返回 nullopt，不抛异常。
     */
    std::optional<std::vector<Tensor>> tryExecuteBackward(
        const ::Node* node, const Tensor& grad,
        const std::vector<Tensor>& forward_inputs = {});

    /**
     * @brief 异步编译 backward 子图
     * @param node 当前 autograd 节点
     * @param grad 下游梯度张量
     * @details 在后台线程中构建 C3 Graph 并编译，编译完成后自动注册到 C3KernelRegistry。
     *          若同一节点类型 + 形状的编译任务已在进行中，去重。
     *          编译失败时静默处理，不影响训练流程。
     */
    void compileBackwardAsync(const ::Node* node, const Tensor& grad);

    /**
     * @brief 为指定输入索引异步编译 backward 单输出 kernel
     * @param node 当前 autograd 节点
     * @param grad 下游梯度张量
     * @param input_index 目标上游输入索引
     * @details tryExecuteBackward 对多输入节点逐 key 查找时，若某输入缺失，
     *          仅触发该输入的编译，其余已编译输入保持缓存复用。
     */
    void compileBackwardAsyncForInput(const ::Node* node, const Tensor& grad,
                                      size_t input_index);

    /**
     * @brief 构建反向子图的 C3 Graph
     * @param node 当前 autograd 节点
     * @param grad_desc 下游梯度的 TensorDesc
     * @param input_descs forward 输入的 TensorDesc 列表
     * @return 构建好的 C3 Graph；若无法构建（节点类型不支持），返回 nullopt
     * @details 根据 node 的类型，选择对应的反向规则，构建等价的 C3 Graph。
     *          例如 ReLUNode 构建 Mul(Gt(x, 0), grad) 的图。
     */
    std::optional<Graph> buildBackwardGraph(
        const ::Node* node,
        const TensorDesc& grad_desc,
        const std::vector<TensorDesc>& input_descs);

    /**
     * @brief 为指定输入索引构建反向子图的单输出 C3 Graph
     * @param node 当前 autograd 节点
     * @param input_index 目标上游输入索引（多输入节点 0..N-1；单输入节点必须为 0）
     * @param grad_desc 下游梯度的 TensorDesc
     * @param input_descs forward 输入的 TensorDesc 列表
     * @return 单输出 C3 Graph（仅计算目标输入的梯度）；不支持则返回 nullopt
     * @details 多输入节点（Add/Mul/MatMul/Sub/Div）的 backward 需产生多个梯度，
     *          每个梯度编译为独立单输出 kernel，用 input_index 区分。
     *          单输入节点（ReLU/Sigmoid/Tanh/Neg/Exp/Log）仅支持 input_index == 0。
     */
    std::optional<Graph> buildBackwardGraphForInput(
        const ::Node* node,
        size_t input_index,
        const TensorDesc& grad_desc,
        const std::vector<TensorDesc>& input_descs);

    /**
     * @brief 按节点类型字符串 + 输入索引构建反向子图的单输出 Graph
     * @param node_type 节点类型字符串（如 "ReLUNode"、"AddNode"）
     * @param input_index 目标上游输入索引
     * @param grad_desc 下游梯度的 TensorDesc
     * @param input_descs forward 输入的 TensorDesc 列表
     * @return 单输出 C3 Graph；不支持则返回 nullopt
     * @details 供异步编译线程使用（不持有 Node 指针，规避 UAF），
     *          与 buildBackwardGraphForInput 逻辑一致，仅改用字符串分发。
     */
    std::optional<BackwardGraph> buildBackwardGraphForTypeAndIndex(
        const std::string& node_type,
        size_t input_index,
        const TensorDesc& grad_desc,
        const std::vector<TensorDesc>& input_descs);

    /**
     * @brief 检查节点类型是否支持 C3 backward 编译
     * @param node_type 节点类型字符串
     * @return true 如果支持
     */
    static bool supportsNodeType(const std::string& node_type);

    /** @brief 获取编译统计信息 */
    struct Stats {
        size_t capture_count = 0;      ///< 捕获次数
        size_t compile_count = 0;      ///< 编译次数
        size_t cache_hit_count = 0;    ///< 缓存命中次数
        size_t cache_miss_count = 0;   ///< 缓存未命中次数
        size_t execution_failures = 0; ///< 执行失败次数
        size_t fusion_compile_count = 0; ///< 融合编译次数
        size_t fusion_hit_count = 0;    ///< 融合执行命中次数
        size_t fusion_miss_count = 0;   ///< 融合尝试未命中次数
    };

    Stats getStats() const;

    // ======================= 反向融合检测 (Phase 2) =======================

    /**
     * @brief 记录一个 backward 节点（用于序列检测）
     * @param node_type 节点类型字符串
     * @param grad_shape 下游梯度形状
     * @param input_shape forward 输入形状
     * @details 在 ComputeCore::backward 中每次处理完一个节点后调用。
     *          当连续多个 backward 节点形成 fusion 模式时，触发融合编译。
     */
    void recordBackwardNode(const std::string& node_type,
                             const std::vector<size_t>& grad_shape,
                             const std::vector<size_t>& input_shape,
                             const std::vector<Tensor>& forward_inputs);

    /**
     * @brief 尝试执行已编译好的反向融合 kernel
     * @param node 当前 autograd 节点（用于拿到 forward_inputs）
     * @param grad 最下游梯度张量
     * @param forward_inputs forward 阶段输入张量列表
     * @return 若命中反向融合缓存，返回最终梯度（单张量，对应最上游的目标输入）；否则返回 nullopt
     * @details 优先于逐输入单 kernel 执行。
     *          融合 kernel 对应一段连续的 backward 序列（如 ReLU → Sigmoid → Mul），
     *          命中后一次性跑完整个序列，节省中间写读。
     */
    std::optional<Tensor> tryExecuteFusedBackward(
        const ::Node* node,
        const Tensor& grad,
        const std::vector<Tensor>& forward_inputs);

private:
    C3BackwardCapture() = default;

    // ======================= 反向 Graph 构建助手 =======================

    /**
     * @brief 构建 ReLU backward 的 C3 Graph
     * @param grad_desc 下游梯度描述符
     * @param input_desc forward 输入描述符
     * @return C3 Graph: Mul(Gt(x, 0), grad)
     */
    BackwardGraph buildReLUBackwardGraph(const TensorDesc& grad_desc,
                                  const TensorDesc& input_desc);

    /**
     * @brief 构建 Sigmoid backward 的 C3 Graph
     * @param grad_desc 下游梯度描述符
     * @param input_desc forward 输入描述符
     * @return C3 Graph: Mul(Mul(Sigmoid(x), Sub(1, Sigmoid(x))), grad)
     */
    BackwardGraph buildSigmoidBackwardGraph(const TensorDesc& grad_desc,
                                     const TensorDesc& input_desc);

    /**
     * @brief 构建 Tanh backward 的 C3 Graph
     * @param grad_desc 下游梯度描述符
     * @param input_desc forward 输入描述符
     * @return C3 Graph: Mul(Sub(1, Mul(Tanh(x), Tanh(x))), grad)
     */
    BackwardGraph buildTanhBackwardGraph(const TensorDesc& grad_desc,
                                  const TensorDesc& input_desc);

    /**
     * @brief 构建 Add backward 的 C3 Graph（单输出：指定输入索引）
     * @param grad_desc 下游梯度描述符
     * @param lhs_desc forward 左输入描述符
     * @param rhs_desc forward 右输入描述符
     * @param input_index 目标上游输入索引（0=左, 1=右）
     * @return C3 Graph: grad（广播时 SumReduce 缩小）
     */
    BackwardGraph buildAddBackwardGraph(const TensorDesc& grad_desc,
                                 const TensorDesc& lhs_desc,
                                 const TensorDesc& rhs_desc,
                                 size_t input_index);

    /**
     * @brief 构建 Mul backward 的 C3 Graph（单输出：指定输入索引）
     * @param grad_desc 下游梯度描述符
     * @param a_desc forward 左输入描述符
     * @param b_desc forward 右输入描述符
     * @param input_index 目标上游输入索引（0=左, 1=右）
     * @return C3 Graph: Mul(grad, B) 或 Mul(A, grad)
     */
    BackwardGraph buildMulBackwardGraph(const TensorDesc& grad_desc,
                                 const TensorDesc& a_desc,
                                 const TensorDesc& b_desc,
                                 size_t input_index);

    /**
     * @brief 构建 MatMul backward 的 C3 Graph（单输出：指定输入索引）
     * @param grad_desc 下游梯度描述符
     * @param a_desc forward 左输入描述符
     * @param b_desc forward 右输入描述符
     * @param input_index 目标上游输入索引（0=左, 1=右）
     * @return C3 Graph: MatMul(grad, Transpose(B)) 或 MatMul(Transpose(A), grad)
     */
    BackwardGraph buildMatMulBackwardGraph(const TensorDesc& grad_desc,
                                    const TensorDesc& a_desc,
                                    const TensorDesc& b_desc,
                                    size_t input_index);

    /**
     * @brief 构建 Neg backward 的 C3 Graph
     * @param grad_desc 下游梯度描述符
     * @return C3 Graph: Neg(grad)
     */
    BackwardGraph buildNegBackwardGraph(const TensorDesc& grad_desc);

    /**
     * @brief 构建 Sub backward 的 C3 Graph（单输出：指定输入索引）
     * @param grad_desc 下游梯度描述符
     * @param input_index 目标上游输入索引（0=左, 1=右）
     * @return C3 Graph: grad 或 Neg(grad)
     */
    BackwardGraph buildSubBackwardGraph(const TensorDesc& grad_desc,
                                 size_t input_index);

    /**
     * @brief 构建 Div backward 的 C3 Graph（单输出：指定输入索引）
     * @param grad_desc 下游梯度描述符
     * @param a_desc forward 左输入（被除数）描述符
     * @param b_desc forward 右输入（除数）描述符
     * @param input_index 目标上游输入索引（0=左, 1=右）
     * @return C3 Graph: Div(grad, B) 或 Neg(Mul(A, Div(grad, Mul(B, B))))
     */
    BackwardGraph buildDivBackwardGraph(const TensorDesc& grad_desc,
                                 const TensorDesc& a_desc,
                                 const TensorDesc& b_desc,
                                 size_t input_index);

    /**
     * @brief 构建 Exp backward 的 C3 Graph
     * @param grad_desc 下游梯度描述符
     * @param input_desc forward 输入描述符
     * @param output_desc forward 输出描述符（exp(x) 的值）
     * @return C3 Graph: Mul(Exp(x), grad) = Mul(output, grad)
     */
    BackwardGraph buildExpBackwardGraph(const TensorDesc& grad_desc,
                                 const TensorDesc& input_desc,
                                 const TensorDesc& output_desc);

    /**
     * @brief 构建 Log backward 的 C3 Graph
     * @param grad_desc 下游梯度描述符
     * @param input_desc forward 输入描述符
     * @return C3 Graph: Div(grad, x)
     */
    BackwardGraph buildLogBackwardGraph(const TensorDesc& grad_desc,
                                 const TensorDesc& input_desc);

    /**
     * @brief 根据节点类型字符串构建 backward Graph（用于融合编译）
     * @param node_type 节点类型字符串（如 "ReLUNode", "AddNode"）
     * @param grad_desc 下游梯度描述符
     * @param input_descs forward 输入描述符列表
     * @return 构建好的 C3 Graph；若类型不支持，返回 nullopt
     * @details 与 buildBackwardGraph 功能相同，但使用字符串匹配而非 typeid。
     *          用于融合编译场景，此时只有节点类型字符串，无实际节点对象。
     */
    std::optional<BackwardGraph> buildBackwardGraphForType(
        const std::string& node_type,
        const TensorDesc& grad_desc,
        const std::vector<TensorDesc>& input_descs);

    // ======================= 工具函数 =======================

    /**
     * @brief 判断是否需要 SumReduce（用于广播反向）
     * @param grad_shape 梯度的形状（当前接收到的）
     * @param target_shape 上游节点期望的形状
     * @return true 如果 grad_shape 需要 SumReduce 才能匹配 target_shape
     */
    static bool needsSumReduce(const std::vector<size_t>& grad_shape,
                                const std::vector<size_t>& target_shape);

    /**
     * @brief 构建 SumReduce 的 axis 参数
     * @param grad_shape 当前梯度形状
     * @param target_shape 目标形状
     * @return 需要 reduce 的 axis（从 grad_shape 到 target_shape）
     */
    static int computeReduceAxis(const std::vector<size_t>& grad_shape,
                                  const std::vector<size_t>& target_shape);

    // ======================= 统计信息 =======================

    mutable std::mutex stats_mutex_;
    size_t capture_count_ = 0;
    size_t compile_count_ = 0;
    size_t cache_hit_count_ = 0;
    size_t cache_miss_count_ = 0;
    size_t execution_failures_ = 0;
    size_t fusion_compile_count_ = 0;

    // 去重 map：正在编译中的 (node_type + shape_hash)
    std::mutex pending_mutex_;
    std::unordered_map<std::string, bool> pending_compiles_;

    // ======================= 反向融合检测 (Phase 2) =======================

    static constexpr size_t kFusionWindowSize = 4; ///< 融合检测窗口大小
    static constexpr int kFusionThreshold = 1;     ///< 触发热融合的频次阈值（=1 表示第一次出现就异步编译，降低用户冷启动首跳延迟）

    /// 反向序列条目
    struct BackwardSequence {
        std::vector<std::string> node_types;         ///< 节点类型序列（[0]=最下游, [N-1]=最上游, 反向传播执行顺序与 ComputeCore 一致）
        std::vector<std::vector<size_t>> grad_shapes; ///< 与 node_types 对齐：每个节点收到的下游 grad 形状（grad_shapes[0] 为最下游端 dL/dy 形状）
        std::vector<std::vector<size_t>> input_shapes;///< 与 node_types 对齐：每个节点的首个 forward 输入形状（input_shapes.back() 为最上游端 data 形状）
        int frequency = 0;                           ///< 出现频次
        bool compiling = false;                      ///< 是否正在编译
    };

    /// 最近的 backward 节点序列（RingBuffer）
    std::deque<std::string> recent_sequence_;
    std::deque<std::vector<size_t>> recent_grad_shapes_;  ///< 与 recent_sequence_ 对齐：每个节点的下游 grad 形状
    std::deque<std::vector<size_t>> recent_input_shapes_; ///< 与 recent_sequence_ 对齐：每个节点的首个 forward 输入形状
    std::deque<std::vector<Tensor>> recent_forward_inputs_; ///< 与 recent_sequence_ 对齐：每个节点的完整 forward inputs（执行融合时按序取）

    /**
     * @brief 融合拦截的待取结果：N0 执行融合时一次性算出 outs[w]=w 个节点的 upstream grad，
     *        outs[1..w-1] 存到这里；当 N1..Nw-1 依次进入 tryExecuteBackward 时直接取出返回，
     *        避免重复计算 & 严格对齐 ComputeCore 的 grad-pack 分发流程。
     * value = {节点类型名, 对应 upstream grad Tensor}：type 校验防止 raw ptr 地址复用的误命中。
     */
    std::unordered_map<const ::Node*, std::pair<std::string, Tensor>> pending_intercepted_;
    mutable std::shared_mutex intercepted_mutex_; ///< 读写锁：miss 路径只读拿共享锁（大幅降低开销），写入 pending 时才拿独占锁

    /**
     * @brief 本轮 backward 中已确认 "fusion lookup 失败" 的节点指针标记。
     *        在同一个 backward 轮次中，一个节点若已经走过路径 B 的 upstream traversal 且 lookup 失败，
     *        则后续再次被访问（例如 wrapper 节点转发）时直接跳过 B 路径，避免重复图遍历开销。
     */
    std::unordered_set<const ::Node*> miss_marker_nodes_;
    mutable std::mutex miss_marker_mutex_;

    /// 已观察到的序列及其频次
    std::unordered_map<std::string, BackwardSequence> sequence_counts_;

    /// 融合检测 mutex
    std::mutex fusion_mutex_;

    /**
     * @brief 检查节点类型是否为元素操作（可融合）
     */
    static bool isElementWiseBackward(const std::string& node_type);

    /**
     * @brief 构建序列 key
     */
    static std::string makeSequenceKey(const std::vector<std::string>& types);

    /**
     * @brief 检查序列是否可融合
     * @param types 节点类型序列
     * @return true 如果序列中的所有节点都是元素操作且可融合
     */
    static bool isFusableSequence(const std::vector<std::string>& types);

    /**
     * @brief 为融合序列构建 fused backward Graph 并异步编译
     */
    void compileFusedBackwardAsync(const BackwardSequence& seq);

    // ======================= 融合查找辅助 =======================

    /**
     * @brief 构造带形状签名的反向融合注册/查找 key（注册与查找共用，保证格式对齐）
     * @param seq_key 由 makeSequenceKey 得到的短序列 key（如 "ReLU+Sigmoid"）
     * @param grad_shape 下游梯度形状
     * @param input_shape 首个 forward 输入形状
     */
    static std::string makeFusedBackwardKey(const std::string& seq_key,
                                             const std::vector<size_t>& grad_shape,
                                             const std::vector<size_t>& input_shape);

    /**
     * @brief 从 recent_sequence_ 尾部取长度 len 的最新子序列
     * @param out_types 输出：最新的 len 个节点类型
     * @param len 要求的序列长度（>= 2）
     * @return true 若 recent_sequence_ 中元素 >= len
     */
    bool getLatestSequenceTail(std::vector<std::string>& out_types, size_t len) const;

    /**
     * @brief 遍历所有可能的反向融合窗口（从长到短），尝试在 C3KernelRegistry 中命中一个
     * @param grad 下游梯度（取 shape）
     * @param first_forward_input_shape 首个 forward 输入形状
     * @param out_key 输出：命中的注册 key
     * @return true 若命中
     */
    bool tryLookupFusedBackwardKey(const Tensor& grad,
                                    const std::vector<size_t>& first_forward_input_shape,
                                    std::string& out_key);

    // ======================= 统计字段新增 =======================
    size_t fusion_hit_count_ = 0;
    size_t fusion_miss_count_ = 0;
};

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_BACKWARD_CAPTURE_H