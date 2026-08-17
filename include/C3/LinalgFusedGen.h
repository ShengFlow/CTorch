/**
 * @file LinalgFusedGen.h
 * @generation JIT-2.0 声明式区域融合后端（Linalg 纯逐元素融合设计）
 * @brief 用 linalg.generic 声明式生成任意多节点/多输出逐元素融合 kernel（JIT 2.0 路线 A 大一统）
 *
 * 背景：多节点前向和反向融合链包含大量的逐元素算子。目前是通过 `buildFusedMultiNode` 手写 LLVM 标量/向量循环。
 * 本组件将任何“纯逐元素”（只含 Add/Sub/Mul/Div/Neg/ReLU/Sigmoid/Tanh/Exp/Log/Gt）的计算图，
 * 不论单节点还是多节点、单输出还是多输出，统一描述为一个标准的 `linalg.generic` 操作，
 * 经标准 lowering pipeline 编译，实现 100% 声明式区域融合。
 *
 * 调用 ABI：与 LinalgElementwiseGen 一致，通过 memref 展开加 `ExecutionEngine::invokePacked` 传入裸指针与元素数。
 *
 * @date 2026/08/15
 */

#ifndef CTORCH_C3_LINALG_FUSED_GEN_H
#define CTORCH_C3_LINALG_FUSED_GEN_H

#include <cstddef>
#include <memory>
#include <vector>
#include "C3/Graph.h"

namespace ct {
namespace c3 {

/// 判断一个计算图是否是“纯逐元素且可被 linalg.generic 编译”的
/// 判定条件：
///   1. 图非空且所有计算节点均属于逐元素算子（Add/Sub/Mul/Div/Neg/ReLU/Sigmoid/Tanh/Exp/Log/Gt）；
///   2. 所有算子输入没有复杂的周期广播（只支持完全同尺寸，此时 linalg 内部无脑 identity map）；
///   3. 逃生开关 `C3_LINALG_FUSED` 允许退避。
bool isPureElementwiseGraph(const Graph& graph);

/// 编译并持有任意多节点逐元素融合 kernel（多 memref<?xf32> 签名 + invokePacked ABI）
/// 线程安全：编译后 execute 可并发调用
class LinalgFusedKernel {
public:
    /// 编译融合 kernel
    explicit LinalgFusedKernel(const Graph& graph, int opt_level = 3);
    ~LinalgFusedKernel();
    LinalgFusedKernel(const LinalgFusedKernel&) = delete;
    LinalgFusedKernel& operator=(const LinalgFusedKernel&) = delete;
    LinalgFusedKernel(LinalgFusedKernel&&) noexcept;
    LinalgFusedKernel& operator=(LinalgFusedKernel&&) noexcept;

    size_t numInputs() const { return num_inputs_; }
    size_t numOutputs() const { return num_outputs_; }

    /// 执行融合 kernel：in_ptrs（输入指针列表），out_ptrs（输出指针列表），n（元素数）
    void execute(const float* const* in_ptrs, float* const* out_ptrs, size_t n) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    size_t num_inputs_;
    size_t num_outputs_;
};

/// 共享融合 kernel 缓存工厂：同一 graph_key 只编译一次。
/// 逃生开关：`C3_LINALG_CACHE=0` 关闭缓存
std::shared_ptr<LinalgFusedKernel> getCachedLinalgFusedKernel(
    const Graph& graph, const std::string& graph_key, int opt_level = 3);

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_LINALG_FUSED_GEN_H
