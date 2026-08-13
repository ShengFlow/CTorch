/**
 * @file HandwrittenKernelGen.h
 * @brief C3 JIT 手写 kernel 生成器内部接口
 * @details 供 C3Engine.cpp 调用，不对外暴露。
 * @date 2026/7/31
 */

#ifndef CTORCH_C3_HANDWRITTEN_KERNEL_GEN_H
#define CTORCH_C3_HANDWRITTEN_KERNEL_GEN_H

#include <functional>
#include "../../include/C3/Graph.h"
#include "../../include/C3/C3KernelRegistry.h"

namespace ct {
namespace c3 {

// C3KernelFunc 类型定义在 C3KernelRegistry.h 中，此处复用

/**
 * @struct GeneratedKernel
 * @brief 编译产物的统一接口，HandwrittenKernelGen 和 MLIRKernelGen 共用。
 */
struct GeneratedKernel {
    C3KernelFunc func = nullptr;
    FusedKernelFunc fused_func = nullptr; ///< 融合 kernel 函数指针（is_fused=true 时使用）
    MultiNodeKernelFunc multi_func = nullptr; ///< 多节点 kernel 函数指针（is_multi_node=true 时使用）
    void* handle = nullptr;               ///< 资源句柄（dlopen handle 或 ExecutionEngine 引用）
    std::function<void()> deleter;        ///< 析构回调：释放 handle 指向的资源
    bool is_matmul = false;
    bool is_fused = false;                ///< 是否为融合 kernel
    bool is_multi_node = false;           ///< 是否为多节点 kernel
    size_t num_inputs = 2;                ///< 外部输入数量
    size_t M = 0, K = 0, N = 0;          ///< MatMul 维度
    size_t elem_n = 0;                    ///< 逐元素操作的元素数
    size_t scratch_size = 0;              ///< JIT scratchpad 暂存大小 (in floats)
    /// DEBT-NEW-7 候选 A:融合 kernel 的真实输出 shape(从 FusedNode.out_desc 提取)
    /// 让 FusedCompiledKernel::execute() 能正确分配 output buffer(支持 MatMul-rooted region)
    std::vector<size_t> fused_out_shape;
};

/// 从 Graph 生成 JIT kernel
GeneratedKernel generateFromGraph(const Graph& graph);

} // namespace c3
} // namespace ct

#endif