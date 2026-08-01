/**
 * @file HandwrittenKernelGen.h
 * @brief C3 JIT 手写 kernel 生成器内部接口
 * @details 供 JITEngine.cpp 调用，不对外暴露。
 * @date 2026/7/31
 */

#ifndef CTORCH_JIT_HANDWRITTEN_KERNEL_GEN_H
#define CTORCH_JIT_HANDWRITTEN_KERNEL_GEN_H

#include "../../include/JIT/Graph.h"
#include "../../include/JIT/C3KernelRegistry.h"

namespace ct {
namespace jit {

// C3KernelFunc 类型定义在 C3KernelRegistry.h 中，此处复用

struct GeneratedKernel {
    C3KernelFunc func = nullptr;
    void* dl_handle = nullptr;
    bool is_matmul = false;
    size_t M = 0, K = 0, N = 0;
};

/// 从 Graph 生成 JIT kernel
GeneratedKernel generateFromGraph(const Graph& graph);

} // namespace jit
} // namespace ct

#endif