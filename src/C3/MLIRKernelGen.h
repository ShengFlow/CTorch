/**
 * @file MLIRKernelGen.h
 * @brief C3 JIT MLIR kernel 生成器（Phase 1: MLIR/LLVM 后端）
 * @details 将 Graph 编译为 MLIR module，通过标准 lowering pipeline 降至 LLVM IR，
 *          再经 ExecutionEngine JIT 编译为原生函数指针。替代 HandwrittenKernelGen。
 *          输出与 HandwrittenKernelGen 完全相同的 GeneratedKernel 结构体，
 *          下游 C3KernelRegistry / CtorchScheduler 无需任何改动。
 *
 *          Pipeline: Graph → MLIR (arith+scf+memref+func) → LLVM dialect → JIT
 * @date 2026/8/1
 */

#ifndef CTORCH_C3_MLIR_KERNEL_GEN_H
#define CTORCH_C3_MLIR_KERNEL_GEN_H

#include "HandwrittenKernelGen.h"  // 复用 GeneratedKernel 定义

namespace ct {
namespace c3 {

/**
 * @brief 从 Graph 生成 MLIR 编译的 kernel（Phase 1 LLVM 后端）
 * @param graph 经过 canonicalize 的计算图
 * @return GeneratedKernel 包含函数指针和资源管理回调
 * @throw std::runtime_error 编译失败时抛出
 */
GeneratedKernel generateFromGraphMLIR(const Graph& graph);

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_MLIR_KERNEL_GEN_H