/**
 * @file MLIRKernelGen.h
 * @generation JIT-2.x MLIR 标量单算子 IR 后端
 * @note generateFromGraphMLIR 开头内嵌 3.0 路由 tryBuildLinalgElementwise
 *       （命中单节点逐元素则改走 LinalgElementwiseGen），待下轮物理分家时迁出。
 * @brief C3 JIT MLIR kernel 生成器（Phase 1: MLIR/LLVM 后端）
 * @details 将 Graph 编译为 MLIR module，通过标准 lowering pipeline 降至 LLVM IR，
 *          再经 ExecutionEngine JIT 编译为原生函数指针。替代 HandwrittenKernelGen。
 *          输出与 HandwrittenKernelGen 完全相同的 GeneratedKernel 结构体，
 *          下游 C3KernelRegistry / CtorchScheduler 无需任何改动。
 *
 *          Pipeline: Graph → MLIR (arith+scf+memref+func) → LLVM dialect → JIT
 * @date 2026/8/1
 *
 * [Dev] v0.5.2 DCU 接入 refactor (2026-08-10):
 *   抽 buildMLIRModule + applyLoweringPipeline 公开 API, 让 MLIRToLLVMIR.cpp
 *   的 mlirToLLVMIRFromGraph 复用同一份 build / lower 逻辑 (不重复代码).
 */

#ifndef CTORCH_C3_MLIR_KERNEL_GEN_H
#define CTORCH_C3_MLIR_KERNEL_GEN_H

#include "C3/GeneratedKernel.h"  // 复用 GeneratedKernel 定义 (SHARED)
#include <mutex>

namespace mlir {
    class MLIRContext;
    class ModuleOp;
    template <typename T> class OwningOpRef;
}

namespace ct {
namespace c3 {

extern std::mutex c3_global_mlir_mutex;

/**
 * @brief 从 Graph 生成 MLIR 编译的 kernel（Phase 1 LLVM 后端）
 * @param graph 经过 canonicalize 的计算图
 * @param opt_level LLVM 优化级别（0=O0, 1=O1, 2=O2, 3=O3/Ofast，默认 2）
 * @return GeneratedKernel 包含函数指针和资源管理回调
 * @throw std::runtime_error 编译失败时抛出
 */
GeneratedKernel generateFromGraphMLIR(const Graph& graph, int opt_level = 2);

/**
 * @brief [v0.5.2 公开] 从 C3 Graph 构建 MLIR Module
 * @details 之前是 file-static, 现抽公开让 MLIRToLLVMIR.cpp 复用
 *
 * @param context 已注册必要 dialect (arith/math/scf/func/memref/LLVM) 的 MLIRContext
 * @param graph C3 Graph
 * @return OwningOpRef<ModuleOp>
 * @throw std::runtime_error 当 graph 校验失败
 */
mlir::OwningOpRef<mlir::ModuleOp> buildMLIRModule(mlir::MLIRContext& context, const Graph& graph);

/**
 * @brief [v0.5.2 公开] 对 MLIR Module 跑标准 lowering pipeline
 * @details Pipeline 顺序: Canonicalizer → CSE → LICM → SCFToCF → ArithToLLVM →
 *          CFToLLVM → FuncToLLVM → MemRefToLLVM → ReconcileUnrealizedCasts
 *          跑完 module 在 LLVM dialect, 可直接喂 mlir::translateModuleToLLVMIR
 */
void applyLoweringPipeline(mlir::ModuleOp module, int opt_level = 3);

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_MLIR_KERNEL_GEN_H