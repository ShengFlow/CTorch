/**
 * @file MLIRToLLVMIR.h
 * @brief MLIR → LLVM IR 转换 helper (v0.5 DCU 接入基础设施, 2026-08-10)
 * @details 把 C3 MLIR backend 输出的 MLIR module 转成 LLVM IR (text 或 bitcode),
 *          作为 Plan A (GCVM) / Plan B (dcc) / Plan C (CPU JIT) 的共同输入。
 *
 * 设计:
 *   - 高级 API (mlirToLLVMIRFromGraph): 接受 Graph, 跑完整 pipeline
 *     (buildMLIRModule + applyLoweringPipeline + emit), 跟 generateFromGraphMLIR 一致
 *   - 底层 API (mlirModuleToLLVMIRText / mlirModuleToLLVMBitcode): 接受
 *     已经 lowered 到 LLVM dialect 的 mlir::ModuleOp, 纯 emit, 无 pipeline
 *
 * 依赖:
 *   - CT_ENABLE_MLIR=ON (因为要 #include mlir 头)
 *   - LLVM/MLIR 工具链 (MLIRKernelGen 已有依赖, 共享)
 *
 * 跟 JITCache 解耦:
 *   - JITCache 是 "build + lower + write bitcode to disk + JIT"
 *   - 本 helper 是 "build + lower + emit text/bitcode in memory"
 *   - 共享 MLIR 构建 + lowering 逻辑 (未来可考虑提取公共 base, 暂不重构)
 */
#pragma once

#include <string>
#include <vector>
#include <cstdint>

#ifdef CT_ENABLE_MLIR
    // MLIR forward declarations (避免头文件强依赖, 加速编译)
    namespace mlir {
        class ModuleOp;
    }
#endif

namespace ct {
namespace c3 {

class Graph;  // 前置声明, 避免 include Graph.h

#ifdef CT_ENABLE_MLIR

/// 转换选项
struct MLIRToLLVMIROptions {
    /// 优化级别 (0=None, 1=Less, 2=Default, 3+=Aggressive)
    /// 跟 MLIRKernelGen::generateFromGraphMLIR 内部逻辑对齐
    int opt_level = 2;

    /// 调试: 是否 dump MLIR module 到 stderr (lowering 前 + 后)
    bool dump_mlir = false;

    /// 调试: 验证 emit 出来的 LLVM IR 是否 well-formed
    bool verify_llvm_ir = true;
};

/// 转换结果
struct MLIRToLLVMIRResult {
    /// 成功标志
    bool success = false;

    /// 错误信息 (success=false 时填)
    std::string error_message;

    /// LLVM IR text (默认填, 可直接喂 LLVM JIT 或 GCVM C API)
    std::string text;

    /// LLVM bitcode (按需填, opt_level 高时体积更小, 适合 dcc 命令行输入)
    /// 注: 即使 emit_bitcode=false 也可能填, 用于双重验证
    std::vector<uint8_t> bitcode;

    /// 转换耗时 (毫秒, 给 perf baseline 用)
    double elapsed_ms = 0.0;
};

/**
 * @brief 高层 API: 接受 Graph, 走完整 MLIR→LLVM IR pipeline
 * @param graph C3 Graph (含输入/输出/算子节点)
 * @param opts 转换选项
 * @return MLIRToLLVMIRResult
 *
 * 内部流程 (跟 MLIRKernelGen::generateFromGraphMLIR 一致):
 *   1. 创建 MLIRContext + 注册必要 dialect (arith/math/scf/func/memref/LLVM)
 *   2. buildMLIRModule(context, graph) → MLIR ModuleOp
 *   3. applyLoweringPipeline(module) → lower 到 LLVM dialect
 *   4. registerBuiltinDialectTranslation + registerLLVMDialectTranslation
 *   5. translateModuleToLLVMIR(module, ctx) → llvm::Module
 *   6. emit LLVM IR text (raw_ostream → string) + 可选 bitcode
 *
 * 注意: 本函数是 MLIRKernelGen::generateFromGraphMLIR 的 "无 ExecutionEngine" 版本
 *       共用 buildMLIRModule + applyLoweringPipeline, 拆分点就在 ExecutionEngine::create 之前
 */
MLIRToLLVMIRResult mlirToLLVMIRFromGraph(const Graph& graph,
                                          const MLIRToLLVMIROptions& opts = {});

/**
 * @brief 底层 API: 把已经 lowered 的 MLIR module emit 成 LLVM IR text
 * @param module 已 lowered 到 LLVM dialect 的 mlir::ModuleOp
 * @return LLVM IR text (空字符串表示失败)
 *
 * 假设: 调用方已 applyLoweringPipeline, 且 module.context 已 registerLLVMDialectTranslation
 */
std::string mlirModuleToLLVMIRText(mlir::ModuleOp module);

/**
 * @brief 底层 API: 把已经 lowered 的 MLIR module emit 成 LLVM bitcode
 * @param module 已 lowered 到 LLVM dialect 的 mlir::ModuleOp
 * @return LLVM bitcode bytes (空 vector 表示失败)
 */
std::vector<uint8_t> mlirModuleToLLVMBitcode(mlir::ModuleOp module);

#endif  // CT_ENABLE_MLIR

}  // namespace c3
}  // namespace ct
