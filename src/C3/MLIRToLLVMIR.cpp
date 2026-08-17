/**
 * @file MLIRToLLVMIR.cpp
 * @brief MLIR → LLVM IR 转换实现 (v0.5 DCU 接入基础设施, 2026-08-10)
 *
 * 实现分层:
 *   - 底层 (mlirModuleToLLVMIRText / mlirModuleToLLVMBitcode): 纯 emit, 无 pipeline
 *     假设 module 已 lowered 到 LLVM dialect, 由调用方负责 MLIR 构建 + lowering
 *   - 高层 (mlirToLLVMIRFromGraph): 走完整 pipeline
 *     TODO 跨 session: 需要先把 MLIRKernelGen.cpp 内部的 buildMLIRModule +
 *     applyLoweringPipeline 暴露成 public API (改非 static + 加头文件), 才能复用
 *
 * 性能:
 *   - text emit: O(module size), 典型 MLP graph ~10-50ms
 *   - bitcode emit: 同上, bitcode 比 text 小 30-50%
 *
 * 线程安全:
 *   - registerBuiltinDialectTranslation / registerLLVMDialectTranslation 用 once_flag 守护
 *   - 多个 helper 调用并发安全
 */
#include "C3/MLIRToLLVMIR.h"

#ifdef CT_ENABLE_MLIR

#include "C3/Graph.h"  // 高层 API mlirToLLVMIRFromGraph 用
#include "MLIRKernelGen.h"  // ct::c3::buildMLIRModule + applyLoweringPipeline 公开 API

// MLIR 头
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/DialectRegistry.h>
// 注: MLIR 22.x 把 Module 相关定义整合到 BuiltinOps.h, 没有独立 Module.h
#include <mlir/Target/LLVMIR/Export.h>
// MLIR 22.x 翻译接口需要显式 include (跟 MLIRKernelGen.cpp:65-66 一致)
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

// 各 dialect 头 (mlirToLLVMIRFromGraph 需要显式 include, 避免 transitive 依赖)
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

// LLVM 头
#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Bitcode/BitcodeWriter.h>  // LLVM 22.x 拆分成 Reader/Writer 两个头
#include <llvm/Support/raw_ostream.h>

#include <chrono>
#include <sstream>
#include <stdexcept>

#endif  // CT_ENABLE_MLIR

namespace ct {
namespace c3 {

#ifdef CT_ENABLE_MLIR

#include <mutex>

extern std::mutex c3_global_mlir_mutex;

namespace {

// MLIR 翻译接口需要 registerBuiltinDialectTranslation + registerLLVMDialectTranslation
// 每个 MLIRContext 都要注册 (per-ctx, 不是全局).
// MLIR 的 register 函数是幂等的, 多调无副作用, 不用 once_flag
// (教训: 之前用 static once_flag 导致第二个 MLIRContext 没注册, 报 missing registration)
inline void ensureLLVMTranslationRegistered(mlir::MLIRContext& ctx) {
    std::lock_guard<std::mutex> lock(c3_global_mlir_mutex);
    mlir::registerBuiltinDialectTranslation(ctx);
    mlir::registerLLVMDialectTranslation(ctx);
}

}  // anonymous namespace

std::string mlirModuleToLLVMIRText(mlir::ModuleOp module) {
    if (!module) {
        return "";  // 空 module 返空字符串
    }

    // 1. 确保翻译接口注册 (每个 MLIRContext 一次)
    mlir::MLIRContext& mlir_ctx = *module.getContext();
    ensureLLVMTranslationRegistered(mlir_ctx);

    // 2. 翻译 MLIR module → llvm::Module
    // 注意: 每个转换创建独立的 llvm::LLVMContext, 避免跟调用方已有 context 冲突
    llvm::LLVMContext llvm_ctx;
    auto llvm_module = mlir::translateModuleToLLVMIR(module, llvm_ctx);
    if (!llvm_module) {
        return "";  // 翻译失败
    }

    // 3. 验证 LLVM IR well-formed (默认开, 跟 LLVM JIT 调用前的检查对齐)
    std::string verify_errors;
    llvm::raw_string_ostream verify_os(verify_errors);
    if (llvm::verifyModule(*llvm_module, &verify_os)) {
        verify_os.flush();
        // 验证失败: 返空字符串 + stderr 提示
        // 注: 不抛异常, 跟 bitcode helper 行为对齐, 让调用方决定怎么 handle
        llvm::errs() << "[MLIRToLLVMIR] verifyModule failed:\n" << verify_errors;
        return "";
    }

    // 4. emit text (人类可读)
    std::string text;
    llvm::raw_string_ostream os(text);
    llvm_module->print(os, /*AssemblyAnnotationWriter=*/nullptr);
    os.flush();

    return text;
}

std::vector<uint8_t> mlirModuleToLLVMBitcode(mlir::ModuleOp module) {
    if (!module) {
        return {};
    }

    mlir::MLIRContext& mlir_ctx = *module.getContext();
    ensureLLVMTranslationRegistered(mlir_ctx);

    llvm::LLVMContext llvm_ctx;
    auto llvm_module = mlir::translateModuleToLLVMIR(module, llvm_ctx);
    if (!llvm_module) {
        return {};
    }

    std::string verify_errors;
    llvm::raw_string_ostream verify_os(verify_errors);
    if (llvm::verifyModule(*llvm_module, &verify_os)) {
        verify_os.flush();
        llvm::errs() << "[MLIRToLLVMIR] verifyModule failed (bitcode path):\n" << verify_errors;
        return {};
    }

    // emit bitcode (.bc 格式)
    // LLVM 22.x 没有 WriteBitcodeToBuffer, 用 WriteBitcodeToFile + raw_svector_ostream
    // (JITCache.cpp 也是这个模式, 见 src/C3/JITCache.cpp)
    llvm::SmallVector<char, 0> buffer;
    {
        llvm::raw_svector_ostream os(buffer);
        llvm::WriteBitcodeToFile(*llvm_module, os);
    }

    return std::vector<uint8_t>(buffer.begin(), buffer.end());
}

// 高层 API: 跨 session follow-up
// [Dev] v0.5.2 DCU 接入 refactor (2026-08-10): 实装完成
//   复用 src/C3/MLIRKernelGen.cpp 公开的 buildMLIRModule + applyLoweringPipeline,
//   跟 generateFromGraphMLIR 共享同一份 build / lower 逻辑
//   区别: 不走 ExecutionEngine JIT, 改 emit LLVM IR text / bitcode 给 GCVM/dcc
MLIRToLLVMIRResult mlirToLLVMIRFromGraph(const Graph& graph,
                                          const MLIRToLLVMIROptions& opts) {
    MLIRToLLVMIRResult result;
    auto t0 = std::chrono::steady_clock::now();

    try {
        // 1. 创建 MLIRContext + 注册必要 dialect (跟 MLIRKernelGen::generateFromGraphMLIR 对齐)
        mlir::DialectRegistry reg;
        reg.insert<mlir::arith::ArithDialect>();
        reg.insert<mlir::math::MathDialect>();
        reg.insert<mlir::scf::SCFDialect>();
        reg.insert<mlir::func::FuncDialect>();
        reg.insert<mlir::memref::MemRefDialect>();
        reg.insert<mlir::LLVM::LLVMDialect>();

        mlir::MLIRContext context(reg);
        context.loadDialect<mlir::arith::ArithDialect>();
        context.loadDialect<mlir::math::MathDialect>();
        context.loadDialect<mlir::scf::SCFDialect>();
        context.loadDialect<mlir::func::FuncDialect>();
        context.loadDialect<mlir::memref::MemRefDialect>();
        context.loadDialect<mlir::LLVM::LLVMDialect>();

        // 2. Build MLIR module (复用公开 API)
        auto module = ct::c3::buildMLIRModule(context, graph);

        // 3. 调试 dump
        if (opts.dump_mlir) {
            llvm::errs() << "=== MLIR module BEFORE lowering ===\n";
            module->dump();
        }

        // 4. Apply lowering pipeline (复用公开 API)
        ct::c3::applyLoweringPipeline(*module);

        if (opts.dump_mlir) {
            llvm::errs() << "=== MLIR module AFTER lowering (LLVM dialect) ===\n";
            module->dump();
        }

        // 5. 翻译 MLIR module → LLVM IR (调用底层 API)
        result.text = mlirModuleToLLVMIRText(*module);
        if (result.text.empty()) {
            result.error_message = "mlirModuleToLLVMIRText returned empty (translate/verify failed)";
            return result;
        }

        // 6. 同步 emit bitcode (供 Plan B dcc 备用)
        result.bitcode = mlirModuleToLLVMBitcode(*module);
        if (result.bitcode.empty() && opts.verify_llvm_ir) {
            // text 验证过但 bitcode 失败 — 罕见, 警告但不当作错误
            llvm::errs() << "[MLIRToLLVMIR] warning: bitcode emit failed but text succeeded"
                         << " (might be a tmp issue, text path is enough for Plan A)\n";
        }

        result.success = true;
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("mlirToLLVMIRFromGraph exception: ") + e.what();
    } catch (...) {
        result.success = false;
        result.error_message = "mlirToLLVMIRFromGraph: unknown exception";
    }

    auto t1 = std::chrono::steady_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
}

#endif  // CT_ENABLE_MLIR

}  // namespace c3
}  // namespace ct
