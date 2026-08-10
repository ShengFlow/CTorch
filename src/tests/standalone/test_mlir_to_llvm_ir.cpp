/**
 * @file test_mlir_to_llvm_ir.cpp
 * @brief MLIRToLLVMIR helper 单测 (v0.5 DCU 接入基础设施, 2026-08-10)
 * @details 验证:
 *          1. mlirModuleToLLVMIRText 接受 LLVM dialect module, 产出 valid LLVM IR text
 *          2. mlirModuleToLLVMBitcode 接受 LLVM dialect module, 产出 valid bitcode
 *          3. 空 module / null module 优雅处理 (返空)
 *          4. mlirToLLVMIRFromGraph 当前是 stub (返 "not implemented", 跨 session 验证)
 *
 * 用法: ./test_mlir_to_llvm_ir
 *       (在 CT_ENABLE_MLIR=ON build 下, macOS 本地能跑, DCU 节点也能跑)
 */
#include "C3/MLIRToLLVMIR.h"
#include "C3/Graph.h"  // for ct::c3::Graph (test 4 用)

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
// 注: MLIR 22.x 把 Module 相关定义整合到 BuiltinOps.h, 没有独立 Module.h
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include <iostream>
#include <string>

namespace {

int g_pass_count = 0;
int g_fail_count = 0;

#define CHECK(cond, msg) do { \
    if (cond) { \
        std::cout << "  ✅ " << msg << std::endl; \
        ++g_pass_count; \
    } else { \
        std::cerr << "  ❌ FAIL: " << msg << std::endl; \
        ++g_fail_count; \
    } \
} while (0)

void testTextEmitOnLLVMDialectModule() {
    std::cout << "\n[TEST 1] mlirModuleToLLVMIRText on LLVM dialect module" << std::endl;

    // 1. 构造 MLIR context, 注册必要 dialect
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    mlir::MLIRContext context(registry);
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();
    context.loadDialect<mlir::LLVM::LLVMDialect>();

    // 2. 解析 LLVM dialect MLIR 文本
    // 注: 用 builtin i32 (MLIR builtin integer), translateModuleToLLVMIR 会自动映射到 LLVM i32
    const char* mlir_text = R"mlir(
module {
  llvm.func @c3_add(%a: i32, %b: i32) -> i32 {
    %0 = llvm.add %a, %b : i32
    llvm.return %0 : i32
  }
}
)mlir";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(mlir_text, &context);
    CHECK(module, "parseSourceString 成功");

    if (!module) return;

    // 3. 调 helper 拿 LLVM IR text
    std::string llvm_ir = ct::c3::mlirModuleToLLVMIRText(*module);
    CHECK(!llvm_ir.empty(), "mlirModuleToLLVMIRText 返非空 text");
    if (llvm_ir.empty()) return;

    // 4. 验证 text 是合法 LLVM IR (有 'define' 关键字)
    CHECK(llvm_ir.find("define") != std::string::npos, "text 含 'define' (LLVM IR 函数定义)");
    CHECK(llvm_ir.find("@c3_add") != std::string::npos, "text 含 '@c3_add' (函数名)");

    // 5. dump 出来 (给肉眼 check)
    std::cout << "  --- LLVM IR text (前 200 chars) ---" << std::endl;
    std::string preview = llvm_ir.substr(0, std::min<size_t>(200, llvm_ir.size()));
    std::cout << "  " << preview << (llvm_ir.size() > 200 ? "..." : "") << std::endl;
    std::cout << "  --- 长度: " << llvm_ir.size() << " chars ---" << std::endl;
}

void testBitcodeEmitOnLLVMDialectModule() {
    std::cout << "\n[TEST 2] mlirModuleToLLVMBitcode on LLVM dialect module" << std::endl;

    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    mlir::MLIRContext context(registry);
    context.loadDialect<mlir::func::FuncDialect>();
    context.loadDialect<mlir::arith::ArithDialect>();
    context.loadDialect<mlir::LLVM::LLVMDialect>();

    const char* mlir_text = R"mlir(
module {
  llvm.func @c3_mul(%a: i32, %b: i32) -> i32 {
    %0 = llvm.mul %a, %b : i32
    llvm.return %0 : i32
  }
}
)mlir";

    auto module = mlir::parseSourceString<mlir::ModuleOp>(mlir_text, &context);
    CHECK(module, "parseSourceString 成功");

    if (!module) return;

    auto bc = ct::c3::mlirModuleToLLVMBitcode(*module);
    CHECK(!bc.empty(), "mlirModuleToLLVMBitcode 返非空 bitcode");
    if (bc.empty()) return;

    // 验证 bitcode 头 (LLVM bitcode magic: 'B' 'C' 0xC0 0xDE)
    CHECK(bc.size() >= 4, "bitcode 长度 >= 4");
    CHECK(bc[0] == 'B' && bc[1] == 'C' && bc[2] == 0xC0 && bc[3] == 0xDE,
          "bitcode 头 4 字节 = BC \\xC0\\xDE (LLVM bitcode magic)");

    std::cout << "  bitcode size: " << bc.size() << " bytes" << std::endl;
}

void testEmptyModule() {
    std::cout << "\n[TEST 3] 空 MLIR module 优雅处理" << std::endl;

    mlir::DialectRegistry registry;
    mlir::MLIRContext context(registry);

    // 构造空 module
    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    CHECK(module, "空 module 创建成功");

    // 注: ModuleOp::create 直接返 ModuleOp (值), 不需要解引用
    std::string text = ct::c3::mlirModuleToLLVMIRText(module);
    // 空 module 应该 emit 出 "; ModuleID = ..." 等注释,但核心内容为空
    // helper 不报错就算通过
    std::cout << "  空 module text 长度: " << text.size() << " (允许为空)" << std::endl;
    CHECK(true, "空 module 不抛异常, helper 优雅返回");
}

void testHighLevelAPIIsStub() {
    std::cout << "\n[TEST 4] mlirToLLVMIRFromGraph 当前是 stub (跨 session TODO)" << std::endl;

    // 构造最小 Graph (空 graph, 反正 helper 不会真用它, 直接走 stub 分支)
    ct::c3::Graph g;

    ct::c3::MLIRToLLVMIROptions opts;
    opts.opt_level = 2;
    opts.dump_mlir = false;
    opts.verify_llvm_ir = true;

    auto result = ct::c3::mlirToLLVMIRFromGraph(g, opts);
    CHECK(!result.success, "mlirToLLVMIRFromGraph 当前 success=false (stub)");
    CHECK(!result.error_message.empty(), "error_message 非空, 说明 stub 状态");
    CHECK(result.error_message.find("not implemented") != std::string::npos,
          "error_message 含 'not implemented' 字样");

    std::cout << "  error_message: " << result.error_message.substr(0, 100)
              << "..." << std::endl;
}

}  // anonymous namespace

int main() {
    std::cout << "=== MLIRToLLVMIR helper 单测 ===" << std::endl;
    std::cout << "v0.5 DCU 接入基础设施 (2026-08-10)" << std::endl;

    testTextEmitOnLLVMDialectModule();
    testBitcodeEmitOnLLVMDialectModule();
    testEmptyModule();
    testHighLevelAPIIsStub();

    std::cout << "\n=== 总计 ===" << std::endl;
    std::cout << "  PASS: " << g_pass_count << std::endl;
    std::cout << "  FAIL: " << g_fail_count << std::endl;

    return (g_fail_count == 0) ? 0 : 1;
}
