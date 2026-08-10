/**
 * @file GCVMBridge.h
 * @brief 海光 DCU GCVM C API 桥接 (v0.5 DCU 接入, 2026-08-10)
 * @details C3 MLIR → LLVM IR → GCVM → gfx906 Code Object (HSACO) 桥接层
 *          基于 IREE 接入 GCVM 官方示例 (per docs/dcu-docs/knowledge_base)
 *
 * 注意: 真实 GCVM C API 函数签名以 /opt/dtk/llvm/gcvm/include/gcvm.h 为准
 *       本文件 API 是按 IREE 接入示例推断的, 探针 (probe-dcu-dtk24.sh) 回来后调整
 *
 * 编译: 必须 WITH_DCU=ON + 链接 libgcvm.so (在 DCU 节点)
 *       macOS 上 WITH_DCU 默认 OFF, 不需要 GCVM 库
 */
#ifndef CTORCH_C3_GCVM_BRIDGE_H
#define CTORCH_C3_GCVM_BRIDGE_H

#include <cstdint>
#include <string>

// 仅在 WITH_DCU 时包含 GCVM 头文件
#ifdef WITH_DCU
    // 路径: /opt/dtk/llvm/gcvm/include/gcvm.h (DTK 24.04+)
    // macOS 上 WITH_DCU 默认 OFF, 不需要 GCVM 头
    #include <gcvm.h>  // GCVM C API (verified via CMakeLists try_compile)
#endif

namespace ct {
namespace c3 {

/// GCVM 编译结果
struct GCVMCompileResult {
    bool success = false;
    std::string error_message;
    std::string code_object;     // HSACO / Code Object bytes
    std::string kernel_name;     // 默认 "c3_kernel" (跟 MLIR c3_kernel 对齐)
};

/// 把 LLVM IR (来自 C3 MLIR translateModuleToLLVMIR) 编译成 gfx906 Code Object
///
/// @param llvm_ir_source LLVM IR 序列化字符串 (从 llvmModule->print 拿)
/// @param kernel_name    kernel symbol 名字 (默认 "c3_kernel", 跟 MLIRKernelGen.cpp 对齐)
/// @param opt_level      0=None, 1=Less, 2=Default, 3=Aggressive
/// @return GCVMCompileResult {success, error_message, code_object, kernel_name}
GCVMCompileResult compileLLVMToDCUObject(const std::string& llvm_ir_source,
                                          const std::string& kernel_name = "c3_kernel",
                                          int opt_level = 2);

/// 检测 GCVM 是否可用 (WITH_DCU 编译 + libgcvm.so 链接成功)
/// @return true if GCVM C API 可调用, false otherwise
bool isGCVMAvailable();

}  // namespace c3
}  // namespace ct

#endif  // CTORCH_C3_GCVM_BRIDGE_H
