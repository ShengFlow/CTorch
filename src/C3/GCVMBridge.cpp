/**
 * @file GCVMBridge.cpp
 * @brief GCVM C API 桥接实现 (v0.5 DCU 接入, 2026-08-10)
 * @details 核心 ~30 行 + 错误处理. 真实 API 签名以 gcvm.h 为准.
 *
 * 关键约束:
 *   - 仅 WITH_DCU 时编译, macOS 上不在 CMakeLists 加这个 .cpp
 *   - API 函数名是按 IREE 接入示例推断, 探针回来后可能调整
 *   - 错误信息要清晰 (return error_message 而非 abort)
 */
#include "../../include/C3/GCVMBridge.h"
#include "../../include/CtorchError.h"

#include <iostream>

#ifdef WITH_DCU
    // GCVM C API (按 IREE 接入示例推断的函数名)
    // 真实函数名以 /opt/dtk/llvm/gcvm/include/gcvm.h 为准
    // 探针 (probe-dcu-dtk24.sh) 跑通后根据 gcvm.h 调整
    extern "C" {
        // IREE 接入示例展示的 3 个 API
        int gcvmCreateProgram(void** prog);          // gcvmProgram* = void*
        int gcvmGetCompiledResult(void* prog, const char** result);
        int gcvmDestroyProgram(void* prog);

        // 推断的 API (探针后调整)
        int gcvmAddLLVMIR(void* prog, const char* ir);
        int gcvmSetTargetTriple(void* prog, const char* triple);
        int gcvmSetOptLevel(void* prog, int opt_level);
        int gcvmCompile(void* prog);
    }
#endif

namespace ct {
namespace c3 {

bool isGCVMAvailable() {
#ifdef WITH_DCU
    return true;  // 编译时已确定 WITH_DCU, 运行时检查移到 probe-dcu-dtk24.sh
#else
    return false;
#endif
}

GCVMCompileResult compileLLVMToDCUObject(const std::string& llvm_ir_source,
                                          const std::string& kernel_name,
                                          int opt_level) {
    GCVMCompileResult result;
    result.kernel_name = kernel_name;

#ifndef WITH_DCU
    result.success = false;
    result.error_message = "GCVM not compiled (WITH_DCU=OFF). Rebuild with -DWITH_DCU=ON on DCU node.";
    return result;
#else
    if (llvm_ir_source.empty()) {
        result.error_message = "Empty LLVM IR source";
        return result;
    }

    // 1. 创建 GCVM Program
    void* gcvm_prog = nullptr;
    int rc = gcvmCreateProgram(&gcvm_prog);
    if (rc != 0 || gcvm_prog == nullptr) {
        result.error_message = "gcvmCreateProgram failed (rc=" + std::to_string(rc) + ")";
        return result;
    }

    // 2. 喂 LLVM IR
    rc = gcvmAddLLVMIR(gcvm_prog, llvm_ir_source.c_str());
    if (rc != 0) {
        result.error_message = "gcvmAddLLVMIR failed (rc=" + std::to_string(rc) + ")";
        gcvmDestroyProgram(gcvm_prog);
        return result;
    }

    // 3. 设置 target triple (gfx906 = Hygon C86 7285)
    const char* target = "amdgcn-amd-amdhsa--gfx906";
    rc = gcvmSetTargetTriple(gcvm_prog, target);
    if (rc != 0) {
        result.error_message = "gcvmSetTargetTriple failed (rc=" + std::to_string(rc) + ", target=" + target + ")";
        gcvmDestroyProgram(gcvm_prog);
        return result;
    }

    // 4. 设置优化级别
    rc = gcvmSetOptLevel(gcvm_prog, opt_level);
    if (rc != 0) {
        result.error_message = "gcvmSetOptLevel failed (rc=" + std::to_string(rc) + ", level=" + std::to_string(opt_level) + ")";
        gcvmDestroyProgram(gcvm_prog);
        return result;
    }

    // 5. 编译
    rc = gcvmCompile(gcvm_prog);
    if (rc != 0) {
        result.error_message = "gcvmCompile failed (rc=" + std::to_string(rc) + ")";
        gcvmDestroyProgram(gcvm_prog);
        return result;
    }

    // 6. 拿编译结果 (Code Object bytes)
    const char* code_object_ptr = nullptr;
    rc = gcvmGetCompiledResult(gcvm_prog, &code_object_ptr);
    if (rc != 0 || code_object_ptr == nullptr) {
        result.error_message = "gcvmGetCompiledResult failed (rc=" + std::to_string(rc) + ")";
        gcvmDestroyProgram(gcvm_prog);
        return result;
    }

    // 7. 复制到 std::string (own copy, 不依赖 GCVM 内部 buffer)
    // 注: GCVM API 没给 result_size, 用 strlen 推测. 探针后可能需要改成带 size 参数
    result.code_object = std::string(code_object_ptr);

    // 8. 清理
    gcvmDestroyProgram(gcvm_prog);

    result.success = !result.code_object.empty();
    if (!result.success) {
        result.error_message = "GCVM returned empty code object";
    }
    return result;
#endif
}

}  // namespace c3
}  // namespace ct
