/**
 * @file GCVMBridge.cpp
 * @brief GCVM C API 桥接实现 (v0.5 DCU 接入, 2026-08-10, 探针后调整)
 * @details 按 gcvm.h 真实 API 重写 (v0.5.1 调整)
 *
 * 关键调整 (vs v0.5.0 推测):
 *   - gcvmAddModuleToProgram: 5 参数 (prog, buffer, size, name, SourceType), 不是 1 参数
 *   - gcvmCompileProgram: 4 参数 (prog, numOptions, options, ResultType), 不是 1 参数
 *   - gcvmSetArch: 用 -arch=gfx906 (CUDA compute_xx 风格可能不适用, 待节点验证)
 *   - ResultType::Hsaco: 拿 HSACO
 *   - 错误码用 gcvmGetErrorString 转字符串
 *
 * 已知风险:
 *   - GCVM IR version 1.6 = LLVM 7.0.1, C3 MLIR 22.1.8 输出 LLVM 14 IR
 *     → IR_VERSION_MISMATCH 风险 (节点实测)
 *   - arch 选项 cuda 命名 vs DCU gfx906 → 实测才知道
 *   - 节点缺 mlir-translate, C3 MLIR → LLVM IR 转换要装 LLVM/MLIR 工具链
 */
#include "../../include/C3/GCVMBridge.h"
#include "../../include/CtorchError.h"

#include <cstring>
#include <vector>

#ifdef WITH_DCU
    // GCVM C API 头文件 (per dcu-probe-dtk24-b02r2n11.md, DTK 26.04 路径)
    #include <gcvm.h>
#endif

namespace ct {
namespace c3 {

bool isGCVMAvailable() {
#ifdef WITH_DCU
    return true;
#else
    return false;
#endif
}

#ifdef WITH_DCU

/// 把 GCVM API 错误码转可读字符串 (代替直接用 magic number)
static std::string gcvmErrorToString(gcvmResult rc) {
    const char* msg = gcvmGetErrorString(rc);
    return std::string(msg ? msg : "unknown") + " (rc=" + std::to_string(rc) + ")";
}

#endif

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
    gcvmProgram prog = nullptr;
    gcvmResult rc = gcvmCreateProgram(&prog);
    if (rc != GCVM_SUCCESS || prog == nullptr) {
        result.error_message = "gcvmCreateProgram failed: " + gcvmErrorToString(rc);
        return result;
    }

    // RAII 清理: 任何 return 路径前都调 gcvmDestroyProgram
    struct ProgramGuard {
        gcvmProgram* p;
        ~ProgramGuard() { if (p && *p) gcvmDestroyProgram(p); }
    } guard{&prog};

    // 2. 设置 arch (gfx906 = Hygon C86 7285)
    // 注意: GCVM 头注释 arch 是 CUDA compute_xx 风格, DCU arch 名待节点验证
    // 尝试 gfx906 (LLVM/AMDGPU 命名), 失败回退 compute_80 (最高 CUDA compute)
    const char* arch_attempts[] = {"gfx906", "compute_80", nullptr};
    bool arch_set = false;
    for (int i = 0; arch_attempts[i]; ++i) {
        rc = gcvmSetArch(prog, arch_attempts[i]);
        if (rc == GCVM_SUCCESS) {
            arch_set = true;
            break;
        }
    }
    if (!arch_set) {
        result.error_message = "gcvmSetArch failed (tried gfx906, compute_80): " + gcvmErrorToString(rc);
        return result;
    }

    // 3. 设置优化级别
    rc = gcvmSetOptLevel(prog, opt_level);
    if (rc != GCVM_SUCCESS) {
        result.error_message = "gcvmSetOptLevel failed: " + gcvmErrorToString(rc);
        return result;
    }

    // 4. Add module (LLVM IR text representation)
    // SourceType::LLVMIR 表示 LLVM IR text (vs bitcode)
    rc = gcvmAddModuleToProgram(prog,
                                 llvm_ir_source.c_str(),
                                 llvm_ir_source.size(),
                                 kernel_name.c_str(),
                                 LLVMIR);
    if (rc != GCVM_SUCCESS) {
        result.error_message = "gcvmAddModuleToProgram failed: " + gcvmErrorToString(rc) +
                               " (可能 GCVM IR version 1.6 vs LLVM " +
                               "(C3 MLIR 22.1.8 输出) 不兼容)";
        return result;
    }

    // 5. Compile with options
    // 选项: -opt=N 映射 opt_level, -g 关闭 (release), -ftz=1
    std::vector<const char*> options;
    std::string opt_flag = "-opt=" + std::to_string(opt_level);
    options.push_back(opt_flag.c_str());
    options.push_back("-ftz=1");
    options.push_back("-fma=1");

    rc = gcvmCompileProgram(prog,
                            static_cast<int>(options.size()),
                            options.data(),
                            Hsaco);
    if (rc != GCVM_SUCCESS) {
        // 拿错误日志
        size_t log_size = 0;
        if (gcvmGetProgramLogSize(prog, &log_size) == GCVM_SUCCESS && log_size > 0) {
            std::vector<char> log_buf(log_size);
            if (gcvmGetProgramLog(prog, log_buf.data()) == GCVM_SUCCESS) {
                result.error_message = "gcvmCompileProgram failed: " + gcvmErrorToString(rc) +
                                       " | log: " + std::string(log_buf.data());
            } else {
                result.error_message = "gcvmCompileProgram failed: " + gcvmErrorToString(rc);
            }
        } else {
            result.error_message = "gcvmCompileProgram failed: " + gcvmErrorToString(rc);
        }
        return result;
    }

    // 6. 拿 Code Object 大小
    size_t code_size = 0;
    rc = gcvmGetCompiledResultSize(prog, &code_size);
    if (rc != GCVM_SUCCESS || code_size == 0) {
        result.error_message = "gcvmGetCompiledResultSize failed: " + gcvmErrorToString(rc);
        return result;
    }

    // 7. 拿 Code Object bytes
    std::vector<char> code_buf(code_size);
    rc = gcvmGetCompiledResult(prog, code_buf.data());
    if (rc != GCVM_SUCCESS) {
        result.error_message = "gcvmGetCompiledResult failed: " + gcvmErrorToString(rc);
        return result;
    }

    // 8. 复制到 std::string (own copy)
    result.code_object.assign(code_buf.data(), code_size);
    result.success = true;
    return result;
#endif
}

}  // namespace c3
}  // namespace ct
