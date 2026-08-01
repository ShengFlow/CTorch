/**
 * @file HandwrittenKernelGen.cpp
 * @brief C3 JIT 手写 kernel 生成器（EXP-1 阶段）
 * @details 为简单图（Add/Mul/MatMul）生成 C++ kernel 源码，通过系统 clang++ 编译为
 *          .so 动态库，dlsym 加载。这是 MLIR+LLVM 后端就绪前的临时方案，用于验证
 *          Graph IR → 编译 → 执行 的最小闭环。
 * @date 2026/7/31
 */

#include "HandwrittenKernelGen.h"
#include "../../include/JIT/Graph.h"

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#ifdef __APPLE__
#include <unistd.h>
#endif

namespace ct {
namespace jit {

// ======================= 源码生成 =======================

/// 为 Add 操作生成 kernel 源码
static std::string generateAddKernel() {
    return R"SRC(
#include <cstddef>
extern "C" void c3_kernel(const float* a, const float* b, float* out,
                          size_t n, size_t, size_t, size_t) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}
)SRC";
}

/// 为 Mul 操作生成 kernel 源码
static std::string generateMulKernel() {
    return R"SRC(
#include <cstddef>
extern "C" void c3_kernel(const float* a, const float* b, float* out,
                          size_t n, size_t, size_t, size_t) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}
)SRC";
}

/// 为 MatMul 操作生成 kernel 源码
static std::string generateMatMulKernel() {
    return R"SRC(
#include <cstddef>
extern "C" void c3_kernel(const float* a, const float* b, float* out,
                          size_t, size_t M, size_t K, size_t N) {
    // out = a @ b, a: (M, K), b: (K, N), out: (M, N)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += a[i * K + k] * b[k * N + j];
            }
            out[i * N + j] = sum;
        }
    }
}
)SRC";
}

// ======================= 编译与加载 =======================

/**
 * @brief 从源码字符串编译为 .so 并加载函数指针
 * @param src C++ 源码字符串
 * @param func_name 要加载的函数名
 * @return 函数指针和 dlopen 句柄的 pair
 * @throw std::runtime_error 编译或加载失败时抛出
 */
static std::pair<C3KernelFunc, void*> compileAndLoad(const std::string& src,
                                                      const std::string& func_name) {
    // 生成临时文件名
    char tmpdir_template[] = "/tmp/ctorch_c3_XXXXXX";
    std::string tmpdir = mkdtemp(tmpdir_template);
    if (tmpdir.empty()) {
        throw std::runtime_error("HandwrittenKernelGen: failed to create temp directory");
    }

    std::string src_path = tmpdir + "/kernel.cpp";
    std::string so_path = tmpdir + "/kernel.so";

    // 写入源码
    {
        std::ofstream out(src_path);
        if (!out) {
            throw std::runtime_error("HandwrittenKernelGen: failed to write kernel source");
        }
        out << src;
        out.close();
    }

    // 编译为 .so
    std::ostringstream cmd;
    cmd << "clang++ -O3 -ffast-math -march=native -fPIC -shared -std=c++20 "
        << "-o " << so_path << " " << src_path << " 2>&1";

    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("HandwrittenKernelGen: failed to invoke clang++");
    }

    std::string compile_output;
    char buf[256];
    while (fgets(buf, sizeof(buf), pipe) != nullptr) {
        compile_output += buf;
    }
    int compile_status = pclose(pipe);

    if (compile_status != 0) {
        throw std::runtime_error(
            "HandwrittenKernelGen: compilation failed:\n" + compile_output);
    }

    // 加载 .so
    void* handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        throw std::runtime_error(
            std::string("HandwrittenKernelGen: dlopen failed: ") + dlerror());
    }

    // 清除错误状态
    dlerror();

    // 查找函数
    auto* func_ptr = reinterpret_cast<C3KernelFunc>(dlsym(handle, func_name.c_str()));
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
        dlclose(handle);
        throw std::runtime_error(
            std::string("HandwrittenKernelGen: dlsym failed: ") + dlsym_error);
    }

    return {func_ptr, handle};
}

// ======================= Graph → Kernel 生成 =======================

GeneratedKernel generateFromGraph(const Graph& graph) {
    // 分析图：找到第一个计算节点
    const auto& nodes = graph.nodes();
    if (nodes.size() < 3) {
        throw std::runtime_error("HandwrittenKernelGen: graph has too few nodes");
    }

    // 跳过输入节点，找到第一个计算节点
    const Node* compute_node = nullptr;
    for (const auto& node : nodes) {
        // 输入节点是 ConstNode 占位符且有 inputs_ 标记
        bool is_input = false;
        for (size_t in_id : graph.inputs()) {
            if (node.id == in_id) { is_input = true; break; }
        }
        if (!is_input && !node.inputs.empty()) {
            compute_node = &node;
            break;
        }
    }

    if (!compute_node) {
        throw std::runtime_error("HandwrittenKernelGen: no compute node found in graph");
    }

    const NodeVariant& op = compute_node->op;
    std::string src;
    GeneratedKernel result;

    if (std::holds_alternative<AddNode>(op)) {
        src = generateAddKernel();
    } else if (std::holds_alternative<MulNode>(op)) {
        src = generateMulKernel();
    } else if (std::holds_alternative<MatMulNode>(op)) {
        src = generateMatMulKernel();
        result.is_matmul = true;
        const auto& mm = std::get<MatMulNode>(op);
        // MatMul: lhs (M, K) @ rhs (K, N) -> (M, N)
        const auto& lhs = mm.lhs_desc.shape;
        const auto& rhs = mm.rhs_desc.shape;
        if (lhs.size() == 2 && rhs.size() == 2) {
            result.M = lhs[0];
            result.K = lhs[1];
            result.N = rhs[1];
        }
    } else {
        throw std::runtime_error(
            std::string("HandwrittenKernelGen: unsupported op type: ") +
            std::to_string(op.index()));
    }

    auto [func, handle] = compileAndLoad(src, "c3_kernel");
    result.func = func;
    result.dl_handle = handle;

    return result;
}

} // namespace jit
} // namespace ct