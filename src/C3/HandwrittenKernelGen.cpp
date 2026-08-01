/**
 * @file HandwrittenKernelGen.cpp
 * @brief C3 JIT 手写 kernel 生成器（EXP-1 阶段）
 * @details 为简单图（Add/Mul/MatMul）生成 C++ kernel 源码，通过系统 clang++ 编译为
 *          .so 动态库，dlsym 加载。这是 MLIR+LLVM 后端就绪前的临时方案，用于验证
 *          Graph IR → 编译 → 执行 的最小闭环。
 * @date 2026/7/31
 */

#include "HandwrittenKernelGen.h"
#include "../../include/C3/Graph.h"

#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#ifdef __APPLE__
#include <unistd.h>
#endif

namespace ct {
namespace c3 {

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

/// 为 Neg 操作生成 kernel 源码
static std::string generateNegKernel() {
    return R"SRC(
#include <cstddef>
extern "C" void c3_kernel(const float* a, const float*, float* out,
                          size_t n, size_t, size_t, size_t) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = -a[i];
    }
}
)SRC";
}

/// 为 ReLU 操作生成 kernel 源码
static std::string generateReLUKernel() {
    return R"SRC(
#include <cstddef>
extern "C" void c3_kernel(const float* a, const float*, float* out,
                          size_t n, size_t, size_t, size_t) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] > 0.0f ? a[i] : 0.0f;
    }
}
)SRC";
}

/// 为 Sub 操作生成 kernel 源码
static std::string generateSubKernel() {
    return R"SRC(
#include <cstddef>
extern "C" void c3_kernel(const float* a, const float* b, float* out,
                          size_t n, size_t, size_t, size_t) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] - b[i];
    }
}
)SRC";
}

/// 为 Div 操作生成 kernel 源码
static std::string generateDivKernel() {
    return R"SRC(
#include <cstddef>
extern "C" void c3_kernel(const float* a, const float* b, float* out,
                          size_t n, size_t, size_t, size_t) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] / b[i];
    }
}
)SRC";
}

/// 为融合节点生成 kernel 源码
/// @param ops 融合链中的操作序列（按执行顺序）
/// @param op_inputs 每个 op 的原始输入节点 ID
/// @param arg_node_ids 外部输入对应的原始节点 ID（按顺序）
/// @return C++ 源码字符串
static std::string generateFusedKernel(const std::vector<NodeVariant>& ops,
                                       const std::vector<std::vector<size_t>>& op_inputs,
                                       const std::vector<size_t>& arg_node_ids) {
    // 构建 node_id → arg_index 的映射
    std::unordered_map<size_t, size_t> node_to_arg;
    for (size_t i = 0; i < arg_node_ids.size(); ++i) {
        node_to_arg[arg_node_ids[i]] = i;
    }

    // 收集所有被引用的外部输入 node_id，生成预加载指针声明
    std::set<size_t> referenced_nodes;
    for (size_t op_idx = 0; op_idx < ops.size(); ++op_idx) {
        const auto& inputs_for_op = op_inputs[op_idx];
        for (size_t in_id : inputs_for_op) {
            if (op_idx > 0 && in_id == op_inputs[op_idx][0]) continue;
            referenced_nodes.insert(in_id);
        }
    }

    std::ostringstream ss;
    ss << "#include <cstddef>\n"
       << "extern \"C\" void c3_kernel(const float* const* inputs, float* out, size_t n) {\n";

    // 循环外：预加载所有外部输入指针
    for (size_t node_id : referenced_nodes) {
        ss << "    const float* p" << node_id << " = inputs[" << node_to_arg.at(node_id) << "];\n";
    }

    ss << "    #pragma clang loop vectorize(enable)\n"
       << "    for (size_t i = 0; i < n; ++i) {\n";

    std::string prev = "";  // 上一个操作结果的变量名

    for (size_t op_idx = 0; op_idx < ops.size(); ++op_idx) {
        const NodeVariant& op = ops[op_idx];
        const auto& inputs_for_op = op_inputs[op_idx];
        bool is_last = (op_idx == ops.size() - 1);
        std::string result = is_last ? "out[i]" : ("t" + std::to_string(op_idx));

        // 获取外部输入：inputs_for_op 中不在 chain 内部的节点
        std::vector<size_t> ext_inputs;
        for (size_t in_id : inputs_for_op) {
            // 判断是否为外部输入：不在 chain 的节点 ID 中
            // 简化：对于链式融合，第一个 op 的输入都是外部，后续 op 的第一个输入是 chain 内部
            bool is_chain_internal = false;
            if (op_idx > 0) {
                // 对于后续 op，第一个非 Const 输入是 chain 内部（上一个 op 的结果）
                size_t internal_count = 0;
                for (size_t cid : inputs_for_op) {
                    if (internal_count == 0) {
                        // 假设第一个输入是 chain 内部
                        is_chain_internal = (cid == in_id);
                        break;
                    }
                }
            }
            if (!is_chain_internal) {
                ext_inputs.push_back(in_id);
            }
        }

        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            std::string prefix = is_last ? "        " : "        float ";

            // 辅助：生成预加载指针引用 p<node_id>[i]
            auto extPtr = [&](size_t node_id) -> std::string {
                return "p" + std::to_string(node_id) + "[i]";
            };

            if constexpr (std::is_same_v<T, NegNode>) {
                if (op_idx > 0) {
                    ss << prefix << result << " = -" << prev << ";\n";
                } else {
                    ss << prefix << result << " = -" << extPtr(ext_inputs[0]) << ";\n";
                }
            } else if constexpr (std::is_same_v<T, ReLUNode>) {
                if (op_idx > 0) {
                    ss << prefix << result << " = " << prev << " > 0.0f ? " << prev << " : 0.0f;\n";
                } else {
                    ss << prefix << result << " = " << extPtr(ext_inputs[0])
                       << " > 0.0f ? " << extPtr(ext_inputs[0]) << " : 0.0f;\n";
                }
            } else if constexpr (std::is_same_v<T, AddNode>) {
                if (op_idx > 0) {
                    ss << prefix << result << " = " << prev
                       << " + " << extPtr(ext_inputs[0]) << ";\n";
                } else {
                    ss << prefix << result << " = " << extPtr(ext_inputs[0])
                       << " + " << extPtr(ext_inputs[1]) << ";\n";
                }
            } else if constexpr (std::is_same_v<T, SubNode>) {
                if (op_idx > 0) {
                    ss << prefix << result << " = " << prev
                       << " - " << extPtr(ext_inputs[0]) << ";\n";
                } else {
                    ss << prefix << result << " = " << extPtr(ext_inputs[0])
                       << " - " << extPtr(ext_inputs[1]) << ";\n";
                }
            } else if constexpr (std::is_same_v<T, MulNode>) {
                if (op_idx > 0) {
                    ss << prefix << result << " = " << prev
                       << " * " << extPtr(ext_inputs[0]) << ";\n";
                } else {
                    ss << prefix << result << " = " << extPtr(ext_inputs[0])
                       << " * " << extPtr(ext_inputs[1]) << ";\n";
                }
            } else if constexpr (std::is_same_v<T, DivNode>) {
                if (op_idx > 0) {
                    ss << prefix << result << " = " << prev
                       << " / " << extPtr(ext_inputs[0]) << ";\n";
                } else {
                    ss << prefix << result << " = " << extPtr(ext_inputs[0])
                       << " / " << extPtr(ext_inputs[1]) << ";\n";
                }
            }
        }, op);

        if (!is_last) prev = "t" + std::to_string(op_idx);
    }

    ss << "    }\n"
       << "}\n";
    return ss.str();
}
static std::string generateMatMulKernel() {
    // 分块矩阵乘法：i0/j0/k0 三级分块 + 内层 k 循环展开 + 向量化
    return R"SRC(
#include <cstddef>
#include <algorithm>
extern "C" void c3_kernel(const float* a, const float* b, float* out,
                          size_t, size_t M, size_t K, size_t N) {
    const size_t TILE_M = 64;
    const size_t TILE_N = 64;
    const size_t TILE_K = 64;

    // 初始化 C 为 0
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            out[i * N + j] = 0.0f;
        }
    }

    // 分块循环：i0 → j0 → k0 → i → j → k
    for (size_t i0 = 0; i0 < M; i0 += TILE_M) {
        size_t i_end = (i0 + TILE_M < M) ? i0 + TILE_M : M;
        for (size_t j0 = 0; j0 < N; j0 += TILE_N) {
            size_t j_end = (j0 + TILE_N < N) ? j0 + TILE_N : N;
            for (size_t k0 = 0; k0 < K; k0 += TILE_K) {
                size_t k_end = (k0 + TILE_K < K) ? k0 + TILE_K : K;
                for (size_t i = i0; i < i_end; ++i) {
                    const float* __restrict a_row = a + i * K;
                    float* __restrict out_row = out + i * N;
                    for (size_t j = j0; j < j_end; ++j) {
                        float sum = out_row[j];
                        const float* __restrict b_col = b + j;
                        #pragma clang loop vectorize(enable)
                        #pragma clang loop unroll(enable)
                        for (size_t k = k0; k < k_end; ++k) {
                            sum += a_row[k] * b_col[k * N];
                        }
                        out_row[j] = sum;
                    }
                }
            }
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
    const auto& nodes = graph.nodes();
    if (nodes.size() < 2) {
        throw std::runtime_error("HandwrittenKernelGen: graph has too few nodes");
    }

    // 跳过输入节点，找到第一个计算节点
    const Node* compute_node = nullptr;
    for (const auto& node : nodes) {
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

    // 处理融合节点
    if (std::holds_alternative<FusedNode>(op)) {
        const auto& fnode = std::get<FusedNode>(op);
        src = generateFusedKernel(fnode.ops, fnode.op_inputs, fnode.arg_node_ids);
        result.is_fused = true;
        result.num_inputs = fnode.arg_descs.size();

        // 编译融合 kernel（使用 FusedKernelFunc 签名）
        auto [func_ptr, dl_handle] = compileAndLoad(src, "c3_kernel");
        result.fused_func = reinterpret_cast<FusedKernelFunc>(func_ptr);
        result.handle = dl_handle;
        result.deleter = [dl_handle]() { if (dl_handle) dlclose(dl_handle); };
        return result;
    }

    if (std::holds_alternative<AddNode>(op)) {
        src = generateAddKernel();
    } else if (std::holds_alternative<SubNode>(op)) {
        src = generateSubKernel();
    } else if (std::holds_alternative<MulNode>(op)) {
        src = generateMulKernel();
    } else if (std::holds_alternative<DivNode>(op)) {
        src = generateDivKernel();
    } else if (std::holds_alternative<NegNode>(op)) {
        src = generateNegKernel();
    } else if (std::holds_alternative<ReLUNode>(op)) {
        src = generateReLUKernel();
    } else if (std::holds_alternative<MatMulNode>(op)) {
        src = generateMatMulKernel();
        result.is_matmul = true;
        const auto& mm = std::get<MatMulNode>(op);
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

    auto [func, dl_handle] = compileAndLoad(src, "c3_kernel");
    result.func = func;
    result.handle = dl_handle;
    result.deleter = [dl_handle]() { if (dl_handle) dlclose(dl_handle); };

    return result;
}

} // namespace c3
} // namespace ct