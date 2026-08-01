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
#include <stdexcept>
#include <string>
extern "C" void c3_kernel(const float* a, const float* b, float* out,
                          size_t n, size_t, size_t, size_t) {
    for (size_t i = 0; i < n; ++i) {
        if (b[i] == 0.0f) {
            throw std::runtime_error(
                "C3 DivNode: division by zero at index " + std::to_string(i));
        }
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
       << "#include <stdexcept>\n"
       << "#include <string>\n"
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
                    ss << "        if (" << extPtr(ext_inputs[0]) << " == 0.0f) "
                       << "throw std::runtime_error(\"C3 Fused DivNode: division by zero at index \" + std::to_string(i));\n";
                    ss << prefix << result << " = " << prev
                       << " / " << extPtr(ext_inputs[0]) << ";\n";
                } else {
                    ss << "        if (" << extPtr(ext_inputs[1]) << " == 0.0f) "
                       << "throw std::runtime_error(\"C3 Fused DivNode: division by zero at index \" + std::to_string(i));\n";
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

/// 判断节点是否为计算节点（非输入、非 Const）
static bool isComputeNode(const Node& node, const std::vector<size_t>& input_ids) {
    for (size_t in_id : input_ids) {
        if (node.id == in_id) return false;
    }
    if (node.inputs.empty()) return false;
    if (std::holds_alternative<ConstNode>(node.op)) return false;
    return true;
}

/// 为多节点图生成 kernel 源码
/// @details 收集所有计算节点，按拓扑顺序生成代码，中间节点输出到临时缓冲区，
///          最后一个节点输出到 output。支持 MatMul + 逐元素操作的混合图。
static std::string generateMultiNodeKernel(const Graph& graph) {
    const auto& nodes = graph.nodes();
    const auto& inputs = graph.inputs();
    const auto& outputs = graph.outputs();

    // 步骤 1: 收集计算节点（拓扑顺序）
    std::vector<const Node*> compute_nodes;
    for (const auto& node : nodes) {
        if (isComputeNode(node, inputs)) {
            compute_nodes.push_back(&node);
        }
    }

    if (compute_nodes.empty()) {
        throw std::runtime_error("HandwrittenKernelGen: no compute node in multi-node graph");
    }

    // 步骤 2: 构建 node_id → 外部输入索引 的映射
    std::unordered_map<size_t, size_t> external_input_map;
    for (size_t i = 0; i < inputs.size(); ++i) {
        external_input_map[inputs[i]] = i;
    }

    // 步骤 3: 为每个计算节点分配缓冲区
    // buffer_index[node_id] = 0,1,2,... 表示中间缓冲区索引
    // SIZE_MAX 表示直接写入 output（最终输出节点）
    std::unordered_map<size_t, size_t> node_to_buffer;
    size_t num_intermediates = 0;
    for (size_t i = 0; i < compute_nodes.size(); ++i) {
        size_t node_id = compute_nodes[i]->id;
        bool is_output = false;
        for (size_t out_id : outputs) {
            if (node_id == out_id) { is_output = true; break; }
        }
        if (i == compute_nodes.size() - 1) is_output = true; // 最后一个节点也作为输出

        if (!is_output) {
            node_to_buffer[node_id] = num_intermediates++;
        } else {
            node_to_buffer[node_id] = SIZE_MAX;
        }
    }

    // 步骤 4: 确定维度参数
    // 从图中提取 MatMul 的 M, K, N 和 elem_n
    size_t M = 0, K = 0, N = 0, elem_n = 0;
    for (const auto* node : compute_nodes) {
        if (std::holds_alternative<MatMulNode>(node->op)) {
            const auto& mm = std::get<MatMulNode>(node->op);
            if (mm.lhs_desc.shape.size() == 2 && mm.rhs_desc.shape.size() == 2) {
                M = mm.lhs_desc.shape[0];
                K = mm.lhs_desc.shape[1];
                N = mm.rhs_desc.shape[1];
            }
        }
        elem_n = std::max(elem_n, node->out_desc.numel);
    }

    // 步骤 5: 生成源码
    std::ostringstream ss;
    ss << "#include <cstddef>\n"
       << "#include <cstring>\n"
       << "#include <algorithm>\n"
       << "#include <stdexcept>\n"
       << "#include <string>\n"
       << "extern \"C\" void c3_kernel(const float* const* inputs, float* output,\n"
       << "                          size_t n, size_t M, size_t K, size_t N) {\n";

    // 分配中间缓冲区
    for (size_t i = 0; i < num_intermediates; ++i) {
        ss << "    float* tmp" << i << " = new float[n];\n";
    }

    // 辅助函数：获取节点输入的数据指针名
    auto inputPtrName = [&](size_t in_node_id) -> std::string {
        auto ext_it = external_input_map.find(in_node_id);
        if (ext_it != external_input_map.end()) {
            return "inputs[" + std::to_string(ext_it->second) + "]";
        }
        auto buf_it = node_to_buffer.find(in_node_id);
        if (buf_it != node_to_buffer.end()) {
            if (buf_it->second == SIZE_MAX) {
                return "output";
            }
            return "tmp" + std::to_string(buf_it->second);
        }
        // fallback: 可能是未分配缓冲区的输入节点
        return "inputs[" + std::to_string(in_node_id) + "]";
    };

    // 生成每个计算节点的代码
    for (size_t ci = 0; ci < compute_nodes.size(); ++ci) {
        const Node* node = compute_nodes[ci];
        bool is_last = (ci == compute_nodes.size() - 1);
        std::string out_ptr = is_last ? "output" : ("tmp" + std::to_string(node_to_buffer.at(node->id)));
        const NodeVariant& op = node->op;

        // 跳过 FusedNode（已在 fuse 阶段处理）
        if (std::holds_alternative<FusedNode>(op)) {
            continue;
        }

        // 获取输入指针名
        std::vector<std::string> in_ptrs;
        for (size_t in_id : node->inputs) {
            in_ptrs.push_back(inputPtrName(in_id));
        }

        ss << "    // " << std::visit([](auto&& n) { return n.name; }, op) << "\n";

        // MatMulNode
        if (std::holds_alternative<MatMulNode>(op)) {
            ss << "    {\n"
               << "        const size_t TILE_M = 64, TILE_N = 64, TILE_K = 64;\n"
               << "        for (size_t i = 0; i < M; ++i)\n"
               << "            for (size_t j = 0; j < N; ++j)\n"
               << "                " << out_ptr << "[i * N + j] = 0.0f;\n"
               << "        for (size_t i0 = 0; i0 < M; i0 += TILE_M) {\n"
               << "            size_t i_end = (i0 + TILE_M < M) ? i0 + TILE_M : M;\n"
               << "            for (size_t j0 = 0; j0 < N; j0 += TILE_N) {\n"
               << "                size_t j_end = (j0 + TILE_N < N) ? j0 + TILE_N : N;\n"
               << "                for (size_t k0 = 0; k0 < K; k0 += TILE_K) {\n"
               << "                    size_t k_end = (k0 + TILE_K < K) ? k0 + TILE_K : K;\n"
               << "                    for (size_t i = i0; i < i_end; ++i) {\n"
               << "                        const float* __restrict a_row = " << in_ptrs[0] << " + i * K;\n"
               << "                        float* __restrict out_row = " << out_ptr << " + i * N;\n"
               << "                        for (size_t j = j0; j < j_end; ++j) {\n"
               << "                            float sum = out_row[j];\n"
               << "                            const float* __restrict b_col = " << in_ptrs[1] << " + j;\n"
               << "                            #pragma clang loop vectorize(enable)\n"
               << "                            #pragma clang loop unroll(enable)\n"
               << "                            for (size_t k = k0; k < k_end; ++k) {\n"
               << "                                sum += a_row[k] * b_col[k * N];\n"
               << "                            }\n"
               << "                            out_row[j] = sum;\n"
               << "                        }\n"
               << "                    }\n"
               << "                }\n"
               << "            }\n"
               << "        }\n"
               << "    }\n";
        }
        // AddNode
        else if (std::holds_alternative<AddNode>(op)) {
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < n; ++i) {\n"
               << "        " << out_ptr << "[i] = " << in_ptrs[0] << "[i] + " << in_ptrs[1] << "[i];\n"
               << "    }\n";
        }
        // SubNode
        else if (std::holds_alternative<SubNode>(op)) {
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < n; ++i) {\n"
               << "        " << out_ptr << "[i] = " << in_ptrs[0] << "[i] - " << in_ptrs[1] << "[i];\n"
               << "    }\n";
        }
        // MulNode
        else if (std::holds_alternative<MulNode>(op)) {
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < n; ++i) {\n"
               << "        " << out_ptr << "[i] = " << in_ptrs[0] << "[i] * " << in_ptrs[1] << "[i];\n"
               << "    }\n";
        }
        // DivNode
        else if (std::holds_alternative<DivNode>(op)) {
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < n; ++i) {\n"
               << "        if (" << in_ptrs[1] << "[i] == 0.0f) throw std::runtime_error(\"C3 MultiNode Div: division by zero at index \" + std::to_string(i));\n"
               << "        " << out_ptr << "[i] = " << in_ptrs[0] << "[i] / " << in_ptrs[1] << "[i];\n"
               << "    }\n";
        }
        // NegNode
        else if (std::holds_alternative<NegNode>(op)) {
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < n; ++i) {\n"
               << "        " << out_ptr << "[i] = -" << in_ptrs[0] << "[i];\n"
               << "    }\n";
        }
        // ReLUNode
        else if (std::holds_alternative<ReLUNode>(op)) {
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < n; ++i) {\n"
               << "        " << out_ptr << "[i] = " << in_ptrs[0] << "[i] > 0.0f ? " << in_ptrs[0] << "[i] : 0.0f;\n"
               << "    }\n";
        }
        else {
            ss << "    // unsupported op type " << op.index() << "\n";
        }
    }

    // 释放中间缓冲区
    for (size_t i = 0; i < num_intermediates; ++i) {
        ss << "    delete[] tmp" << i << ";\n";
    }

    ss << "}\n";
    return ss.str();
}

static size_t countComputeNodes(const Graph& graph) {
    size_t count = 0;
    for (const auto& node : graph.nodes()) {
        if (isComputeNode(node, graph.inputs())) {
            count++;
        }
    }
    return count;
}

GeneratedKernel generateFromGraph(const Graph& graph) {
    const auto& nodes = graph.nodes();
    if (nodes.size() < 2) {
        throw std::runtime_error("HandwrittenKernelGen: graph has too few nodes");
    }

    size_t num_compute = countComputeNodes(graph);

    // 多节点图：使用新的多节点 kernel 生成
    if (num_compute > 1) {
        std::string src = generateMultiNodeKernel(graph);
        GeneratedKernel result;
        result.is_multi_node = true;
        result.num_inputs = graph.inputCount();

        // 提取维度信息
        for (const auto& node : nodes) {
            if (isComputeNode(node, graph.inputs())) {
                if (std::holds_alternative<MatMulNode>(node.op)) {
                    const auto& mm = std::get<MatMulNode>(node.op);
                    if (mm.lhs_desc.shape.size() == 2 && mm.rhs_desc.shape.size() == 2) {
                        result.M = mm.lhs_desc.shape[0];
                        result.K = mm.lhs_desc.shape[1];
                        result.N = mm.rhs_desc.shape[1];
                    }
                }
                result.elem_n = std::max(result.elem_n, node.out_desc.numel);
            }
        }

        auto [func_ptr, dl_handle] = compileAndLoad(src, "c3_kernel");
        result.multi_func = reinterpret_cast<MultiNodeKernelFunc>(func_ptr);
        result.handle = dl_handle;
        result.deleter = [dl_handle]() { if (dl_handle) dlclose(dl_handle); };
        return result;
    }

    // 单节点图：使用原有逻辑
    const Node* compute_node = nullptr;
    for (const auto& node : nodes) {
        if (isComputeNode(node, graph.inputs())) {
            compute_node = &node;
            break;
        }
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