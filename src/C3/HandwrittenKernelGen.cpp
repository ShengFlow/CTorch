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

static std::string generateSigmoidKernel() {
    return R"SRC(
#include <cstddef>
#include <cmath>
extern "C" void c3_kernel(const float* a, const float*, float* out,
                          size_t n, size_t, size_t, size_t) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = 1.0f / (1.0f + expf(-a[i]));
    }
}
)SRC";
}

static std::string generateTanhKernel() {
    return R"SRC(
#include <cstddef>
#include <cmath>
extern "C" void c3_kernel(const float* a, const float*, float* out,
                          size_t n, size_t, size_t, size_t) {
    for (size_t i = 0; i < n; ++i) {
        float x = a[i];
        out[i] = (expf(x) - expf(-x)) / (expf(x) + expf(-x));
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
            } else if constexpr (std::is_same_v<T, SigmoidNode>) {
                // Sigmoid: 1.0f / (1.0f + expf(-x))
                if (op_idx > 0) {
                    ss << prefix << result << " = 1.0f / (1.0f + expf(-" << prev << "));\n";
                } else {
                    ss << prefix << result << " = 1.0f / (1.0f + expf(-" << extPtr(ext_inputs[0]) << "));\n";
                }
            } else if constexpr (std::is_same_v<T, TanhNode>) {
                // Tanh: (expf(x) - expf(-x)) / (expf(x) + expf(-x))
                if (op_idx > 0) {
                    ss << prefix << result << " = (expf(" << prev << ") - expf(-" << prev << ")) / (expf(" << prev << ") + expf(-" << prev << "));\n";
                } else {
                    ss << prefix << result << " = (expf(" << extPtr(ext_inputs[0]) << ") - expf(-" << extPtr(ext_inputs[0]) << ")) / (expf(" << extPtr(ext_inputs[0]) << ") + expf(-" << extPtr(ext_inputs[0]) << "));\n";
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
    // 委托 Accelerate BLAS (cblas_sgemm) 获得 AMX 指令集加速
    return R"SRC(
#include <cstddef>
#include <Accelerate/Accelerate.h>
extern "C" void c3_kernel(const float* a, const float* b, float* out,
                          size_t, size_t M, size_t K, size_t N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K,
                1.0f, a, (int)K,
                b, (int)N,
                0.0f, out, (int)N);
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
        << "-o " << so_path << " " << src_path
        << " -framework Accelerate 2>&1";

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

    // 步骤 3: 为每个计算节点分配缓冲区，记录每个缓冲区的 numel
    // buffer_index[node_id] = 0,1,2,... 表示中间缓冲区索引
    // SIZE_MAX 表示直接写入 output（最终输出节点）
    std::unordered_map<size_t, size_t> node_to_buffer;
    std::vector<size_t> buffer_numels; // buffer index → numel
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
            buffer_numels.push_back(compute_nodes[i]->out_desc.numel);
        } else {
            node_to_buffer[node_id] = SIZE_MAX;
        }
    }

    // 步骤 3a: Buffer 原地复用分析（与 MLIR 后端相同逻辑）
    std::unordered_map<size_t, size_t> node_buffer_reuse;
    for (size_t ci = 0; ci + 1 < compute_nodes.size(); ++ci) {
        const Node* cur = compute_nodes[ci];
        const Node* next = compute_nodes[ci + 1];
        size_t cur_id = cur->id;
        size_t next_id = next->id;

        auto cur_buf = node_to_buffer.find(cur_id);
        if (cur_buf == node_to_buffer.end() || cur_buf->second == SIZE_MAX) continue;
        if (std::holds_alternative<FusedNode>(cur->op)) continue;
        if (!std::holds_alternative<FusedNode>(next->op)) continue;

        bool consumes_cur = false;
        for (size_t in_id : next->inputs) {
            if (in_id == cur_id) { consumes_cur = true; break; }
        }
        if (!consumes_cur) continue;
        if (next->out_desc.numel != cur->out_desc.numel) continue;

        node_buffer_reuse[next_id] = cur_buf->second;
    }

    // 步骤 4: 确定 elem_n（所有节点输出 numel 的最大值，用于 MultiNodeKernel 的 elem_n 参数）
    size_t elem_n = 0;
    for (const auto* node : compute_nodes) {
        elem_n = std::max(elem_n, node->out_desc.numel);
    }

    // 步骤 5: 生成源码
    std::ostringstream ss;
    ss << "#include <cstddef>\n"
       << "#include <cstring>\n"
       << "#include <algorithm>\n"
       << "#include <stdexcept>\n"
       << "#include <string>\n"
       << "#include <Accelerate/Accelerate.h>\n"
       << "extern \"C\" void c3_kernel(const float* const* inputs, float* output,\n"
       << "                          size_t n, size_t M, size_t K, size_t N) {\n";

    // 分配中间缓冲区（每个按节点实际输出 numel 分配）
    for (size_t i = 0; i < num_intermediates; ++i) {
        ss << "    float* tmp" << i << " = new float[" << buffer_numels[i] << "];\n";
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
        // 确定输出 buffer：优先使用原地复用的 buffer
        std::string out_ptr;
        if (is_last) {
            out_ptr = "output";
        } else {
            auto reuse_it = node_buffer_reuse.find(node->id);
            if (reuse_it != node_buffer_reuse.end()) {
                out_ptr = "tmp" + std::to_string(reuse_it->second);
            } else {
                out_ptr = "tmp" + std::to_string(node_to_buffer.at(node->id));
            }
        }
        const NodeVariant& op = node->op;

        // FusedNode — 生成融合循环
        if (std::holds_alternative<FusedNode>(op)) {
            const auto& fnode = std::get<FusedNode>(op);
            int64_t node_n = (int64_t)node->out_desc.numel;

            // 构建 arg_node_id → numel 映射（用于广播）
            std::unordered_map<size_t, int64_t> arg_numels;
            for (size_t aidx = 0; aidx < fnode.arg_node_ids.size(); ++aidx) {
                size_t nid = fnode.arg_node_ids[aidx];
                for (const auto& gn : nodes) {
                    if (gn.id == nid) {
                        arg_numels[nid] = (int64_t)gn.out_desc.numel;
                        break;
                    }
                }
            }

            // 辅助 lambda：生成输入访问表达式（支持广播）
            auto loadExpr = [&](size_t node_id, const std::string& idx_var) -> std::string {
                auto it = arg_numels.find(node_id);
                if (it != arg_numels.end() && it->second > 0 && (size_t)it->second < (size_t)node_n) {
                    // 需要广播：用 modulo 索引
                    return inputPtrName(node_id) + "[" + idx_var + " % " + std::to_string(it->second) + "]";
                }
                return inputPtrName(node_id) + "[" + idx_var + "]";
            };

            ss << "    // Fused (" << fnode.ops.size() << " ops)\n";
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < " << node_n << "; ++i) {\n";

            for (size_t op_idx = 0; op_idx < fnode.ops.size(); ++op_idx) {
                const NodeVariant& fop = fnode.ops[op_idx];
                const auto& inputs_for_op = fnode.op_inputs[op_idx];
                bool is_last_op = (op_idx == fnode.ops.size() - 1);

                // 获取外部输入节点 ID（排除 chain 内部连接）
                std::vector<size_t> ext_inputs;
                for (size_t in_id : inputs_for_op) {
                    if (op_idx > 0 && in_id == inputs_for_op[0]) continue;
                    ext_inputs.push_back(in_id);
                }

                std::visit([&](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;
                    std::string lhs, rhs, result_var;

                    if (op_idx > 0) {
                        result_var = "val";
                    } else {
                        result_var = "float val";
                    }

                    if constexpr (std::is_same_v<T, NegNode>) {
                        lhs = (op_idx > 0) ? "val" : loadExpr(ext_inputs[0], "i");
                        ss << "        " << result_var << " = -" << lhs << ";\n";
                    } else if constexpr (std::is_same_v<T, ReLUNode>) {
                        lhs = (op_idx > 0) ? "val" : loadExpr(ext_inputs[0], "i");
                        ss << "        " << result_var << " = " << lhs << " > 0.0f ? " << lhs << " : 0.0f;\n";
                    } else if constexpr (std::is_same_v<T, SigmoidNode>) {
                        lhs = (op_idx > 0) ? "val" : loadExpr(ext_inputs[0], "i");
                        ss << "        " << result_var << " = 1.0f / (1.0f + expf(-" << lhs << "));\n";
                    } else if constexpr (std::is_same_v<T, TanhNode>) {
                        lhs = (op_idx > 0) ? "val" : loadExpr(ext_inputs[0], "i");
                        ss << "        float x" << op_idx << " = " << lhs << ";\n";
                        ss << "        " << result_var << " = (expf(x" << op_idx << ") - expf(-x" << op_idx << ")) / (expf(x" << op_idx << ") + expf(-x" << op_idx << "));\n";
                    } else if constexpr (std::is_same_v<T, AddNode> || std::is_same_v<T, SubNode> ||
                                       std::is_same_v<T, MulNode> || std::is_same_v<T, DivNode>) {
                        if (op_idx > 0) {
                            lhs = "val";
                            rhs = loadExpr(ext_inputs[0], "i");
                        } else {
                            lhs = loadExpr(ext_inputs[0], "i");
                            rhs = loadExpr(ext_inputs[1], "i");
                        }
                        const char* op_str = "";
                        if constexpr (std::is_same_v<T, AddNode>) op_str = "+";
                        else if constexpr (std::is_same_v<T, SubNode>) op_str = "-";
                        else if constexpr (std::is_same_v<T, MulNode>) op_str = "*";
                        else if constexpr (std::is_same_v<T, DivNode>) op_str = "/";
                        ss << "        " << result_var << " = " << lhs << " " << op_str << " " << rhs << ";\n";
                    }
                }, fop);

                if (is_last_op) {
                    ss << "        " << out_ptr << "[i] = val;\n";
                }
            }
            ss << "    }\n";
            continue;
        }

        // 获取输入指针名
        std::vector<std::string> in_ptrs;
        for (size_t in_id : node->inputs) {
            in_ptrs.push_back(inputPtrName(in_id));
        }

        ss << "    // " << std::visit([](auto&& n) { return n.name; }, op) << "\n";

        // 计算广播取模（用于 bias 等向量广播到矩阵的场景）
        auto getBroadcastMod = [&](const NodeVariant& v) -> int64_t {
            auto getRhsShape = [](const NodeVariant& var) -> std::vector<size_t> {
                if (std::holds_alternative<AddNode>(var)) return std::get<AddNode>(var).rhs_desc.shape;
                if (std::holds_alternative<SubNode>(var)) return std::get<SubNode>(var).rhs_desc.shape;
                if (std::holds_alternative<MulNode>(var)) return std::get<MulNode>(var).rhs_desc.shape;
                if (std::holds_alternative<DivNode>(var)) return std::get<DivNode>(var).rhs_desc.shape;
                return {};
            };
            auto getLhsShape = [](const NodeVariant& var) -> std::vector<size_t> {
                if (std::holds_alternative<AddNode>(var)) return std::get<AddNode>(var).lhs_desc.shape;
                if (std::holds_alternative<SubNode>(var)) return std::get<SubNode>(var).lhs_desc.shape;
                if (std::holds_alternative<MulNode>(var)) return std::get<MulNode>(var).lhs_desc.shape;
                if (std::holds_alternative<DivNode>(var)) return std::get<DivNode>(var).lhs_desc.shape;
                return {};
            };
            auto lhs = getLhsShape(v);
            auto rhs = getRhsShape(v);
            if (lhs.empty() || rhs.empty() || lhs == rhs) return 0;
            size_t rhs_numel = 1;
            for (size_t d : rhs) rhs_numel *= d;
            if (rhs_numel == 1) return 1;
            if (rhs.size() == 1 && !lhs.empty() && lhs.back() == rhs[0]) {
                return (int64_t)rhs[0];
            }
            return 0;
        };

        // MatMulNode — 委托 Accelerate BLAS（每个 MatMul 硬编码自己的 M, K, N）
        if (std::holds_alternative<MatMulNode>(op)) {
            const auto& mm = std::get<MatMulNode>(op);
            int64_t matM = (int64_t)mm.lhs_desc.shape[0];
            int64_t matK = (int64_t)mm.lhs_desc.shape[1];
            int64_t matN = (int64_t)mm.rhs_desc.shape[1];
            ss << "    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,\n"
               << "                " << matM << ", " << matN << ", " << matK << ",\n"
               << "                1.0f, " << in_ptrs[0] << ", " << matK << ",\n"
               << "                " << in_ptrs[1] << ", " << matN << ",\n"
               << "                0.0f, " << out_ptr << ", " << matN << ");\n";
        }
        // AddNode
        else if (std::holds_alternative<AddNode>(op)) {
            int64_t bmod = getBroadcastMod(op);
            int64_t node_n = (int64_t)node->out_desc.numel;
            std::string rhs_idx = (bmod > 0) ? ("[i % " + std::to_string(bmod) + "]") : "[i]";
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < " << node_n << "; ++i) {\n"
               << "        " << out_ptr << "[i] = " << in_ptrs[0] << "[i] + " << in_ptrs[1] << rhs_idx << ";\n"
               << "    }\n";
        }
        // SubNode
        else if (std::holds_alternative<SubNode>(op)) {
            int64_t bmod = getBroadcastMod(op);
            int64_t node_n = (int64_t)node->out_desc.numel;
            std::string rhs_idx = (bmod > 0) ? ("[i % " + std::to_string(bmod) + "]") : "[i]";
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < " << node_n << "; ++i) {\n"
               << "        " << out_ptr << "[i] = " << in_ptrs[0] << "[i] - " << in_ptrs[1] << rhs_idx << ";\n"
               << "    }\n";
        }
        // MulNode
        else if (std::holds_alternative<MulNode>(op)) {
            int64_t bmod = getBroadcastMod(op);
            int64_t node_n = (int64_t)node->out_desc.numel;
            std::string rhs_idx = (bmod > 0) ? ("[i % " + std::to_string(bmod) + "]") : "[i]";
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < " << node_n << "; ++i) {\n"
               << "        " << out_ptr << "[i] = " << in_ptrs[0] << "[i] * " << in_ptrs[1] << rhs_idx << ";\n"
               << "    }\n";
        }
        // DivNode
        else if (std::holds_alternative<DivNode>(op)) {
            int64_t bmod = getBroadcastMod(op);
            int64_t node_n = (int64_t)node->out_desc.numel;
            std::string rhs_idx = (bmod > 0) ? ("[i % " + std::to_string(bmod) + "]") : "[i]";
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < " << node_n << "; ++i) {\n"
               << "        if (" << in_ptrs[1] << rhs_idx << " == 0.0f) throw std::runtime_error(\"C3 MultiNode Div: division by zero at index \" + std::to_string(i));\n"
               << "        " << out_ptr << "[i] = " << in_ptrs[0] << "[i] / " << in_ptrs[1] << rhs_idx << ";\n"
               << "    }\n";
        }
        // NegNode
        else if (std::holds_alternative<NegNode>(op)) {
            int64_t node_n = (int64_t)node->out_desc.numel;
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < " << node_n << "; ++i) {\n"
               << "        " << out_ptr << "[i] = -" << in_ptrs[0] << "[i];\n"
               << "    }\n";
        }
        // ReLUNode
        else if (std::holds_alternative<ReLUNode>(op)) {
            int64_t node_n = (int64_t)node->out_desc.numel;
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < " << node_n << "; ++i) {\n"
               << "        " << out_ptr << "[i] = " << in_ptrs[0] << "[i] > 0.0f ? " << in_ptrs[0] << "[i] : 0.0f;\n"
               << "    }\n";
        }
        // SigmoidNode
        else if (std::holds_alternative<SigmoidNode>(op)) {
            int64_t node_n = (int64_t)node->out_desc.numel;
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < " << node_n << "; ++i) {\n"
               << "        " << out_ptr << "[i] = 1.0f / (1.0f + expf(-" << in_ptrs[0] << "[i]));\n"
               << "    }\n";
        }
        // TanhNode
        else if (std::holds_alternative<TanhNode>(op)) {
            int64_t node_n = (int64_t)node->out_desc.numel;
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < " << node_n << "; ++i) {\n"
               << "        float x = " << in_ptrs[0] << "[i];\n"
               << "        " << out_ptr << "[i] = (expf(x) - expf(-x)) / (expf(x) + expf(-x));\n"
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
    } else if (std::holds_alternative<SigmoidNode>(op)) {
        src = generateSigmoidKernel();
    } else if (std::holds_alternative<TanhNode>(op)) {
        src = generateTanhKernel();
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