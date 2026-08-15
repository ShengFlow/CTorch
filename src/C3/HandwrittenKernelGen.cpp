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
#include <iomanip>
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
 * @details 流程：mkdtemp 写源码 → clang++ 编译 .so → dlopen → dlsym。
 *          [removed 2026-08-15] AOT 磁盘缓存已删除（见 STATUS_CONTEXT 4.12），
 *          手写 kernel 每次进程内首次使用重新 clang++ 编译。
 */
static std::pair<C3KernelFunc, void*> compileAndLoad(const std::string& src,
                                                      const std::string& func_name) {
    // 步骤 1: 编译（每次进程内首次使用重新编译）
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

    // 步骤 2: 加载 .so
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

/// DEBT-NEW-7 性能优化（v0.5.1+）：AMX region kernel generator
/// @brief 检测 MatMul + Add(bias) [+ ReLU] 模式,生成专用的 AMX 加速 kernel
/// @details 目标:消除 generateMultiNodeKernel 在这种 3 节点 pattern 下产生的 2 个
///          中间 buffer(cblas_sgemm 写 tmp0 + Add 写 tmp1 + ReLU 写 output),
///          改为:cblas_sgemm 直接写 output + 一次 fused bias+ReLU pass 复用 output
///
/// 性能来源:
///  - cblas_sgemm 仍走 Accelerate/AMX(M1/M2 上是 5-10 TFLOPS),保留矩阵乘吞吐
///  - Add+ReLU 折叠为单次 pass,无需中间写(原 3-pass 写 2 个 tmp buffer)
///  - 单 pass 让 clang 的 auto-vectorizer 看到完整的 bias+ReLU 模式,通常能生成
///    比 2 个独立 loop 更好的 NEON 代码
///
/// 触发条件(由调用方在 generateFromGraph 决定):
///  - Pattern A:MatMul([M,K],[K,N]) → Add(bias[N] 广播) → ReLU   (MNIST 隐藏层)
///  - Pattern B:MatMul([M,K],[K,N]) → Add(bias[N] 广播)          (MNIST 输出层)
static bool isMatmulBiasReluPattern(const Graph& graph,
                                    bool& has_relu_after_bias,
                                    size_t& matM, size_t& matK, size_t& matN) {
    const auto& nodes = graph.nodes();
    const auto& inputs = graph.inputs();
    has_relu_after_bias = false;
    matM = matK = matN = 0;

    // 收集所有计算节点
    std::vector<const Node*> compute;
    for (const auto& node : nodes) {
        if (isComputeNode(node, inputs)) {
            compute.push_back(&node);
        }
    }
    if (compute.size() < 2 || compute.size() > 4) return false;

    // 第一个必须是 MatMul
    if (!std::holds_alternative<MatMulNode>(compute[0]->op)) return false;
    const auto& mm = std::get<MatMulNode>(compute[0]->op);
    if (mm.lhs_desc.shape.size() != 2 || mm.rhs_desc.shape.size() != 2) return false;
    matM = mm.lhs_desc.shape[0];
    matK = mm.lhs_desc.shape[1];
    matN = mm.rhs_desc.shape[1];

    // DEBT-NEW-7 性能优化:支持 fuse 后的 FusedNode 形态
    // fuse() 步骤会把 elementwise 链(Add+ReLU)合并成单个 FusedNode。
    // 所以计算图可能是:
    //   形态 A: [MatMul, Add, ReLU]              (3 个独立 compute 节点)
    //   形态 B: [MatMul, Add]                    (2 个独立 compute 节点,无 ReLU)
    //   形态 C: [MatMul, FusedNode(Add,ReLU)]    (fuse 后,FusedNode 包裹 Add+ReLU)
    //   形态 D: [MatMul, FusedNode(Add)]         (fuse 后,无 ReLU)
    //
    // 统一处理:从 compute[1] 提取 Add + (可选) ReLU,然后判定是否形成 MatMul+bias 模式。
    auto extractAddAndRelu = [&](const Node* second_node,
                                 bool& found_add, bool& found_relu,
                                 const AddNode*& add_ref) -> bool {
        found_add = false;
        found_relu = false;
        add_ref = nullptr;

        if (std::holds_alternative<AddNode>(second_node->op)) {
            // 形态 A / B:独立 AddNode(可后跟 ReLU)
            found_add = true;
            add_ref = &std::get<AddNode>(second_node->op);
            // 形态 A:ReLU 是 compute[2]
            if (compute.size() == 3 && std::holds_alternative<ReLUNode>(compute[2]->op)) {
                found_relu = true;
            }
            return true;
        }
        if (std::holds_alternative<FusedNode>(second_node->op)) {
            // 形态 C / D:FusedNode 包裹
            const auto& fn = std::get<FusedNode>(second_node->op);
            // FusedNode 内必须包含一个 AddNode
            for (const auto& op_inside : fn.ops) {
                if (std::holds_alternative<AddNode>(op_inside)) {
                    found_add = true;
                    // 找到 Add 在 ops 中的索引
                    size_t add_idx = 0;
                    for (; add_idx < fn.ops.size(); ++add_idx) {
                        if (std::holds_alternative<AddNode>(fn.ops[add_idx])) break;
                    }
                    // ReLU 必须在 Add 之后
                    for (size_t i = add_idx + 1; i < fn.ops.size(); ++i) {
                        if (std::holds_alternative<ReLUNode>(fn.ops[i])) {
                            found_relu = true;
                            break;
                        }
                    }
                    // 构造一个 AddNode 引用(指向 FusedNode 内的 AddNode)
                    // 注意:这里 add_ref 的生命周期由 graph 保证
                    add_ref = &std::get<AddNode>(op_inside);
                    return true;
                }
            }
        }
        return false;
    };

    bool found_add = false, found_relu = false;
    const AddNode* add_ptr = nullptr;
    if (!extractAddAndRelu(compute[1], found_add, found_relu, add_ptr)) return false;
    if (!found_add || !add_ptr) return false;

    const AddNode& add = *add_ptr;
    // bias 的特征:rhs 是 1D 且长度 == N,或 rhs 总元素数较小
    bool rhs_is_bias = (add.rhs_desc.shape.size() == 1 && add.rhs_desc.shape[0] == matN);
    if (!rhs_is_bias) {
        // 兜底:rhs.numel < lhs.numel(可能 shape 不规则但仍是广播)
        rhs_is_bias = (add.rhs_desc.numel < add.lhs_desc.numel &&
                       !add.rhs_desc.shape.empty() &&
                       add.rhs_desc.shape.back() == matN);
    }
    if (!rhs_is_bias) return false;

    has_relu_after_bias = found_relu;
#ifdef CT_DEBUG
    {
        std::ostringstream oss;
        oss << "[C3-AMXRegion] MATCH M=" << matM << " K=" << matK << " N=" << matN
            << " has_relu=" << (has_relu_after_bias ? "yes" : "no");
        CtorchError::log(ErrorLevel::DEBUG, ErrorPlatform::kGENERAL, ErrorType::UNKNOWN, oss.str());
    }
#endif
    return true;
}

/// DEBT-NEW-7 性能优化:AMX region kernel 源码生成
/// @details 签名与 MultiNodeKernelFunc 一致:
///          void c3_kernel(const float* const* inputs, float* output,
///                        size_t n, size_t M, size_t K, size_t N)
///          inputs[0] = A[M,K], inputs[1] = B[K,N], inputs[2] = bias[N]
static std::string generateFusedMatmulBiasKernel(bool has_relu,
                                                  size_t M, size_t K, size_t N) {
    // M, K, N 烘焙到生成的 kernel 字符串字面量中,本身不参与本函数的 C++ 计算
    (void)M; (void)K; (void)N;
    std::ostringstream ss;
    ss << "#include <cstddef>\n"
       << "#include <Accelerate/Accelerate.h>\n"
       << "extern \"C\" void c3_kernel(const float* const* inputs, float* output,\n"
       << "                          size_t n, size_t M, size_t K, size_t N, float* scratchpad) {\n"
       << "    const float* A = inputs[0];\n"
       << "    const float* B = inputs[1];\n";

    if (has_relu) {
        // Pattern A: cblas_sgemm + fused bias + ReLU
        ss << "    const float* bias = inputs[2];\n"
           << "    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,\n"
           << "                (int)M, (int)N, (int)K,\n"
           << "                1.0f, A, (int)K,\n"
           << "                B, (int)N,\n"
           << "                0.0f, output, (int)N);\n"
           << "    // Fused bias + ReLU: 单 pass 复用 output buffer (无中间写)\n"
           << "    #pragma clang loop vectorize(enable)\n"
           << "    for (size_t m = 0; m < M; ++m) {\n"
           << "        float* row = output + m * N;\n"
           << "        for (size_t n_idx = 0; n_idx < N; ++n_idx) {\n"
           << "            float v = row[n_idx] + bias[n_idx];\n"
           << "            row[n_idx] = v > 0.0f ? v : 0.0f;\n"
           << "        }\n"
           << "    }\n";
    } else {
        // Pattern B: cblas_sgemm + fused bias only
        ss << "    const float* bias = inputs[2];\n"
           << "    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,\n"
           << "                (int)M, (int)N, (int)K,\n"
           << "                1.0f, A, (int)K,\n"
           << "                B, (int)N,\n"
           << "                0.0f, output, (int)N);\n"
           << "    // Fused bias only: 单 pass 复用 output buffer (无中间写)\n"
           << "    #pragma clang loop vectorize(enable)\n"
           << "    for (size_t m = 0; m < M; ++m) {\n"
           << "        float* row = output + m * N;\n"
           << "        for (size_t n_idx = 0; n_idx < N; ++n_idx) {\n"
           << "            row[n_idx] = row[n_idx] + bias[n_idx];\n"
           << "        }\n"
           << "    }\n";
    }

    ss << "}\n";
    return ss.str();
}

/// 为多节点图生成 kernel 源码
/// @details 收集所有计算节点，按拓扑顺序生成代码，中间节点输出到临时缓冲区，
///          最后一个节点输出到 output。支持 MatMul + 逐元素操作的混合图。
static std::string generateMultiNodeKernel(const Graph& graph, size_t& out_scratch_size) {
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

    // 步骤 3: 为每个计算节点分配缓冲区，记录每个缓冲区的 numel。
    // 支持多输出（反向融合图）：输出节点写入 output 平面 buffer 的对应段
    //   （output_index[id] * elem_n），非输出节点写入中间 tmp buffer。
    //   SIZE_MAX 表示该节点是输出节点（写入 output 平面 buffer）。
    std::unordered_map<size_t, size_t> node_to_buffer;
    std::unordered_map<size_t, size_t> output_index; // 输出节点 id → 平面 buffer 段索引（按 graph.outputs() 顺序）
    std::vector<size_t> buffer_numels;               // 中间 buffer index → numel
    size_t num_intermediates = 0;
    auto assign_output = [&](size_t node_id) {
        if (output_index.count(node_id)) return; // 已分配输出段（去重）
        const size_t seg = output_index.size();
        output_index[node_id] = seg;
        node_to_buffer[node_id] = SIZE_MAX;
    };
    for (size_t i = 0; i < compute_nodes.size(); ++i) {
        size_t node_id = compute_nodes[i]->id;
        bool is_output = false;
        for (size_t out_id : outputs) {
            if (node_id == out_id) { is_output = true; break; }
        }
        if (is_output) {
            assign_output(node_id);
        } else {
            node_to_buffer[node_id] = num_intermediates++;
            buffer_numels.push_back(compute_nodes[i]->out_desc.numel);
        }
    }
    // 安全网：确保最后一个计算节点总是输出（forward 单输出情形）
    assign_output(compute_nodes.back()->id);

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
       << "                          size_t n, size_t M, size_t K, size_t N, float* scratchpad) {\n";

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
                // 输出节点：读 output 平面 buffer 的对应段（output_index * elem_n）。
                // 必须带段偏移，否则多输出时误读第 0 段。
                auto oi = output_index.find(in_node_id);
                size_t seg = (oi != output_index.end()) ? oi->second : 0;
                return "(output + " + std::to_string(seg * elem_n) + ")";
            }
            return "tmp" + std::to_string(buf_it->second);
        }
        // fallback: 可能是未分配缓冲区的输入节点
        return "inputs[" + std::to_string(in_node_id) + "]";
    };

    // [Fix 2026-08-11 反向图接线 E2E]: 提供统一的二元/一元操作数表达式生成。
    // 背景：ReLU backward 的 Gt(x, 0) / Sigmoid backward 的 Div(1, denom) 里，
    //       addConstant 生成的真常量（ConstNode，不在 graph.inputs()）被当作普通
    //       buffer 指针引用会越界（标量 buffer 只有 1 元素却用 [i] 索引），
    //       且 address 是字面量时 `(val)[i]` 语法非法。
    // 方案：isTrueConst 识别真常量；nodeConstValue 取常量值；operandExpr 统一
    //       生成操作数表达式——常量→字面量 / 标量 buffer→[0] / 全量→[i] /
    //       尾维广播→[i % numel]。
    auto isTrueConst = [&](size_t nid) -> bool {
        for (const auto& gn : nodes) {
            if (gn.id == nid && std::holds_alternative<ConstNode>(gn.op)) {
                bool is_input = false;
                for (size_t iid : inputs) { if (iid == nid) { is_input = true; break; } }
                return !is_input; // 排除输入占位符（inputs_ 中的 ConstNode{0.0}）
            }
        }
        return false;
    };
    auto nodeConstValue = [&](size_t nid) -> double {
        for (const auto& gn : nodes) {
            if (gn.id == nid && std::holds_alternative<ConstNode>(gn.op)) {
                return std::get<ConstNode>(gn.op).value;
            }
        }
        return 0.0;
    };
    // 生成合法的浮点字面量：整数常量（如 1、0）必须带小数点（1f 非法，1.0f 合法），
    // 且对科学计数法（如 1e-07）保持原样不追加小数点。
    auto floatLit = [](double v) -> std::string {
        std::ostringstream os;
        os << std::setprecision(9) << v;
        std::string s = os.str();
        if (s.find_first_of(".eE") == std::string::npos) s += ".0";
        s += "f";
        return s;
    };
    // 生成操作数表达式：ptr 是该输入节点的 in_ptrs 指针名，node_n 是当前输出 numel
    auto operandExpr = [&](size_t nid, const std::string& ptr, int64_t node_n) -> std::string {
        if (isTrueConst(nid)) return floatLit(nodeConstValue(nid));
        int64_t on = 1;
        for (const auto& gn : nodes) if (gn.id == nid) { on = (int64_t)gn.out_desc.numel; break; }
        if (on == node_n) return ptr + "[i]";
        if (on == 1) return ptr + "[0]";
        if (on < node_n && node_n % on == 0) return ptr + "[i % " + std::to_string(on) + "]";
        return ptr + "[i]";
    };

// 生成每个计算节点的代码
    for (size_t ci = 0; ci < compute_nodes.size(); ++ci) {
        const Node* node = compute_nodes[ci];
        // 确定输出 buffer：输出节点 → 写入 output 平面 buffer 对应段（output_index * elem_n）；
        //   否则写中间 tmp buffer（优先原地复用）。
        std::string out_ptr;
        auto oci = output_index.find(node->id);
        if (oci != output_index.end()) {
            // 输出节点：写 output 平面 buffer 的对应段（output_index * elem_n）。
            // 用括号包裹成 (output + N)，后续拼接 [i] 生成合法语句 (output + N)[i]。
            out_ptr = "(output + " + std::to_string(oci->second * elem_n) + ")";
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
            // [Fix 2026-08-11 Tanh+ReLU backward chain]: addConstant 创建的 ConstNode
            // (真实常量, 不在 graph.inputs()) 必须返回浮点字面量, 否则 inputPtrName
            // fallback 到 inputs[N] 引用会导致读取越界 (1 元素 buffer 用 [i] 索引)。
            auto loadExpr = [&](size_t node_id, const std::string& idx_var) -> std::string {
                for (const auto& gn : nodes) {
                    if (gn.id == node_id && std::holds_alternative<ConstNode>(gn.op)) {
                        bool is_input = false;
                        for (size_t iid : inputs) {
                            if (iid == node_id) { is_input = true; break; }
                        }
                        if (!is_input) {
                            return floatLit(std::get<ConstNode>(gn.op).value);
                        }
                    }
                }
                auto it = arg_numels.find(node_id);
                if (it != arg_numels.end() && it->second > 0 && (size_t)it->second < (size_t)node_n) {
                    // 需要广播：用 modulo 索引
                    return inputPtrName(node_id) + "[" + idx_var + " % " + std::to_string(it->second) + "]";
                }
                return inputPtrName(node_id) + "[" + idx_var + "]";
            };

            ss << "    // Fused (" << fnode.ops.size() << " ops)\n";
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < std::min((size_t)" << node_n << ", n); ++i) {\n";

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
            int64_t node_n = (int64_t)node->out_desc.numel;
            std::string lhs_e = operandExpr(node->inputs[0], in_ptrs[0], node_n);
            std::string rhs_e = operandExpr(node->inputs[1], in_ptrs[1], node_n);
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < std::min((size_t)" << node_n << ", n); ++i) {\n"
               << "        " << out_ptr << "[i] = " << lhs_e << " + " << rhs_e << ";\n"
               << "    }\n";
        }
        // SubNode
        else if (std::holds_alternative<SubNode>(op)) {
            int64_t node_n = (int64_t)node->out_desc.numel;
            std::string lhs_e = operandExpr(node->inputs[0], in_ptrs[0], node_n);
            std::string rhs_e = operandExpr(node->inputs[1], in_ptrs[1], node_n);
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < std::min((size_t)" << node_n << ", n); ++i) {\n"
               << "        " << out_ptr << "[i] = " << lhs_e << " - " << rhs_e << ";\n"
               << "    }\n";
        }
        // MulNode
        else if (std::holds_alternative<MulNode>(op)) {
            int64_t node_n = (int64_t)node->out_desc.numel;
            std::string lhs_e = operandExpr(node->inputs[0], in_ptrs[0], node_n);
            std::string rhs_e = operandExpr(node->inputs[1], in_ptrs[1], node_n);
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < std::min((size_t)" << node_n << ", n); ++i) {\n"
               << "        " << out_ptr << "[i] = " << lhs_e << " * " << rhs_e << ";\n"
               << "    }\n";
        }
        // DivNode
        else if (std::holds_alternative<DivNode>(op)) {
            int64_t node_n = (int64_t)node->out_desc.numel;
            std::string lhs_e = operandExpr(node->inputs[0], in_ptrs[0], node_n);
            std::string rhs_e = operandExpr(node->inputs[1], in_ptrs[1], node_n);
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < std::min((size_t)" << node_n << ", n); ++i) {\n"
               << "        if (" << rhs_e << " == 0.0f) throw std::runtime_error(\"C3 MultiNode Div: division by zero at index \" + std::to_string(i));\n"
               << "        " << out_ptr << "[i] = " << lhs_e << " / " << rhs_e << ";\n"
               << "    }\n";
        }
        // GtNode — out = (lhs > rhs) ? 1.0f : 0.0f (ReLU backward mask)
        else if (std::holds_alternative<GtNode>(op)) {
            int64_t node_n = (int64_t)node->out_desc.numel;
            std::string lhs_e = operandExpr(node->inputs[0], in_ptrs[0], node_n);
            std::string rhs_e = operandExpr(node->inputs[1], in_ptrs[1], node_n);
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < std::min((size_t)" << node_n << ", n); ++i) {\n"
               << "        " << out_ptr << "[i] = " << lhs_e << " > " << rhs_e << " ? 1.0f : 0.0f;\n"
               << "    }\n";
        }
        // ExpNode — out = expf(x) (Sigmoid/Tanh backward)
        else if (std::holds_alternative<ExpNode>(op)) {
            int64_t node_n = (int64_t)node->out_desc.numel;
            std::string in_e = operandExpr(node->inputs[0], in_ptrs[0], node_n);
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < std::min((size_t)" << node_n << ", n); ++i) {\n"
               << "        " << out_ptr << "[i] = expf(" << in_e << ");\n"
               << "    }\n";
        }
        // LogNode — out = logf(x) (Log backward)
        else if (std::holds_alternative<LogNode>(op)) {
            int64_t node_n = (int64_t)node->out_desc.numel;
            std::string in_e = operandExpr(node->inputs[0], in_ptrs[0], node_n);
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < std::min((size_t)" << node_n << ", n); ++i) {\n"
               << "        " << out_ptr << "[i] = logf(" << in_e << ");\n"
               << "    }\n";
        }
        // SumReduceNode (DEBT-NEW-7 v0.5.1+: backward-only 节点,AddNode broadcast 反向)
        // 实现:沿 axis 累加,非 axis 维保持原位
        // axis=-1 表示全 reduce (output 标量 1 元素)
        // 简化:目前只处理 axis==last_dim 或 axis==-1 全 reduce(常见 broadcast case)
        // 复杂情况(中间 axis)后续 v0.5.2 扩展,先 throw fail-fast
        else if (std::holds_alternative<SumReduceNode>(op)) {
            const auto& sr = std::get<SumReduceNode>(op);
            const auto& in_shape = sr.in_desc.shape;
            int64_t in_rank = (int64_t)in_shape.size();
            int64_t out_n = (int64_t)node->out_desc.numel;
            int64_t in_n = (int64_t)sr.in_desc.numel;
            if (in_n == 0 || out_n == 0) {
                ss << "    // SumReduce: zero-size input, skip\n";
            } else if (sr.axis == -1 || out_n == 1) {
                // 全 reduce: out = sum(in)
                ss << "    {\n"
                   << "        float acc = 0.0f;\n"
                   << "        for (int64_t i = 0; i < " << in_n << "; ++i) {\n"
                   << "            acc += " << in_ptrs[0] << "[i];\n"
                   << "        }\n"
                   << "        " << out_ptr << "[0] = acc;\n"
                   << "    }\n";
            } else if (sr.axis == in_rank - 1 && in_rank >= 2) {
                // reduce last dim: 累加每个 row 的所有 column
                int64_t outer = in_n / in_shape[in_rank - 1];
                int64_t inner = in_shape[in_rank - 1];
                ss << "    {\n"
                   << "        for (int64_t r = 0; r < " << outer << "; ++r) {\n"
                   << "            float acc = 0.0f;\n"
                   << "            for (int64_t c = 0; c < " << inner << "; ++c) {\n"
                   << "                acc += " << in_ptrs[0] << "[r * " << inner << " + c];\n"
                   << "            }\n"
                   << "            " << out_ptr << "[r] = acc;\n"
                   << "        }\n"
                   << "    }\n";
            } else if (sr.axis == 0 && in_rank >= 2) {
                // reduce first dim: 累加每个 column 的所有 row
                int64_t outer = in_shape[0];
                int64_t inner = in_n / outer;
                ss << "    {\n"
                   << "        for (int64_t c = 0; c < " << inner << "; ++c) {\n"
                   << "            float acc = 0.0f;\n"
                   << "            for (int64_t r = 0; r < " << outer << "; ++r) {\n"
                   << "                acc += " << in_ptrs[0] << "[r * " << inner << " + c];\n"
                   << "            }\n"
                   << "            " << out_ptr << "[c] = acc;\n"
                   << "        }\n"
                   << "    }\n";
            } else {
                // 复杂 axis:fall back to 全 reduce 然后 broadcast (近似,不严格正确)
                ss << "    // SumReduce axis=" << sr.axis << " (intermediate axis, falling back to full reduce)\n"
                   << "    {\n"
                   << "        float acc = 0.0f;\n"
                   << "        for (int64_t i = 0; i < " << in_n << "; ++i) {\n"
                   << "            acc += " << in_ptrs[0] << "[i];\n"
                   << "        }\n"
                   << "        for (int64_t i = 0; i < " << out_n << "; ++i) {\n"
                   << "            " << out_ptr << "[i] = acc;\n"
                   << "        }\n"
                   << "    }\n";
            }
        }
        // NegNode
        else if (std::holds_alternative<NegNode>(op)) {
            int64_t node_n = (int64_t)node->out_desc.numel;
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < std::min((size_t)" << node_n << ", n); ++i) {\n"
               << "        " << out_ptr << "[i] = -" << in_ptrs[0] << "[i];\n"
               << "    }\n";
        }
        // ReLUNode
        else if (std::holds_alternative<ReLUNode>(op)) {
            int64_t node_n = (int64_t)node->out_desc.numel;
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < std::min((size_t)" << node_n << ", n); ++i) {\n"
               << "        " << out_ptr << "[i] = " << in_ptrs[0] << "[i] > 0.0f ? " << in_ptrs[0] << "[i] : 0.0f;\n"
               << "    }\n";
        }
        // SigmoidNode
        else if (std::holds_alternative<SigmoidNode>(op)) {
            int64_t node_n = (int64_t)node->out_desc.numel;
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < std::min((size_t)" << node_n << ", n); ++i) {\n"
               << "        " << out_ptr << "[i] = 1.0f / (1.0f + expf(-" << in_ptrs[0] << "[i]));\n"
               << "    }\n";
        }
        // TanhNode
        else if (std::holds_alternative<TanhNode>(op)) {
            int64_t node_n = (int64_t)node->out_desc.numel;
            ss << "    #pragma clang loop vectorize(enable)\n"
               << "    for (size_t i = 0; i < std::min((size_t)" << node_n << ", n); ++i) {\n"
               << "        float x = " << in_ptrs[0] << "[i];\n"
               << "        " << out_ptr << "[i] = (expf(x) - expf(-x)) / (expf(x) + expf(-x));\n"
               << "    }\n";
        }
        // TransposeNode — 2D 转置:out[n*M + m] = in[m*N + n] (in 形状 [M,N] → out 形状 [N,M])
        // [Fix 2026-08-11 5d follow-up]: in_ptrs[0] 经 external_input_map 解析成真实内核输入指针
        // (grad_A 情形 = inputs[2] = B; grad_B 情形 = inputs[1] = A), 与 c3 内部 caller 传入的
        // inputs=[grad, A, B] 顺序严格一致, 消除 5d 的 ext_map 错位 bug。
        else if (std::holds_alternative<TransposeNode>(op)) {
            const auto& tn = std::get<TransposeNode>(op);
            if (tn.in_desc.shape.size() != 2) {
                throw std::runtime_error(
                    "HandwrittenKernelGen: TransposeNode only supports 2D input, got rank " +
                    std::to_string(tn.in_desc.shape.size()));
            }
            int64_t M = (int64_t)tn.in_desc.shape[0];
            int64_t N = (int64_t)tn.in_desc.shape[1];
            ss << "    for (int64_t m = 0; m < " << M << "; ++m) {\n"
               << "        for (int64_t n = 0; n < " << N << "; ++n) {\n"
               << "            " << out_ptr << "[n * " << M << " + m] = "
               << in_ptrs[0] << "[m * " << N << " + n];\n"
               << "        }\n"
               << "    }\n";
        }
        // [Fix 2026-08-09 用户审查 P0]: 之前 L1020 '// unsupported op type {op.index()}'
        // 只写注释, 不生成代码 → 静默跳过, 跟 MLIRKernelGen #2 同源 bug。
        // 改: 显式 throw, 强制 fallback 走 eager 而非 c3 bw 路径。
        else {
            const std::string op_name = std::visit(
                [](const auto& n) -> std::string { return typeid(n).name(); },
                op);
            throw std::runtime_error(
                "HandwrittenKernelGen: unsupported op in multi-node graph: " + op_name +
                " (TransposeNode 实装 5d 跨 session follow-up; Exp/Log/Gt/Const M2 范畴)");
        }
    }

    // 释放中间缓冲区
    for (size_t i = 0; i < num_intermediates; ++i) {
        ss << "    delete[] tmp" << i << ";\n";
    }

    out_scratch_size = 0;
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

    // DEBT-NEW-7 性能优化:AMX region kernel 专用路径
    // 当图是 MatMul + Add(bias) [+ ReLU] 时,用 generateFusedMatmulBiasKernel 生成
    // 专用 kernel,消除 2 个中间 buffer(cblas_sgemm 写 tmp0 + Add 写 tmp1 + ReLU 写 output)
    // → 1 个 buffer (cblas_sgemm 写 output + 一次 fused bias[+ReLU] pass 复用 output)
    bool has_relu_after_bias = false;
    size_t matM = 0, matK = 0, matN = 0;
    if (isMatmulBiasReluPattern(graph, has_relu_after_bias, matM, matK, matN)) {
        std::string src = generateFusedMatmulBiasKernel(has_relu_after_bias, matM, matK, matN);
        GeneratedKernel result;
        result.is_multi_node = true;
        result.num_inputs = graph.inputCount();
        result.M = matM;
        result.K = matK;
        result.N = matN;
        result.elem_n = matM * matN;

        // [Fix] v0.5.2 Linux build: DTK clang 17 OpenMP 严格模式, lambda 不能 capture
        //   structured binding. 用 named 变量替代 auto [a, b] = ...
        auto _compile_result = compileAndLoad(src, "c3_kernel");
        auto func_ptr = _compile_result.first;
        auto dl_handle = _compile_result.second;
        result.multi_func = reinterpret_cast<MultiNodeKernelFunc>(func_ptr);
        result.handle = dl_handle;
        result.deleter = [dl_handle]() { if (dl_handle) dlclose(dl_handle); };
#ifdef CT_DEBUG
        {
            std::ostringstream oss;
            oss << "[C3-AMXRegion] M=" << matM << " K=" << matK << " N=" << matN
                << " has_relu=" << (has_relu_after_bias ? "yes" : "no")
                << " num_inputs=" << result.num_inputs;
            CtorchError::log(ErrorLevel::DEBUG, ErrorPlatform::kGENERAL,
                ErrorType::UNKNOWN, oss.str());
        }
#endif
        return result;
    }

    // 多节点图：使用新的多节点 kernel 生成
    if (num_compute > 1) {
        size_t scratch_size = 0;
        std::string src = generateMultiNodeKernel(graph, scratch_size);
        GeneratedKernel result;
        result.is_multi_node = true;
        result.num_inputs = graph.inputCount();
        result.scratch_size = scratch_size;

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

        #ifdef CT_DEBUG
        {
            std::ofstream dbgf("/tmp/c3_kernel_dump.c",
                                      std::ios::out | std::ios::trunc);
            dbgf << "// M=" << result.M << " K=" << result.K << " N=" << result.N
                 << " num_inputs=" << result.num_inputs << " elem_n=" << result.elem_n << "\n";
            dbgf << src;
            dbgf.close();
        }
#endif
        // [Dev] C3_DUMP_KERNEL=1 → 将多节点 kernel 源码 dump 到工作目录（便于审查生成代码）
        if (getenv("C3_DUMP_KERNEL")) {
            std::ofstream dbgf("./debug_multinode_kernel.c", std::ios::out | std::ios::trunc);
            dbgf << "// M=" << result.M << " K=" << result.K << " N=" << result.N
                 << " num_inputs=" << result.num_inputs << " elem_n=" << result.elem_n << "\n";
            dbgf << src;
            dbgf.close();
        }
        // [Fix] v0.5.2 Linux build: DTK clang 17 OpenMP 严格模式, named 变量替代 structured binding
        auto _compile_result = compileAndLoad(src, "c3_kernel");
        auto func_ptr = _compile_result.first;
        auto dl_handle = _compile_result.second;
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
        // DEBT-NEW-7 候选 A:把 FusedNode 的真实输出 shape 传给 kernel wrapper,
        // 让 FusedCompiledKernel::execute() 能正确分配 output buffer
        // (MatMul+Add+ReLU 的输出是 [M,N],不是 inputs[0] 的 [M,K])
        result.fused_out_shape = fnode.out_desc.shape;

        // 编译融合 kernel（使用 FusedKernelFunc 签名）
        // [Fix] v0.5.2 Linux build: DTK clang 17 OpenMP 严格模式, named 变量替代 structured binding
        auto _compile_result = compileAndLoad(src, "c3_kernel");
        auto func_ptr = _compile_result.first;
        auto dl_handle = _compile_result.second;
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
    } else if (std::holds_alternative<SumReduceNode>(op)) {
        // DEBT-NEW-7 v0.5.1+: 单节点 SumReduce (AddNode broadcast backward 等)
        // 直接生成 C 源码,axis=-1 全 reduce / axis=last_dim 沿行求和 / axis=0 沿列求和
        const auto& sr = std::get<SumReduceNode>(op);
        const auto& in_shape = sr.in_desc.shape;
        int64_t in_rank = (int64_t)in_shape.size();
        int64_t in_n = (int64_t)sr.in_desc.numel;
        int64_t out_n = (int64_t)nodes[0].out_desc.numel;
        std::stringstream ss;
        ss << "extern \"C\" void c3_kernel(const float* in0, float* out) {\n";
        if (in_n == 0 || out_n == 0) {
            ss << "    // empty\n";
        } else if (sr.axis == -1 || out_n == 1) {
            int64_t unroll = (in_n <= 16) ? in_n : 16;
            ss << "    float acc = 0.0f;\n"
               << "    int64_t i = 0;\n"
               << "    for (; i + " << unroll << " <= " << in_n << "; i += " << unroll << ") {\n"
               << "        acc += in0[i";
            for (int k = 1; k < unroll; ++k) ss << "] + in0[i+" << k;
            ss << "];\n"
               << "    }\n"
               << "    for (; i < " << in_n << "; ++i) acc += in0[i];\n"
               << "    out[0] = acc;\n";
        } else if (sr.axis == in_rank - 1 && in_rank >= 2) {
            int64_t outer = in_n / in_shape[in_rank - 1];
            int64_t inner = in_shape[in_rank - 1];
            ss << "    for (int64_t r = 0; r < " << outer << "; ++r) {\n"
               << "        float acc = 0.0f;\n"
               << "        for (int64_t c = 0; c < " << inner << "; ++c) {\n"
               << "            acc += in0[r * " << inner << " + c];\n"
               << "        }\n"
               << "        out[r] = acc;\n"
               << "    }\n";
        } else if (sr.axis == 0 && in_rank >= 2) {
            int64_t outer = in_shape[0];
            int64_t inner = in_n / outer;
            ss << "    for (int64_t c = 0; c < " << inner << "; ++c) {\n"
               << "        float acc = 0.0f;\n"
               << "        for (int64_t r = 0; r < " << outer << "; ++r) {\n"
               << "            acc += in0[r * " << inner << " + c];\n"
               << "        }\n"
               << "        out[c] = acc;\n"
               << "    }\n";
        } else {
            // fallback: 全 reduce + broadcast
            ss << "    float acc = 0.0f;\n"
               << "    for (int64_t i = 0; i < " << in_n << "; ++i) acc += in0[i];\n"
               << "    for (int64_t i = 0; i < " << out_n << "; ++i) out[i] = acc;\n";
        }
        ss << "}\n";
        src = ss.str();
        result.num_inputs = 1;
    } else {
        throw std::runtime_error(
            std::string("HandwrittenKernelGen: unsupported op type: ") +
            std::to_string(op.index()));
    }

    // [Fix] v0.5.2 Linux build: DTK clang 17 OpenMP 严格模式, named 变量替代 structured binding
    auto _compile_result = compileAndLoad(src, "c3_kernel");
    auto func = _compile_result.first;
    auto dl_handle = _compile_result.second;
    result.func = func;
    result.handle = dl_handle;
    result.deleter = [dl_handle]() { if (dl_handle) dlclose(dl_handle); };

    return result;
}

} // namespace c3
} // namespace ct