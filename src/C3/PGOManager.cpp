/**
 * @file PGOManager.cpp
 * @brief PGO 三层异步编译流水线实现
 * @details 实现 CaaS 架构：第一次调用即触发异步编译链，逐级提升优化级别。
 * @date 2026/08/02
 */

#include "../../include/C3/PGOManager.h"
#include "../../include/C3/Graph.h"
#include "../../include/CtorchError.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <future>
#include <stdexcept>
#include <thread>

#include "CtorchScheduler.h"

namespace ct {
namespace c3 {

// ======================= 辅助函数 =======================

namespace {

/// 将 C3 NodeVariant 映射到调度器 op 枚举（与 C3Engine.cpp 中定义一致）
static std::optional<op> nodeVariantToOp(const NodeVariant& nv) {
    switch (nv.index()) {
        case 0:  return op::Add;
        case 1:  return op::Sub;
        case 2:  return op::Mul;
        case 3:  return op::Div;
        case 4:  return op::MatMul;
        case 5:  return op::Neg;
        case 6:  return op::ReLU;
        case 7:  return op::Sigmoid;
        case 8:  return op::Tanh;
        default: return std::nullopt;
    }
}

/// 判断是否为二元算子
static bool isBinaryOp(const NodeVariant& nv) {
    auto op_type = nodeVariantToOp(nv);
    if (!op_type.has_value()) return false;
    switch (op_type.value()) {
        case op::Add:
        case op::Sub:
        case op::Mul:
        case op::Div:
        case op::MatMul:
            return true;
        default:
            return false;
    }
}

/// 从 Graph 提取 KernelShapeInfo（与 C3Engine.cpp 中定义一致）
static KernelShapeInfo graphToShapeInfo(const Graph& graph) {
    KernelShapeInfo info;
    if (graph.outputCount() == 0) return info;

    auto& out_node = graph.node(graph.outputs()[0]);
    info.out_shape = out_node.out_desc.shape;

    auto& input_ids = graph.inputs();
    if (input_ids.size() >= 1) {
        info.lhs_shape = graph.node(input_ids[0]).out_desc.shape;
    }
    if (input_ids.size() >= 2) {
        info.rhs_shape = graph.node(input_ids[1]).out_desc.shape;
    }

    if (auto* mm = std::get_if<MatMulNode>(&out_node.op)) {
        info.is_matmul = true;
        info.M = mm->lhs_desc.shape[0];
        info.K = mm->lhs_desc.shape[1];
        info.N = mm->rhs_desc.shape[1];
    }

    return info;
}

} // anonymous namespace

// ======================= PGOCompiledKernel =======================

PGOCompiledKernel::PGOCompiledKernel(
    const Graph& graph,
    CompileOptions options,
    std::string cache_key,
    std::shared_ptr<ProfileData> profile_data,
    C3Engine& engine)
    : graph_(graph)
    , options_(std::move(options))
    , cache_key_(std::move(cache_key))
    , profile_data_(std::move(profile_data))
    , engine_(engine)
{
    CtorchError::log(ErrorLevel::DEBUG, ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
        "PGOCompiledKernel ctor: nodes=" + std::to_string(graph_.nodeCount()) +
        " inputs=" + std::to_string(graph_.inputCount()) +
        " outputs=" + std::to_string(graph_.outputCount()) +
        " cache_key=" + cache_key_);
}

std::vector<Tensor> PGOCompiledKernel::execute(const std::vector<Tensor>& inputs) {
    // ====== Deoptimization 支持 (ADR-006) ======
    // 优先级 1: Ofast kernel（若未被 deopt 禁用）
    if (auto k = ofast_kernel_) {
        if (!ofast_disabled_.load(std::memory_order_acquire)) {
            try {
                auto start = std::chrono::steady_clock::now();
                auto result = k->execute(inputs);
                auto end = std::chrono::steady_clock::now();
                auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
                profile_data_->record(ns);
                return result;
            } catch (const std::exception& e) {
                // Ofast 运行时异常 → deopt 到 O2
                recordDeopt("ofast", e.what());
            } catch (...) {
                recordDeopt("ofast", "unknown exception");
            }
            // 显式 disable 防止后续重试（单次失败即永久 deopt）
            ofast_disabled_.store(true, std::memory_order_release);
        }
        // 显式 drop 局部 shared_ptr，加速 ofast_kernel_ 释放（如果不空）
        // 注：成员 ofast_kernel_ 仍保留，避免影响其他观察者；下次 execute 会重新检查 disabled
    }

    // 优先级 2: O2 kernel（若未被 deopt 禁用）
    if (auto k = o2_kernel_) {
        if (!o2_disabled_.load(std::memory_order_acquire)) {
            try {
                auto start = std::chrono::steady_clock::now();
                auto result = k->execute(inputs);
                auto end = std::chrono::steady_clock::now();
                auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
                profile_data_->record(ns);
                return result;
            } catch (const std::exception& e) {
                // O2 运行时异常 → deopt 到 Eager
                recordDeopt("o2", e.what());
            } catch (...) {
                recordDeopt("o2", "unknown exception");
            }
            o2_disabled_.store(true, std::memory_order_release);
        }
    }

    // 优先级 3: Eager 解释执行（Tier 1，零编译延迟）
    auto start = std::chrono::steady_clock::now();
    auto result = executeInterpreted(inputs);
    auto end = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    profile_data_->record(ns);

    // 第一次调用时触发异步编译链
    if (!compilation_triggered_.exchange(true)) {
        triggerCompilationChain();
    }

    return result;
}

void PGOCompiledKernel::recordDeopt(const char* tier, const std::string& reason) {
    deopt_count_.fetch_add(1, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(deopt_mutex_);
        last_deopt_reason_ = std::string(tier) + ": " + reason;
    }
    CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL, ErrorType::KERNEL_LAUNCH,
        "PGO: deopt " + std::string(tier) + " kernel for " + cache_key_ + " — " + reason);
}

void PGOCompiledKernel::recordCompileError(const char* tier, const std::string& reason) {
    // 截断到 1KB
    std::string truncated_reason = reason;
    const size_t kMaxLen = 1024;
    if (truncated_reason.size() > kMaxLen) {
        truncated_reason = truncated_reason.substr(0, kMaxLen) +
            "... [truncated, original=" + std::to_string(reason.size()) + " bytes]";
    }
    {
        std::lock_guard<std::mutex> lock(compile_error_mutex_);
        last_compile_error_ = std::string(tier) + ": " + truncated_reason;
    }
    CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL, ErrorType::KERNEL_LAUNCH,
        "PGO: compile error " + std::string(tier) + " for " + cache_key_ + " — " +
        truncated_reason);
    // 同时回传到 C3Engine 全局 last_compile_error_（ADR-010）：
    // 让调用方通过 C3Engine::getLastCompileError() 也能查到 PGO 编译失败，
    // 而不需要保留 PGOCompiledKernel 指针来调用本 kernel 的 lastCompileError()。
    // 注意：这里会覆盖前一次记录，但前一次记录通常也是 PGO 链中的更早一环，
    //      因此 latest-wins 语义是合理的。
    try {
        engine_.recordCompileError(tier, truncated_reason);
    } catch (...) {
        // recordCompileError 理论上不会抛（只有 mutex 锁），但双保险吞错，
        // 避免 PGO 错误日志自身抛异常把 compileO2/Ofast 流程搞砸
    }
}

bool PGOCompiledKernel::installIntoRegistry(op op_type, const KernelShapeInfo& shapes) {
    // PGOCompiledKernel 本身不直接安装到注册表
    // 编译后的 kernel 由编译链完成时安装
    (void)op_type;
    (void)shapes;
    return false;
}

void PGOCompiledKernel::triggerCompilationChain() {
    auto& pgo = PGOManager::getInstance();
    auto self = shared_from_this();
    if (!pgo.canAcceptCompilation()) {
        // 编译队列背压触发：将任务推入优先级队列，按热度评分排序
        double heat = computeHeatScore();
        std::string desc = "PGO: " + cache_key_;

        // 创建编译任务 lambda
        auto compile_task = [self, &pgo]() {
            pgo.notifyCompilationStarted();
            self->compileO2();
            self->compileOfast();
            pgo.notifyCompilationCompleted();
        };

        // 推入优先级队列
        {
            std::lock_guard<std::mutex> lock(pgo.queue_mutex());
            pgo.task_queue().push(CompilationTask{
                heat, std::move(compile_task),
                std::chrono::steady_clock::now(), std::move(desc)
            });
        }

        // 记录背压拒绝
        pgo.recordQueueRejection();

        // 下次调用时尝试从队列中取出
        compilation_triggered_.store(false, std::memory_order_release);
        return;
    }

    // 通知 PGOManager 编译任务开始
    pgo.notifyCompilationStarted();

    // 启动异步编译链（O2 → Ofast）
    if (PGOManager::getInstance().config().async_compilation) {
        // 集中 future 管理，避免 std::thread::detach() 引发 UAF
        // （线程在 main 退出后仍会 lock 已析构 the PGOManager mutex）
        std::future<void> fut = std::async(std::launch::async, [self]() {
            try {
                CtorchError::log(ErrorLevel::DEBUG, ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
                    "PGO: starting async compile chain for " + self->cache_key_);
                self->compileO2();
                self->compileOfast();
            } catch (const std::exception& e) {
                // 防止后台线程异常导致 std::terminate；同时避免 lock 已析构 mutex
                CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL,
                    ErrorType::KERNEL_LAUNCH,
                    std::string("PGO: async compile chain exception for ") + self->cache_key_ +
                    ": " + e.what());
            } catch (...) {
                CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL,
                    ErrorType::KERNEL_LAUNCH,
                    "PGO: async compile chain unknown exception for " + self->cache_key_);
            }
            try {
                PGOManager::getInstance().notifyCompilationCompleted();
            } catch (...) {
                // notifyCompilationCompleted 可能 lock 已析构 PGOManager mutex
                // （main 退出 + 用户未调 PGOManager::shutdown 的极端场景）
                // 静默吞错：仅统计作用，不影响业务正确性
            }
        });
        std::lock_guard<std::mutex> lock(PGOManager::getInstance().futures_mutex());
        PGOManager::getInstance().compile_futures().push_back(std::move(fut));
    } else {
        // 同步编译（用于测试）
        CtorchError::log(ErrorLevel::DEBUG, ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
            "PGO: starting sync compile chain for " + cache_key_);
        compileO2();
        compileOfast();
        PGOManager::getInstance().notifyCompilationCompleted();
    }
}

void PGOCompiledKernel::compileO2() {
    // 编译 O2 级别 kernel
    CtorchError::log(ErrorLevel::DEBUG, ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
        "PGO: compileO2 ENTER for " + cache_key_);
    try {
        CompileOptions o2_opts = options_;
        o2_opts.opt_level = 2;
        o2_opts.pgo_mode = false;       // 避免递归创建 PGOCompiledKernel
        o2_opts.enable_cache = true;    // 启用缓存

        auto kernel = engine_.compile(graph_, o2_opts);
        if (!kernel) {
            recordCompileError("o2", "compile returned nullptr");
        }

        {
            std::lock_guard<std::mutex> lock(compile_mutex_);
            o2_kernel_ = std::move(kernel);
        }

        // 尝试安装到 C3 注册表（仅支持单节点图）
        if (graph_.outputCount() > 0) {
            auto& out_node = graph_.node(graph_.outputs()[0]);
            auto op_type = nodeVariantToOp(out_node.op);
            if (op_type.has_value()) {
                auto shapes = graphToShapeInfo(graph_);
                if (o2_kernel_) {
                    o2_kernel_->installIntoRegistry(op_type.value(), shapes);
                }
            }
        }
    } catch (const std::exception& e) {
        // 编译失败：记录到 last_compile_error_（ADR-007），调用方可通过
        // PGOCompiledKernel::lastCompileError() 查询。同时静默处理，继续 Eager。
        recordCompileError("o2", e.what());
    } catch (...) {
        recordCompileError("o2", "unknown exception");
    }
}

void PGOCompiledKernel::compileOfast() {
    // 编译 Ofast 级别 kernel（在 O2 编译完成后调用）
    try {
        CompileOptions ofast_opts = options_;
        ofast_opts.opt_level = 3;
        ofast_opts.pgo_mode = false;    // 避免递归创建 PGOCompiledKernel
        ofast_opts.enable_cache = true; // 启用缓存

        auto kernel = engine_.compile(graph_, ofast_opts);
        if (!kernel) {
            recordCompileError("ofast", "compile returned nullptr");
            return; // 编译失败，静默处理
        }

        {
            std::lock_guard<std::mutex> lock(compile_mutex_);
            ofast_kernel_ = std::move(kernel);
        }

        // 尝试安装到 C3 注册表（仅支持单节点图）
        if (graph_.outputCount() > 0) {
            auto& out_node = graph_.node(graph_.outputs()[0]);
            auto op_type = nodeVariantToOp(out_node.op);
            if (op_type.has_value()) {
                auto shapes = graphToShapeInfo(graph_);
                ofast_kernel_->installIntoRegistry(op_type.value(), shapes);
            }
        }
    } catch (const std::exception& e) {
        // 编译失败：记录到 last_compile_error_（ADR-007）
        recordCompileError("ofast", e.what());
    } catch (...) {
        recordCompileError("ofast", "unknown exception");
    }
}

double PGOCompiledKernel::computeHeatScore() const {
    uint64_t calls = profile_data_->call_count.load(std::memory_order_relaxed);
    if (calls == 0) return 0.0;

    uint64_t total_time = profile_data_->total_time_ns.load(std::memory_order_relaxed);

    // 调用次数评分：对数增长，趋向饱和
    const uint64_t window = PGOManager::getInstance().config().heat_score_window_calls;
    double call_score = std::min(1.0, std::log2(1.0 + static_cast<double>(calls)) /
                                        std::log2(1.0 + static_cast<double>(window)));

    // 累计时间评分：归一化到 1 秒
    double time_score = std::min(1.0, static_cast<double>(total_time) / 1e9);

    // 综合评分：调用次数 40% + 累计时间 60%（时间更能反映实际计算量）
    return 0.4 * call_score + 0.6 * time_score;
}

std::vector<Tensor> PGOCompiledKernel::executeInterpreted(const std::vector<Tensor>& inputs) {
    auto& scheduler = CtorchScheduler::getInstance();

    // 值映射：node_id → Tensor
    std::unordered_map<size_t, Tensor> values;
    values.reserve(graph_.nodeCount());

    // 1. 将函数输入映射到图输入节点
    const auto& input_ids = graph_.inputs();
    if (inputs.size() < input_ids.size()) {
        throw std::runtime_error(
            "PGOCompiledKernel: need " + std::to_string(input_ids.size()) +
            " inputs, got " + std::to_string(inputs.size()));
    }
    for (size_t i = 0; i < input_ids.size(); ++i) {
        values[input_ids[i]] = inputs[i];
    }

    // 2. 按拓扑顺序遍历所有节点，解释执行
    const auto& nodes = graph_.nodes();
    // 预计算 input 节点 ID 集合（input 节点用 ConstNode{0.0} 作占位符，遍历时需优先匹配 input）
    const std::unordered_set<size_t> input_set(graph_.inputs().begin(), graph_.inputs().end());
    for (const auto& node : nodes) {
        // input 节点已在 step 1 映射到 inputs[i]，跳过即可（防止被下方 ConstNode 分支误判为 0.0）
        if (input_set.count(node.id)) continue;

        // 跳过 ConstNode（常量折叠后的残留节点）
        if (std::holds_alternative<ConstNode>(node.op)) {
            if (node.out_desc.numel == 1) {
                float val = static_cast<float>(std::get<ConstNode>(node.op).value);
                Tensor t(ShapeTag{}, {1});
                t.data_write<float>()[0] = val;
                values[node.id] = std::move(t);
            } else {
                Tensor t(ShapeTag{}, node.out_desc.shape);
                std::memset(t.data_write<float>(), 0, t.numel() * sizeof(float));
                values[node.id] = std::move(t);
            }
            continue;
        }

        // 处理 FusedNode
        if (std::holds_alternative<FusedNode>(node.op)) {
            values[node.id] = executeFusedNodeInterpreted(
                std::get<FusedNode>(node.op), values);
            continue;
        }

        // 获取算子类型
        auto op_type = nodeVariantToOp(node.op);
        if (!op_type.has_value()) {
            throw std::runtime_error(
                "PGOCompiledKernel: unsupported node type at node " +
                std::to_string(node.id));
        }

        // 解析输入张量并调度执行
        if (isBinaryOp(node.op)) {
            if (node.inputs.size() < 2) {
                throw std::runtime_error(
                    "PGOCompiledKernel: binary op needs 2 inputs at node " +
                    std::to_string(node.id));
            }
            auto it_lhs = values.find(node.inputs[0]);
            auto it_rhs = values.find(node.inputs[1]);
            if (it_lhs == values.end() || it_rhs == values.end()) {
                throw std::runtime_error(
                    "PGOCompiledKernel: input not found for node " +
                    std::to_string(node.id));
            }
            values[node.id] = scheduler.dispatch(it_lhs->second, it_rhs->second, op_type.value());
        } else {
            // 一元算子
            if (node.inputs.empty()) {
                throw std::runtime_error(
                    "PGOCompiledKernel: unary op needs 1 input at node " +
                    std::to_string(node.id));
            }
            auto it = values.find(node.inputs[0]);
            if (it == values.end()) {
                throw std::runtime_error(
                    "PGOCompiledKernel: input not found for node " +
                    std::to_string(node.id));
            }
            values[node.id] = scheduler.dispatch(it->second, op_type.value());
        }
    }

    // 3. 收集输出张量
    const auto& output_ids = graph_.outputs();
    std::vector<Tensor> results;
    results.reserve(output_ids.size());
    for (size_t out_id : output_ids) {
        auto it = values.find(out_id);
        if (it == values.end()) {
            throw std::runtime_error(
                "PGOCompiledKernel: output node " + std::to_string(out_id) + " not found");
        }
        results.push_back(it->second);
    }

    return results;
}

Tensor PGOCompiledKernel::executeFusedNodeInterpreted(
    const FusedNode& fnode,
    const std::unordered_map<size_t, Tensor>& values)
{
    auto& scheduler = CtorchScheduler::getInstance();

    // 1. 解析外部输入：arg_node_ids[i] -> values[arg_node_ids[i]]
    //    同时构建 arg_node_id -> arg_index 映射，用于后续 op_inputs 解析
    std::vector<Tensor> ext_inputs;
    ext_inputs.reserve(fnode.arg_node_ids.size());
    std::unordered_map<size_t, size_t> arg_id_to_idx;
    for (size_t i = 0; i < fnode.arg_node_ids.size(); ++i) {
        size_t arg_id = fnode.arg_node_ids[i];
        auto it = values.find(arg_id);
        if (it == values.end()) {
            throw std::runtime_error(
                "PGOCompiledKernel: FusedNode external input node " +
                std::to_string(arg_id) + " not found");
        }
        ext_inputs.push_back(it->second);
        arg_id_to_idx[arg_id] = i;
    }

    // 2. 按顺序执行融合链中的每个操作
    //    op_inputs[i] 是 Graph 节点 ID（不是 FusedNode 内部索引！）
    //    假设（与 HandwrittenKernelGen 保持一致）：
    //      - op[0] 的所有 input_ids 都是外部输入
    //      - op[i>0] 的 input_ids[0] 是 chain 内部（即 op_outputs[i-1]）
    //      - op[i>0] 的 input_ids[1..] 是外部输入
    //    用 op_outputs[] 数组记录每个 op 的输出，支持任意长度的 chain。
    std::vector<Tensor> op_outputs;
    op_outputs.reserve(fnode.ops.size());

    for (size_t i = 0; i < fnode.ops.size(); ++i) {
        const auto& op = fnode.ops[i];
        const auto& input_ids = fnode.op_inputs[i];  // Graph 节点 ID 列表

        auto op_type = nodeVariantToOp(op);
        if (!op_type.has_value()) {
            throw std::runtime_error(
                "PGOCompiledKernel: unsupported op in FusedNode at index " +
                std::to_string(i));
        }

        // 解析 input：第一个位置（i>0）是 chain 内部，其他是外部
        auto resolveByPosition = [&](size_t pos) -> Tensor {
            size_t in_id = input_ids[pos];
            if (i > 0 && pos == 0) {
                // chain 内部：引用上一个 op 的输出
                return op_outputs[i - 1];
            }
            // 外部输入：用 arg_id_to_idx 映射
            auto it = arg_id_to_idx.find(in_id);
            if (it == arg_id_to_idx.end()) {
                throw std::runtime_error(
                    "PGOCompiledKernel: FusedNode input node " + std::to_string(in_id) +
                    " not found in arg_node_ids at op " + std::to_string(i) +
                    " pos " + std::to_string(pos));
            }
            return ext_inputs[it->second];
        };

        Tensor cur_output;
        if (isBinaryOp(op)) {
            if (input_ids.size() < 2) {
                throw std::runtime_error(
                    "PGOCompiledKernel: FusedNode binary op needs 2 inputs at op " +
                    std::to_string(i));
            }
            auto lhs = resolveByPosition(0);
            auto rhs = resolveByPosition(1);
            cur_output = scheduler.dispatch(lhs, rhs, op_type.value());
        } else {
            if (input_ids.empty()) {
                throw std::runtime_error(
                    "PGOCompiledKernel: FusedNode unary op needs 1 input at op " +
                    std::to_string(i));
            }
            auto input = resolveByPosition(0);
            cur_output = scheduler.dispatch(input, op_type.value());
        }
        op_outputs.push_back(std::move(cur_output));
    }

    if (op_outputs.empty()) {
        throw std::runtime_error("PGOCompiledKernel: empty FusedNode");
    }

    // 返回最后一个 op 的输出（与原语义一致：FusedNode 链终点即整体输出）
    return op_outputs.back();
}

// ======================= PGOManager =======================

PGOManager& PGOManager::getInstance() {
    static PGOManager instance;
    return instance;
}

std::shared_ptr<PGOCompiledKernel> PGOManager::registerKernel(
    const Graph& graph,
    const CompileOptions& options,
    std::string cache_key,
    std::shared_ptr<ProfileData> profile_data,
    C3Engine& engine)
{
    std::lock_guard<std::mutex> lock(mutex_);

    // 1. 缓存命中：相同 cache_key 复用既有 PGOCompiledKernel
    //    （profile_data 仍由调用方持有并复用，确保 PGO 统计在多次调用间累计）
    auto cache_it = cache_.find(cache_key);
    if (cache_it != cache_.end()) {
        if (cache_it->second) {
            return cache_it->second;
        }
        // 缓存条目已失效，移除
        cache_.erase(cache_it);
    }

    // 2. 缓存未命中：创建新 PGOCompiledKernel
    //    注意：cache_key 是值类型，std::make_shared 构造函数把它 move 到 PGOCompiledKernel，
    //    因此写入缓存前需要把 cache_key 复制一份。
    const std::string key_copy = cache_key;
    auto kernel = std::make_shared<PGOCompiledKernel>(
        graph, options, std::move(cache_key),
        std::move(profile_data), engine);

    // 3. entries_ 直接持有 shared_ptr（无需清理过期 weak_ptr）
    entries_.push_back(kernel);

    // 4. 写入缓存（使用 shared_ptr 确保 kernel 一直存活）
    cache_[key_copy] = kernel;

    return kernel;
}

bool PGOManager::canAcceptCompilation() const {
    uint64_t active = active_compilations_.load(std::memory_order_acquire);
    if (active >= config_.max_concurrent_compilations) {
        return false;
    }

    // 检查队列长度（Anti-Windup 背压）
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (task_queue_.size() >= config_.queue_backpressure_threshold) {
        return false;
    }

    return true;
}

void PGOManager::notifyCompilationStarted() {
    active_compilations_.fetch_add(1, std::memory_order_acq_rel);
}

void PGOManager::notifyCompilationCompleted() {
    active_compilations_.fetch_sub(1, std::memory_order_acq_rel);

    // 编译完成后，尝试从优先级队列中取出下一个任务执行
    processQueue();
}

void PGOManager::processQueue() {
    CompilationTask task;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (task_queue_.empty()) return;
        task = std::move(const_cast<CompilationTask&>(task_queue_.top()));
        task_queue_.pop();
    }

    // 执行编译任务（在当前线程中，不额外启动新线程）
    try {
        if (task.task) {
            task.task();
        }
    } catch (const std::exception&) {
        // 队列任务执行失败，静默处理
    }
}

PGOManager::Stats PGOManager::getStats() const {
    Stats s;
    std::lock_guard<std::mutex> lock(mutex_);
    s.total_registered = entries_.size();
    s.active_compilations = active_compilations_.load(std::memory_order_acquire);
    s.queue_rejections = total_queue_rejections_;
    for (const auto& sp : entries_) {
        if (sp) {
            bool has_o2 = sp->o2Kernel() != nullptr;
            bool has_ofast = sp->ofastKernel() != nullptr;
            if (has_ofast) s.ofast_ready++;
            if (has_o2)    s.o2_ready++;
            if (!has_o2 && !has_ofast) s.pending++;
        }
    }
    return s;
}

void PGOCompiledKernel::promote() {
    // 强制触发编译链
    if (compilation_triggered_.exchange(true)) {
        // 已经触发过，但可能还没完成
        // 如果 O2 和 Ofast 都没有，则重新触发
        if (!o2_kernel_ && !ofast_kernel_) {
            // 直接在当前线程中编译
            compileO2();
            compileOfast();
        }
        return;
    }

    // 还没触发过，直接在当前线程中编译
    compileO2();
    compileOfast();
}

void PGOManager::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    cache_.clear();
    active_compilations_.store(0, std::memory_order_release);
    total_queue_rejections_ = 0;
    {
        std::lock_guard<std::mutex> qlock(queue_mutex_);
        // priority_queue 没有 clear()，用 swap 技巧
        std::priority_queue<CompilationTask> empty;
        std::swap(task_queue_, empty);
    }
}

void PGOManager::promoteAll() {
    // 强制所有待编译 kernel 立即启动编译链
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& sp : entries_) {
        if (sp) {
            sp->promote();
        }
    }
}

void PGOManager::shutdown() {
    // 等待所有后台 PGO 编译任务完成（30s 超时，与 C3Engine::shutdown 同步）
    std::vector<std::future<void>> futures;
    {
        std::lock_guard<std::mutex> lock(futures_mutex());
        futures = std::move(compile_futures());
    }
    for (auto& f : futures) {
        if (f.valid()) {
            auto status = f.wait_for(std::chrono::seconds(30));
            if (status == std::future_status::ready) {
                try { f.get(); } catch (...) {} // 吸收异常
            } else {
                CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
                    "PGOManager::shutdown: background compile did not finish in 30s, "
                    "future abandoned (may cause UAF if main exits before thread finishes)");
            }
        }
    }
}

// ======================= 访问器实现 =======================

std::mutex& PGOManager::queue_mutex() { return queue_mutex_; }
std::priority_queue<CompilationTask>& PGOManager::task_queue() { return task_queue_; }
std::mutex& PGOManager::futures_mutex() { return futures_mutex_; }
std::vector<std::future<void>>& PGOManager::compile_futures() { return compile_futures_; }

} // namespace c3
} // namespace ct