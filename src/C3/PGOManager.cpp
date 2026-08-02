/**
 * @file PGOManager.cpp
 * @brief PGO 三层异步编译流水线实现
 * @details 实现 CaaS 架构：第一次调用即触发异步编译链，逐级提升优化级别。
 * @date 2026/08/02
 */

#include "../../include/C3/PGOManager.h"
#include "../../include/C3/Graph.h"

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
}

std::vector<Tensor> PGOCompiledKernel::execute(const std::vector<Tensor>& inputs) {
    // 优先级 1: 如果有 Ofast kernel，直接使用（最高优化级别）
    if (auto k = ofast_kernel_) {
        auto start = std::chrono::steady_clock::now();
        auto result = k->execute(inputs);
        auto end = std::chrono::steady_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        profile_data_->record(ns);
        return result;
    }

    // 优先级 2: 如果有 O2 kernel，使用 O2（编译完成但 Ofast 还在编译）
    if (auto k = o2_kernel_) {
        auto start = std::chrono::steady_clock::now();
        auto result = k->execute(inputs);
        auto end = std::chrono::steady_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        profile_data_->record(ns);
        return result;
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

bool PGOCompiledKernel::installIntoRegistry(op op_type, const KernelShapeInfo& shapes) {
    // PGOCompiledKernel 本身不直接安装到注册表
    // 编译后的 kernel 由编译链完成时安装
    (void)op_type;
    (void)shapes;
    return false;
}

void PGOCompiledKernel::triggerCompilationChain() {
    auto& pgo = PGOManager::getInstance();
    if (!pgo.canAcceptCompilation()) {
        // 编译队列背压触发：将任务推入优先级队列，按热度评分排序
        double heat = computeHeatScore();
        std::string desc = "PGO: " + cache_key_;

        // 创建编译任务 lambda
        auto compile_task = [this, &pgo]() {
            pgo.notifyCompilationStarted();
            compileO2();
            compileOfast();
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
        std::thread([this]() {
            std::cerr << "[PGO] Starting async compile chain for " << cache_key_ << std::endl;
            compileO2();
            compileOfast();
            PGOManager::getInstance().notifyCompilationCompleted();
        }).detach();
    } else {
        // 同步编译（用于测试）
        std::cerr << "[PGO] Starting sync compile chain for " << cache_key_ << std::endl;
        compileO2();
        std::cerr << "[PGO] After compileO2: o2_kernel_=" << (o2_kernel_ ? "valid" : "nullptr") << std::endl;
        compileOfast();
        std::cerr << "[PGO] After compileOfast: ofast_kernel_=" << (ofast_kernel_ ? "valid" : "nullptr") << std::endl;
        PGOManager::getInstance().notifyCompilationCompleted();
    }
}

void PGOCompiledKernel::compileO2() {
    // 编译 O2 级别 kernel
    std::cerr << "[PGO] compileO2 ENTER for " << cache_key_ << std::endl;
    try {
        CompileOptions o2_opts = options_;
        o2_opts.opt_level = 2;
        o2_opts.pgo_mode = false;       // 避免递归创建 PGOCompiledKernel
        o2_opts.enable_cache = true;    // 启用缓存

        std::cerr << "[PGO] compileO2: calling engine_.compile() with opt_level=2" << std::endl;
        auto kernel = engine_.compile(graph_, o2_opts);
        std::cerr << "[PGO] compileO2: engine_.compile() returned " << (kernel ? "valid" : "nullptr") << std::endl;

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
                o2_kernel_->installIntoRegistry(op_type.value(), shapes);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[PGO] O2 compile exception: " << e.what() << std::endl;
        // 编译失败，静默处理——继续使用 Eager 解释执行
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
            std::cerr << "[PGO] Ofast compile failed for " << cache_key_ << std::endl;
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
        std::cerr << "[PGO] Ofast compile exception: " << e.what() << std::endl;
        // 编译失败，静默处理——继续使用 O2 或 Eager
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
    for (const auto& node : nodes) {
        // 跳过已处理的输入节点
        if (values.count(node.id)) continue;

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

    // 解析外部输入
    std::vector<Tensor> ext_inputs;
    ext_inputs.reserve(fnode.arg_node_ids.size());
    for (size_t arg_id : fnode.arg_node_ids) {
        auto it = values.find(arg_id);
        if (it == values.end()) {
            throw std::runtime_error(
                "PGOCompiledKernel: FusedNode external input node " +
                std::to_string(arg_id) + " not found");
        }
        ext_inputs.push_back(it->second);
    }

    // 按顺序执行融合链中的每个操作
    Tensor last_output;
    bool has_last = false;

    for (size_t i = 0; i < fnode.ops.size(); ++i) {
        const auto& op = fnode.ops[i];
        const auto& input_indices = fnode.op_inputs[i];

        auto op_type = nodeVariantToOp(op);
        if (!op_type.has_value()) {
            throw std::runtime_error(
                "PGOCompiledKernel: unsupported op in FusedNode at index " +
                std::to_string(i));
        }

        auto resolveInput = [&](size_t idx) -> Tensor {
            if (idx < ext_inputs.size()) {
                return ext_inputs[idx];
            }
            size_t prev_idx = idx - ext_inputs.size();
            if (prev_idx == 0 && has_last) {
                return last_output;
            }
            throw std::runtime_error(
                "PGOCompiledKernel: FusedNode input index " + std::to_string(idx) +
                " out of range");
        };

        if (isBinaryOp(op)) {
            if (input_indices.size() < 2) {
                throw std::runtime_error(
                    "PGOCompiledKernel: FusedNode binary op needs 2 inputs");
            }
            auto lhs = resolveInput(input_indices[0]);
            auto rhs = resolveInput(input_indices[1]);
            last_output = scheduler.dispatch(lhs, rhs, op_type.value());
        } else {
            if (input_indices.empty()) {
                throw std::runtime_error(
                    "PGOCompiledKernel: FusedNode unary op needs 1 input");
            }
            auto input = resolveInput(input_indices[0]);
            last_output = scheduler.dispatch(input, op_type.value());
        }
        has_last = true;
    }

    if (!has_last) {
        throw std::runtime_error("PGOCompiledKernel: empty FusedNode");
    }

    return last_output;
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
    auto kernel = std::make_shared<PGOCompiledKernel>(
        graph, options, std::move(cache_key),
        std::move(profile_data), engine);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        // 清理失效的 weak_ptr
        entries_.erase(
            std::remove_if(entries_.begin(), entries_.end(),
                           [](const std::weak_ptr<PGOCompiledKernel>& wp) {
                               return wp.expired();
                           }),
            entries_.end());
        entries_.push_back(kernel);
    }

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
    for (const auto& wp : entries_) {
        if (auto sp = wp.lock()) {
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
    for (const auto& wp : entries_) {
        if (auto sp = wp.lock()) {
            sp->promote();
        }
    }
}

} // namespace c3
} // namespace ct