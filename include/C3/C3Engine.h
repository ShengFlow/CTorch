/**
 * @file C3Engine.h
 * @generation SHARED 跨代编译引擎接口（doCompile 内串起 JIT-2.0/2.x/3.0）
 * @brief CTorch JIT 编译引擎公共接口
 * @details 提供将计算图（Graph）编译为后端 kernel 的能力，并管理编译产物缓存。
 *          当前为公共接口层，具体 Graph 定义与 kernel 实现位于 src/JIT 模块。
 * @date 2026/7/31
 */

#ifndef CTORCH_C3_JITENGINE_H
#define CTORCH_C3_JITENGINE_H

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../Ctools.h"
#include "../Tensor.h"
#include "C3KernelRegistry.h"
#include "GraphMerger.h"
#include "Tracer.h"

namespace ct {
namespace c3 {

// 前向声明：计算图定义由 src/JIT 模块提供，避免在公共头文件中暴露实现细节。
class Graph;
struct AutoTunerConfig;
/// 内部状态结构前向声明（P0-2 修复：EngineState 由 C3Engine 拥有，避免 TU 间 static 析构顺序 UB）
struct EngineState;

/**
 * @enum C3Backend
 * @brief JIT 编译后端选择
 * @note [已弃用] Handwritten 模式（JIT 1.0 clang++ 编译落盘后端）已彻底废弃删除。
 *       目前默认且强制采用 MLIR 统一后端（支持 JIT 2.0/3.0），
 *       即便显式指定为 Handwritten，在编译期或运行期亦会安全重定向到 MLIR。
 */
enum class C3Backend {
    /** @brief [已废弃] JIT 1.0 手写 C++ kernel 后端 (不推荐使用，内部强制重定向到 MLIR) */
    Handwritten = 0,
    /** @brief JIT 2.x/3.0 统一 MLIR 后端（图算子 -> C3 Dialect -> Linalg/SCF -> LLVM JIT） */
    MLIR = 1,
};

/**
 * @struct CompileOptions
 * @brief JIT 编译选项
 * @details 控制编译目标设备、后端选择、优化级别、算子融合策略与缓存行为。
 */
struct CompileOptions {
    /** @brief JIT 编译后端，默认采用 3.0 MLIR 后端 */
    C3Backend backend = C3Backend::MLIR;
    /** @brief 目标设备，默认 CPU */
    DeviceType target_device = DeviceType::kCPU;
    /** @brief 优化级别：0=关闭优化，1=基础优化，2=O2，3=O3/Ofast（默认，MLIR 后端生产优化级别） */
    int opt_level = 3;
    /** @brief 是否启用算子融合（默认开启） */
    bool enable_fusion = true;
    /** @brief 是否启用编译缓存（默认开启） */
    bool enable_cache = true;
    /** @brief 是否启用自动调优（默认关闭，首次编译时运行 QEA 搜索最优分块） */
    bool enable_autotune = false;
    /** @brief 是否启用 PGO 性能分析（默认关闭，开启后记录每次 execute 耗时） */
    bool enable_profiling = false;
    /** @brief 是否启用 PGO 两阶段编译（默认关闭）
     *
     *  Tier 1：解释器模式（Eager 逐节点执行），无编译开销
     *  Tier 2：当热路径检测到后，自动编译为 MLIR kernel 并透明切换
     *
     *  启用后，compile() 返回 PGOCompiledKernel，初始阶段通过 Eager 执行，
     *  收集 profile 数据，当调用次数/时间超过阈值后自动提升到编译模式。
     */
    bool pgo_mode = false;
    /** @brief 自定义缓存键；为空时由引擎根据图结构与选项生成 */
    std::string cache_key_override;
};

/**
 * @struct C3CacheStats
 * @brief JIT 编译缓存统计信息
 */
struct C3CacheStats {
    /** @brief 缓存条目总数 */
    size_t total_entries = 0;
    /** @brief 缓存命中次数 */
    size_t hits = 0;
    /** @brief 缓存未命中次数 */
    size_t misses = 0;
    /** @brief 被逐出的条目数 */
    size_t evictions = 0;
    /** @brief 缓存占用的近似字节数 */
    size_t bytes_used = 0;
    /** @brief 异步编译任务数（进行中） */
    size_t pending_compiles = 0;
    /** @brief 异步编译完成数 */
    size_t async_completions = 0;
    /** @brief 异步编译失败数 */
    size_t async_failures = 0;
};

/**
 * @struct ProfileData
 * @brief PGO 性能分析数据，记录 kernel 执行时间统计
 * @details 线程安全：所有字段使用 atomic 支持并发记录。
 *          由 ProfiledCompiledKernel 在每次 execute() 时自动更新。
 */
struct ProfileData {
    /** @brief 调用次数 */
    std::atomic<uint64_t> call_count{0};
    /** @brief 累计执行时间（纳秒） */
    std::atomic<uint64_t> total_time_ns{0};
    /** @brief 单次最短执行时间（纳秒） */
    std::atomic<uint64_t> min_time_ns{UINT64_MAX};
    /** @brief 单次最长执行时间（纳秒） */
    std::atomic<uint64_t> max_time_ns{0};
    /** @brief 最近一次执行时间（纳秒） */
    std::atomic<uint64_t> last_time_ns{0};

    /** @brief 记录一次执行耗时 */
    void record(uint64_t ns) {
        call_count.fetch_add(1, std::memory_order_relaxed);
        total_time_ns.fetch_add(ns, std::memory_order_relaxed);
        last_time_ns.store(ns, std::memory_order_relaxed);
        uint64_t old_min = min_time_ns.load(std::memory_order_relaxed);
        while (ns < old_min &&
               !min_time_ns.compare_exchange_weak(old_min, ns, std::memory_order_relaxed)) {}
        uint64_t old_max = max_time_ns.load(std::memory_order_relaxed);
        while (ns > old_max &&
               !max_time_ns.compare_exchange_weak(old_max, ns, std::memory_order_relaxed)) {}
    }

    /** @brief 平均执行时间（纳秒），无调用时返回 0 */
    [[nodiscard]] uint64_t avgTimeNs() const {
        uint64_t cnt = call_count.load(std::memory_order_relaxed);
        if (cnt == 0) return 0;
        return total_time_ns.load(std::memory_order_relaxed) / cnt;
    }
};

/**
 * @class CompiledKernel
 * @brief 编译后的可执行 kernel 抽象
 * @details 由 C3Engine::compile() 产出，封装了后端特定的可执行代码。
 *          通过 execute() 接收输入张量并返回输出张量。
 */
class CompiledKernel {
public:
    virtual ~CompiledKernel() = default;

    /**
     * @brief 执行编译后的 kernel
     * @param inputs 输入张量列表，顺序与 Graph 输入一致
     * @return 输出张量列表，顺序与 Graph 输出一致
     * @throw std::runtime_error 当输入不匹配或执行失败时抛出
     */
    virtual std::vector<Tensor> execute(const std::vector<Tensor>& inputs) = 0;

    /** @brief 返回该 kernel 的缓存键 */
    [[nodiscard]] virtual const std::string& cacheKey() const = 0;

    /** @brief 返回该 kernel 的目标设备 */
    [[nodiscard]] virtual DeviceType targetDevice() const = 0;

    /** @brief 返回执行时所需的工作空间字节数（0 表示不需要额外内存） */
    [[nodiscard]] virtual size_t workspaceBytes() const = 0;

    /**
     * @brief v0.5.2 (2026-08-09): 返回该 kernel 的输出 shape（注册时记录的"真实"out_shape）
     * @details 默认返回 std::nullopt,execute() 沿用 a.shape() / M×N 等启发式。
     *          Concrete/Fused/Multi 三个派生类覆写后,execute() 优先用此字段构造 Tensor,
     *          解决 backward 路径 grad 形状 ≠ forward output 形状的 bug (out_shape 修复)。
     *          - FusedCompiledKernel: 已有 out_shape_ 字段,1 行覆写
     *          - MultiNodeCompiledKernel: 用 M×N 或 elem_n
     *          - ConcreteCompiledKernel: 加 out_shape_ 字段,install 时透传
     * @return optional,空表示 kernel 不声明 out_shape (沿用启发式)
     */
    [[nodiscard]] virtual std::optional<std::vector<size_t>> outShape() const {
        return std::nullopt;
    }

    /**
     * @brief 将编译产物安装到 C3 内核注册表，启用热替换
     * @param op_type 对应算子类型
     * @param shapes 形状签名
     * @return 安装成功返回 true；默认实现返回 false（未实现）
     */
    virtual bool installIntoRegistry(op op_type, const KernelShapeInfo& shapes) {
        (void)op_type; (void)shapes;
        return false;
    }

    /**
     * @brief 返回该 kernel 的编译优化级别 (0=O0, 1=O1, 2=O2, 3=O3, 4=Ofast)
     */
    [[nodiscard]] virtual int optLevel() const { return opt_level_; }

protected:
    int opt_level_ = 3;
};

/**
 * @class ProfiledCompiledKernel
 * @brief PGO 性能分析包装器：装饰 CompiledKernel 并自动记录每次 execute 耗时
 * @details 透明代理模式，所有方法委托给内部 kernel，仅 execute() 额外计时代码。
 *          当 CompileOptions::enable_profiling 为 true 时，C3Engine 自动包装。
 */
class ProfiledCompiledKernel : public CompiledKernel {
public:
    ProfiledCompiledKernel(std::shared_ptr<CompiledKernel> inner,
                           std::shared_ptr<ProfileData> data)
        : inner_(std::move(inner)), data_(std::move(data)) {}

    std::vector<Tensor> execute(const std::vector<Tensor>& inputs) override {
        auto start = std::chrono::steady_clock::now();
        auto result = inner_->execute(inputs);
        auto end = std::chrono::steady_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        data_->record(ns);
        return result;
    }

    [[nodiscard]] const std::string& cacheKey() const override { return inner_->cacheKey(); }
    [[nodiscard]] DeviceType targetDevice() const override { return inner_->targetDevice(); }
    [[nodiscard]] size_t workspaceBytes() const override { return inner_->workspaceBytes(); }

    bool installIntoRegistry(op op_type, const KernelShapeInfo& shapes) override {
        return inner_->installIntoRegistry(op_type, shapes);
    }

    /** @brief 获取内部 kernel 的引用 */
    CompiledKernel& inner() { return *inner_; }
    /** @brief 获取 profile 数据 */
    const ProfileData& profileData() const { return *data_; }

private:
    std::shared_ptr<CompiledKernel> inner_;
    std::shared_ptr<ProfileData> data_;
};

/// 异步编译结果的 future 类型（shared_future 支持多等待者）
using CompileFuture = std::shared_future<std::shared_ptr<CompiledKernel>>;

/**
 * @class C3Engine
 * @brief CTorch JIT 编译引擎
 * @details 单例类，负责将 Graph 编译为 CompiledKernel，并维护编译产物缓存。
 *          线程安全：所有公共方法内部同步。
 */
class C3Engine {
public:
    /** @brief 获取 C3Engine 单例实例 */
    static C3Engine& getInstance();

    /**
     * @brief 编译计算图
     * @param graph 待编译的计算图
     * @param options 编译选项
     * @return 编译后的 kernel；若缓存命中则直接返回缓存产物
     * @throw std::runtime_error 当图不合法或目标设备不支持时抛出
     */
    std::shared_ptr<CompiledKernel> compile(const Graph& graph, const CompileOptions& options = {});

    /**
     * @brief 异步编译计算图（非阻塞，立即返回 future）
     * @param graph 待编译的计算图
     * @param options 编译选项
     * @return CompileFuture，调用方可通过 .get() / .wait() 获取编译产物
     * @details 编译在后台线程中执行。若同一 cache key 已有编译任务进行中，
     *          则返回同一个 future（去重）。编译完成后自动写入缓存并热替换。
     *          编译失败时 future 中存储 nullptr，不抛异常。
     */
    CompileFuture compileAsync(const Graph& graph, const CompileOptions& options = {});

    /**
     * @brief 并行编译多个独立子图
     * @param graphs 待编译的子图列表（各子图必须独立，无数据依赖）
     * @param options 编译选项
     * @return 编译后的 kernel 列表，顺序与 graphs 一致
     * @details 内部使用 std::async 并行编译所有子图，所有子图编译完成后返回。
     *          每个子图独立走 compile() 流程（独立优化、缓存检查、编译）。
     *          适用于 MLP 各层独立编译等场景，显著减少冷启动总编译时间。
     * @throw std::runtime_error 当任一子图编译失败时抛出
     */
    std::vector<std::shared_ptr<CompiledKernel>> compileParallel(
        const std::vector<Graph>& graphs,
        const CompileOptions& options = {});

    // ======================= compileMerged 系列：多层融合编译 =======================

    /**
     * @brief 多子图融合编译：将 N 个独立子图按链接规格合并为单个图后统一编译
     * @param sub_graphs 待融合的子图列表（至少 1 个，N>=1）
     * @param spec 子图之间的链接规格
     * @param options 编译选项（enable_fusion 会触发额外的图内优化）
     * @return 编译后的单 kernel（封装整个融合图）；若 sub_graphs.size()==1 则等价于 compile()
     * @details 这是"逐层编译 + 后台全图融合"模式的核心入口：
     *          1. 调用 GraphMerger::merge() 合并子图为单个 Graph
     *          2. 复用 compile() 的完整流程：canonicalize → eliminateDeadCode → fuse → cache → compile
     *          3. 合并后的图 hash 作为 cache key，因此子图分别缓存后再次融合会命中同一 cache
     *
     *          典型用法（MLP 全图融合）：
     *          @code
     *          std::vector<Graph> layers = {layer1, layer2, layer3};
     *          auto fused = engine.compileMerged(layers, MergeSpec{}); // 纯顺序链接
     *          auto out = fused->execute({input});
     *          @endcode
     *
     * @throw std::invalid_argument 当 sub_graphs 为空或 spec 与子图数量不匹配时
     * @throw std::runtime_error 当图不合法、链接不兼容或编译失败时
     */
    std::shared_ptr<CompiledKernel> compileMerged(
        const std::vector<Graph>& sub_graphs,
        const MergeSpec& spec,
        const CompileOptions& options = {});

    /**
     * @brief 简化版：纯顺序链接的多子图融合编译（MLP 等典型场景）
     * @param sub_graphs 子图列表，每层要求单输入单输出
     * @param options 编译选项
     * @return 编译后的单 kernel
     * @details 等价于 compileMerged(sub_graphs, GraphMerger::makeSequentialSpec(sub_graphs), options)
     */
    std::shared_ptr<CompiledKernel> compileMergedSequential(
        const std::vector<Graph>& sub_graphs,
        const CompileOptions& options = {});

    /**
     * @brief 异步版多子图融合编译
     * @param sub_graphs 待融合的子图列表
     * @param spec 链接规格
     * @param options 编译选项
     * @return CompileFuture，可通过 .get() 获取编译产物
     * @details 在后台线程中执行 merge + compile。返回的 future 与对同一 (sub_graphs+spec) 后续调用
     *          共享（去重），适用于多层网络冷启动时同时发起逐层编译和全图编译的场景。
     *
     *          **生命周期要求**：后台 std::async 任务依赖 C3Engine 单例持有的 EngineState。
     *          编译器失败时异常通过 future.get() 传播（不静默吞错）。
     *          **调用方应在 main() 退出前调用 `C3Engine::getInstance().shutdown()`**
     *          以等待所有后台编译完成，避免 C3Engine 单例析构时与后台线程产生
     *          mutex 析构顺序冲突。
     */
    CompileFuture compileMergedAsync(
        const std::vector<Graph>& sub_graphs,
        const MergeSpec& spec,
        const CompileOptions& options = {});

    /**
     * @brief 多子图融合编译 + PGO 三层异步升级（"逐层编译 + 后台全图融合"模式核心入口）
     * @param sub_graphs 待融合的子图列表
     * @param spec 链接规格
     * @param options 编译选项（会强制启用 pgo_mode=true，opt_level/opt backend 等其他字段透传）
     * @return 编译后的 PGOCompiledKernel（merged graph 版本）
     * @details 典型工作流（多层网络如 MLP）：
     *          1. 客户端逐层调用 compile(layer_i, opts) 编译每个子图（首屏快）
     *          2. 同时调用 compileMergedPGO(layers, spec) 启动全图 PGO 编译链
     *          3. 首次 execute(merged_kernel, {x, w1, b1, ..., wN, bN})：
     *             - PGO 包装器内部用 Eager 调度器逐算子执行 merged_graph（不阻塞）
     *             - 同时触发 O2 + Ofast 异步编译
     *          4. O2 编译完成：下一次 execute 原子切换到 O2 merged kernel
     *          5. Ofast 编译完成：下一次 execute 原子切换到 Ofast merged kernel
     *          6. 通过 isPromoted() / o2Kernel() / ofastKernel() 监控升级状态
     *
     *          缓存语义：merged graph 拥有独立的 cache key（基于 mergedCacheKey + 编译维度），
     *          与子图独立缓存的条目互不冲突；同一 (sub_graphs+spec) 重复调用会命中缓存。
     *
     * @throw std::invalid_argument 当 sub_graphs 为空或 spec 与子图数量不匹配时
     * @throw std::runtime_error 当图不合法、链接不兼容或编译失败时
     */
    std::shared_ptr<CompiledKernel> compileMergedPGO(
        const std::vector<Graph>& sub_graphs,
        const MergeSpec& spec,
        const CompileOptions& options = {});

    /**
     * @brief 简化版：纯顺序链接的多子图融合 + PGO 异步升级（MLP 典型场景）
     * @param sub_graphs 子图列表，每层要求单输入单输出
     * @param options 编译选项
     * @return PGOCompiledKernel（merged graph 版本）
     */
    std::shared_ptr<CompiledKernel> compileMergedPGOSequential(
        const std::vector<Graph>& sub_graphs,
        const CompileOptions& options = {});

    /**
     * @brief 编译计算图并自动安装到 C3 内核注册表
     * @param graph 待编译的计算图
     * @param options 编译选项
     * @return 编译后的 kernel
     * @details 编译后的 kernel 自动注册到 C3KernelRegistry，下次调度器 dispatch
     *          相同操作时自动使用 C3 kernel 而非 Eager kernel。
     *          支持单节点图（Add/Mul/MatMul/Neg/ReLU/Sigmoid/Tanh）自动注入。
     *          对于融合节点（FusedNode）或多节点图，调用方需手动指定 op_type。
     * @throw std::runtime_error 当图不合法或无法推断 op_type 时抛出
     */
    std::shared_ptr<CompiledKernel> compileAndInject(
        const Graph& graph, const CompileOptions& options = {});

    // ======================= traceAndInject 系列 =======================

    /**
     * @brief 追踪表达式 → 编译 → 注入注册表，一站式完成
     * @tparam F 可调用对象类型（接收 ProxyTensor 并返回 ProxyTensor）
     * @param fn 表达式 lambda，如 [](auto& x) { return x.relu(); }
     * @param desc 输入张量描述符
     * @param options 编译选项
     * @return 编译后的 kernel（已注入调度器注册表）
     * @details traceAndInject 将 Tracer::trace() + compile() + inject() 合并为一步。
     *          使用方式：
     *          @code
     *          auto kernel = engine.traceAndInject(
     *              [](auto& x) { return x.relu(); }, desc);
     *          auto result = kernel->execute({input_tensor});
     *          @endcode
     */
    template <typename F>
    std::shared_ptr<CompiledKernel> traceAndInject(
        F&& fn, const TensorDesc& desc,
        const CompileOptions& options = {}) {
        auto graph = Tracer::trace(std::forward<F>(fn), desc);
        return compileAndInject(graph, options);
    }

    /// 双输入版本
    template <typename F>
    std::shared_ptr<CompiledKernel> traceAndInject(
        F&& fn, const TensorDesc& a_desc, const TensorDesc& b_desc,
        const CompileOptions& options = {}) {
        auto graph = Tracer::trace(std::forward<F>(fn), a_desc, b_desc);
        return compileAndInject(graph, options);
    }

    /// 三输入版本（适合 FC 层：input, weight, bias）
    template <typename F>
    std::shared_ptr<CompiledKernel> traceAndInject(
        F&& fn, const TensorDesc& a_desc, const TensorDesc& b_desc,
        const TensorDesc& c_desc,
        const CompileOptions& options = {}) {
        auto graph = Tracer::trace(std::forward<F>(fn), a_desc, b_desc, c_desc);
        return compileAndInject(graph, options);
    }

    /**
     * @brief 异步追踪表达式 → 编译 → 注入（非阻塞）
     * @tparam F 可调用对象类型
     * @param fn 表达式 lambda
     * @param desc 输入张量描述符
     * @param options 编译选项
     * @return CompileFuture，可通过 .get() 获取编译后的 kernel
     * @details 编译在后台线程中执行，主线程可继续执行其他工作。
     *          编译完成后自动注入调度器注册表。
     */
    template <typename F>
    CompileFuture traceAndInjectAsync(
        F&& fn, const TensorDesc& desc,
        const CompileOptions& options = {}) {
        auto graph = Tracer::trace(std::forward<F>(fn), desc);
        return compileAsync(graph, options);
    }

    /// 双输入异步版本
    template <typename F>
    CompileFuture traceAndInjectAsync(
        F&& fn, const TensorDesc& a_desc, const TensorDesc& b_desc,
        const CompileOptions& options = {}) {
        auto graph = Tracer::trace(std::forward<F>(fn), a_desc, b_desc);
        return compileAsync(graph, options);
    }

    /// 三输入异步版本
    template <typename F>
    CompileFuture traceAndInjectAsync(
        F&& fn, const TensorDesc& a_desc, const TensorDesc& b_desc,
        const TensorDesc& c_desc,
        const CompileOptions& options = {}) {
        auto graph = Tracer::trace(std::forward<F>(fn), a_desc, b_desc, c_desc);
        return compileAsync(graph, options);
    }

    /**
     * @brief 根据缓存键获取已编译 kernel
     * @param cache_key 缓存键
     * @return 对应 kernel；若不存在则返回 nullptr
     */
    [[nodiscard]] std::shared_ptr<CompiledKernel> getKernel(const std::string& cache_key) const;

    /** @brief 查询当前缓存状态 */
    [[nodiscard]] C3CacheStats getCacheStats() const;

    /**
     * @brief 根据缓存键获取 PGO profile 数据
     * @param cache_key 缓存键
     * @return 对应 profile 数据的共享指针；若 profiling 未启用或键不存在则返回 nullptr
     * @details 返回的 ProfileData 指针在 kernel 生命周期内有效。
     *          调用方可通过 ProfileData 的原子接口安全读取统计信息。
     */
    [[nodiscard]] std::shared_ptr<ProfileData> getProfileData(const std::string& cache_key) const;

    /**
     * @brief 获取最近一次编译失败的错误信息（覆盖所有编译路径：sync / async / PGO / Merged）
     * @return 错误信息字符串；编译成功或引擎刚启动时返回空字符串
     * @details 线程安全：内部 mutex 保护。
     *          该 API 用于诊断"silent fail"问题：以前编译失败只 log warning，调用方无法查询。
     *          现在所有编译失败都会记录到 EngineState.last_compile_error_，
     *          调用方可通过本 API 显式查询。
     *
     *          错误信息可能包含 tier 前缀：
     *          - `[o2] ...`：PGO O2 编译失败
     *          - `[ofast] ...`：PGO Ofast 编译失败
     *          - `[merge] ...`：GraphMerger 合并失败
     *          - `[async] ...`：compileAsync 后台编译失败
     *          - 其他：sync 编译失败（一般通过异常传播，调用方可能 catch 后调用此 API）
     *
     *          错误信息最大 1KB，超出截断（避免 OOM）。
     */
    [[nodiscard]] std::string getLastCompileError() const;

    /**
     * @brief 显式清空 last_compile_error_ 状态
     * @details 主要用于测试或重试场景。
     *          注意：不会撤销已发生的失败（kernel 仍是 nullptr / disabled 状态），
     *          仅清空错误消息以便后续观察。
     */
    void clearLastCompileError();

    /**
     * @brief 记录编译错误到全局 last_compile_error_（供内部子系统使用，如 PGOCompiledKernel）
     * @param prefix 错误来源前缀，如 "o2"、"ofast"、"async-merge"
     * @param err 错误信息字符串
     * @details 线程安全：内部 mutex 保护。
     *          这是内部 callback hook，主要被 PGOCompiledKernel::recordCompileError() 调用，
     *          让 PGO O2/Ofast 编译失败也能被 C3Engine::getLastCompileError() 查到。
     *          错误信息最大 1KB，超出截断。
     */
    void recordCompileError(const std::string& prefix, const std::string& err);

    /**
     * @brief 查询最近一次 async compile 是否因 watchdog 超时返回 nullptr（P0-3 修复）
     * @param cache_key 待查询的 cache key（与 compileAsync 调用对应的图编译产生的 key）
     * @return true 表示曾因 timeout 返回 nullptr；false 表示正常完成 / 未触发 / 缓存已覆盖
     * @details 背景：watchdog 超时后调用方 `future.get()` 立即拿到 nullptr，但**实际编译线程
     *          仍在后台跑**（clang++/MLIR 不可取消），跑完后会写入 cache 供后续命中。
     *          用户体感上"首次失败、二次莫名 hit"——加这个 API 让调用方能区分 nullptr 的原因：
     *          - 本 API 返回 true：本次是 timeout，未来同 key 应当直接命中 cache
     *          - 本 API 返回 false：nullptr 是真的编译失败
     *
     *          实现：通过解析 `last_compile_error_` 中 `[async-timeout] compile exceeded Xms for <cache_key>`
     *          格式（watchdog L761 写入），找到匹配的 cache_key 即返回 true。
     *          注意：last_compile_error_ 是**全局最近一次**错误，可能被后续编译覆盖，**仅在
     *          future.get() 返回 nullptr 之后立即查询才可靠**。
     */
    [[nodiscard]] bool wasAsyncCompileTimedOut(const std::string& cache_key) const;

    /**
     * @brief 设置异步编译超时（毫秒），默认 30000ms (30s)
     * @param ms 超时毫秒数，0 表示永不超时（不推荐，可能 thread pool 永远增长）
     * @details 仅对 compileAsync 路径生效（同步 compile() 仍然由调用方控制）。
     *          超时行为（ADR-011）：
     *          - watchdog 线程在 timeout_ms 内没拿到 kernel → 视为超时
     *          - 用户 future 立即返回 nullptr
     *          - last_compile_error_ 记录 "[async-timeout] compile exceeded Xms for <cache_key>"
     *          - **实际 compile 线程继续跑**（无法取消 clang++/MLIR），跑完后写入 cache
     *            供后续相同 cache_key 命中
     *
     *          适用场景：冷启动时多图并发编译，其中某张图触发 MLIR 复杂优化卡死，
     *          watchdog 防止主线程 / shutdown() 永远等待。
     *
     *          调优建议：
     *          - 开发环境：60s（避免被合法大图误杀）
     *          - 生产环境：10-15s（快速失败，依赖 cache 复用）
     *          - 测试环境：2-3s（让超时测试可以快速跑完）
     */
    void setCompileTimeoutMs(uint32_t ms);

    /**
     * @brief 获取当前异步编译超时配置
     */
    [[nodiscard]] uint32_t getCompileTimeoutMs() const;

    // ======================= AOT (Ahead-Of-Time) 持久化 =======================
    //
    // [removed 2026-08-15] AOTCache（.so 磁盘缓存）已删除，详见 STATUS_CONTEXT。
    // 跨进程复用改由 JITCache（LLVM bitcode 磁盘缓存）承担，路径：
    //   LinalgElementwiseGen / MLIRKernelGen → JITCache（JIT 编译产物的磁盘版）。
    //
    // ======================= 编译缓存管理 =======================

    /** @brief 清空全部编译缓存（不取消进行中的异步编译） */
    void clearCache();

    /**
     * @brief 运行自动调优，搜索当前机器最优 MatMul 分块参数
     * @details 使用 QEA 量子启发算法在搜索空间中寻找最优 TILE_M/N/K/unroll 组合。
     *          调优结果写入 TuningState，后续所有编译自动使用最优参数。
     *          仅需调用一次；重复调用会跳过（已调优检查）。
     * @param config 调优配置（可选，默认使用 AutoTunerConfig 默认值）
     */
    void autoTune(const struct AutoTunerConfig& config);

    /**
     * @brief 等待所有后台编译完成并回收线程资源
     * @details 应在程序退出前调用，确保所有异步编译任务完成。
     *          带 30 秒超时（覆盖长编译/链接场景），超时后会 abandon 未完成 future
     *          并记录 WARN。
     *
     *          P0-2 修复：析构函数也会自动调用 shutdown()，调用方**不再需要**显式调用。
     *          EngineState 现在是 C3Engine 的 std::unique_ptr 成员（不再是函数内 static），
     *          析构顺序由 C++ 类成员声明顺序决定：state_ 必须在 C3Engine 自身销毁之后才释放，
     *          因此析构函数调 shutdown() 安全（state_ 仍存活，async task 可被 join）。
     *          老代码中"不要在 main 末尾调 shutdown() 的劝告"已废弃——可以调也可以不调。
     */
    void shutdown();

    /// P0-2 修复：返回内部 EngineState 引用（static member）。
    /// 设为 static 的关键原因：让 C3Engine.cpp 内的 33 个 unqualified getState() 调用点
    /// （其中大量在 std::async lambda 内部）无需 capture `this` 即可工作。static member 在
    /// unqualified lookup 中是 class-scope 名字，调用时不需要 `this`，因此 lambda 写
    /// `[...]() { getState(); }` 即可，不需要 `[..., this]() { this->getState(); }`。
    /// 路径：C3Engine::getState() → C3Engine::getInstance().state_。
    /// 前提：state_ 在 C3Engine() 构造时已初始化。
    static EngineState& getState();

private:
    C3Engine();
    // P0-2 修复：自定义析构函数 = 调 shutdown() 等待后台任务，然后 unique_ptr 自动释放 state_
    ~C3Engine();
    C3Engine(const C3Engine&) = delete;
    C3Engine& operator=(const C3Engine&) = delete;

    /// P0-2 修复：EngineState 由 C3Engine 拥有，析构顺序可控
    /// 之前是函数内 static EngineState + Meyers C3Engine 单例，跨 TU 析构顺序未定义 → SIGABRT
    /// 现在 std::unique_ptr 成员在 C3Engine 析构时**最后**释放（unique_ptr 析构在 C3Engine 自身之后）
    std::unique_ptr<EngineState> state_;
};

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_JITENGINE_H
