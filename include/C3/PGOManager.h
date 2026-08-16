/**
 * @file PGOManager.h
 * @brief PGO (Profile-Guided Optimization) 三层异步编译流水线
 * @details 实现 CaaS（Compilation as a Service）架构：
 *
 *          三层流水线（第一次调用即触发）：
 *          Tier 1 (Eager): 第一次执行 kernel，通过 Eager 调度器解释执行，零延迟
 *          Tier 2 (O2):    异步编译 O2 优化级别的 kernel，完成后透明替换
 *          Tier 3 (Ofast): O2 编译完成后自动触发 Ofast 异步编译，完成后透明替换
 *
 *          编译队列管理（博弈论 + 反馈系统跨域模式）：
 *          - 优先级队列：按 kernel 热度评分分配编译资源
 *          - Anti-Windup 背压：队列超过阈值时降级到 Eager 直通
 *
 *          对调用方完全透明——PGOCompiledKernel 看起来就像普通 CompiledKernel。
 * @date 2026/08/02
 */

#ifndef CTORCH_C3_PGO_MANAGER_H
#define CTORCH_C3_PGO_MANAGER_H

#include "C3Engine.h"
#include "Graph.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace ct {
namespace c3 {

// 前向声明
class C3Engine;

/**
 * @struct PGOConfig
 * @brief PGO 编译流水线配置
 */
struct PGOConfig {
    /** @brief 最大并发编译任务数（默认 4） */
    uint64_t max_concurrent_compilations = 4;
    /** @brief 编译队列背压阈值（默认 32），超过后新请求降级为 Eager 直通 */
    uint64_t queue_backpressure_threshold = 32;
    /** @brief 热度评分采用的调用次数窗口（默认 1000 次） */
    uint64_t heat_score_window_calls = 1000;
    /** @brief 是否启用异步编译（默认 true） */
    bool async_compilation = true;
};

/**
 * @struct CompilationTask
 * @brief 编译任务描述符，用于优先级队列
 */
struct CompilationTask {
    double priority;                   ///< 热度评分（越高越优先）
    std::function<void()> task;        ///< 编译任务
    std::chrono::steady_clock::time_point created_at; ///< 创建时间
    std::string description;           ///< 任务描述

    /** @brief 优先级队列比较（最高优先级的先出队） */
    bool operator<(const CompilationTask& other) const {
        return priority < other.priority;
    }
};

/**
 * @class PGOCompiledKernel
 * @brief PGO 三层异步编译 kernel
 *
 * 执行流程：
 * 1. 第一次调用：Eager 解释执行 + 立即触发异步编译链
 * 2. O2 编译完成后：透明切换到 O2 kernel，同时自动触发 Ofast 编译
 * 3. Ofast 编译完成后：透明切换到 Ofast kernel
 *
 * 优先级：Ofast > O2 > Eager，编译完成后下一调用自动使用更高优化级别。
 */
class PGOCompiledKernel : public CompiledKernel, public std::enable_shared_from_this<PGOCompiledKernel> {
    // =================== Test-only access (ADR-006 deopt PoC) ===================
    // 仅 PoC 测试用：允许测试代码直接注入 mock kernel 模拟运行时崩溃。
    // 正式版本应保留此 API 以便 fuzz / chaos 测试。
    friend class PGOCompiledKernelTestAccess;

public:
    PGOCompiledKernel(
        const Graph& graph,
        CompileOptions options,
        std::string cache_key,
        std::shared_ptr<ProfileData> profile_data,
        C3Engine& engine);

    ~PGOCompiledKernel() override = default;

    std::vector<Tensor> execute(const std::vector<Tensor>& inputs) override;

    [[nodiscard]] const std::string& cacheKey() const override { return cache_key_; }
    [[nodiscard]] DeviceType targetDevice() const override { return options_.target_device; }
    [[nodiscard]] size_t workspaceBytes() const override { return 0; }
    bool installIntoRegistry(op op_type, const KernelShapeInfo& shapes) override;

    /** @brief 检查是否已有 O2 或 Ofast kernel */
    [[nodiscard]] bool isPromoted() const {
        return ofast_kernel_ != nullptr || o2_kernel_ != nullptr;
    }

    /** @brief 获取 O2 kernel（可能为 nullptr） */
    [[nodiscard]] std::shared_ptr<CompiledKernel> o2Kernel() const {
        std::lock_guard<std::mutex> lock(compile_mutex_);
        return o2_kernel_;
    }

    /** @brief 获取 Ofast kernel（可能为 nullptr） */
    [[nodiscard]] std::shared_ptr<CompiledKernel> ofastKernel() const {
        std::lock_guard<std::mutex> lock(compile_mutex_);
        return ofast_kernel_;
    }

    /** @brief 获取 profile 数据 */
    [[nodiscard]] const ProfileData& profileData() const { return *profile_data_; }

    // =================== Deoptimization (ADR-006) ===================

    /** @brief 总 deopt 次数（O2 或 Ofast 运行时抛异常 + disable） */
    [[nodiscard]] uint64_t deoptCount() const {
        return deopt_count_.load(std::memory_order_relaxed);
    }

    /** @brief 最近一次 deopt 的原因（含 tier 标签，如 "ofast: ..." / "o2: ..."） */
    [[nodiscard]] const std::string& lastDeoptReason() const {
        std::lock_guard<std::mutex> lock(deopt_mutex_);
        return last_deopt_reason_;
    }

    /** @brief Ofast kernel 是否已被 deopt 禁用（true = 永久回退到 O2 或 Eager） */
    [[nodiscard]] bool isOfastDisabled() const {
        return ofast_disabled_.load(std::memory_order_acquire);
    }

    /** @brief O2 kernel 是否已被 deopt 禁用（true = 永久回退到 Eager） */
    [[nodiscard]] bool isO2Disabled() const {
        return o2_disabled_.load(std::memory_order_acquire);
    }

    /** @brief 强制触发编译链（供 PGOManager::promoteAll 使用） */
    void promote();

    /** @brief 计算热度评分（0.0 ~ 1.0），用于编译优先级排序 */
    [[nodiscard]] double computeHeatScore() const;

    // =================== Compile Error Observability (ADR-007) ===================

    /**
     * @brief 获取该 kernel 最近一次编译失败原因（含 tier 前缀，如 "o2: ..." / "ofast: ..."）
     * @return 错误信息字符串；无失败时返回空字符串
     * @details 与 lastDeoptReason() 区别：
     *          - lastDeoptReason() 记录**运行时**失败（kernel execute 抛异常）
     *          - lastCompileError() 记录**编译时**失败（compileO2/Ofast 失败）
     *
     *          线程安全：内部 mutex 保护。
     *          编译成功时不会自动清空（用户可调用 clearLastCompileError() 重置）。
     */
    [[nodiscard]] const std::string& lastCompileError() const {
        std::lock_guard<std::mutex> lock(compile_error_mutex_);
        return last_compile_error_;
    }

    /** @brief 显式清空 last_compile_error_ */
    void clearLastCompileError() {
        std::lock_guard<std::mutex> lock(compile_error_mutex_);
        last_compile_error_.clear();
    }

private:
    /** @brief Tier 1：Eager 解释执行图节点 */
    std::vector<Tensor> executeInterpreted(const std::vector<Tensor>& inputs);

    /** @brief 触发异步编译链（O2 → Ofast） */
    void triggerCompilationChain();

    /** @brief 编译 O2 级别 kernel */
    void compileO2();

    /** @brief 编译 Ofast 级别 kernel（在 O2 完成后调用） */
    void compileOfast();

    /** @brief 解释执行 FusedNode */
    Tensor executeFusedNodeInterpreted(const FusedNode& fnode,
                                        const std::unordered_map<size_t, Tensor>& values);

    /** @brief 记录一次 deopt 事件（原子 +1 计数 + 加锁更新原因） */
    void recordDeopt(const char* tier, const std::string& reason);

    /** @brief 记录一次编译错误（带 tier 前缀，更新 last_compile_error_） */
    void recordCompileError(const char* tier, const std::string& reason);

    Graph graph_;
    CompileOptions options_;
    std::string cache_key_;
    std::shared_ptr<ProfileData> profile_data_;
    C3Engine& engine_;

    mutable std::mutex compile_mutex_;
    std::shared_ptr<CompiledKernel> o2_kernel_;      ///< O2 编译结果
    std::shared_ptr<CompiledKernel> ofast_kernel_;    ///< Ofast 编译结果

    std::atomic<bool> compilation_triggered_{false};  ///< 是否已触发编译链

    // =================== Deoptimization 状态 (ADR-006) ===================
    // 运行时失败（kernel execute 抛异常）→ 自动 disable 永久回退到下一级
    std::atomic<bool> ofast_disabled_{false};        ///< Ofast 是否已被 deopt
    std::atomic<bool> o2_disabled_{false};            ///< O2 是否已被 deopt
    std::atomic<uint64_t> deopt_count_{0};            ///< 总 deopt 次数
    mutable std::mutex deopt_mutex_;                  ///< 保护 last_deopt_reason_
    std::string last_deopt_reason_;                   ///< 最近一次 deopt 原因

    // =================== Compile Error 状态 (ADR-007) ===================
    // 编译失败（compileO2/Ofast 失败）→ 记录到 last_compile_error_（独立 mutex，
    // 避免与 deopt_mutex_ 互锁）
    mutable std::mutex compile_error_mutex_;
    std::string last_compile_error_;
};

/**
 * @class PGOManager
 * @brief PGO 管理器，管理编译队列和资源
 *
 * 基于跨域模式设计：
 * - 博弈论 VCG 机制：按热度评分分配编译资源
 * - 反馈系统 Anti-Windup：编译队列背压防止积压
 */
class PGOManager {
public:
    static PGOManager& getInstance();

    /** @brief 注册一个 PGO 管理的 kernel */
    std::shared_ptr<PGOCompiledKernel> registerKernel(
        const Graph& graph,
        const CompileOptions& options,
        std::string cache_key,
        std::shared_ptr<ProfileData> profile_data,
        C3Engine& engine);

    /** @brief 获取配置 */
    PGOConfig& config() { return config_; }
    const PGOConfig& config() const { return config_; }

    /** @brief 启用/禁用 PGO */
    void setEnabled(bool enabled) { enabled_.store(enabled, std::memory_order_release); }
    bool isEnabled() const { return enabled_.load(std::memory_order_acquire); }

    /** @brief 编译队列管理 */
    bool canAcceptCompilation() const;
    void notifyCompilationStarted();
    void notifyCompilationCompleted();

    /** @brief 记录队列背压拒绝 */
    void recordQueueRejection() { total_queue_rejections_++; }

    /** @brief 处理优先级队列中的下一个任务 */
    void processQueue();

    /** @brief 清除所有注册的 kernel（主要用于测试） */
    void clear();

    /** @brief 统计信息 */
    struct Stats {
        size_t total_registered = 0;
        size_t o2_ready = 0;
        size_t ofast_ready = 0;
        size_t pending = 0;
        uint64_t active_compilations = 0;
        uint64_t queue_rejections = 0;
    };
    Stats getStats() const;

    /** @brief 强制所有待编译的 kernel 立即编译 */
    void promoteAll();

    /**
     * @brief 等待所有后台 PGO 编译任务完成
     * @details 必须由用户在 main() 退出前显式调用。
     *          后台 PGO 编译通过 std::async 启动，若 main() 退出前未等待，
     *          单例析构后线程继续运行会 lock 已析构的 mutex 导致 UAF。
     *          调用方应在 C3Engine::shutdown() 之后或之前调用本方法。
     */
    void shutdown();

    /** @brief 获取队列互斥锁（PGOCompiledKernel 需要访问） */
    std::mutex& queue_mutex();

    /** @brief 获取优先级队列（PGOCompiledKernel 需要访问） */
    std::priority_queue<CompilationTask>& task_queue();

    /** @brief 后台编译 future 的互斥锁（PGOCompiledKernel::triggerCompilationChain 需要） */
    std::mutex& futures_mutex();

    /** @brief 后台编译 future 列表（PGOCompiledKernel::triggerCompilationChain 需要） */
    std::vector<std::future<void>>& compile_futures();
private:
    PGOManager() = default;
    ~PGOManager() { shutdown(); }  // RAII：析构时自动等待后台编译完成

    mutable std::mutex mutex_;
    // entries_ 持有 shared_ptr 以保持 kernel 存活（类似 C3Engine cache 的语义）。
    // 使用 shared_ptr 确保 PGOManager 注册的 kernel 在整个程序运行期间可用，
    // 避免 kernel 被销毁后 weak_ptr 过期导致 entries_.size 不准。
    std::vector<std::shared_ptr<PGOCompiledKernel>> entries_;
    // PGO 缓存：相同 cache_key 复用同一 PGOCompiledKernel 实例，
    // 保证重复调用 compileMergedPGO/compileMergedPGOSequential 返回相同对象，
    // 同时 profile_data 在多次调用间累计。
    std::unordered_map<std::string, std::shared_ptr<PGOCompiledKernel>> cache_;
    PGOConfig config_;
    std::atomic<bool> enabled_{false};

    // 编译队列状态
    mutable std::mutex queue_mutex_;
    std::atomic<uint64_t> active_compilations_{0};
    uint64_t total_queue_rejections_{0};

    // 优先级队列（未来扩展：实际编译任务调度）
    std::priority_queue<CompilationTask> task_queue_;

    // 后台 PGO 编译任务 future 列表（用于 shutdown 等待）
    std::mutex futures_mutex_;
    std::vector<std::future<void>> compile_futures_;
};

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_PGO_MANAGER_H