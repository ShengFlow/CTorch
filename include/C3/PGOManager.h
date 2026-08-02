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
class PGOCompiledKernel : public CompiledKernel {
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

    /** @brief 强制触发编译链（供 PGOManager::promoteAll 使用） */
    void promote();

    /** @brief 计算热度评分（0.0 ~ 1.0），用于编译优先级排序 */
    [[nodiscard]] double computeHeatScore() const;

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

    Graph graph_;
    CompileOptions options_;
    std::string cache_key_;
    std::shared_ptr<ProfileData> profile_data_;
    C3Engine& engine_;

    mutable std::mutex compile_mutex_;
    std::shared_ptr<CompiledKernel> o2_kernel_;      ///< O2 编译结果
    std::shared_ptr<CompiledKernel> ofast_kernel_;    ///< Ofast 编译结果

    std::atomic<bool> compilation_triggered_{false};  ///< 是否已触发编译链
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

    /** @brief 获取队列互斥锁（PGOCompiledKernel 需要访问） */
    std::mutex& queue_mutex() { return queue_mutex_; }

    /** @brief 获取优先级队列（PGOCompiledKernel 需要访问） */
    std::priority_queue<CompilationTask>& task_queue() { return task_queue_; }

private:
    PGOManager() = default;

    mutable std::mutex mutex_;
    std::vector<std::weak_ptr<PGOCompiledKernel>> entries_;
    PGOConfig config_;
    std::atomic<bool> enabled_{false};

    // 编译队列状态
    mutable std::mutex queue_mutex_;
    std::atomic<uint64_t> active_compilations_{0};
    uint64_t total_queue_rejections_{0};

    // 优先级队列（未来扩展：实际编译任务调度）
    std::priority_queue<CompilationTask> task_queue_;
};

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_PGO_MANAGER_H