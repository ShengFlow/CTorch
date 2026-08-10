/**
 * @file QuorumManager.h
 * @brief Quorum NRW 管理器 — 继承 Gen 1 Quorum 的后端感知 Quorum 管理
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 * @version v0.1 (Gen 2 BANT)
 *
 * @details 本文件实现 QuorumManager，是 Gen 1 Quorum NRW（读/写 Quorum）
 *          在 Gen 2 中的独立实现和增强。
 *
 *          设计原则（分布式数据库理论）：
 *          1. NRW 模型：N 个副本，W 个写确认，R 个读确认
 *          2. 强一致性：W + R > N
 *          3. 后端覆盖 Quorum：需要至少 K 种不同后端参与
 *
 *          本模块与 GradientAggregator 解耦：
 *          QuorumManager 只管理 Quorum 决策（是否达到阈值），
 *          不参与梯度聚合计算。
 *          GradientAggregator 调用 QuorumManager 做决策，
 *          聚合计算由 GradientAggregator 自行完成。
 */

#ifndef CTORCH_DISTRIBUTED_QUORUM_MANAGER_H
#define CTORCH_DISTRIBUTED_QUORUM_MANAGER_H

#include "Ctools.h"
#include "CtorchError.h"
#include "DeviceBackend.h"

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <functional>

namespace ct {
namespace distributed {

/**
 * @brief Quorum 状态枚举
 */
enum class QuorumStatus : uint8_t {
    Pending,     ///< 等待更多节点提交
    Achieved,    ///< 已满足 Quorum 条件
    Timeout,     ///< 超时
    Failed,      ///< 无法达到 Quorum
};

/**
 * @struct QuorumRequest
 * @brief Quorum 请求
 */
struct QuorumRequest {
    uint64_t request_id;                    ///< 请求 ID
    size_t write_quorum;                    ///< 写 Quorum (W)
    size_t read_quorum;                     ///< 读 Quorum (R)
    size_t total_replicas;                  ///< 总副本数 (N)
    size_t backend_coverage_quorum;         ///< 后端覆盖 Quorum
    size_t current_acks;                    ///< 当前确认数
    size_t current_backend_coverage;        ///< 当前后端覆盖数
    std::unordered_set<DeviceType> covered_backends; ///< 已覆盖的后端类型
    std::chrono::steady_clock::time_point created_at; ///< 创建时间
    std::chrono::steady_clock::time_point deadline;   ///< 截止时间
    QuorumStatus status;                    ///< 当前状态
};

/**
 * @struct QuorumConfig
 * @brief Quorum 配置
 */
struct QuorumConfig {
    size_t default_write_quorum;            ///< 默认写 Quorum (W)
    size_t default_read_quorum;             ///< 默认读 Quorum (R)
    size_t default_backend_coverage;        ///< 默认后端覆盖数
    float quorum_timeout_ms;                ///< Quorum 超时时间 (ms)
    bool enable_backend_coverage;           ///< 是否启用后端覆盖
    bool enable_adaptive_quorum;            ///< 是否自适应 Quorum 大小
    float adaptive_quorum_factor;           ///< 自适应因子

    static QuorumConfig defaultConfig() {
        return QuorumConfig{
            2,          // default_write_quorum
            2,          // default_read_quorum
            1,          // default_backend_coverage
            5000.0f,    // quorum_timeout_ms (5s)
            true,       // enable_backend_coverage
            true,       // enable_adaptive_quorum
            0.75f       // adaptive_quorum_factor
        };
    }
};

/**
 * @struct QuorumStats
 * @brief Quorum 统计信息
 */
struct QuorumStats {
    size_t total_requests;                   ///< 总请求数
    size_t achieved_count;                   ///< 成功次数
    size_t timeout_count;                    ///< 超时次数
    size_t failed_count;                     ///< 失败次数
    double avg_achievement_time_ms;          ///< 平均达成时间 (ms)
    double avg_ack_count;                    ///< 平均确认数
    size_t backend_coverage_triggers;        ///< 后端覆盖触发次数

    void reset() {
        total_requests = 0;
        achieved_count = 0;
        timeout_count = 0;
        failed_count = 0;
        avg_achievement_time_ms = 0.0;
        avg_ack_count = 0.0;
        backend_coverage_triggers = 0;
    }
};

/**
 * @class QuorumManager
 * @brief Quorum NRW 管理器
 *
 * 管理梯度聚合的 Quorum 决策，提供：
 * 1. NRW 模型：N 总副本，W 写确认，R 读确认
 * 2. 后端覆盖 Quorum：至少 K 种不同后端参与
 * 3. 自适应 Quorum：根据活跃节点数动态调整 W/R
 * 4. 超时管理：超时后自动降级
 * 5. 请求生命周期管理
 *
 * 使用方式：
 *   1. createRequest() 创建 Quorum 请求
 *   2. recordAck() 记录每个节点确认
 *   3. checkStatus() 检查当前状态
 *   4. 当状态为 Achieved 时执行聚合
 *   5. 超时或失败时执行降级策略
 */
class QuorumManager {
public:
    /**
     * @brief 构造 Quorum 管理器
     * @param config Quorum 配置
     */
    explicit QuorumManager(QuorumConfig config = QuorumConfig::defaultConfig());

    ~QuorumManager() = default;

    // ======================= 请求生命周期 =======================

    /**
     * @brief 创建新的 Quorum 请求
     * @param total_replicas 总副本数 N
     * @param write_quorum 写 Quorum W（0 = 使用默认值）
     * @param backend_coverage_quorum 后端覆盖 Quorum（0 = 使用默认值）
     * @return 请求 ID
     *
     * 如果启用自适应 Quorum，W = ceil(N * adaptive_quorum_factor)。
     */
    uint64_t createRequest(size_t total_replicas,
                            size_t write_quorum = 0,
                            size_t backend_coverage_quorum = 0);

    /**
     * @brief 记录一个确认
     * @param request_id 请求 ID
     * @param node_id 确认节点 ID
     * @param backend_type 确认节点的后端类型
     * @return 当前 QuorumStatus
     *
     * 每次确认后自动检查是否达到 Quorum 条件。
     * 如果达到 → 状态变为 Achieved
     * 如果超时 → 状态变为 Timeout
     */
    QuorumStatus recordAck(uint64_t request_id, uint32_t node_id,
                            DeviceType backend_type);

    /**
     * @brief 检查请求状态
     * @param request_id 请求 ID
     * @return 当前状态
     *
     * 同时检查超时条件。
     */
    QuorumStatus checkStatus(uint64_t request_id) const;

    /**
     * @brief 获取请求信息
     * @param request_id 请求 ID
     * @return 请求信息（如果不存在返回 nullptr）
     */
    std::shared_ptr<const QuorumRequest> getRequest(uint64_t request_id) const;

    /**
     * @brief 移除请求
     * @param request_id 请求 ID
     */
    void removeRequest(uint64_t request_id);

    /**
     * @brief 清理所有超时请求
     * @return 被清理的请求数量
     */
    size_t cleanupTimedOut();

    // ======================= Quorum 决策 =======================

    /**
     * @brief 检查是否达到 Quorum
     * @param request_id 请求 ID
     * @return true 如果达到 Quorum
     *
     * 同时检查写 Quorum 和后端覆盖 Quorum。
     */
    bool hasQuorum(uint64_t request_id) const;

    /**
     * @brief 检查是否达到写 Quorum
     * @param current_acks 当前确认数
     * @param write_quorum 写 Quorum
     * @return true 如果 current_acks >= write_quorum
     */
    static bool hasWriteQuorum(size_t current_acks, size_t write_quorum) {
        return current_acks >= write_quorum;
    }

    /**
     * @brief 检查是否达到后端覆盖 Quorum
     * @param covered_backends 已覆盖的后端类型集合
     * @param backend_coverage_quorum 后端覆盖 Quorum
     * @return true 如果覆盖数 >= 后端覆盖 Quorum
     */
    static bool hasBackendCoverage(
        const std::unordered_set<DeviceType>& covered_backends,
        size_t backend_coverage_quorum) {
        return covered_backends.size() >= backend_coverage_quorum;
    }

    /**
     * @brief 计算自适应 Quorum
     * @param total_replicas 总副本数
     * @return 自适应写 Quorum
     *
     * W = ceil(N * factor)，其中 factor 是配置中的 adaptive_quorum_factor。
     * 保证 W >= 1 且 W <= N。
     */
    size_t computeAdaptiveQuorum(size_t total_replicas) const;

    /**
     * @brief 计算最小需要的 Quorum
     * @param total_replicas 总副本数
     * @return 最小写 Quorum（N/2 + 1，保证 W + R > N）
     */
    static size_t computeMinQuorum(size_t total_replicas) {
        return total_replicas / 2 + 1;
    }

    // ======================= 配置 =======================

    /**
     * @brief 设置 Quorum 配置
     * @param config Quorum 配置
     */
    void setConfig(const QuorumConfig& config) { _config = config; }

    /**
     * @brief 获取当前 Quorum 配置
     * @return Quorum 配置
     */
    const QuorumConfig& config() const { return _config; }

    // ======================= 统计信息 =======================

    /**
     * @brief 获取 Quorum 统计信息
     * @return 统计信息
     */
    QuorumStats stats() const { return _stats; }

    /**
     * @brief 重置统计信息
     */
    void resetStats() { _stats.reset(); }

private:
    QuorumConfig _config;
    QuorumStats _stats;
    mutable std::mutex _mtx;

    // 请求表
    std::unordered_map<uint64_t, QuorumRequest> _requests;

    // 请求 ID 生成器
    uint64_t _next_request_id;

    /**
     * @brief 检查请求是否超时
     * @param request 请求
     * @return true 如果已超时
     */
    bool isTimedOut(const QuorumRequest& request) const;
};

} // namespace distributed
} // namespace ct

#endif // CTORCH_DISTRIBUTED_QUORUM_MANAGER_H