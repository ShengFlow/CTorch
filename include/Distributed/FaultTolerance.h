/**
 * @file FaultTolerance.h
 * @brief 异步容错模块 — 继承 Gen 1 VCAO 的 CRDT 状态恢复
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 * @version v0.1 (Gen 2 BANT)
 *
 * @details 本文件实现 FaultTolerance，是 Gen 1 VCAO（CRDT 异步容错）
 *          在 Gen 2 中的独立实现和增强。
 *
 *          设计原则：
 *          1. CRDT 语义：所有状态使用 Conflict-free Replicated Data Types
 *          2. 状态快照：定期保存状态快照用于故障恢复
 *          3. 分区恢复：网络分区恢复后自动合并状态
 *          4. 优雅降级：部分节点故障时系统继续运行
 *
 *          本模块与 DistributedOptimizer 解耦：
 *          FaultTolerance 管理 CRDT 状态的快照和恢复，
 *          DistributedOptimizer 使用 CRDTState 进行参数更新。
 *          FaultTolerance 不直接操作优化器逻辑。
 */

#ifndef CTORCH_DISTRIBUTED_FAULT_TOLERANCE_H
#define CTORCH_DISTRIBUTED_FAULT_TOLERANCE_H

#include "Ctools.h"
#include "CtorchError.h"
#include "Tensor.h"

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <functional>
#include <mutex>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>

namespace ct {
namespace distributed {

/**
 * @brief 容错策略枚举
 */
enum class FaultToleranceStrategy : uint8_t {
    BestEffort,     ///< 尽力而为：部分节点故障时继续训练
    Strict,         ///< 严格模式：任何节点故障触发全局暂停
    Graceful,       ///< 优雅降级：故障节点降权，其他节点继续
    CRDTAsync,      ///< CRDT 异步：完全异步，故障后 CRDT 合并恢复
};

/**
 * @brief 分区状态枚举
 */
enum class PartitionStatus : uint8_t {
    Connected,     ///< 正常连接
    Suspected,     ///< 可疑分区
    Confirmed,     ///< 确认分区
    Recovered,     ///< 分区恢复
};

/**
 * @struct CRDTSnapshot
 * @brief CRDT 状态快照
 */
struct CRDTSnapshot {
    uint64_t snapshot_id = 0;                    ///< 快照 ID
    std::chrono::steady_clock::time_point timestamp{}; ///< 快照时间
    std::vector<float> momentum;             ///< 动量向量 (LWW)
    std::vector<uint64_t> version_vector;    ///< 版本向量
    uint64_t global_step = 0;                    ///< 全局步数
    uint64_t local_step = 0;                     ///< 本地步数
    std::vector<uint64_t> grad_counter;      ///< 梯度计数器 (G-Counter)
    std::vector<uint8_t> serialized_params;  ///< 序列化的参数（可选）

    /**
     * @brief 序列化快照为字节流
     * @return 字节流
     */
    std::vector<uint8_t> serialize() const;

    /**
     * @brief 从字节流反序列化为快照
     * @param data 字节流
     * @return 反序列化后的快照
     */
    static CRDTSnapshot deserialize(const std::vector<uint8_t>& data);
};

/**
 * @struct PartitionEntry
 * @brief 分区记录
 */
struct PartitionEntry {
    uint32_t node_id;                        ///< 对端节点 ID
    PartitionStatus status;                  ///< 分区状态
    std::chrono::steady_clock::time_point detected_at; ///< 检测时间
    std::chrono::steady_clock::time_point recovered_at;///< 恢复时间
    CRDTSnapshot last_known_state;           ///< 最后已知状态
    size_t missed_heartbeats;                ///< 丢失心跳数
};

/**
 * @struct RecoveryPlan
 * @brief 恢复计划
 */
struct RecoveryPlan {
    uint32_t target_node_id;                 ///< 目标节点 ID
    CRDTSnapshot local_snapshot;             ///< 本地快照
    CRDTSnapshot remote_snapshot;            ///< 远程快照
    std::vector<uint64_t> merged_version;    ///< 合并后的版本向量
    uint64_t merged_global_step;             ///< 合并后的全局步数
    size_t param_bytes_to_sync;              ///< 需要同步的参数字节数
    bool requires_full_sync;                 ///< 是否需要全量同步
};

/**
 * @struct FaultToleranceConfig
 * @brief 容错配置
 */
struct FaultToleranceConfig {
    FaultToleranceStrategy strategy;         ///< 容错策略
    float snapshot_interval_s;               ///< 快照间隔 (秒)
    size_t max_snapshots;                    ///< 最大快照保留数
    bool enable_auto_recovery;               ///< 是否自动恢复
    float recovery_timeout_s;                ///< 恢复超时 (秒)
    bool enable_partition_detection;         ///< 是否启用分区检测
    float partition_threshold_s;             ///< 分区判定阈值 (秒)
    bool enable_incremental_sync;            ///< 是否启用增量同步

    static FaultToleranceConfig defaultConfig() {
        return FaultToleranceConfig{
            FaultToleranceStrategy::CRDTAsync,  // strategy
            60.0f,                              // snapshot_interval_s (1min)
            10,                                 // max_snapshots
            true,                               // enable_auto_recovery
            30.0f,                              // recovery_timeout_s
            true,                               // enable_partition_detection
            10.0f,                              // partition_threshold_s
            true,                               // enable_incremental_sync
        };
    }
};

/**
 * @struct FaultToleranceStats
 * @brief 容错统计信息
 */
struct FaultToleranceStats {
    size_t total_snapshots;                    ///< 总快照数
    size_t total_recoveries;                   ///< 总恢复次数
    size_t full_syncs;                         ///< 全量同步次数
    size_t incremental_syncs;                  ///< 增量同步次数
    size_t partitions_detected;                ///< 检测到的分区数
    size_t partitions_recovered;               ///< 恢复的分区数
    size_t crdt_merges;                        ///< CRDT 合并次数
    double avg_recovery_time_ms;               ///< 平均恢复时间 (ms)
    double avg_snapshot_size_bytes;            ///< 平均快照大小 (byte)

    void reset() {
        total_snapshots = 0;
        total_recoveries = 0;
        full_syncs = 0;
        incremental_syncs = 0;
        partitions_detected = 0;
        partitions_recovered = 0;
        crdt_merges = 0;
        avg_recovery_time_ms = 0.0;
        avg_snapshot_size_bytes = 0.0;
    }
};

/**
 * @brief 状态提供回调 — 获取当前 CRDT 状态
 */
using CRDTStateProvider = std::function<CRDTSnapshot()>;

/**
 * @brief 状态应用回调 — 应用恢复后的 CRDT 状态
 */
using CRDTStateApplier = std::function<void(const CRDTSnapshot&)>;

/**
 * @brief 参数同步回调 — 在节点间同步参数
 */
using ParamSyncCallback = std::function<void(
    uint32_t target_node, const std::vector<uint8_t>& params)>;

/**
 * @class FaultTolerance
 * @brief 异步容错与 CRDT 状态恢复
 *
 * 基于 CRDT 的异步容错模块，提供：
 * 1. CRDT 状态快照和恢复
 * 2. 网络分区检测和恢复
 * 3. 增量/全量参数同步
 * 4. 优雅降级策略
 * 5. 自动恢复机制
 *
 * 使用方式：
 *   1. 注册状态提供回调和应用回调
 *   2. 定期调用 takeSnapshot() 保存状态
 *   3. 节点故障时调用 recover() 恢复
 *   4. 分区恢复后调用 mergePartition() 合并
 */
class FaultTolerance {
public:
    /**
     * @brief 构造容错模块
     * @param local_node_id 本地节点 ID
     * @param config 容错配置
     */
    explicit FaultTolerance(uint32_t local_node_id,
                             FaultToleranceConfig config = FaultToleranceConfig::defaultConfig());

    ~FaultTolerance() = default;

    // ======================= 快照管理 =======================

    /**
     * @brief 创建 CRDT 状态快照
     * @param params 当前参数（可选，用于序列化保存）
     * @return 快照 ID
     *
     * 快照保存到环形缓冲区，超出 max_snapshots 时覆盖最旧的。
     * 快照间隔由 snapshot_interval_s 控制。
     */
    uint64_t takeSnapshot(const std::vector<Tensor*>& params = {});

    /**
     * @brief 获取最新的快照
     * @return 最新快照（如果没有则返回默认构造）
     */
    CRDTSnapshot latestSnapshot() const;

    /**
     * @brief 获取指定 ID 的快照
     * @param snapshot_id 快照 ID
     * @return 快照（如果不存在则返回默认构造）
     */
    CRDTSnapshot getSnapshot(uint64_t snapshot_id) const;

    /**
     * @brief 获取所有快照
     * @return 快照列表
     */
    std::vector<CRDTSnapshot> allSnapshots() const;

    /**
     * @brief 检查是否需要创建新快照
     * @return true 如果距离上次快照超过 snapshot_interval
     */
    bool needsSnapshot() const;

    // ======================= 分区检测 =======================

    /**
     * @brief 检测网络分区
     * @param alive_nodes 当前活跃节点 ID 列表
     * @return 新检测到的分区节点 ID 列表
     *
     * 检查对端节点是否在 partition_threshold 秒内未通信。
     * 如果超过阈值，标记为 Suspected 或 Confirmed。
     */
    std::vector<uint32_t> detectPartitions(
        const std::vector<uint32_t>& alive_nodes);

    /**
     * @brief 记录对端节点状态
     * @param node_id 对端节点 ID
     * @param snapshot 对端节点的最近状态快照
     *
     * 用于分区恢复时比较版本向量。
     */
    void recordPeerState(uint32_t node_id, const CRDTSnapshot& snapshot);

    /**
     * @brief 标记分区恢复
     * @param node_id 恢复的节点 ID
     * @return 恢复计划
     *
     * 比较本地和远程的版本向量，确定需要同步的数据。
     */
    RecoveryPlan markRecovery(uint32_t node_id);

    // ======================= 恢复 =======================

    /**
     * @brief 执行故障恢复
     * @param node_id 故障节点 ID
     * @return 恢复计划
     *
     * 1. 获取本地最新快照
     * 2. 与远程节点交换状态
     * 3. 比较版本向量
     * 4. 确定需要同步的数据
     * 5. 执行 CRDT 合并
     */
    RecoveryPlan recover(uint32_t node_id);

    /**
     * @brief 执行 CRDT 状态合并
     * @param local 本地快照
     * @param remote 远程快照
     * @return 合并后的快照
     *
     * 合并规则：
     * - 版本向量：逐元素取 max
     * - 全局步数：取 max
     * - 本地步数：取 max（LWW）
     * - 梯度计数器：逐元素取 max（G-Counter）
     * - 动量：取版本较新的（LWW）
     */
    static CRDTSnapshot mergeCRDT(
        const CRDTSnapshot& local, const CRDTSnapshot& remote);

    /**
     * @brief 检查是否需要全量同步
     * @param local 本地快照
     * @param remote 远程快照
     * @return true 如果版本向量在所有维度上差异过大
     *
     * 如果任一维度差异 > 3 步，需要全量同步。
     * 否则可以使用增量同步。
     */
    static bool needsFullSync(
        const CRDTSnapshot& local, const CRDTSnapshot& remote);

    // ======================= 回调注册 =======================

    /**
     * @brief 注册状态提供回调
     * @param provider 回调函数
     */
    void setStateProvider(CRDTStateProvider provider) {
        _state_provider = std::move(provider);
    }

    /**
     * @brief 注册状态应用回调
     * @param applier 回调函数
     */
    void setStateApplier(CRDTStateApplier applier) {
        _state_applier = std::move(applier);
    }

    /**
     * @brief 注册参数同步回调
     * @param callback 回调函数
     */
    void setParamSyncCallback(ParamSyncCallback callback) {
        _param_sync_callback = std::move(callback);
    }

    // ======================= 配置 =======================

    /**
     * @brief 设置容错配置
     * @param config 容错配置
     */
    void setConfig(const FaultToleranceConfig& config) { _config = config; }

    /**
     * @brief 获取当前容错配置
     * @return 容错配置
     */
    const FaultToleranceConfig& config() const { return _config; }

    // ======================= 统计信息 =======================

    /**
     * @brief 获取容错统计信息
     * @return 统计信息
     */
    FaultToleranceStats stats() const { return _stats; }

    /**
     * @brief 重置统计信息
     */
    void resetStats() { _stats.reset(); }

private:
    uint32_t _local_node_id;
    FaultToleranceConfig _config;
    mutable std::mutex _mtx;
    FaultToleranceStats _stats;

    // 快照环形缓冲区
    std::vector<CRDTSnapshot> _snapshots;
    uint64_t _next_snapshot_id;
    std::chrono::steady_clock::time_point _last_snapshot_time;

    // 分区记录
    std::unordered_map<uint32_t, PartitionEntry> _partitions;

    // 回调
    CRDTStateProvider _state_provider;
    CRDTStateApplier _state_applier;
    ParamSyncCallback _param_sync_callback;

    /**
     * @brief 检查快照是否需要覆盖最旧的
     */
    void pruneSnapshots();
};

} // namespace distributed
} // namespace ct

#endif // CTORCH_DISTRIBUTED_FAULT_TOLERANCE_H