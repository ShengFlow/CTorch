/**
 * @file NodeDiscovery.h
 * @brief 自动节点发现与健康检查 — 分布式集群节点管理
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 * @version v0.1 (Gen 2 BANT)
 *
 * @details 本文件实现 NodeDiscovery，负责分布式集群中的自动节点发现、
 *          心跳检测、健康检查和节点状态管理。
 *
 *          设计原则：
 *          1. Gossip 协议风格的心跳传播
 *          2. 可配置的健康检查超时和重试策略
 *          3. 后端能力发现：节点加入时自动交换 BackendCapability
 *          4. 故障检测：基于 Phi Accrual Failure Detector 算法
 *
 *          本模块与 TopologyManager 解耦：
 *          NodeDiscovery 负责"发现"节点，
 *          TopologyManager 负责"管理"节点关系。
 *          NodeDiscovery 发现新节点后通知 TopologyManager 注册。
 */

#ifndef CTORCH_DISTRIBUTED_NODE_DISCOVERY_H
#define CTORCH_DISTRIBUTED_NODE_DISCOVERY_H

#include "Ctools.h"
#include "CtorchError.h"
#include "DeviceBackend.h"
#include "BackendManager.h"

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <mutex>
#include <functional>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>

namespace ct {
namespace distributed {

/**
 * @brief 节点状态枚举
 */
enum class NodeStatus : uint8_t {
    Unknown,     ///< 未知状态（未发现）
    Alive,       ///< 节点存活
    Suspect,     ///< 可疑（心跳丢失，但未确认死亡）
    Dead,        ///< 节点死亡
    Left,        ///< 节点主动离开
};

/**
 * @brief 发现协议类型
 */
enum class DiscoveryProtocol : uint8_t {
    Manual,      ///< 手动注册（调试/测试用）
    Static,      ///< 静态配置文件发现
    Gossip,      ///< Gossip 协议传播
    Broadcast,   ///< UDP 广播发现
    Centralized, ///< 中心化的发现服务
};

/**
 * @struct NodeEndpoint
 * @brief 节点网络端点
 */
struct NodeEndpoint {
    uint32_t node_id;          ///< 节点 ID
    std::string host;          ///< 主机名或 IP
    uint16_t port;             ///< 端口
    DiscoveryProtocol protocol;///< 发现协议
    DeviceType backend_type;   ///< 后端类型
    std::string version;       ///< CTorch 版本号
};

/**
 * @struct HeartbeatMessage
 * @brief 心跳消息
 */
struct HeartbeatMessage {
    uint32_t node_id;                           ///< 发送节点 ID
    uint64_t sequence_number;                   ///< 序列号（单调递增）
    std::chrono::steady_clock::time_point timestamp; ///< 发送时间戳
    float load_factor;                          ///< 当前负载因子 (0.0 ~ 1.0)
    DeviceType backend_type;                    ///< 后端类型
    std::vector<uint32_t> gossip_nodes;         ///< 随心跳传播的节点列表
};

/**
 * @struct FailureDetectorState
 * @brief Phi Accrual Failure Detector 状态
 *
 * Phi Accrual 算法：
 * 1. 维护最近 N 个心跳间隔的滑动窗口
 * 2. 假设心跳间隔符合正态分布，计算当前间隔的 Phi 值
 * 3. Phi = -log10(P(当前间隔 | 分布))
 * 4. Phi > 阈值 → 判定为故障
 */
struct FailureDetectorState {
    std::vector<double> interval_history;  ///< 最近心跳间隔历史 (ms)
    size_t max_history_size;               ///< 最大历史记录数
    double phi_threshold;                  ///< Phi 阈值（默认 8.0）
    double last_heartbeat_time_ms;         ///< 最后心跳时间 (ms)

    FailureDetectorState() : max_history_size(100), phi_threshold(8.0),
                             last_heartbeat_time_ms(0.0) {}
};

/**
 * @struct DiscoveryConfig
 * @brief 节点发现配置
 */
struct DiscoveryConfig {
    DiscoveryProtocol protocol;          ///< 发现协议
    float heartbeat_interval_ms;         ///< 心跳间隔 (ms)
    float phi_threshold;                 ///< Phi Accrual 阈值
    size_t phi_history_size;             ///< Phi 历史窗口大小
    size_t max_heartbeat_loss;           ///< 最大心跳丢失次数
    float discovery_timeout_ms;          ///< 发现超时 (ms)
    bool enable_gossip;                  ///< 是否启用 Gossip 传播
    size_t gossip_fanout;                ///< Gossip 扇出数
    float gossip_interval_ms;            ///< Gossip 传播间隔 (ms)

    static DiscoveryConfig defaultConfig() {
        return DiscoveryConfig{
            DiscoveryProtocol::Gossip,  // protocol
            1000.0f,                    // heartbeat_interval_ms (1s)
            8.0f,                       // phi_threshold
            100,                        // phi_history_size
            3,                          // max_heartbeat_loss
            5000.0f,                    // discovery_timeout_ms (5s)
            true,                       // enable_gossip
            3,                          // gossip_fanout
            2000.0f,                    // gossip_interval_ms (2s)
        };
    }
};

/**
 * @struct DiscoveryStats
 * @brief 发现统计信息
 */
struct DiscoveryStats {
    size_t total_nodes_discovered;       ///< 总发现节点数
    size_t total_nodes_lost;             ///< 总丢失节点数
    size_t heartbeats_sent;              ///< 发送心跳数
    size_t heartbeats_received;          ///< 接收心跳数
    size_t false_positives;              ///< 误判故障数
    size_t phi_failures;                 ///< Phi 检测触发数
    size_t gossip_messages;              ///< Gossip 消息数
    double avg_phi_value;                ///< 平均 Phi 值

    void reset() {
        total_nodes_discovered = 0;
        total_nodes_lost = 0;
        heartbeats_sent = 0;
        heartbeats_received = 0;
        false_positives = 0;
        phi_failures = 0;
        gossip_messages = 0;
        avg_phi_value = 0.0;
    }
};

/**
 * @brief 节点状态变化回调
 */
using NodeStatusCallback = std::function<void(
    uint32_t node_id, NodeStatus old_status, NodeStatus new_status)>;

/**
 * @brief 节点发现回调
 */
using NodeDiscoveryCallback = std::function<void(
    const NodeEndpoint& endpoint)>;

/**
 * @class NodeDiscovery
 * @brief 自动节点发现与健康检查
 *
 * 基于 Phi Accrual Failure Detector 的节点发现模块：
 * 1. 自动发现：支持 Gossip 协议和静态配置
 * 2. 健康检查：Phi Accrual 算法检测故障
 * 3. 心跳传播：定期发送心跳，携带 Gossip 节点信息
 * 4. 状态管理：跟踪节点状态变化
 * 5. 回调通知：节点状态变化时通知外部
 *
 * Phi Accrual 算法：
 * - 维护每个节点的心跳间隔历史
 * - 假设间隔符合正态分布
 * - 计算当前间隔的 Phi = -log10(P)
 * - Phi > 阈值 → 判定为故障
 * - 自适应：随历史数据增加，检测精度提高
 */
class NodeDiscovery {
public:
    /**
     * @brief 构造节点发现器
     * @param local_node_id 本地节点 ID
     * @param config 发现配置
     */
    explicit NodeDiscovery(uint32_t local_node_id,
                            DiscoveryConfig config = DiscoveryConfig::defaultConfig());

    ~NodeDiscovery() = default;

    // ======================= 节点注册 =======================

    /**
     * @brief 注册种子节点（用于初始发现）
     * @param endpoint 种子节点端点
     */
    void registerSeedNode(const NodeEndpoint& endpoint);

    /**
     * @brief 注册一组种子节点
     * @param endpoints 种子节点端点列表
     */
    void registerSeedNodes(const std::vector<NodeEndpoint>& endpoints);

    /**
     * @brief 记录节点发现
     * @param endpoint 发现的节点端点
     *
     * 如果节点已存在，更新状态为 Alive。
     * 如果节点是新节点，触发 discovery_callback。
     */
    void recordDiscovery(const NodeEndpoint& endpoint);

    /**
     * @brief 标记节点离开
     * @param node_id 节点 ID
     */
    void recordLeave(uint32_t node_id);

    // ======================= 心跳管理 =======================

    /**
     * @brief 生成心跳消息
     * @return 心跳消息
     *
     * 包含序列号、时间戳、负载因子和 Gossip 节点列表。
     */
    HeartbeatMessage generateHeartbeat();

    /**
     * @brief 处理接收到的远程心跳
     * @param heartbeat 远程心跳消息
     *
     * 1. 更新节点状态为 Alive
     * 2. 更新心跳间隔历史
     * 3. 传播 Gossip 信息
     */
    void processHeartbeat(const HeartbeatMessage& heartbeat);

    // ======================= 故障检测 =======================

    /**
     * @brief 计算节点的 Phi 值
     * @param node_id 节点 ID
     * @return Phi 值（越高越可能故障）
     *
     * Phi = -log10(P(当前间隔 | 历史分布))
     * Phi > threshold → 判定为故障
     */
    double computePhi(uint32_t node_id) const;

    /**
     * @brief 执行故障检测轮次
     * @return 新判定为故障的节点 ID 列表
     *
     * 对所有节点计算 Phi 值，
     * Phi > 阈值 → 标记为 Suspect 或 Dead。
     */
    std::vector<uint32_t> detectFailures();

    /**
     * @brief 获取节点状态
     * @param node_id 节点 ID
     * @return 节点状态
     */
    NodeStatus getNodeStatus(uint32_t node_id) const;

    /**
     * @brief 获取所有存活节点
     * @return 存活节点 ID 列表
     */
    std::vector<uint32_t> aliveNodes() const;

    /**
     * @brief 获取所有可疑节点
     * @return 可疑节点 ID 列表
     */
    std::vector<uint32_t> suspectNodes() const;

    /**
     * @brief 检查节点是否存活
     * @param node_id 节点 ID
     * @return true 如果节点状态为 Alive
     */
    bool isAlive(uint32_t node_id) const;

    // ======================= 回调 =======================

    /**
     * @brief 设置节点状态变化回调
     * @param callback 回调函数
     */
    void setNodeStatusCallback(NodeStatusCallback callback) {
        _status_callback = std::move(callback);
    }

    /**
     * @brief 设置节点发现回调
     * @param callback 回调函数
     */
    void setNodeDiscoveryCallback(NodeDiscoveryCallback callback) {
        _discovery_callback = std::move(callback);
    }

    // ======================= 配置 =======================

    /**
     * @brief 设置发现配置
     * @param config 发现配置
     */
    void setConfig(const DiscoveryConfig& config) { _config = config; }

    /**
     * @brief 获取当前发现配置
     * @return 发现配置
     */
    const DiscoveryConfig& config() const { return _config; }

    // ======================= 统计信息 =======================

    /**
     * @brief 获取发现统计信息
     * @return 统计信息
     */
    DiscoveryStats stats() const { return _stats; }

    /**
     * @brief 重置统计信息
     */
    void resetStats() { _stats.reset(); }

private:
    uint32_t _local_node_id;
    DiscoveryConfig _config;
    mutable std::mutex _mtx;
    DiscoveryStats _stats;

    // 节点状态表
    std::unordered_map<uint32_t, NodeStatus> _node_statuses;

    // 节点端点表
    std::unordered_map<uint32_t, NodeEndpoint> _node_endpoints;

    // 种子节点
    std::vector<NodeEndpoint> _seed_nodes;

    // Phi Accrual 检测器状态
    std::unordered_map<uint32_t, FailureDetectorState> _phi_states;

    // 心跳序列号
    uint64_t _heartbeat_seq;

    // 负载因子估计
    float _estimated_load;

    // 回调
    NodeStatusCallback _status_callback;
    NodeDiscoveryCallback _discovery_callback;

    // 随机数生成器（用于 Gossip 扇出选择）
    mutable std::mt19937 _rng;

    /**
     * @brief 更新节点状态并触发回调
     * @param node_id 节点 ID
     * @param new_status 新状态
     */
    void updateNodeStatus(uint32_t node_id, NodeStatus new_status);

    /**
     * @brief 更新 Phi 状态（记录心跳间隔）
     * @param node_id 节点 ID
     * @param interval_ms 心跳间隔 (ms)
     */
    void updatePhiState(uint32_t node_id, double interval_ms);

    /**
     * @brief 计算 Phi 值（基于历史分布）
     * @param state 故障检测器状态
     * @return Phi 值
     */
    double computePhiValue(const FailureDetectorState& state) const;

    /**
     * @brief 选择 Gossip 扇出节点
     * @return 选中节点的 ID 列表
     */
    std::vector<uint32_t> selectGossipFanout();

    /**
     * @brief 估计当前负载因子
     * @return 负载因子 (0.0 ~ 1.0)
     */
    float estimateLoadFactor() const;

    /**
     * @brief 检查节点是否在种子节点列表中
     * @param node_id 节点 ID
     * @return true 如果是种子节点
     */
    bool isSeedNode(uint32_t node_id) const;
};

} // namespace distributed
} // namespace ct

#endif // CTORCH_DISTRIBUTED_NODE_DISCOVERY_H