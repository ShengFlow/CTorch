/**
 * @file TopologyManager.h
 * @brief 动态拓扑管理器 — 继承 Gen 1 TASS 的后端感知拓扑管理
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 * @version v0.1 (Gen 2 BANT)
 *
 * @details 本文件实现 TopologyManager，是 Gen 1 TASS（动态拓扑扩展）
 *          在 Gen 2 中的独立实现和增强。
 *
 *          设计原则：
 *          1. 后端感知：邻居选择从"按延迟"扩展为"按延迟+后端兼容性+带宽"
 *          2. 动态重构：根据网络条件变化自动调整拓扑
 *          3. 兼容性评分：使用后端类型匹配和算力差异计算兼容性
 *
 *          本模块与 CommEngine 解耦：
 *          TopologyManager 只管理拓扑信息（节点间关系），
 *          不参与数据传输。CommEngine 调用 TopologyManager 获取
 *          邻居选择建议，但实际传输由 CommEngine 自行完成。
 */

#ifndef CTORCH_DISTRIBUTED_TOPOLOGY_MANAGER_H
#define CTORCH_DISTRIBUTED_TOPOLOGY_MANAGER_H

#include "Ctools.h"
#include "CtorchError.h"
#include "DeviceBackend.h"
#include "BackendManager.h"

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <map>
#include <string>
#include <mutex>
#include <cmath>
#include <algorithm>
#include <memory>
#include <chrono>

namespace ct {
namespace distributed {

/**
 * @brief 节点 ID 类型
 */
using TopoNodeId = uint32_t;

/**
 * @brief 拓扑连接类型
 */
enum class TopologyLinkType : uint8_t {
    Direct,         ///< 直接连接（低延迟，高带宽）
    Relay,          ///< 中继连接（通过中间节点）
    Overlay,        ///< 覆盖连接（逻辑连接，物理路径未知）
    Virtual,        ///< 虚拟连接（仅用于路由决策）
};

/**
 * @struct TopologyNode
 * @brief 拓扑节点信息
 */
struct TopologyNode {
    TopoNodeId id;                              ///< 节点 ID
    DeviceType backend_type;                     ///< 后端类型
    std::string address;                         ///< 网络地址
    float compute_throughput;                    ///< 算力 (TFLOPS)
    float memory_bandwidth;                      ///< 内存带宽 (GB/s)
    bool unified_memory;                         ///< 是否统一内存架构
    float numerical_precision;                   ///< 数值精度 (1.0 = float32)
    uint32_t max_batch_size;                     ///< 最大 batch size
    bool is_active;                              ///< 是否活跃
    std::chrono::steady_clock::time_point last_seen; ///< 最后可见时间
};

/**
 * @struct TopologyLink
 * @brief 拓扑连接信息
 */
struct TopologyLink {
    TopoNodeId node_a;                           ///< 节点 A
    TopoNodeId node_b;                           ///< 节点 B
    TopologyLinkType link_type;                  ///< 连接类型
    float rtt_ms;                                ///< 往返延迟 (ms)
    float bandwidth_mbps;                        ///< 带宽 (Mbps)
    float stability_score;                       ///< 稳定性评分 (0.0 ~ 1.0)
    float compatibility_score;                   ///< 后端兼容性评分 (0.0 ~ 1.0)
    std::chrono::steady_clock::time_point measured_at; ///< 测量时间
};

/**
 * @struct TopologySnapshot
 * @brief 拓扑快照
 */
struct TopologySnapshot {
    std::vector<TopologyNode> nodes;             ///< 所有节点
    std::vector<TopologyLink> links;             ///< 所有连接
    size_t num_active_nodes;                     ///< 活跃节点数
    size_t num_links;                            ///< 连接总数
    float avg_latency_ms;                        ///< 平均延迟
    float max_latency_ms;                        ///< 最大延迟
    float avg_bandwidth_mbps;                    ///< 平均带宽
    float graph_connectivity;                    ///< 图连通性 (0.0 ~ 1.0)
};

/**
 * @struct TopologyConfig
 * @brief 拓扑管理配置
 */
struct TopologyConfig {
    size_t max_neighbors;                        ///< 最大邻居数
    float latency_weight;                        ///< 延迟权重 (0.0 ~ 1.0)
    float bandwidth_weight;                      ///< 带宽权重 (0.0 ~ 1.0)
    float compatibility_weight;                  ///< 兼容性权重 (0.0 ~ 1.0)
    float stability_weight;                      ///< 稳定性权重 (0.0 ~ 1.0)
    float stale_timeout_s;                       ///< 节点超时时间 (秒)
    float reconfiguration_interval_s;            ///< 拓扑重构间隔 (秒)
    float min_compatibility_threshold;           ///< 最小兼容性阈值

    static TopologyConfig defaultConfig() {
        return TopologyConfig{
            8,          // max_neighbors
            0.4f,       // latency_weight
            0.3f,       // bandwidth_weight
            0.2f,       // compatibility_weight
            0.1f,       // stability_weight
            30.0f,      // stale_timeout_s
            60.0f,      // reconfiguration_interval_s
            0.3f        // min_compatibility_threshold
        };
    }
};

/**
 * @struct TopologyStats
 * @brief 拓扑统计信息
 */
struct TopologyStats {
    size_t total_nodes_registered;               ///< 总注册节点数
    size_t total_links_discovered;               ///< 总发现连接数
    size_t reconfigurations;                     ///< 拓扑重构次数
    size_t node_failures_detected;               ///< 检测到的节点故障数
    size_t link_failures_detected;               ///< 检测到的连接故障数
    float avg_neighbor_count;                    ///< 平均邻居数
    float avg_graph_diameter;                    ///< 平均图直径

    void reset() {
        total_nodes_registered = 0;
        total_links_discovered = 0;
        reconfigurations = 0;
        node_failures_detected = 0;
        link_failures_detected = 0;
        avg_neighbor_count = 0.0f;
        avg_graph_diameter = 0.0f;
    }
};

/**
 * @class TopologyManager
 * @brief 动态拓扑管理器
 *
 * 管理分布式集群的拓扑结构，提供：
 * 1. 节点注册/注销/状态管理
 * 2. 后端感知的邻居评分和选择
 * 3. 动态拓扑重构（基于网络条件变化）
 * 4. 拓扑快照和统计信息
 * 5. 图连通性分析
 *
 * 拓扑评分公式：
 * score(a, b) = w_latency * (1 - rtt_norm) +
 *               w_bandwidth * bw_norm +
 *               w_compat * compat(a, b) +
 *               w_stability * stability
 *
 * 兼容性评分考虑后端类型匹配度、算力差异和内存模型差异。
 */
class TopologyManager {
public:
    /**
     * @brief 构造拓扑管理器
     * @param local_node_id 本地节点 ID
     * @param config 拓扑配置
     */
    explicit TopologyManager(TopoNodeId local_node_id,
                              TopologyConfig config = TopologyConfig::defaultConfig());

    ~TopologyManager() = default;

    // ======================= 节点管理 =======================

    /**
     * @brief 注册或更新节点
     * @param node 节点信息
     */
    void registerNode(const TopologyNode& node);

    /**
     * @brief 注销节点
     * @param node_id 节点 ID
     */
    void unregisterNode(TopoNodeId node_id);

    /**
     * @brief 获取节点信息
     * @param node_id 节点 ID
     * @return 节点信息（如果不存在则返回 nullptr 的 shared_ptr）
     */
    std::shared_ptr<const TopologyNode> getNode(TopoNodeId node_id) const;

    /**
     * @brief 获取所有活跃节点
     * @return 节点列表
     */
    std::vector<TopologyNode> activeNodes() const;

    /**
     * @brief 获取本地节点 ID
     * @return 本地节点 ID
     */
    TopoNodeId localNodeId() const { return _local_node_id; }

    // ======================= 连接管理 =======================

    /**
     * @brief 注册或更新连接
     * @param link 连接信息
     */
    void registerLink(const TopologyLink& link);

    /**
     * @brief 移除连接
     * @param node_a 节点 A
     * @param node_b 节点 B
     */
    void removeLink(TopoNodeId node_a, TopoNodeId node_b);

    /**
     * @brief 更新延迟测量
     * @param node_id 目标节点
     * @param rtt_ms 测量到的 RTT
     */
    void updateLatency(TopoNodeId node_id, float rtt_ms);

    /**
     * @brief 更新带宽测量
     * @param node_id 目标节点
     * @param bandwidth_mbps 测量到的带宽
     */
    void updateBandwidth(TopoNodeId node_id, float bandwidth_mbps);

    /**
     * @brief 更新稳定性评分
     * @param node_id 目标节点
     * @param success 最近一次通信是否成功
     */
    void updateStability(TopoNodeId node_id, bool success);

    // ======================= 邻居选择 =======================

    /**
     * @brief 获取最优邻居（按综合评分排序）
     * @param max_neighbors 最大邻居数（0 = 使用配置值）
     * @return 节点 ID 列表，按评分降序
     */
    std::vector<TopoNodeId> getBestNeighbors(size_t max_neighbors = 0) const;

    /**
     * @brief 获取后端兼容的邻居
     * @param target_backend 目标后端类型
     * @param min_score 最小兼容性评分
     * @param max_neighbors 最大邻居数
     * @return 节点 ID 列表
     */
    std::vector<TopoNodeId> getCompatibleNeighbors(
        DeviceType target_backend, float min_score = 0.5f,
        size_t max_neighbors = 0) const;

    /**
     * @brief 计算两个节点间的综合评分
     * @param node_a 节点 A
     * @param node_b 节点 B
     * @return 综合评分 (0.0 ~ 1.0)
     */
    float computeScore(TopoNodeId node_a, TopoNodeId node_b) const;

    /**
     * @brief 计算后端兼容性评分
     * @param backend_a 后端类型 A
     * @param backend_b 后端类型 B
     * @return 兼容性评分 (0.0 ~ 1.0)
     *
     * 同类型后端 = 1.0，不同类型 = 0.5 ~ 0.8
     * 取决于算力差异和内存模型差异。
     */
    float computeBackendCompatibility(DeviceType backend_a,
                                       DeviceType backend_b) const;

    // ======================= 拓扑重构 =======================

    /**
     * @brief 检测并标记失效节点
     * @return 新标记为失效的节点 ID 列表
     *
     * 根据 stale_timeout 检查所有节点的 last_seen 时间。
     * 超时未见的节点标记为 is_active = false。
     */
    std::vector<TopoNodeId> detectStaleNodes();

    /**
     * @brief 执行拓扑重构
     * @return 是否发生了拓扑变化
     *
     * 1. 移除失效节点
     * 2. 重新评估所有连接的评分
     * 3. 修剪低评分连接
     * 4. 如果可能，建立新的连接
     */
    bool reconfigure();

    /**
     * @brief 检查是否需要重构
     * @return true 如果距离上次重构超过 reconfiguration_interval
     */
    bool needsReconfiguration() const;

    // ======================= 拓扑分析 =======================

    /**
     * @brief 获取拓扑快照
     * @return 当前拓扑的快照
     */
    TopologySnapshot getSnapshot() const;

    /**
     * @brief 计算图连通性
     * @return 连通性 (0.0 ~ 1.0)
     *
     * 使用 BFS 计算最大连通分量占比。
     */
    float computeGraphConnectivity() const;

    /**
     * @brief 计算图直径
     * @return 图直径（最长最短路径的边数）
     *
     * 使用 Floyd-Warshall 或 BFS 计算。
     * 如果图不连通，返回最大连通分量的直径。
     */
    size_t computeGraphDiameter() const;

    /**
     * @brief 检查两个节点是否直接相连
     * @param node_a 节点 A
     * @param node_b 节点 B
     * @return true 如果存在直接连接
     */
    bool hasDirectLink(TopoNodeId node_a, TopoNodeId node_b) const;

    // ======================= 配置 =======================

    /**
     * @brief 设置拓扑配置
     * @param config 拓扑配置
     */
    void setConfig(const TopologyConfig& config) { _config = config; }

    /**
     * @brief 获取当前拓扑配置
     * @return 拓扑配置
     */
    const TopologyConfig& config() const { return _config; }

    // ======================= 统计信息 =======================

    /**
     * @brief 获取拓扑统计信息
     * @return 统计信息
     */
    TopologyStats stats() const { return _stats; }

    /**
     * @brief 重置统计信息
     */
    void resetStats() { _stats.reset(); }

private:
    TopoNodeId _local_node_id;
    TopologyConfig _config;
    mutable std::mutex _mtx;
    TopologyStats _stats;

    // 节点表和连接表
    std::unordered_map<TopoNodeId, TopologyNode> _nodes;
    std::map<std::pair<TopoNodeId, TopoNodeId>, TopologyLink> _links;

    // 重构时间戳
    std::chrono::steady_clock::time_point _last_reconfiguration;

    /**
     * @brief 归一化延迟评分
     * @param rtt_ms 延迟 (ms)
     * @return 归一化评分 (0.0 ~ 1.0)
     */
    float normalizeLatency(float rtt_ms) const {
        // 延迟 < 0.1ms → 1.0, 延迟 > 100ms → 0.0
        const float min_lat = 0.1f;
        const float max_lat = 100.0f;
        if (rtt_ms <= min_lat) return 1.0f;
        if (rtt_ms >= max_lat) return 0.0f;
        return 1.0f - (rtt_ms - min_lat) / (max_lat - min_lat);
    }

    /**
     * @brief 归一化带宽评分
     * @param bandwidth_mbps 带宽 (Mbps)
     * @return 归一化评分 (0.0 ~ 1.0)
     */
    float normalizeBandwidth(float bandwidth_mbps) const {
        // 带宽 < 10 Mbps → 0.0, 带宽 > 10000 Mbps → 1.0
        const float min_bw = 10.0f;
        const float max_bw = 10000.0f;
        if (bandwidth_mbps <= min_bw) return 0.0f;
        if (bandwidth_mbps >= max_bw) return 1.0f;
        return std::log2(bandwidth_mbps / min_bw) / std::log2(max_bw / min_bw);
    }

    /**
     * @brief 获取两个节点间连接信息的内部辅助函数
     * @param node_a 节点 A
     * @param node_b 节点 B
     * @return 连接信息（如果不存在返回 nullptr）
     */
    std::shared_ptr<const TopologyLink> getLinkInternal(
        TopoNodeId node_a, TopoNodeId node_b) const;
};

} // namespace distributed
} // namespace ct

#endif // CTORCH_DISTRIBUTED_TOPOLOGY_MANAGER_H