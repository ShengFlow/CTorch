/**
 * @file GTCScheduler.h
 * @brief 博弈论驱动跨后端调度器 — GTCS (Game-Theoretic Cross-Backend Scheduler)
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 * @version v0.1 (Gen 2 GTCS)
 *
 * @details 本文件实现 GTCS，是 Gen 2 分布式系统的调度层增强组件。
 *          GTCS 使用博弈论机制（VCG、比例分配、Shapley 值）来
 *          优化跨后端任务分配和梯度聚合权重。
 *
 *          核心机制（Algorithmic Game Theory #5-6, #10, #17）：
 *          1. VCG 机制：每个节点上报后端处理单位 batch 的预期时间，
 *             调度器求解社会福利最大化分配
 *          2. 比例分配：梯度聚合权重 = 算力 × 后端精度 / 总加权算力
 *             理论保证 PoA ≤ 4/3
 *          3. Shapley 值：评估每个后端节点对模型性能的边际贡献
 *
 *          GTCS 作为 BANT 架构的调度层增强，与 DeviceBackend 无关，
 *          只查询 BackendManager 获取后端能力信息。
 */

#ifndef CTORCH_DISTRIBUTED_GTC_SCHEDULER_H
#define CTORCH_DISTRIBUTED_GTC_SCHEDULER_H

#include "DeviceBackend.h"
#include "BackendManager.h"
#include "Tensor.h"

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <mutex>
#include <algorithm>
#include <numeric>

namespace ct {
namespace distributed {

/**
 * @struct VCGAllocation
 * @brief VCG 分配结果
 */
struct VCGAllocation {
    uint32_t node_id;              ///< 节点 ID
    DeviceType backend;            ///< 后端类型
    size_t batch_size;             ///< 分配的 batch size
    float reported_time_ms;        ///< 上报的预期时间
    float vcg_payment;             ///< VCG 支付（外部性内化）
    float social_welfare;          ///< 社会福利贡献
};

/**
 * @struct ShapleyValue
 * @brief Shapley 值计算结果
 */
struct ShapleyValue {
    uint32_t node_id;              ///< 节点 ID
    DeviceType backend;            ///< 后端类型
    float marginal_contribution;   ///< 边际贡献（Shapley 值）
    float rank_score;              ///< 综合排名评分
};

/**
 * @class GTCScheduler
 * @brief 博弈论驱动跨后端调度器
 *
 * GTCS 负责：
 * 1. 收集节点后端的 VCG 报价（处理单位 batch 的预期时间）
 * 2. 求解社会福利最大化的 batch 分配方案
 * 3. 计算比例分配梯度权重（算力 × 后端精度校正）
 * 4. 评估节点贡献（Shapley 值）
 *
 * @note GTCS 是纯调度器，不参与梯度计算或参数更新。
 *       它只提供调度决策，执行由 CommEngine 和 DistributedOptimizer 完成。
 */
class GTCScheduler {
public:
    GTCScheduler() = default;
    ~GTCScheduler() = default;

    // ======================= VCG 调度 =======================

    /**
     * @brief 设置节点报价
     * @param node_id 节点 ID
     * @param backend 后端类型
     * @param time_ms 处理单位 batch 的预期时间 (ms)
     * @param max_batch 最大可处理 batch size
     */
    void setNodeBid(uint32_t node_id, DeviceType backend,
                     float time_ms, size_t max_batch);

    /**
     * @brief 求解社会福利最大化的 batch 分配
     * @param total_batch 总 batch size
     * @return 分配方案列表
     *
     * 使用 VCG 机制：
     * 1. 求解原始社会福利最大化问题
     * 2. 对每个节点，求解移除该节点后的社会福利
     * 3. VCG 支付 = 移除后的社会福利 - 原始社会福利（不含该节点）
     */
    std::vector<VCGAllocation> solveAllocation(size_t total_batch);

    // ======================= 比例分配权重 =======================

    /**
     * @brief 计算比例分配梯度聚合权重
     * @param node_id 节点 ID
     * @return 聚合权重 (0.0 ~ 1.0)
     *
     * 权重公式：w_r = θ_r · ε_r / Σ(θ_s · ε_s)
     * 其中 θ_r = compute_throughput, ε_r = numerical_precision
     * 理论保证 Price of Anarchy (PoA) ≤ 4/3
     */
    float getAggregationWeight(uint32_t node_id) const;

    /**
     * @brief 获取所有节点聚合权重
     * @return node_id → weight 映射
     */
    std::unordered_map<uint32_t, float> getAllAggregationWeights() const;

    // ======================= Shapley 值评估 =======================

    /**
     * @brief 计算所有节点的 Shapley 值
     * @param performance_scores 每个节点对模型性能的贡献分数
     * @return Shapley 值列表
     *
     * Shapley 值衡量每个节点对模型性能的"公平"边际贡献。
     * 用于节点选择和资源分配决策。
     */
    std::vector<ShapleyValue> computeShapleyValues(
        const std::unordered_map<uint32_t, float>& performance_scores);

    // ======================= 节点管理 =======================

    /**
     * @brief 注册节点到调度器
     * @param node_id 节点 ID
     * @param backend 后端类型
     *
     * 自动从 BackendManager 查询后端能力信息。
     */
    void registerNode(uint32_t node_id, DeviceType backend);

    /**
     * @brief 注销节点
     * @param node_id 节点 ID
     */
    void unregisterNode(uint32_t node_id);

    /**
     * @brief 获取所有注册节点
     * @return 节点 ID 列表
     */
    std::vector<uint32_t> registeredNodes() const;

    // ======================= 效率分析 =======================

    /**
     * @brief 计算 Price of Anarchy (PoA)
     * @return PoA 值（≥1.0，越接近 1.0 效率越高）
     *
     * PoA = 最优社会福利 / 实际社会福利
     * 比例分配机制的理论保证 PoA ≤ 4/3
     */
    float computePriceOfAnarchy() const;

    /**
     * @brief 校验节点报价真实性
     * @param actual_times 实际完成时间
     * @return 偏差 >20% 的节点 ID 列表
     *
     * 用于检测节点是否策略性虚报后端能力。
     */
    std::vector<uint32_t> detectBidCheating(
        const std::unordered_map<uint32_t, float>& actual_times);

private:
    /**
     * @struct NodeBid
     * @brief 内部节点报价结构
     */
    struct NodeBid {
        uint32_t node_id;
        DeviceType backend;
        float reported_time_ms;
        size_t max_batch;
        float compute_score;    ///< 综合算力评分
        float precision;        ///< 数值精度
    };

    std::unordered_map<uint32_t, NodeBid> _bids;
    mutable std::mutex _mtx;

    /**
     * @brief 求解 LP 分配问题（贪心近似）
     * @param nodes 节点列表
     * @param total_batch 总 batch size
     * @return 分配方案
     *
     * 使用贪心算法：按"单位时间吞吐量"降序分配。
     * 公式：throughput_per_ms = 1.0 / reported_time_ms
     */
    std::vector<VCGAllocation> greedyAllocation(
        const std::vector<NodeBid>& nodes, size_t total_batch);
};

} // namespace distributed
} // namespace ct

#endif // CTORCH_DISTRIBUTED_GTC_SCHEDULER_H