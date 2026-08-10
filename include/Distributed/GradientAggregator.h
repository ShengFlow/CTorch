/**
 * @file GradientAggregator.h
 * @brief 梯度聚合器 — 中立空间梯度聚合
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 * @version v0.1 (Gen 2 BANT)
 *
 * @details 本文件实现 GradientAggregator，是 BANT 架构中
 *          "梯度聚合"的专用组件。
 *
 *          核心设计原则（CTFP #16 — 自然性条件）：
 *          梯度聚合必须在"中立空间"（CPU buffer）中执行，
 *          以保证聚合操作与设备迁移可交换。
 *
 *          支持多种聚合策略：
 *          1. 简单平均 (SimpleAverage)
 *          2. 加权平均 (WeightedAverage) — 按算力×精度加权
 *          3. Quorum 聚合 (QuorumAggregation) — 继承 Gen 1 的 Quorum NRW
 *          4. 鲁棒聚合 (RobustAggregation) — 去除离群梯度
 */

#ifndef CTORCH_DISTRIBUTED_GRADIENT_AGGREGATOR_H
#define CTORCH_DISTRIBUTED_GRADIENT_AGGREGATOR_H

#include "Tensor.h"

#include <vector>
#include <cstdint>
#include <functional>
#include <mutex>

namespace ct {
namespace distributed {

/**
 * @brief 聚合策略枚举
 */
enum class AggregationStrategy {
    SimpleAverage,     ///< 简单平均: grad = (1/N) * Σ grad_i
    WeightedAverage,   ///< 加权平均: grad = Σ w_i * grad_i / Σ w_i
    QuorumNRW,         ///< Quorum NRW: 只需 W 个节点提交即可聚合
    RobustMedian,      ///< 中位数聚合: 对离群梯度鲁棒
    RobustTrimmedMean, ///< 截尾均值: 去除头部和尾部 k% 的极端值
};

/**
 * @class GradientAggregator
 * @brief 梯度聚合器 — 在中立空间（CPU buffer）中执行聚合
 *
 * 所有聚合操作在 CPU 缓冲区中执行，保证了自然性条件：
 * 聚合操作与设备迁移可交换。
 *
 * 线程安全：支持多线程并发聚合。
 */
class GradientAggregator {
public:
    /**
     * @brief 构造梯度聚合器
     * @param strategy 聚合策略，默认加权平均
     */
    explicit GradientAggregator(AggregationStrategy strategy = AggregationStrategy::WeightedAverage);

    ~GradientAggregator() = default;

    // ======================= 聚合操作 =======================

    /**
     * @brief 聚合一组梯度（所有梯度移至 CPU 后聚合）
     * @param gradients 梯度张量列表
     * @param weights 权重列表（与 gradients 一一对应，仅加权平均策略使用）
     * @return 聚合后的梯度张量（在 CPU 上）
     * @throws CtorchError 如果梯度列表为空或形状不匹配
     */
    Tensor aggregate(const std::vector<Tensor>& gradients,
                     const std::vector<float>& weights = {});

    /**
     * @brief 带 Quorum 的聚合（继承 Gen 1 的 Quorum NRW）
     * @param gradients 梯度张量列表
     * @param write_quorum 写 Quorum：需要 W 个节点提交
     * @param weights 权重列表
     * @return 聚合后的梯度张量
     *
     * 当可用梯度数 < write_quorum 时，返回空 Tensor 表示聚合不可用。
     */
    Tensor aggregateWithQuorum(const std::vector<Tensor>& gradients,
                                size_t write_quorum,
                                const std::vector<float>& weights = {});

    /**
     * @brief 后端感知的 Quorum 聚合
     * @param gradients 梯度张量列表
     * @param write_quorum 写 Quorum
     * @param backend_coverage_quorum 后端覆盖 Quorum：需要至少覆盖多少种后端
     * @param weights 权重列表
     * @return 聚合后的梯度张量
     *
     * 扩展自 Gen 1 的 Quorum NRW：阈值从"节点数"扩展为"节点数+后端覆盖"。
     * 例如，在 MPS+CUDA 混合集群中，即使 3/4 节点已提交，如果只有 1 种后端
     * 被覆盖，也等待第 4 个节点（不同后端）。
     */
    Tensor aggregateWithBackendQuorum(const std::vector<Tensor>& gradients,
                                       size_t write_quorum,
                                       size_t backend_coverage_quorum,
                                       const std::vector<float>& weights = {});

    // ======================= 策略管理 =======================

    /**
     * @brief 设置聚合策略
     * @param strategy 聚合策略
     */
    void setStrategy(AggregationStrategy strategy) { _strategy = strategy; }

    /**
     * @brief 获取当前聚合策略
     * @return 聚合策略
     */
    AggregationStrategy strategy() const { return _strategy; }

    /**
     * @brief 设置截尾均值参数（仅 RobustTrimmedMean 策略）
     * @param trim_fraction 截尾比例 (0.0 ~ 0.5), 默认 0.1
     */
    void setTrimFraction(float trim_fraction) { _trim_fraction = trim_fraction; }

    // ======================= 统计信息 =======================

    /**
     * @brief 聚合统计信息
     */
    struct Stats {
        size_t total_aggregations;     ///< 总聚合次数
        size_t total_gradients;        ///< 总处理梯度数
        size_t quorum_timeouts;        ///< Quorum 超时次数
        double avg_aggregation_time_ms; ///< 平均聚合时间 (ms)

        void reset() {
            total_aggregations = 0;
            total_gradients = 0;
            quorum_timeouts = 0;
            avg_aggregation_time_ms = 0.0;
        }
    };

    /**
     * @brief 获取统计信息
     * @return 统计信息
     */
    Stats stats() const { return _stats; }

    /**
     * @brief 重置统计信息
     */
    void resetStats() { _stats.reset(); }

private:
    AggregationStrategy _strategy;
    float _trim_fraction;
    mutable std::mutex _mtx;
    Stats _stats;

    /**
     * @brief 内部实现：简单平均聚合
     */
    Tensor simpleAverage(const std::vector<Tensor>& cpu_grads);

    /**
     * @brief 内部实现：加权平均聚合
     */
    Tensor weightedAverage(const std::vector<Tensor>& cpu_grads,
                            const std::vector<float>& weights);

    /**
     * @brief 内部实现：中位数聚合
     */
    Tensor robustMedian(const std::vector<Tensor>& cpu_grads);

    /**
     * @brief 内部实现：截尾均值聚合
     */
    Tensor robustTrimmedMean(const std::vector<Tensor>& cpu_grads);

    /**
     * @brief 获取张量所在的 DeviceType
     */
    static DeviceType getTensorDevice(const Tensor& t) { return t.device(); }

    /**
     * @brief 确保张量在 CPU 上
     */
    static Tensor ensureCPU(const Tensor& t) {
        if (t.device() == DeviceType::kCPU) return t;
        return t.to(DeviceType::kCPU);
    }
};

} // namespace distributed
} // namespace ct

#endif // CTORCH_DISTRIBUTED_GRADIENT_AGGREGATOR_H