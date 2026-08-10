/**
 * @file DistributedOptimizer.h
 * @brief 分布式优化器 — 后端无关的分布式训练优化器
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 * @version v0.1 (Gen 2 BANT)
 *
 * @details 本文件实现 DistributedOptimizer，是 BANT 架构中
 *          "优化逻辑"的核心组件。
 *
 *          设计原则（分离原理 — Feedback Systems #11）：
 *          DistributedOptimizer 只操作"逻辑梯度"（Tensor），
 *          不关心梯度在哪个后端、如何传输。
 *          通信逻辑由 CommEngine 完全接管。
 *
 *          DistributedOptimizer 的核心职责：
 *          1. 管理全局参数和梯度
 *          2. 执行 Local-SGD（本地多步计算，周期性全局同步）
 *          3. 维护 CRDT 优化器状态（继承 Gen 1 的 VCAO）
 *          4. 协调 CommEngine 和 GradientAggregator
 *
 *          与 CommEngine 的交互：
 *          DistributedOptimizer 调用 CommEngine::sendGradient() 发送本地梯度，
 *          CommEngine 收到远程梯度后通过回调通知 DistributedOptimizer。
 *          两者通过回调接口解耦，无直接依赖。
 */

#ifndef CTORCH_DISTRIBUTED_OPTIMIZER_H
#define CTORCH_DISTRIBUTED_OPTIMIZER_H

#include "Tensor.h"
#include "CommEngine.h"
#include "GradientAggregator.h"

#include <memory>
#include <vector>
#include <atomic>
#include <chrono>
#include <mutex>

namespace ct {
namespace distributed {

/**
 * @struct OptimizerConfig
 * @brief 分布式优化器配置
 */
struct OptimizerConfig {
    float learning_rate;           ///< 学习率
    float momentum;                ///< 动量系数 (0 = 无动量)
    float weight_decay;            ///< 权重衰减 (L2 正则化)
    size_t local_steps;            ///< Local-SGD 本地步数 K
    size_t warmup_steps;           ///< 预热步数
    float gradient_clip_norm;      ///< 梯度裁剪阈值 (0 = 不裁剪)

    static OptimizerConfig defaultConfig() {
        return OptimizerConfig{
            0.01f,      // learning_rate
            0.9f,       // momentum
            0.0001f,    // weight_decay
            10,         // local_steps
            100,        // warmup_steps
            1.0f        // gradient_clip_norm
        };
    }
};

/**
 * @struct CRDTState
 * @brief CRDT 优化器状态（继承 Gen 1 的 VCAO）
 *
 * 使用 CRDT（Conflict-free Replicated Data Type）语义，
 * 使优化器状态可以在异步网络中安全合并。
 * 动量状态使用 LWW（Last-Writer-Wins）寄存器，
 * 梯度计数器使用 G-Counter（Grow-only Counter）。
 */
struct CRDTState {
    /// 动量向量（LWW 寄存器）
    std::vector<float> momentum;
    /// 版本向量（每个节点一个计数器）
    std::vector<uint64_t> version_vector;
    /// 本地步数计数器
    uint64_t local_step;
    /// 全局步数计数器
    uint64_t global_step;
    /// 梯度累积计数器（G-Counter）
    std::vector<uint64_t> grad_counter;

    /**
     * @brief 检查本状态是否"领先于"另一个状态
     * @param other 另一个 CRDT 状态
     * @return true 如果本状态的版本向量在所有维度上 ≥ other
     */
    bool dominates(const CRDTState& other) const {
        if (version_vector.size() != other.version_vector.size()) return false;
        for (size_t i = 0; i < version_vector.size(); ++i) {
            if (version_vector[i] < other.version_vector[i]) return false;
        }
        return true;
    }

    /**
     * @brief 合并两个 CRDT 状态（取逐元素最大值）
     * @param a 状态 A
     * @param b 状态 B
     * @return 合并后的状态
     */
    static CRDTState merge(const CRDTState& a, const CRDTState& b) {
        CRDTState result = a;
        if (b.version_vector.size() > result.version_vector.size()) {
            result.version_vector.resize(b.version_vector.size(), 0);
        }
        for (size_t i = 0; i < b.version_vector.size() && i < result.version_vector.size(); ++i) {
            result.version_vector[i] = std::max(result.version_vector[i], b.version_vector[i]);
        }
        result.global_step = std::max(result.global_step, b.global_step);
        if (b.grad_counter.size() > result.grad_counter.size()) {
            result.grad_counter.resize(b.grad_counter.size(), 0);
        }
        for (size_t i = 0; i < b.grad_counter.size() && i < result.grad_counter.size(); ++i) {
            result.grad_counter[i] = std::max(result.grad_counter[i], b.grad_counter[i]);
        }
        // 动量：取"版本较新"的那个（LWW）
        if (b.local_step > a.local_step) {
            result.momentum = b.momentum;
            result.local_step = b.local_step;
        }
        return result;
    }
};

/**
 * @class DistributedOptimizer
 * @brief 分布式优化器 — 后端无关的 Local-SGD + CRDT 异步容错
 *
 * 核心工作流：
 * 1. 本地计算：每个节点独立计算 K 步梯度（Local-SGD）
 * 2. 梯度交换：通过 CommEngine 发送/接收梯度
 * 3. 中立聚合：GradientAggregator 在 CPU 中聚合梯度
 * 4. 全局更新：聚合后的梯度写回各后端，更新参数
 * 5. CRDT 合并：异步容错场景下使用 CRDT 状态合并
 *
 * 优化器完全与后端无关：所有参数和梯度以 Tensor 形式存在，
 * 具体在哪个后端由上游代码决定。
 */
class DistributedOptimizer {
public:
    /**
     * @brief 构造分布式优化器
     * @param params 模型参数列表
     * @param config 优化器配置
     * @param comm_engine 通信引擎（外部传入，共享所有权）
     */
    DistributedOptimizer(std::vector<Tensor*> params,
                          OptimizerConfig config = OptimizerConfig::defaultConfig(),
                          std::shared_ptr<CommEngine> comm_engine = nullptr);

    ~DistributedOptimizer() = default;

    // ======================= 训练循环 =======================

    /**
     * @brief 执行一步优化
     * @param loss 当前 loss 值（用于梯度计算）
     *
     * 内部流程：
     * 1. 检查是否到达同步点（local_step % K == 0）
     * 2. 若到达同步点：梯度聚合 → 全局更新 → 广播
     * 3. 若未到达同步点：本地更新
     */
    void step(float loss);

    /**
     * @brief 执行 Local-SGD 的一个本地步
     * @param grads 本地计算的梯度列表
     *
     * 累积本地梯度，当累积步数达到 K 时触发全局同步。
     */
    void localStep(const std::vector<Tensor>& grads);

    /**
     * @brief 触发全局同步
     *
     * 1. 将本地累积梯度发送到 CommEngine
     * 2. 等待 CommEngine 收集远程梯度
     * 3. 调用 GradientAggregator 聚合
     * 4. 更新全局参数
     * 5. 重置本地梯度累积
     */
    void synchronize();

    // ======================= 参数管理 =======================

    /**
     * @brief 获取当前参数
     * @return 参数张量列表
     */
    const std::vector<Tensor*>& parameters() const { return _params; }

    /**
     * @brief 更新单个参数（基于其对应的梯度）
     * @param param_idx 参数索引
     * @param grad 该参数的梯度张量
     */
    void updateParameter(size_t param_idx, const Tensor& grad);

    /**
     * @brief 零化梯度
     */
    void zeroGrad();

    // ======================= 配置管理 =======================

    /**
     * @brief 设置本地步数 K
     * @param k 本地步数
     */
    void setLocalSteps(size_t k) { _config.local_steps = k; }

    /**
     * @brief 获取本地步数 K
     * @return 本地步数
     */
    size_t localSteps() const { return _config.local_steps; }

    /**
     * @brief 设置学习率
     * @param lr 学习率
     */
    void setLearningRate(float lr) { _config.learning_rate = lr; }

    /**
     * @brief 获取当前学习率
     * @return 学习率
     */
    float learningRate() const { return _config.learning_rate; }

    // ======================= CRDT 状态管理 =======================

    /**
     * @brief 获取当前 CRDT 状态
     * @return CRDT 状态快照
     */
    CRDTState getCRDTState() const;

    /**
     * @brief 合并远程 CRDT 状态
     * @param remote_state 远程节点的 CRDT 状态
     *
     * 用于异步容错场景：当网络分区恢复后，
     * 通过 CRDT 合并恢复一致的优化器状态。
     */
    void mergeCRDTState(const CRDTState& remote_state);

    // ======================= 统计信息 =======================

    /**
     * @brief 优化器统计信息
     */
    struct Stats {
        size_t total_steps;          ///< 总步数
        size_t local_steps;          ///< 本地步数
        size_t syncs;                ///< 同步次数
        size_t crdt_merges;          ///< CRDT 合并次数
        float avg_loss;              ///< 平均 loss
        float current_lr;            ///< 当前学习率

        void reset() {
            total_steps = 0;
            local_steps = 0;
            syncs = 0;
            crdt_merges = 0;
            avg_loss = 0.0f;
            current_lr = 0.0f;
        }
    };

    /**
     * @brief 获取统计信息
     * @return 统计信息
     */
    Stats stats() const { return _stats; }

private:
    // 参数和配置
    std::vector<Tensor*> _params;
    OptimizerConfig _config;
    CRDTState _crdt_state;

    // 组件
    std::shared_ptr<CommEngine> _comm_engine;
    GradientAggregator _aggregator;

    // 梯度累积
    std::vector<Tensor> _accumulated_grads;
    size_t _local_step_counter;

    // 统计
    Stats _stats;

    // 远程梯度队列
    mutable std::mutex _remote_grads_mtx;
    std::vector<Tensor> _pending_remote_grads;

    /**
     * @brief 应用梯度裁剪
     * @param grads 梯度列表
     */
    void clipGradients(std::vector<Tensor>& grads);

    /**
     * @brief 应用权重衰减
     * @param grads 梯度列表
     */
    void applyWeightDecay(std::vector<Tensor>& grads);

    /**
     * @brief 梯度接收回调（由 CommEngine 调用）
     */
    void onGradientReceived(NodeId source, const Tensor& gradient);
};

} // namespace distributed
} // namespace ct

#endif // CTORCH_DISTRIBUTED_OPTIMIZER_H