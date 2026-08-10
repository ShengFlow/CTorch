/**
 * @file DistributedTrainer.h
 * @brief 分布式训练器 — 编排本地/全局同步训练循环
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 * @version v0.1 (Gen 2 BANT)
 *
 * DistributedTrainer 是 Gen 2 分布式系统的"胶水层"，
 * 负责任务编排而非核心计算逻辑。
 *
 * 设计原则（分离原理 — Feedback Systems #11）：
 * - DistributedTrainer 薄层编排，不重复子模块逻辑
 * - 训练循环由 DistributedTrainer 控制，但梯度计算、优化、通信
 *   分别委托给具体的子模块
 * - 所有子模块可独立替换：替换 CommEngine 实现不影响优化器，
 *   替换聚合策略不影响训练循环
 *
 * 子模块职责边界：
 * ┌──────────────────────────────────────────────────┐
 * │ DistributedTrainer (编排层)                      │
 * │  ├─ 控制训练循环何时前向/反向/同步                │
 * │  ├─ 管理检查点保存/加载                           │
 * │  └─ 收集训练指标                                  │
 * ├──────────────────────────────────────────────────┤
 * │ DistributedOptimizer  ← 参数更新、CRDT 状态       │
 * │ CommEngine           ← 梯度传输、序列化           │
 * │ GradientAggregator   ← 梯度聚合策略               │
 * │ CheckpointManager    ← 检查点持久化               │
 * │ EntropyAwareCompressor ← 自适应压缩              │
 * │ GTCScheduler         ← 博弈论调度决策             │
 * └──────────────────────────────────────────────────┘
 */

#ifndef CTORCH_DISTRIBUTED_TRAINER_H
#define CTORCH_DISTRIBUTED_TRAINER_H

#include "DistributedOptimizer.h"
#include "CommEngine.h"
#include "GradientAggregator.h"
#include "CheckpointManager.h"
#include "EntropyAwareCompressor.h"
#include "GTCScheduler.h"
#include "Tensor.h"

#include <memory>
#include <vector>
#include <functional>
#include <chrono>
#include <cstdint>
#include <string>
#include <limits>

namespace ct {
namespace distributed {

/**
 * @struct TrainerConfig
 * @brief 训练器配置
 */
struct TrainerConfig {
    // ======================= 训练参数 =======================
    size_t local_steps = 10;             ///< Local-SGD 本地步数 K
    float learning_rate = 0.01f;         ///< 学习率
    float momentum = 0.9f;               ///< 动量系数
    float weight_decay = 0.0001f;        ///< 权重衰减
    float gradient_clip_norm = 1.0f;     ///< 梯度裁剪阈值 (0 = 不裁剪)
    size_t warmup_steps = 0;             ///< 预热步数

    // ======================= 检查点 =======================
    size_t checkpoint_interval = 1000;   ///< 每 N 步保存一次检查点 (0 = 不保存)
    std::string checkpoint_dir = "./checkpoints";  ///< 检查点目录
    std::string checkpoint_tag = "distributed";    ///< 检查点标签

    // ======================= 梯度聚合 =======================
    AggregationStrategy agg_strategy = AggregationStrategy::SimpleAverage;

    // ======================= 压缩 =======================
    bool enable_compression = false;     ///< 是否启用梯度压缩
};

/**
 * @struct TrainingMetrics
 * @brief 训练器运行时指标
 */
struct TrainingMetrics {
    size_t global_step = 0;              ///< 全局步数
    size_t local_step = 0;               ///< 当前 Local-SGD 周期内的本地步数
    float current_loss = 0.0f;           ///< 当前 loss
    float best_loss = std::numeric_limits<float>::max();  ///< 历史最佳 loss
    size_t num_syncs = 0;                ///< 全局同步次数
    size_t checkpoints_saved = 0;        ///< 已保存检查点数
    double avg_step_time_ms = 0.0;       ///< 平均单步时间 (ms)

    void reset() {
        global_step = 0;
        local_step = 0;
        current_loss = 0.0f;
        best_loss = std::numeric_limits<float>::max();
        num_syncs = 0;
        checkpoints_saved = 0;
        avg_step_time_ms = 0.0;
    }
};

/**
 * @class DistributedTrainer
 * @brief 分布式训练器 — 编排本地/全局同步训练循环
 *
 * 使用方式：
 * @code
 *   // 1. 创建参数
 *   Tensor param(ShapeTag{}, {128, 64}, DType::kFloat, DeviceType::kCPU, false);
 *   std::vector<Tensor*> params = {&param};
 *
 *   // 2. 配置训练器
 *   TrainerConfig config;
 *   config.local_steps = 10;
 *   config.learning_rate = 0.01f;
 *   config.checkpoint_interval = 500;
 *
 *   // 3. 创建训练器
 *   DistributedTrainer trainer(params, config);
 *
 *   // 4. 手动单步训练
 *   Tensor grad = computeGradient(param);
 *   trainer.step({grad}, 0.5f);
 *
 *   // 5. 或自动训练循环
 *   trainer.fit(1000, [](size_t step) -> std::pair<std::vector<Tensor>, float> {
 *       auto [grads, loss] = forwardBackward(step);
 *       return {grads, loss};
 *   });
 * @endcode
 */
class DistributedTrainer {
public:
    /**
     * @brief 构造训练器
     * @param params 模型参数列表（指针，训练器不拥有所有权）
     * @param config 训练器配置
     */
    explicit DistributedTrainer(std::vector<Tensor*>& params,
                                 const TrainerConfig& config = TrainerConfig{});

    ~DistributedTrainer() = default;

    // ======================= 训练循环 =======================

    /**
     * @brief 执行一步训练
     * @param gradients 当前 batch 的梯度列表
     * @param loss 当前 batch 的 loss 值
     *
     * 内部流程：
     * 1. 梯度累积到 DistributedOptimizer
     * 2. 达到 local_steps 时触发全局同步（聚合+参数更新）
     * 3. 达到 checkpoint_interval 时保存检查点
     * 4. 更新训练指标
     */
    void step(const std::vector<Tensor>& gradients, float loss);

    /**
     * @brief 自动训练循环
     * @param num_steps 训练步数
     * @param step_fn 每步计算函数 (global_step) -> (gradients, loss)
     *
     * step_fn 由用户提供，负责前向/反向传播。
     * 训练器只负责控制训练循环节奏和编排子模块。
     */
    void fit(size_t num_steps,
             std::function<std::pair<std::vector<Tensor>, float>(size_t)> step_fn);

    /**
     * @brief 触发全局同步（手动）
     *
     * 强制将当前累积的梯度同步到所有节点并聚合更新。
     * 通常由 step() 自动调用，但如果需要手动控制同步点也可调用。
     */
    void synchronize();

    // ======================= 检查点 =======================

    /**
     * @brief 保存当前训练状态
     * @param tag 自定义标签
     */
    void save(const std::string& tag = "");

    /**
     * @brief 加载检查点恢复训练
     * @param checkpoint_id 检查点 ID (0 = 最新)
     * @return true 加载成功
     */
    bool load(uint64_t checkpoint_id = 0);

    // ======================= 配置 =======================

    /**
     * @brief 更新学习率
     * @param lr 新学习率
     */
    void setLearningRate(float lr) {
        _optimizer->setLearningRate(lr);
    }

    /**
     * @brief 获取当前学习率
     * @return 学习率
     */
    float learningRate() const {
        return _optimizer->learningRate();
    }

    /**
     * @brief 设置本地步数 K
     * @param k 本地步数
     */
    void setLocalSteps(size_t k) {
        _config.local_steps = k;
        _optimizer->setLocalSteps(k);
    }

    // ======================= 状态查询 =======================

    /**
     * @brief 获取训练指标
     * @return 当前指标快照
     */
    TrainingMetrics metrics() const { return _metrics; }

    /**
     * @brief 获取模型参数
     * @return 参数指针列表
     */
    const std::vector<Tensor*>& parameters() const { return _params; }

    /**
     * @brief 获取全局步数
     * @return 全局步数
     */
    size_t globalStep() const { return _metrics.global_step; }

    /**
     * @brief 获取历史最佳 loss
     * @return 最佳 loss
     */
    float bestLoss() const { return _metrics.best_loss; }

    // ======================= 子模块访问 =======================

    /**
     * @brief 获取内部优化器（用于高级配置）
     * @return 优化器指针
     */
    DistributedOptimizer* optimizer() { return _optimizer.get(); }

    /**
     * @brief 获取内部 CommEngine（用于高级配置）
     * @return CommEngine 指针
     */
    CommEngine* commEngine() { return _comm_engine.get(); }

private:
    /**
     * @brief 检查是否应触发全局同步
     */
    bool shouldSync() const {
        return _config.local_steps > 0 &&
               _metrics.local_step >= _config.local_steps;
    }

    /**
     * @brief 检查是否应保存检查点
     */
    bool shouldSave() const {
        return _config.checkpoint_interval > 0 &&
               _metrics.global_step > 0 &&
               _metrics.global_step % _config.checkpoint_interval == 0;
    }

    /**
     * @brief 更新步时统计
     */
    void updateStepTiming();

    // ======================= 成员 =======================

    std::vector<Tensor*> _params;
    TrainerConfig _config;
    TrainingMetrics _metrics;

    // 子模块（所有权归属训练器）
    std::unique_ptr<DistributedOptimizer> _optimizer;
    std::unique_ptr<GradientAggregator> _aggregator;
    std::unique_ptr<CommEngine> _comm_engine;
    std::unique_ptr<CheckpointManager> _checkpoint_mgr;
    std::unique_ptr<EntropyAwareCompressor> _compressor;
    std::unique_ptr<GTCScheduler> _scheduler;

    // 计时
    std::chrono::steady_clock::time_point _step_start;
    bool _timing_initialized = false;
};

} // namespace distributed
} // namespace ct

#endif // CTORCH_DISTRIBUTED_TRAINER_H