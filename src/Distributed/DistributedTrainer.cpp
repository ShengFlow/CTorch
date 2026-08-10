/**
 * @file DistributedTrainer.cpp
 * @brief 分布式训练器实现
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 *
 * DistributedTrainer 是薄层编排器，所有核心逻辑委托给子模块。
 * 训练器负责：
 * 1. 控制训练循环节奏（step/fit）
 * 2. 管理检查点保存/加载
 * 3. 收集训练指标
 * 4. 梯度压缩（可选）
 */

#include "Distributed/DistributedTrainer.h"
#include "CtorchError.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace ct {
namespace distributed {

// ======================= 构造 =======================

DistributedTrainer::DistributedTrainer(std::vector<Tensor*>& params,
                                         const TrainerConfig& config)
    : _params(params)
    , _config(config)
{
    // 创建 CommEngine（本地节点 ID 0）
    _comm_engine = std::make_unique<CommEngine>(0);

    // 创建梯度聚合器
    _aggregator = std::make_unique<GradientAggregator>(config.agg_strategy);

    // 创建优化器（持有 CommEngine 共享指针）
    OptimizerConfig opt_config;
    opt_config.learning_rate = config.learning_rate;
    opt_config.momentum = config.momentum;
    opt_config.weight_decay = config.weight_decay;
    opt_config.local_steps = config.local_steps;
    opt_config.gradient_clip_norm = config.gradient_clip_norm;
    opt_config.warmup_steps = config.warmup_steps;

    _optimizer = std::make_unique<DistributedOptimizer>(
        params, opt_config, std::shared_ptr<CommEngine>(_comm_engine.get(), [](void*){}));

    // 创建检查点管理器
    CheckpointConfig cp_cfg = CheckpointConfig::defaultConfig();
    cp_cfg.checkpoint_dir = config.checkpoint_dir;
    _checkpoint_mgr = std::make_unique<CheckpointManager>(cp_cfg);

    // 创建压缩器（可选）
    if (config.enable_compression) {
        _compressor = std::make_unique<EntropyAwareCompressor>();
    }

    // 创建调度器（可选）
    _scheduler = std::make_unique<GTCScheduler>();
}

// ======================= 训练循环 =======================

void DistributedTrainer::step(const std::vector<Tensor>& gradients, float loss) {
    // 计时
    updateStepTiming();

    // 更新指标
    _metrics.current_loss = loss;
    _metrics.global_step++;

    // 更新最佳 loss
    if (loss < _metrics.best_loss) {
        _metrics.best_loss = loss;
    }

    // 可选：梯度压缩
    std::vector<Tensor> working_grads = gradients;
    if (_config.enable_compression && _compressor) {
        for (auto& g : working_grads) {
            // 压缩梯度以降低传输带宽
            auto compressed = _compressor->compress(g);
            // 训练模式下，压缩后立即解压用于本地累积
            g = _compressor->decompressToTensor(
                compressed.compressed_data, {g.numel()});
        }
    }

    // 累积梯度到优化器
    _optimizer->localStep(working_grads);
    _metrics.local_step++;

    // 检查是否需要全局同步
    if (shouldSync()) {
        synchronize();
    }

    // 检查是否需要保存检查点
    if (shouldSave()) {
        save();
    }
}

void DistributedTrainer::fit(
    size_t num_steps,
    std::function<std::pair<std::vector<Tensor>, float>(size_t)> step_fn) {

    for (size_t i = 0; i < num_steps; ++i) {
        // 调用用户提供的步函数
        auto [gradients, loss] = step_fn(i);

        // 执行一步训练
        step(gradients, loss);
    }
}

void DistributedTrainer::synchronize() {
    // 委托给优化器执行全局同步
    _optimizer->synchronize();

    // 更新指标
    _metrics.num_syncs++;
    _metrics.local_step = 0;
}

// ======================= 检查点 =======================

void DistributedTrainer::save(const std::string& tag) {
    // 构建标签
    std::string checkpoint_tag = _config.checkpoint_tag;
    if (!tag.empty()) {
        checkpoint_tag = tag;
    }

    // 保存检查点
    std::vector<Tensor*> params_ptr = _params;
    auto checkpoint_id = _checkpoint_mgr->save(
        params_ptr,
        _metrics.global_step,
        _metrics.current_loss,
        _metrics.best_loss,
        CheckpointTrigger::StepInterval,
        {{"local_step", std::to_string(_metrics.local_step)},
         {"num_syncs", std::to_string(_metrics.num_syncs)},
         {"tag", checkpoint_tag}}
    );

    if (checkpoint_id > 0) {
        _metrics.checkpoints_saved++;
    }
}

bool DistributedTrainer::load(uint64_t checkpoint_id) {
    // 加载检查点
    auto metadata = _checkpoint_mgr->load(_params, checkpoint_id);

    if (metadata.global_step == 0 && checkpoint_id > 0) {
        // 加载失败
        return false;
    }

    // 恢复训练状态
    _metrics.global_step = metadata.global_step;
    _metrics.current_loss = metadata.loss;
    _metrics.best_loss = metadata.best_loss;

    // 从 tags 中恢复 local_step
    auto it = metadata.tags.find("local_step");
    if (it != metadata.tags.end()) {
        _metrics.local_step = std::stoul(it->second);
    }

    it = metadata.tags.find("num_syncs");
    if (it != metadata.tags.end()) {
        _metrics.num_syncs = std::stoul(it->second);
    }

    // 确保优化器的 CRDT 状态与加载的步数一致
    _optimizer->setLocalSteps(_config.local_steps);

    return true;
}

// ======================= 内部辅助 =======================

void DistributedTrainer::updateStepTiming() {
    auto now = std::chrono::steady_clock::now();

    if (_timing_initialized) {
        auto elapsed = std::chrono::duration<double, std::milli>(
            now - _step_start).count();

        // 指数移动平均
        const double alpha = 0.1;
        if (_metrics.avg_step_time_ms == 0.0) {
            _metrics.avg_step_time_ms = elapsed;
        } else {
            _metrics.avg_step_time_ms = (1.0 - alpha) * _metrics.avg_step_time_ms
                                      + alpha * elapsed;
        }
    }

    _step_start = now;
    _timing_initialized = true;
}

} // namespace distributed
} // namespace ct