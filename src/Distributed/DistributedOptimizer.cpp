#include "Distributed/DistributedOptimizer.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ct {
namespace distributed {

DistributedOptimizer::DistributedOptimizer(std::vector<Tensor*> params,
                                             OptimizerConfig config,
                                             std::shared_ptr<CommEngine> comm_engine)
    : _params(std::move(params))
    , _config(config)
    , _comm_engine(std::move(comm_engine))
    , _aggregator(AggregationStrategy::WeightedAverage)
    , _local_step_counter(0)
{
    // 初始化 CRDT 状态
    _crdt_state.local_step = 0;
    _crdt_state.global_step = 0;
    // 动量向量按参数总元素数分配（per-element 动量）
    {
        size_t total_elements = 0;
        for (auto* p : _params) {
            if (p) total_elements += p->numel();
        }
        _crdt_state.momentum.resize(total_elements, 0.0f);
    }
    _crdt_state.version_vector = {0};
    _crdt_state.grad_counter = {0};

    // 初始化累积梯度
    _accumulated_grads.reserve(_params.size());
    for (auto* p : _params) {
        if (p) {
            _accumulated_grads.push_back(Tensor(ShapeTag{}, p->shape(), p->dtype(), DeviceType::kCPU, true));
        }
    }

    // 注册 CommEngine 回调
    if (_comm_engine) {
        _comm_engine->setGradientCallback(
            [this](NodeId source, const Tensor& gradient) {
                onGradientReceived(source, gradient);
            });
    }

    _stats = {};
    _stats.current_lr = _config.learning_rate;
}

void DistributedOptimizer::step(float loss) {
    _stats.total_steps++;
    _stats.avg_loss = (_stats.avg_loss * (_stats.total_steps - 1) + loss) / _stats.total_steps;
    _local_step_counter++;

    // 检查是否到达同步点
    if (_local_step_counter >= _config.local_steps) {
        synchronize();
        _local_step_counter = 0;
    }
}

void DistributedOptimizer::localStep(const std::vector<Tensor>& grads) {
    if (grads.size() != _accumulated_grads.size()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
            "DistributedOptimizer: gradient count mismatch");
    }

    // 累积梯度（在 CPU 中立空间中）
    for (size_t i = 0; i < grads.size(); ++i) {
        Tensor cpu_grad = (grads[i].device() == DeviceType::kCPU)
            ? grads[i] : grads[i].to(DeviceType::kCPU);

        float* acc_data = _accumulated_grads[i].data_write<float>();
        const float* grad_data = cpu_grad.data_read<float>();
        size_t n = _accumulated_grads[i].numel();

        for (size_t j = 0; j < n; ++j) {
            acc_data[j] += grad_data[j];
        }
    }

    _stats.local_steps++;
    _crdt_state.local_step++;
}

void DistributedOptimizer::synchronize() {
    if (_accumulated_grads.empty() || _params.empty()) return;

    // 1. 裁剪梯度
    clipGradients(_accumulated_grads);

    // 2. 应用权重衰减
    applyWeightDecay(_accumulated_grads);

    // 3. 累积梯度除以 local_steps 得到平均梯度
    float inv_k = 1.0f / static_cast<float>(_config.local_steps);
    for (auto& grad : _accumulated_grads) {
        float* data = grad.data_write<float>();
        size_t n = grad.numel();
        for (size_t j = 0; j < n; ++j) {
            data[j] *= inv_k;
        }
    }

    // 4. 发送平均梯度到远程节点
    if (_comm_engine) {
        for (const auto& grad : _accumulated_grads) {
            _comm_engine->broadcastGradient(grad);
        }
    }

    // 5. 逐参数更新（聚合本地 + 远程梯度）
    _aggregator.setStrategy(AggregationStrategy::SimpleAverage);
    {
        std::lock_guard<std::mutex> lock(_remote_grads_mtx);
        for (size_t i = 0; i < _accumulated_grads.size(); ++i) {
            // 收集本地和远程梯度（按 shape 匹配）
            std::vector<Tensor> all_grads = {_accumulated_grads[i]};
            for (auto it = _pending_remote_grads.begin(); it != _pending_remote_grads.end();) {
                if (it->shape() == _accumulated_grads[i].shape()) {
                    all_grads.push_back(*it);
                    it = _pending_remote_grads.erase(it);
                } else {
                    ++it;
                }
            }

            // 简单平均聚合所有可用梯度
            Tensor aggregated = (all_grads.size() > 1)
                ? _aggregator.aggregate(all_grads)
                : all_grads[0];
            updateParameter(i, aggregated);
        }
    }

    // 6. 重置累积梯度
    for (auto& grad : _accumulated_grads) {
        grad.zero();
    }

    // 7. 更新 CRDT 状态
    _crdt_state.global_step++;
    if (!_crdt_state.version_vector.empty()) {
        _crdt_state.version_vector[0]++;
    }

    _stats.syncs++;
}

void DistributedOptimizer::updateParameter(size_t param_idx, const Tensor& grad) {
    if (param_idx >= _params.size() || !_params[param_idx]) return;

    Tensor* p = _params[param_idx];
    size_t numel = p->numel();
    const float* grad_data = grad.data_read<float>();
    float* param_data = p->data_write<float>();

    // 计算该参数在动量向量中的偏移量
    size_t offset = 0;
    for (size_t i = 0; i < param_idx; ++i) {
        if (_params[i]) offset += _params[i]->numel();
    }

    for (size_t i = 0; i < numel; ++i) {
        size_t mi = offset + i;
        // 动量更新
        _crdt_state.momentum[mi] = _config.momentum * _crdt_state.momentum[mi]
                                 + _config.learning_rate * grad_data[i];
        // 参数更新
        param_data[i] -= _crdt_state.momentum[mi];
    }

    _stats.current_lr = _config.learning_rate;
}

void DistributedOptimizer::zeroGrad() {
    for (auto& grad : _accumulated_grads) {
        grad.zero();
    }
}

void DistributedOptimizer::clipGradients(std::vector<Tensor>& grads) {
    if (_config.gradient_clip_norm <= 0.0f) return;

    float total_norm = 0.0f;
    for (const auto& g : grads) {
        const float* data = g.data_read<float>();
        size_t n = g.numel();
        float norm = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            norm += data[i] * data[i];
        }
        total_norm += norm;
    }
    total_norm = std::sqrt(total_norm);

    if (total_norm > _config.gradient_clip_norm) {
        float scale = _config.gradient_clip_norm / total_norm;
        for (auto& g : grads) {
            float* data = g.data_write<float>();
            size_t n = g.numel();
            for (size_t i = 0; i < n; ++i) {
                data[i] *= scale;
            }
        }
    }
}

void DistributedOptimizer::applyWeightDecay(std::vector<Tensor>& grads) {
    if (_config.weight_decay <= 0.0f) return;

    for (size_t i = 0; i < grads.size() && i < _params.size(); ++i) {
        const float* param_data = _params[i]->data_read<float>();
        float* grad_data = grads[i].data_write<float>();
        size_t n = grads[i].numel();

        for (size_t j = 0; j < n; ++j) {
            grad_data[j] += _config.weight_decay * param_data[j];
        }
    }
}

CRDTState DistributedOptimizer::getCRDTState() const {
    return _crdt_state;
}

void DistributedOptimizer::mergeCRDTState(const CRDTState& remote_state) {
    _crdt_state = CRDTState::merge(_crdt_state, remote_state);
    _stats.crdt_merges++;
}

void DistributedOptimizer::onGradientReceived(NodeId source, const Tensor& gradient) {
    (void)source;
    // 将远程梯度加入待处理队列，等待下次 synchronize() 时聚合
    std::lock_guard<std::mutex> lock(_remote_grads_mtx);
    _pending_remote_grads.push_back(gradient);
}

} // namespace distributed
} // namespace ct