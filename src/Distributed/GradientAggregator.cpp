#include "Distributed/GradientAggregator.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <chrono>

namespace ct {
namespace distributed {

GradientAggregator::GradientAggregator(AggregationStrategy strategy)
    : _strategy(strategy)
    , _trim_fraction(0.1f)
{
}

Tensor GradientAggregator::aggregate(const std::vector<Tensor>& gradients,
                                       const std::vector<float>& weights) {
    if (gradients.empty()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
            "GradientAggregator: empty gradient list");
    }

    auto start = std::chrono::steady_clock::now();

    // 将所有梯度移至 CPU（中立空间）
    std::vector<Tensor> cpu_grads;
    cpu_grads.reserve(gradients.size());
    for (const auto& g : gradients) {
        cpu_grads.push_back(ensureCPU(g));
    }

    // 验证形状一致性
    const auto& ref_shape = cpu_grads[0].shape();
    for (size_t i = 1; i < cpu_grads.size(); ++i) {
        if (cpu_grads[i].shape() != ref_shape) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                "GradientAggregator: gradient shape mismatch");
        }
    }

    Tensor result;
    switch (_strategy) {
        case AggregationStrategy::SimpleAverage:
            result = simpleAverage(cpu_grads);
            break;
        case AggregationStrategy::WeightedAverage:
            result = weightedAverage(cpu_grads, weights);
            break;
        case AggregationStrategy::RobustMedian:
            result = robustMedian(cpu_grads);
            break;
        case AggregationStrategy::RobustTrimmedMean:
            result = robustTrimmedMean(cpu_grads);
            break;
        default:
            result = simpleAverage(cpu_grads);
            break;
    }

    // 更新统计
    _stats.total_aggregations++;
    _stats.total_gradients += cpu_grads.size();
    auto end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    _stats.avg_aggregation_time_ms = (_stats.avg_aggregation_time_ms * (_stats.total_aggregations - 1) + elapsed_ms)
                                     / _stats.total_aggregations;

    return result;
}

Tensor GradientAggregator::aggregateWithQuorum(const std::vector<Tensor>& gradients,
                                                  size_t write_quorum,
                                                  const std::vector<float>& weights) {
    if (gradients.size() < write_quorum) {
        _stats.quorum_timeouts++;
        CtorchError::log(ErrorLevel::INFO, ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
            "GradientAggregator: quorum not met — have " + std::to_string(gradients.size())
            + ", need " + std::to_string(write_quorum));
        return Tensor();  // 返回空 Tensor 表示聚合不可用
    }
    // 只取前 write_quorum 个梯度进行聚合
    std::vector<Tensor> subset(gradients.begin(), gradients.begin() + write_quorum);
    std::vector<float> subset_weights;
    if (!weights.empty()) {
        subset_weights.assign(weights.begin(), weights.begin() + write_quorum);
    }
    return aggregate(subset, subset_weights);
}

Tensor GradientAggregator::aggregateWithBackendQuorum(const std::vector<Tensor>& gradients,
                                                         size_t write_quorum,
                                                         size_t backend_coverage_quorum,
                                                         const std::vector<float>& weights) {
    if (gradients.size() < write_quorum) {
        _stats.quorum_timeouts++;
        CtorchError::log(ErrorLevel::INFO, ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
            "GradientAggregator: backend quorum write not met — have "
            + std::to_string(gradients.size()) + ", need " + std::to_string(write_quorum));
        return Tensor();
    }

    // 检查后端覆盖
    std::unordered_set<DeviceType> backend_types;
    for (const auto& g : gradients) {
        backend_types.insert(g.device());
    }
    if (backend_types.size() < backend_coverage_quorum) {
        _stats.quorum_timeouts++;
        CtorchError::log(ErrorLevel::INFO, ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
            "GradientAggregator: backend coverage quorum not met — have "
            + std::to_string(backend_types.size()) + ", need "
            + std::to_string(backend_coverage_quorum));
        return Tensor();
    }

    return aggregate(gradients, weights);
}

Tensor GradientAggregator::simpleAverage(const std::vector<Tensor>& cpu_grads) {
    size_t n = cpu_grads.size();
    if (n == 0) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
            "GradientAggregator: empty gradient list for simple average");
    }
    size_t numel = cpu_grads[0].numel();
    if (numel == 0) {
        return Tensor();
    }
    Tensor result(ShapeTag{}, cpu_grads[0].shape(), cpu_grads[0].dtype(), DeviceType::kCPU, true);
    float* result_data = result.data_write<float>();

    for (size_t i = 0; i < numel; ++i) {
        float sum = 0.0f;
        for (const auto& g : cpu_grads) {
            sum += g.data_read<float>()[i];
        }
        result_data[i] = sum / static_cast<float>(n);
    }
    return result;
}

Tensor GradientAggregator::weightedAverage(const std::vector<Tensor>& cpu_grads,
                                              const std::vector<float>& weights) {
    size_t n = cpu_grads.size();
    if (n == 0) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
            "GradientAggregator: empty gradient list for weighted average");
    }
    size_t numel = cpu_grads[0].numel();
    if (numel == 0) {
        return Tensor();
    }

    // 如果没有提供权重，使用均匀权重
    std::vector<float> actual_weights = weights;
    if (actual_weights.empty()) {
        actual_weights.assign(n, 1.0f / n);
    }

    // 归一化权重
    float w_sum = std::accumulate(actual_weights.begin(), actual_weights.end(), 0.0f);
    if (w_sum <= 0.0f) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::MEMORY,
            "GradientAggregator: invalid weights (sum <= 0)");
    }
    for (auto& w : actual_weights) w /= w_sum;

    Tensor result(ShapeTag{}, cpu_grads[0].shape(), cpu_grads[0].dtype(), DeviceType::kCPU, true);
    float* result_data = result.data_write<float>();

    for (size_t i = 0; i < numel; ++i) {
        float weighted_sum = 0.0f;
        for (size_t j = 0; j < n; ++j) {
            weighted_sum += actual_weights[j] * cpu_grads[j].data_read<float>()[i];
        }
        result_data[i] = weighted_sum;
    }
    return result;
}

Tensor GradientAggregator::robustMedian(const std::vector<Tensor>& cpu_grads) {
    size_t n = cpu_grads.size();
    if (n == 0) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
            "GradientAggregator: empty gradient list for robust median");
    }
    size_t numel = cpu_grads[0].numel();
    if (numel == 0) {
        return Tensor();
    }
    Tensor result(ShapeTag{}, cpu_grads[0].shape(), cpu_grads[0].dtype(), DeviceType::kCPU, true);
    float* result_data = result.data_write<float>();

    std::vector<float> values(n);
    for (size_t i = 0; i < numel; ++i) {
        for (size_t j = 0; j < n; ++j) {
            values[j] = cpu_grads[j].data_read<float>()[i];
        }
        std::nth_element(values.begin(), values.begin() + n / 2, values.end());
        result_data[i] = values[n / 2];
    }
    return result;
}

Tensor GradientAggregator::robustTrimmedMean(const std::vector<Tensor>& cpu_grads) {
    size_t n = cpu_grads.size();
    if (n == 0) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
            "GradientAggregator: empty gradient list for robust trimmed mean");
    }
    size_t numel = cpu_grads[0].numel();
    if (numel == 0) {
        return Tensor();
    }
    size_t trim_count = static_cast<size_t>(n * _trim_fraction);
    if (trim_count * 2 >= n) trim_count = n / 4;  // 确保至少保留一半数据

    Tensor result(ShapeTag{}, cpu_grads[0].shape(), cpu_grads[0].dtype(), DeviceType::kCPU, true);
    float* result_data = result.data_write<float>();

    std::vector<float> values(n);
    for (size_t i = 0; i < numel; ++i) {
        for (size_t j = 0; j < n; ++j) {
            values[j] = cpu_grads[j].data_read<float>()[i];
        }
        std::sort(values.begin(), values.end());
        float sum = 0.0f;
        size_t count = 0;
        for (size_t j = trim_count; j < n - trim_count; ++j) {
            sum += values[j];
            ++count;
        }
        result_data[i] = (count > 0) ? (sum / static_cast<float>(count)) : 0.0f;
    }
    return result;
}

} // namespace distributed
} // namespace ct