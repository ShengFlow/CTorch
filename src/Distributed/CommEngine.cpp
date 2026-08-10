#include "Distributed/CommEngine.h"
#include "Distributed/CDTF.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ct {
namespace distributed {

CommEngine::CommEngine(NodeId local_node_id)
    : _local_node_id(local_node_id)
    , _compression_config(CompressionConfig::defaultConfig())
{
}

void CommEngine::registerNode(const NodeInfo& node_info) {
    std::lock_guard<std::mutex> lock(_nodes_mtx);
    _nodes[node_info.id] = node_info;
}

void CommEngine::unregisterNode(NodeId node_id) {
    std::lock_guard<std::mutex> lock(_nodes_mtx);
    _nodes.erase(node_id);
}

std::vector<NodeInfo> CommEngine::activeNodes() const {
    std::lock_guard<std::mutex> lock(_nodes_mtx);
    std::vector<NodeInfo> result;
    result.reserve(_nodes.size());
    for (const auto& [_, info] : _nodes) {
        if (info.is_active) {
            result.push_back(info);
        }
    }
    return result;
}

void CommEngine::sendGradient(const Tensor& grad, NodeId target, uint16_t flags) {
    // 自适应选择压缩标志
    uint16_t actual_flags = flags;
    if (flags == CDTF_FLAG_NONE) {
        actual_flags = adaptiveCompressionFlags(grad);
    }

    // 序列化为 CDTF 字节流
    auto data = CDTF::serialize(grad, actual_flags);

    // 更新统计信息
    _stats.bytes_sent += data.size();
    _stats.messages_sent++;
    _stats.raw_bytes_sent += grad.numel() * sizeof(float);
    _stats.compressed_bytes_sent += data.size();
    if (actual_flags & CDTF_FLAG_QUANTIZE_8) {
        _stats.quantize_8_count++;
        _stats.compression_count++;
    } else if (actual_flags & CDTF_FLAG_QUANTIZE_16) {
        _stats.quantize_16_count++;
        _stats.compression_count++;
    }
    _stats.compression_ratio = _stats.effective_ratio();

    // 传输
    transmitData(data, target);
}

void CommEngine::broadcastGradient(const Tensor& grad, uint16_t flags) {
    std::vector<NodeId> targets;
    {
        std::lock_guard<std::mutex> lock(_nodes_mtx);
        for (const auto& [id, info] : _nodes) {
            if (id != _local_node_id && info.is_active) {
                targets.push_back(id);
            }
        }
    }

    uint16_t actual_flags = flags;
    if (flags == CDTF_FLAG_NONE) {
        actual_flags = adaptiveCompressionFlags(grad);
    }

    auto data = CDTF::serialize(grad, actual_flags);
    _stats.bytes_sent += data.size() * targets.size();
    _stats.messages_sent += targets.size();
    _stats.raw_bytes_sent += grad.numel() * sizeof(float) * targets.size();
    _stats.compressed_bytes_sent += data.size() * targets.size();
    if (actual_flags & CDTF_FLAG_QUANTIZE_8) {
        _stats.quantize_8_count++;
        _stats.compression_count++;
    } else if (actual_flags & CDTF_FLAG_QUANTIZE_16) {
        _stats.quantize_16_count++;
        _stats.compression_count++;
    }
    _stats.compression_ratio = _stats.effective_ratio();

    for (auto target : targets) {
        transmitData(data, target);
    }
}

uint16_t CommEngine::adaptiveCompressionFlags(const Tensor& grad) {
    // 简单梯度熵估计：使用梯度的绝对值分布
    Tensor cpu_grad = (grad.device() == DeviceType::kCPU)
        ? grad : grad.to(DeviceType::kCPU);

    const float* data = cpu_grad.data_read<float>();
    size_t n = cpu_grad.numel();
    if (!data || n == 0) return CDTF_FLAG_NONE;

    // 计算梯度绝对值均值和方差作为熵的代理
    float sum_abs = 0.0f;
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float v = std::abs(data[i]);
        sum_abs += v;
        sum_sq += v * v;
    }
    float mean_abs = sum_abs / n;
    float variance = (sum_sq / n) - (mean_abs * mean_abs);

    // 熵估计：低方差 → 低熵 → 可强压缩
    float entropy_estimate = (variance < 1e-10f) ? 0.0f
        : std::log2(1.0f + std::sqrt(variance) / (mean_abs + 1e-10f));

    if (entropy_estimate < _compression_config.entropy_threshold * 0.5f) {
        // 极低熵：8-bit 量化 + 熵编码
        return CDTF_FLAG_QUANTIZE_8 | CDTF_FLAG_COMPRESSED;
    } else if (entropy_estimate < _compression_config.entropy_threshold) {
        // 低熵：16-bit 量化
        return CDTF_FLAG_QUANTIZE_16;
    }
    // 高熵：不压缩
    return CDTF_FLAG_NONE;
}

void CommEngine::updateLatency(NodeId node_id, float rtt_ms) {
    std::lock_guard<std::mutex> lock(_nodes_mtx);
    auto it = _nodes.find(node_id);
    if (it != _nodes.end()) {
        it->second.rtt_ms = rtt_ms;
        it->second.last_seen = std::chrono::steady_clock::now();
    }
}

std::vector<NodeId> CommEngine::getNeighbors(size_t max_neighbors) const {
    std::lock_guard<std::mutex> lock(_nodes_mtx);
    std::vector<std::pair<NodeId, float>> sorted;
    sorted.reserve(_nodes.size());
    for (const auto& [id, info] : _nodes) {
        if (id != _local_node_id && info.is_active) {
            sorted.emplace_back(id, info.rtt_ms);
        }
    }
    // 按延迟升序排序
    std::sort(sorted.begin(), sorted.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

    std::vector<NodeId> result;
    size_t count = (max_neighbors > 0) ? max_neighbors : sorted.size();
    for (size_t i = 0; i < count && i < sorted.size(); ++i) {
        result.push_back(sorted[i].first);
    }
    return result;
}

std::vector<NodeId> CommEngine::getCompatibleNeighbors(DeviceType target_backend,
                                                         size_t max_neighbors) const {
    std::lock_guard<std::mutex> lock(_nodes_mtx);
    std::vector<std::pair<NodeId, float>> sorted;
    sorted.reserve(_nodes.size());
    for (const auto& [id, info] : _nodes) {
        if (id != _local_node_id && info.is_active) {
            // 兼容性评分：同后端类型 = 100 分，不同后端 = 后端兼容性评分
            // 综合排序 = 延迟权重 * 0.6 + 兼容性权重 * 0.4
            float compat = (info.backend_type == target_backend) ? 100.0f
                          : static_cast<float>(info.compatibility_score);
            float score = info.rtt_ms * 0.6f + (100.0f - compat) * 0.4f;
            sorted.emplace_back(id, score);
        }
    }
    std::sort(sorted.begin(), sorted.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

    std::vector<NodeId> result;
    size_t count = (max_neighbors > 0) ? max_neighbors : sorted.size();
    for (size_t i = 0; i < count && i < sorted.size(); ++i) {
        result.push_back(sorted[i].first);
    }
    return result;
}

void CommEngine::transmitData(const std::vector<uint8_t>& data, NodeId target) {
    // 如果设置了 TCP 传输层，使用真实网络传输
    if (_transport) {
        if (!_transport->send(static_cast<uint32_t>(target), data)) {
            CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API,
                "CommEngine: TCP send failed to node " + std::to_string(target));
        }
        return;
    }

    // 占位实现：使用本地回调模拟网络传输
    if (_gradient_callback) {
        try {
            Tensor grad = CDTF::deserialize(data);
            _gradient_callback(target, grad);
            _stats.bytes_received += data.size();
            _stats.messages_received++;
        } catch (const std::exception& e) {
            CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API,
                std::string("CommEngine: failed to deserialize received gradient: ") + e.what());
        }
    }
}

std::vector<uint8_t> CommEngine::receiveData(NodeId source) {
    // 占位实现
    (void)source;
    return {};
}

} // namespace distributed
} // namespace ct