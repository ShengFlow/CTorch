#include "Distributed/GTCScheduler.h"

#include <cmath>
#include <set>
#include <sstream>

namespace ct {
namespace distributed {

void GTCScheduler::setNodeBid(uint32_t node_id, DeviceType backend,
                                float time_ms, size_t max_batch) {
    std::lock_guard<std::mutex> lock(_mtx);

    // 从 BackendManager 查询后端能力
    auto backend_ptr = BackendManager::getInstance().getBackend(backend);
    float compute_score = 0.5f;  // 默认值
    float precision = 1.0f;      // 默认值

    if (backend_ptr) {
        auto cap = backend_ptr->capability();
        compute_score = cap.compositeScore();
        precision = cap.numerical_precision;
    }

    _bids[node_id] = NodeBid{
        node_id, backend, time_ms, max_batch, compute_score, precision
    };
}

std::vector<VCGAllocation> GTCScheduler::solveAllocation(size_t total_batch) {
    std::lock_guard<std::mutex> lock(_mtx);
    if (_bids.empty()) return {};

    // 收集节点列表
    std::vector<NodeBid> nodes;
    nodes.reserve(_bids.size());
    for (const auto& [_, bid] : _bids) {
        nodes.push_back(bid);
    }

    // 原始分配
    auto allocation = greedyAllocation(nodes, total_batch);

    // 计算原始社会福利
    float original_social_welfare = 0.0f;
    for (const auto& alloc : allocation) {
        original_social_welfare += alloc.social_welfare;
    }

    // 对每个节点计算 VCG 支付
    for (auto& alloc : allocation) {
        // 移除该节点后重新分配
        std::vector<NodeBid> remaining;
        for (const auto& bid : nodes) {
            if (bid.node_id != alloc.node_id) {
                remaining.push_back(bid);
            }
        }

        auto without_node = greedyAllocation(remaining, total_batch);
        float welfare_without = 0.0f;
        for (const auto& wa : without_node) {
            welfare_without += wa.social_welfare;
        }

        // VCG 支付 = 移除后的社会福利 - 原始社会福利 + 该节点的贡献
        alloc.vcg_payment = welfare_without
                          - (original_social_welfare - alloc.social_welfare);
        if (alloc.vcg_payment < 0) alloc.vcg_payment = 0;
    }

    return allocation;
}

float GTCScheduler::getAggregationWeight(uint32_t node_id) const {
    std::lock_guard<std::mutex> lock(_mtx);

    auto it = _bids.find(node_id);
    if (it == _bids.end()) return 0.0f;

    // 计算总加权分
    float total_weighted = 0.0f;
    for (const auto& [_, bid] : _bids) {
        total_weighted += bid.compute_score * bid.precision;
    }

    if (total_weighted <= 0.0f) return 1.0f / _bids.size();

    return (it->second.compute_score * it->second.precision) / total_weighted;
}

std::unordered_map<uint32_t, float> GTCScheduler::getAllAggregationWeights() const {
    std::lock_guard<std::mutex> lock(_mtx);

    float total_weighted = 0.0f;
    for (const auto& [_, bid] : _bids) {
        total_weighted += bid.compute_score * bid.precision;
    }

    std::unordered_map<uint32_t, float> weights;
    if (total_weighted <= 0.0f) {
        float uniform = 1.0f / _bids.size();
        for (const auto& [id, _] : _bids) {
            weights[id] = uniform;
        }
    } else {
        for (const auto& [id, bid] : _bids) {
            weights[id] = (bid.compute_score * bid.precision) / total_weighted;
        }
    }
    return weights;
}

std::vector<ShapleyValue> GTCScheduler::computeShapleyValues(
    const std::unordered_map<uint32_t, float>& performance_scores)
{
    std::lock_guard<std::mutex> lock(_mtx);

    std::vector<uint32_t> node_ids;
    node_ids.reserve(_bids.size());
    for (const auto& [id, _] : _bids) {
        node_ids.push_back(id);
    }

    size_t n = node_ids.size();
    std::vector<ShapleyValue> shapley_values;
    shapley_values.reserve(n);

    // 简化 Shapley 值计算：使用排列平均
    // 对每个节点，计算其加入所有可能子集时的边际贡献均值
    for (auto id : node_ids) {
        float shapley = 0.0f;
        size_t count = 0;

        // 枚举所有包含该节点的子集
        size_t total_subsets = 1 << n;
        for (size_t mask = 0; mask < total_subsets; ++mask) {
            if (!(mask & (1 << (std::find(node_ids.begin(), node_ids.end(), id) - node_ids.begin())))) {
                continue;
            }

            // 计算该子集的总性能
            float with_score = 0.0f;
            float without_score = 0.0f;
            auto it = performance_scores.find(id);
            float node_score = (it != performance_scores.end()) ? it->second : 0.0f;

            for (size_t j = 0; j < n; ++j) {
                if (mask & (1 << j)) {
                    auto sit = performance_scores.find(node_ids[j]);
                    with_score += (sit != performance_scores.end()) ? sit->second : 0.0f;
                    if (node_ids[j] != id) {
                        without_score += (sit != performance_scores.end()) ? sit->second : 0.0f;
                    }
                }
            }

            shapley += (with_score - without_score);
            count++;
        }

        if (count > 0) {
            shapley /= count;
        }

        ShapleyValue sv;
        sv.node_id = id;
        auto bit = _bids.find(id);
        sv.backend = (bit != _bids.end()) ? bit->second.backend : DeviceType::kCPU;
        sv.marginal_contribution = shapley;
        sv.rank_score = shapley;  // 直接使用 Shapley 值作为排名评分
        shapley_values.push_back(sv);
    }

    // 按 Shapley 值降序排序
    std::sort(shapley_values.begin(), shapley_values.end(),
        [](const ShapleyValue& a, const ShapleyValue& b) {
            return a.marginal_contribution > b.marginal_contribution;
        });

    return shapley_values;
}

void GTCScheduler::registerNode(uint32_t node_id, DeviceType backend) {
    auto backend_ptr = BackendManager::getInstance().getBackend(backend);
    float time_ms = 10.0f;  // 默认值
    size_t max_batch = 32;  // 默认值

    if (backend_ptr) {
        auto cap = backend_ptr->capability();
        // 根据算力估算处理时间
        time_ms = (cap.compute_throughput > 0.0f)
            ? (100.0f / cap.compute_throughput) : 10.0f;
        max_batch = static_cast<size_t>(
            cap.available_memory / (1024 * 1024));  // 简单估算
        if (max_batch < 1) max_batch = 1;
        if (max_batch > 1024) max_batch = 1024;
    }

    setNodeBid(node_id, backend, time_ms, max_batch);
}

void GTCScheduler::unregisterNode(uint32_t node_id) {
    std::lock_guard<std::mutex> lock(_mtx);
    _bids.erase(node_id);
}

std::vector<uint32_t> GTCScheduler::registeredNodes() const {
    std::lock_guard<std::mutex> lock(_mtx);
    std::vector<uint32_t> ids;
    ids.reserve(_bids.size());
    for (const auto& [id, _] : _bids) {
        ids.push_back(id);
    }
    return ids;
}

float GTCScheduler::computePriceOfAnarchy() const {
    std::lock_guard<std::mutex> lock(_mtx);
    if (_bids.empty()) return 1.0f;

    // 最优社会福利：所有 batch 分配给最快的节点
    float best_time = std::numeric_limits<float>::max();
    size_t total_batch = 0;
    for (const auto& [_, bid] : _bids) {
        best_time = std::min(best_time, bid.reported_time_ms);
        total_batch += bid.max_batch;
    }

    if (best_time >= std::numeric_limits<float>::max()) return 1.0f;

    float optimal_welfare = static_cast<float>(total_batch) / best_time;

    // 实际社会福利：按比例分配
    float actual_welfare = 0.0f;
    for (const auto& [_, bid] : _bids) {
        actual_welfare += static_cast<float>(bid.max_batch) / bid.reported_time_ms;
    }

    return (actual_welfare > 0.0f) ? (optimal_welfare / actual_welfare) : 1.0f;
}

std::vector<uint32_t> GTCScheduler::detectBidCheating(
    const std::unordered_map<uint32_t, float>& actual_times)
{
    std::lock_guard<std::mutex> lock(_mtx);
    std::vector<uint32_t> cheaters;

    for (const auto& [node_id, actual_time] : actual_times) {
        auto it = _bids.find(node_id);
        if (it != _bids.end()) {
            float reported = it->second.reported_time_ms;
            if (reported > 0.0f) {
                float deviation = std::abs(actual_time - reported) / reported;
                if (deviation > 0.20f) {  // 偏差 > 20%
                    cheaters.push_back(node_id);
                }
            }
        }
    }
    return cheaters;
}

std::vector<VCGAllocation> GTCScheduler::greedyAllocation(
    const std::vector<NodeBid>& nodes, size_t total_batch)
{
    // 按"单位时间吞吐量"降序排序
    std::vector<NodeBid> sorted = nodes;
    std::sort(sorted.begin(), sorted.end(),
        [](const NodeBid& a, const NodeBid& b) {
            float ta = (a.reported_time_ms > 0.0f) ? (1.0f / a.reported_time_ms) : 0.0f;
            float tb = (b.reported_time_ms > 0.0f) ? (1.0f / b.reported_time_ms) : 0.0f;
            return ta > tb;
        });

    std::vector<VCGAllocation> result;
    result.reserve(sorted.size());

    size_t remaining = total_batch;
    for (const auto& node : sorted) {
        if (remaining == 0) break;

        VCGAllocation alloc;
        alloc.node_id = node.node_id;
        alloc.backend = node.backend;
        alloc.reported_time_ms = node.reported_time_ms;
        alloc.batch_size = std::min(remaining, node.max_batch);
        // 社会福利 = 吞吐量 / 时间
        alloc.social_welfare = (node.reported_time_ms > 0.0f)
            ? (static_cast<float>(alloc.batch_size) / node.reported_time_ms) : 0.0f;
        alloc.vcg_payment = 0.0f;

        remaining -= alloc.batch_size;
        result.push_back(alloc);
    }

    return result;
}

} // namespace distributed
} // namespace ct