#include "Distributed/TopologyManager.h"

#include <queue>
#include <limits>
#include <algorithm>
#include <cmath>

namespace ct {
namespace distributed {

// ======================= 构造与析构 =======================

TopologyManager::TopologyManager(TopoNodeId local_node_id,
                                   TopologyConfig config)
    : _local_node_id(local_node_id)
    , _config(config)
    , _last_reconfiguration(std::chrono::steady_clock::now())
{
    // 初始化统计信息
    _stats.reset();
}

// ======================= 节点管理 =======================

void TopologyManager::registerNode(const TopologyNode& node) {
    std::lock_guard<std::mutex> lock(_mtx);

    auto it = _nodes.find(node.id);
    if (it == _nodes.end()) {
        // 新节点
        _stats.total_nodes_registered++;
    }

    // upsert 节点
    _nodes[node.id] = node;
    _nodes[node.id].last_seen = std::chrono::steady_clock::now();
}

void TopologyManager::unregisterNode(TopoNodeId node_id) {
    std::lock_guard<std::mutex> lock(_mtx);

    _nodes.erase(node_id);

    // 移除所有与该节点相关的连接
    for (auto it = _links.begin(); it != _links.end(); ) {
        if (it->first.first == node_id || it->first.second == node_id) {
            _stats.link_failures_detected++;
            it = _links.erase(it);
        } else {
            ++it;
        }
    }
}

std::shared_ptr<const TopologyNode> TopologyManager::getNode(TopoNodeId node_id) const {
    std::lock_guard<std::mutex> lock(_mtx);

    auto it = _nodes.find(node_id);
    if (it != _nodes.end()) {
        return std::make_shared<const TopologyNode>(it->second);
    }
    return nullptr;
}

std::vector<TopologyNode> TopologyManager::activeNodes() const {
    std::lock_guard<std::mutex> lock(_mtx);

    std::vector<TopologyNode> result;
    for (const auto& [id, node] : _nodes) {
        if (node.is_active) {
            result.push_back(node);
        }
    }
    return result;
}

// ======================= 连接管理 =======================

void TopologyManager::registerLink(const TopologyLink& link) {
    std::lock_guard<std::mutex> lock(_mtx);

    auto key = std::make_pair(link.node_a, link.node_b);
    auto it = _links.find(key);

    TopologyLink updated_link = link;
    // 计算兼容性评分
    if (_nodes.find(link.node_a) != _nodes.end() &&
        _nodes.find(link.node_b) != _nodes.end()) {
        updated_link.compatibility_score = computeBackendCompatibility(
            _nodes[link.node_a].backend_type,
            _nodes[link.node_b].backend_type);
    } else {
        updated_link.compatibility_score = 0.5f;
    }

    if (it == _links.end()) {
        // 新连接
        _stats.total_links_discovered++;
    }

    _links[key] = updated_link;
}

void TopologyManager::removeLink(TopoNodeId node_a, TopoNodeId node_b) {
    std::lock_guard<std::mutex> lock(_mtx);

    auto key = std::make_pair(node_a, node_b);
    auto it = _links.find(key);
    if (it != _links.end()) {
        _stats.link_failures_detected++;
        _links.erase(it);
    }
}

void TopologyManager::updateLatency(TopoNodeId node_id, float rtt_ms) {
    std::lock_guard<std::mutex> lock(_mtx);

    // 尝试两种顺序查找链接
    auto key1 = std::make_pair(_local_node_id, node_id);
    auto key2 = std::make_pair(node_id, _local_node_id);

    auto it = _links.find(key1);
    if (it != _links.end()) {
        it->second.rtt_ms = rtt_ms;
        it->second.measured_at = std::chrono::steady_clock::now();
        return;
    }

    it = _links.find(key2);
    if (it != _links.end()) {
        it->second.rtt_ms = rtt_ms;
        it->second.measured_at = std::chrono::steady_clock::now();
        return;
    }
}

void TopologyManager::updateBandwidth(TopoNodeId node_id, float bandwidth_mbps) {
    std::lock_guard<std::mutex> lock(_mtx);

    auto key1 = std::make_pair(_local_node_id, node_id);
    auto key2 = std::make_pair(node_id, _local_node_id);

    auto it = _links.find(key1);
    if (it != _links.end()) {
        it->second.bandwidth_mbps = bandwidth_mbps;
        it->second.measured_at = std::chrono::steady_clock::now();
        return;
    }

    it = _links.find(key2);
    if (it != _links.end()) {
        it->second.bandwidth_mbps = bandwidth_mbps;
        it->second.measured_at = std::chrono::steady_clock::now();
        return;
    }
}

void TopologyManager::updateStability(TopoNodeId node_id, bool success) {
    std::lock_guard<std::mutex> lock(_mtx);

    const float alpha = 0.3f;  // EMA 平滑因子
    float new_value = success ? 1.0f : 0.0f;

    auto key1 = std::make_pair(_local_node_id, node_id);
    auto key2 = std::make_pair(node_id, _local_node_id);

    auto it = _links.find(key1);
    if (it != _links.end()) {
        float old_stability = it->second.stability_score;
        it->second.stability_score = alpha * new_value + (1.0f - alpha) * old_stability;
        it->second.measured_at = std::chrono::steady_clock::now();
        return;
    }

    it = _links.find(key2);
    if (it != _links.end()) {
        float old_stability = it->second.stability_score;
        it->second.stability_score = alpha * new_value + (1.0f - alpha) * old_stability;
        it->second.measured_at = std::chrono::steady_clock::now();
        return;
    }
}

// ======================= 邻居选择 =======================

std::vector<TopoNodeId> TopologyManager::getBestNeighbors(size_t max_neighbors) const {
    std::lock_guard<std::mutex> lock(_mtx);

    size_t k = (max_neighbors == 0) ? _config.max_neighbors : max_neighbors;

    // 收集所有邻居节点及其评分
    std::vector<std::pair<TopoNodeId, float>> scored_neighbors;

    // 遍历所有链接，找到与本地节点相连的邻居
    for (const auto& [key, link] : _links) {
        TopoNodeId neighbor_id;
        if (key.first == _local_node_id) {
            neighbor_id = key.second;
        } else if (key.second == _local_node_id) {
            neighbor_id = key.first;
        } else {
            continue;
        }

        // 检查邻居节点是否存在且活跃
        auto node_it = _nodes.find(neighbor_id);
        if (node_it == _nodes.end() || !node_it->second.is_active) {
            continue;
        }

        float score = computeScore(_local_node_id, neighbor_id);
        scored_neighbors.emplace_back(neighbor_id, score);
    }

    // 按评分降序排序
    std::sort(scored_neighbors.begin(), scored_neighbors.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    // 取前 k 个
    std::vector<TopoNodeId> result;
    result.reserve(std::min(k, scored_neighbors.size()));
    for (size_t i = 0; i < std::min(k, scored_neighbors.size()); ++i) {
        result.push_back(scored_neighbors[i].first);
    }

    return result;
}

std::vector<TopoNodeId> TopologyManager::getCompatibleNeighbors(
    DeviceType target_backend, float min_score, size_t max_neighbors) const {
    std::lock_guard<std::mutex> lock(_mtx);

    size_t k = (max_neighbors == 0) ? _config.max_neighbors : max_neighbors;

    std::vector<std::pair<TopoNodeId, float>> compatible_neighbors;

    for (const auto& [key, link] : _links) {
        TopoNodeId neighbor_id;
        if (key.first == _local_node_id) {
            neighbor_id = key.second;
        } else if (key.second == _local_node_id) {
            neighbor_id = key.first;
        } else {
            continue;
        }

        auto node_it = _nodes.find(neighbor_id);
        if (node_it == _nodes.end() || !node_it->second.is_active) {
            continue;
        }

        float compat = computeBackendCompatibility(target_backend,
                                                    node_it->second.backend_type);
        if (compat >= min_score) {
            float score = computeScore(_local_node_id, neighbor_id);
            compatible_neighbors.emplace_back(neighbor_id, score);
        }
    }

    std::sort(compatible_neighbors.begin(), compatible_neighbors.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    std::vector<TopoNodeId> result;
    result.reserve(std::min(k, compatible_neighbors.size()));
    for (size_t i = 0; i < std::min(k, compatible_neighbors.size()); ++i) {
        result.push_back(compatible_neighbors[i].first);
    }

    return result;
}

float TopologyManager::computeScore(TopoNodeId node_a, TopoNodeId node_b) const {
    // 获取链接信息
    auto link = getLinkInternal(node_a, node_b);
    if (!link) {
        return 0.0f;
    }

    // 获取节点信息
    auto it_a = _nodes.find(node_a);
    auto it_b = _nodes.find(node_b);
    if (it_a == _nodes.end() || it_b == _nodes.end()) {
        return 0.0f;
    }

    // 归一化延迟评分：延迟越低越好
    float norm_latency = normalizeLatency(link->rtt_ms);

    // 归一化带宽评分：带宽越高越好
    float norm_bandwidth = normalizeBandwidth(link->bandwidth_mbps);

    // 兼容性评分
    float compat = link->compatibility_score;

    // 稳定性评分
    float stability = link->stability_score;

    // 加权综合评分
    float score = _config.latency_weight * norm_latency +
                  _config.bandwidth_weight * norm_bandwidth +
                  _config.compatibility_weight * compat +
                  _config.stability_weight * stability;

    // 归一化到 [0.0, 1.0]
    float total_weight = _config.latency_weight +
                         _config.bandwidth_weight +
                         _config.compatibility_weight +
                         _config.stability_weight;

    if (total_weight > 0.0f) {
        score /= total_weight;
    }

    return std::clamp(score, 0.0f, 1.0f);
}

float TopologyManager::computeBackendCompatibility(DeviceType backend_a,
                                                    DeviceType backend_b) const {
    // 相同后端类型 → 完全兼容
    if (backend_a == backend_b) {
        return 1.0f;
    }

    // 定义后端兼容性组
    // CPU 族：kCPU, kSIMD
    // GPU 族：kCUDA, kMPS
    // 通用族：kUNKNOWN, kGENERAL
    // 特殊：kAMX（Apple 矩阵加速器）

    auto isCPU = [](DeviceType t) -> bool {
        return t == DeviceType::kCPU || t == DeviceType::kSIMD;
    };

    auto isGPU = [](DeviceType t) -> bool {
        return t == DeviceType::kCUDA || t == DeviceType::kMPS;
    };

    auto isGeneric = [](DeviceType t) -> bool {
        return t == DeviceType::kUNKNOWN || t == DeviceType::kGENERAL;
    };

    // 同一族内 → 兼容
    if ((isCPU(backend_a) && isCPU(backend_b)) ||
        (isGPU(backend_a) && isGPU(backend_b)) ||
        (isGeneric(backend_a) && isGeneric(backend_b))) {
        return 0.7f;
    }

    // 通用类型与任何类型兼容性中等
    if (isGeneric(backend_a) || isGeneric(backend_b)) {
        return 0.5f;
    }

    // AMX 与 GPU 有一定兼容性（都是 Apple 加速硬件）
    if ((backend_a == DeviceType::kAMX && isGPU(backend_b)) ||
        (backend_b == DeviceType::kAMX && isGPU(backend_a))) {
        return 0.5f;
    }

    // AMX 与其他类型兼容性较低
    if (backend_a == DeviceType::kAMX || backend_b == DeviceType::kAMX) {
        return 0.3f;
    }

    // CPU 与 GPU 跨族 → 不兼容
    return 0.3f;
}

// ======================= 拓扑重构 =======================

std::vector<TopoNodeId> TopologyManager::detectStaleNodes() {
    std::lock_guard<std::mutex> lock(_mtx);

    std::vector<TopoNodeId> stale_nodes;
    auto now = std::chrono::steady_clock::now();

    for (auto& [id, node] : _nodes) {
        if (!node.is_active) {
            continue;  // 跳过已标记为不活跃的节点
        }

        auto elapsed = std::chrono::duration_cast<std::chrono::duration<float>>(
            now - node.last_seen).count();

        if (elapsed > _config.stale_timeout_s) {
            node.is_active = false;
            stale_nodes.push_back(id);
            _stats.node_failures_detected++;
        }
    }

    return stale_nodes;
}

bool TopologyManager::reconfigure() {
    std::lock_guard<std::mutex> lock(_mtx);

    bool topology_changed = false;

    // 1. 检测并移除失效节点
    auto stale_nodes = detectStaleNodes();
    if (!stale_nodes.empty()) {
        topology_changed = true;
        for (auto node_id : stale_nodes) {
            // 移除与该节点相关的所有连接
            for (auto it = _links.begin(); it != _links.end(); ) {
                if (it->first.first == node_id || it->first.second == node_id) {
                    _stats.link_failures_detected++;
                    it = _links.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

    // 2. 修剪低评分连接
    size_t pruned_links = 0;
    for (auto it = _links.begin(); it != _links.end(); ) {
        // 检查连接的两端节点是否都存在且活跃
        auto node_a_it = _nodes.find(it->first.first);
        auto node_b_it = _nodes.find(it->first.second);

        if (node_a_it == _nodes.end() || node_b_it == _nodes.end() ||
            !node_a_it->second.is_active || !node_b_it->second.is_active) {
            // 节点不存在或不活跃，移除连接
            _stats.link_failures_detected++;
            it = _links.erase(it);
            pruned_links++;
            topology_changed = true;
            continue;
        }

        // 检查连接评分是否低于阈值
        float score = computeScore(it->first.first, it->first.second);
        if (score < _config.min_compatibility_threshold) {
            _stats.link_failures_detected++;
            it = _links.erase(it);
            pruned_links++;
            topology_changed = true;
            continue;
        }

        ++it;
    }

    // 3. 更新统计信息
    if (topology_changed) {
        _stats.reconfigurations++;
        // 更新平均邻居数
        size_t total_neighbors = 0;
        size_t node_count = 0;
        for (const auto& [id, node] : _nodes) {
            if (node.is_active) {
                node_count++;
                size_t neighbor_count = 0;
                for (const auto& [key, _] : _links) {
                    if (key.first == id || key.second == id) {
                        neighbor_count++;
                    }
                }
                total_neighbors += neighbor_count;
            }
        }
        if (node_count > 0) {
            _stats.avg_neighbor_count = static_cast<float>(total_neighbors) /
                                         static_cast<float>(node_count);
        }

        // 更新图直径
        _stats.avg_graph_diameter = static_cast<float>(computeGraphDiameter());
    }

    // 更新重构时间戳
    _last_reconfiguration = std::chrono::steady_clock::now();

    return topology_changed;
}

bool TopologyManager::needsReconfiguration() const {
    std::lock_guard<std::mutex> lock(_mtx);

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<float>>(
        now - _last_reconfiguration).count();

    return elapsed > _config.reconfiguration_interval_s;
}

// ======================= 拓扑分析 =======================

TopologySnapshot TopologyManager::getSnapshot() const {
    std::lock_guard<std::mutex> lock(_mtx);

    TopologySnapshot snapshot;
    snapshot.num_active_nodes = 0;
    snapshot.num_links = _links.size();

    // 收集所有节点
    snapshot.nodes.reserve(_nodes.size());
    for (const auto& [id, node] : _nodes) {
        snapshot.nodes.push_back(node);
        if (node.is_active) {
            snapshot.num_active_nodes++;
        }
    }

    // 收集所有连接并计算延迟/带宽统计
    snapshot.links.reserve(_links.size());
    float total_latency = 0.0f;
    float total_bandwidth = 0.0f;
    snapshot.max_latency_ms = 0.0f;

    for (const auto& [key, link] : _links) {
        snapshot.links.push_back(link);
        total_latency += link.rtt_ms;
        total_bandwidth += link.bandwidth_mbps;
        if (link.rtt_ms > snapshot.max_latency_ms) {
            snapshot.max_latency_ms = link.rtt_ms;
        }
    }

    snapshot.avg_latency_ms = (_links.size() > 0)
        ? total_latency / static_cast<float>(_links.size())
        : 0.0f;

    snapshot.avg_bandwidth_mbps = (_links.size() > 0)
        ? total_bandwidth / static_cast<float>(_links.size())
        : 0.0f;

    // 计算图连通性
    snapshot.graph_connectivity = computeGraphConnectivity();

    return snapshot;
}

float TopologyManager::computeGraphConnectivity() const {
    if (_nodes.empty()) {
        return 0.0f;
    }

    // 构建邻接表
    std::unordered_map<TopoNodeId, std::vector<TopoNodeId>> adj;
    for (const auto& [id, node] : _nodes) {
        if (node.is_active) {
            adj[id] = {};
        }
    }

    for (const auto& [key, link] : _links) {
        auto it_a = _nodes.find(key.first);
        auto it_b = _nodes.find(key.second);
        if (it_a != _nodes.end() && it_b != _nodes.end() &&
            it_a->second.is_active && it_b->second.is_active) {
            adj[key.first].push_back(key.second);
            adj[key.second].push_back(key.first);
        }
    }

    if (adj.empty()) {
        return 0.0f;
    }

    // BFS 寻找最大连通分量
    std::unordered_set<TopoNodeId> visited;
    size_t largest_component = 0;

    for (const auto& [start_id, _] : adj) {
        if (visited.find(start_id) != visited.end()) {
            continue;
        }

        // BFS
        size_t component_size = 0;
        std::queue<TopoNodeId> q;
        q.push(start_id);
        visited.insert(start_id);

        while (!q.empty()) {
            TopoNodeId current = q.front();
            q.pop();
            component_size++;

            for (const auto& neighbor : adj[current]) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);
                    q.push(neighbor);
                }
            }
        }

        if (component_size > largest_component) {
            largest_component = component_size;
        }
    }

    return static_cast<float>(largest_component) /
           static_cast<float>(adj.size());
}

size_t TopologyManager::computeGraphDiameter() const {
    if (_nodes.empty()) {
        return 0;
    }

    // 构建邻接表（仅活跃节点）
    std::unordered_map<TopoNodeId, std::vector<TopoNodeId>> adj;
    for (const auto& [id, node] : _nodes) {
        if (node.is_active) {
            adj[id] = {};
        }
    }

    for (const auto& [key, link] : _links) {
        auto it_a = _nodes.find(key.first);
        auto it_b = _nodes.find(key.second);
        if (it_a != _nodes.end() && it_b != _nodes.end() &&
            it_a->second.is_active && it_b->second.is_active) {
            adj[key.first].push_back(key.second);
            adj[key.second].push_back(key.first);
        }
    }

    if (adj.empty()) {
        return 0;
    }

    // 对每个节点执行 BFS，找到最长最短路径
    size_t diameter = 0;

    for (const auto& [start_id, _] : adj) {
        // BFS 求从 start_id 到所有节点的最短路径
        std::unordered_map<TopoNodeId, size_t> dist;
        std::queue<TopoNodeId> q;

        dist[start_id] = 0;
        q.push(start_id);

        while (!q.empty()) {
            TopoNodeId current = q.front();
            q.pop();

            for (const auto& neighbor : adj[current]) {
                if (dist.find(neighbor) == dist.end()) {
                    dist[neighbor] = dist[current] + 1;
                    q.push(neighbor);
                    if (dist[neighbor] > diameter) {
                        diameter = dist[neighbor];
                    }
                }
            }
        }
    }

    return diameter;
}

bool TopologyManager::hasDirectLink(TopoNodeId node_a, TopoNodeId node_b) const {
    std::lock_guard<std::mutex> lock(_mtx);

    auto key = std::make_pair(node_a, node_b);
    return _links.find(key) != _links.end();
}

std::shared_ptr<const TopologyLink> TopologyManager::getLinkInternal(
    TopoNodeId node_a, TopoNodeId node_b) const {
    // 尝试两种顺序
    auto key1 = std::make_pair(node_a, node_b);
    auto it = _links.find(key1);
    if (it != _links.end()) {
        return std::make_shared<const TopologyLink>(it->second);
    }

    auto key2 = std::make_pair(node_b, node_a);
    it = _links.find(key2);
    if (it != _links.end()) {
        return std::make_shared<const TopologyLink>(it->second);
    }

    return nullptr;
}

} // namespace distributed
} // namespace ct