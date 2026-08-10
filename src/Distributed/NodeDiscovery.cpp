#include "Distributed/NodeDiscovery.h"

#include <numeric>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>

namespace ct {
namespace distributed {

// ======================= 辅助函数 =======================

/**
 * @brief 获取当前时间戳（毫秒）
 */
static inline double nowMs() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();
}

/**
 * @brief 标准正态分布 CDF 近似
 */
static inline double normalCDF(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// ======================= 构造与析构 =======================

NodeDiscovery::NodeDiscovery(uint32_t local_node_id, DiscoveryConfig config)
    : _local_node_id(local_node_id)
    , _config(config)
    , _heartbeat_seq(0)
    , _estimated_load(0.5f)
{
    std::random_device rd;
    _rng.seed(rd());
}

// ======================= 节点注册 =======================

void NodeDiscovery::registerSeedNode(const NodeEndpoint& endpoint) {
    std::lock_guard<std::mutex> lock(_mtx);

    // 添加到种子节点列表
    _seed_nodes.push_back(endpoint);

    // 记录发现（如果尚未存在）
    auto it = _node_statuses.find(endpoint.node_id);
    if (it == _node_statuses.end()) {
        _node_statuses[endpoint.node_id] = NodeStatus::Alive;
        _node_endpoints[endpoint.node_id] = endpoint;
        _stats.total_nodes_discovered++;
        if (_discovery_callback) {
            _discovery_callback(endpoint);
        }
    }
}

void NodeDiscovery::registerSeedNodes(const std::vector<NodeEndpoint>& endpoints) {
    for (const auto& ep : endpoints) {
        registerSeedNode(ep);
    }
}

// ======================= 发现与离开 =======================

void NodeDiscovery::recordDiscovery(const NodeEndpoint& endpoint) {
    std::lock_guard<std::mutex> lock(_mtx);

    auto it = _node_statuses.find(endpoint.node_id);
    if (it != _node_statuses.end()) {
        // 节点已存在 — 如果状态是 Dead/Left，恢复为 Alive
        if (it->second == NodeStatus::Dead || it->second == NodeStatus::Left) {
            NodeStatus old = it->second;
            it->second = NodeStatus::Alive;
            _node_endpoints[endpoint.node_id] = endpoint;
            if (_status_callback) {
                _status_callback(endpoint.node_id, old, NodeStatus::Alive);
            }
        }
    } else {
        // 新节点
        _node_statuses[endpoint.node_id] = NodeStatus::Alive;
        _node_endpoints[endpoint.node_id] = endpoint;
        _stats.total_nodes_discovered++;
        if (_discovery_callback) {
            _discovery_callback(endpoint);
        }
    }
}

void NodeDiscovery::recordLeave(uint32_t node_id) {
    std::lock_guard<std::mutex> lock(_mtx);

    auto it = _node_statuses.find(node_id);
    if (it != _node_statuses.end() && it->second != NodeStatus::Dead) {
        NodeStatus old = it->second;
        it->second = NodeStatus::Left;
        if (_status_callback) {
            _status_callback(node_id, old, NodeStatus::Left);
        }
    }
}

// ======================= 心跳管理 =======================

HeartbeatMessage NodeDiscovery::generateHeartbeat() {
    std::lock_guard<std::mutex> lock(_mtx);

    _heartbeat_seq++;

    // 收集存活节点（不调用 aliveNodes() 以避免重入锁）
    std::vector<uint32_t> alive;
    alive.reserve(_node_statuses.size());
    for (const auto& [nid, status] : _node_statuses) {
        if (status == NodeStatus::Alive && nid != _local_node_id) {
            alive.push_back(nid);
        }
    }

    HeartbeatMessage msg;
    msg.node_id = _local_node_id;
    msg.sequence_number = _heartbeat_seq;
    msg.timestamp = std::chrono::steady_clock::now();
    msg.load_factor = _estimated_load;
    msg.gossip_nodes = std::move(alive);

    _stats.heartbeats_sent++;
    return msg;
}

void NodeDiscovery::processHeartbeat(const HeartbeatMessage& heartbeat) {
    std::lock_guard<std::mutex> lock(_mtx);

    uint32_t node_id = heartbeat.node_id;

    // 1. 未知节点 → 记录为新发现
    auto it = _node_statuses.find(node_id);
    if (it == _node_statuses.end()) {
        NodeEndpoint endpoint;
        endpoint.node_id = node_id;
        endpoint.protocol = DiscoveryProtocol::Gossip;
        // host/port/backend_type/version 在仅有心跳时不可知
        _node_statuses[node_id] = NodeStatus::Alive;
        _node_endpoints[node_id] = endpoint;
        _stats.total_nodes_discovered++;
        if (_discovery_callback) {
            _discovery_callback(endpoint);
        }
    } else {
        // 更新状态为 Alive
        NodeStatus old = it->second;
        it->second = NodeStatus::Alive;
        if (old != NodeStatus::Alive && _status_callback) {
            _status_callback(node_id, old, NodeStatus::Alive);
        }
    }

    // 2. 计算心跳间隔并更新 Phi 状态
    auto& phi_state = _phi_states[node_id];
    double now_ms_val = nowMs();
    double interval_ms = 0.0;
    if (phi_state.last_heartbeat_time_ms > 0.0) {
        interval_ms = now_ms_val - phi_state.last_heartbeat_time_ms;
    }
    phi_state.last_heartbeat_time_ms = now_ms_val;

    if (interval_ms > 0.0) {
        phi_state.interval_history.push_back(interval_ms);
        if (phi_state.interval_history.size() > phi_state.max_history_size) {
            phi_state.interval_history.erase(phi_state.interval_history.begin());
        }
    }

    // 3. 处理 Gossip 节点
    for (uint32_t gossip_id : heartbeat.gossip_nodes) {
        if (gossip_id == _local_node_id) continue;
        if (_node_statuses.find(gossip_id) == _node_statuses.end()) {
            NodeEndpoint gossip_ep;
            gossip_ep.node_id = gossip_id;
            gossip_ep.protocol = DiscoveryProtocol::Gossip;
            _node_statuses[gossip_id] = NodeStatus::Alive;
            _node_endpoints[gossip_id] = gossip_ep;
            _stats.total_nodes_discovered++;
            if (_discovery_callback) {
                _discovery_callback(gossip_ep);
            }
        }
    }

    _stats.heartbeats_received++;
}

// ======================= 故障检测 =======================

double NodeDiscovery::computePhi(uint32_t node_id) const {
    std::lock_guard<std::mutex> lock(_mtx);

    auto it = _phi_states.find(node_id);
    if (it == _phi_states.end()) {
        return 0.0;
    }
    return computePhiValue(it->second);
}

std::vector<uint32_t> NodeDiscovery::detectFailures() {
    std::lock_guard<std::mutex> lock(_mtx);

    std::vector<uint32_t> newly_dead;
    double now_ms_val = nowMs();

    for (auto& [node_id, status] : _node_statuses) {
        if (node_id == _local_node_id) continue;

        if (status == NodeStatus::Alive) {
            // 计算 Phi 值，判断是否变为 Suspect
            auto phi_it = _phi_states.find(node_id);
            if (phi_it != _phi_states.end()) {
                double phi = computePhiValue(phi_it->second);
                if (phi > _config.phi_threshold) {
                    status = NodeStatus::Suspect;
                    _stats.phi_failures++;
                    if (_status_callback) {
                        _status_callback(node_id, NodeStatus::Alive, NodeStatus::Suspect);
                    }
                }
            }
        } else if (status == NodeStatus::Suspect) {
            // 检查心跳丢失次数是否超过阈值
            auto phi_it = _phi_states.find(node_id);
            if (phi_it != _phi_states.end()) {
                double elapsed_ms = now_ms_val - phi_it->second.last_heartbeat_time_ms;
                size_t missed = static_cast<size_t>(elapsed_ms / _config.heartbeat_interval_ms);
                if (missed >= _config.max_heartbeat_loss) {
                    status = NodeStatus::Dead;
                    _stats.total_nodes_lost++;
                    newly_dead.push_back(node_id);
                    if (_status_callback) {
                        _status_callback(node_id, NodeStatus::Suspect, NodeStatus::Dead);
                    }
                }
            }
        }
    }

    return newly_dead;
}

// ======================= 状态查询 =======================

NodeStatus NodeDiscovery::getNodeStatus(uint32_t node_id) const {
    std::lock_guard<std::mutex> lock(_mtx);

    auto it = _node_statuses.find(node_id);
    if (it == _node_statuses.end()) {
        return NodeStatus::Unknown;
    }
    return it->second;
}

std::vector<uint32_t> NodeDiscovery::aliveNodes() const {
    std::lock_guard<std::mutex> lock(_mtx);

    std::vector<uint32_t> result;
    result.reserve(_node_statuses.size());
    for (const auto& [nid, status] : _node_statuses) {
        if (status == NodeStatus::Alive && nid != _local_node_id) {
            result.push_back(nid);
        }
    }
    return result;
}

std::vector<uint32_t> NodeDiscovery::suspectNodes() const {
    std::lock_guard<std::mutex> lock(_mtx);

    std::vector<uint32_t> result;
    for (const auto& [nid, status] : _node_statuses) {
        if (status == NodeStatus::Suspect) {
            result.push_back(nid);
        }
    }
    return result;
}

bool NodeDiscovery::isAlive(uint32_t node_id) const {
    std::lock_guard<std::mutex> lock(_mtx);

    auto it = _node_statuses.find(node_id);
    return it != _node_statuses.end() && it->second == NodeStatus::Alive;
}

// ======================= 内部方法 =======================

void NodeDiscovery::updateNodeStatus(uint32_t node_id, NodeStatus new_status) {
    std::lock_guard<std::mutex> lock(_mtx);

    auto it = _node_statuses.find(node_id);
    NodeStatus old = (it != _node_statuses.end()) ? it->second : NodeStatus::Unknown;
    _node_statuses[node_id] = new_status;
    if (_status_callback) {
        _status_callback(node_id, old, new_status);
    }
}

void NodeDiscovery::updatePhiState(uint32_t node_id, double interval_ms) {
    std::lock_guard<std::mutex> lock(_mtx);

    auto& state = _phi_states[node_id];
    state.interval_history.push_back(interval_ms);
    if (state.interval_history.size() > state.max_history_size) {
        state.interval_history.erase(state.interval_history.begin());
    }
    state.last_heartbeat_time_ms = nowMs();
}

double NodeDiscovery::computePhiValue(const FailureDetectorState& state) const {
    if (state.interval_history.empty()) {
        return 0.0;
    }

    // 计算均值
    double sum = 0.0;
    for (double interval : state.interval_history) {
        sum += interval;
    }
    double mean = sum / static_cast<double>(state.interval_history.size());

    // 计算标准差
    double sq_sum = 0.0;
    for (double interval : state.interval_history) {
        double diff = interval - mean;
        sq_sum += diff * diff;
    }
    double variance = sq_sum / static_cast<double>(state.interval_history.size());
    double stddev = std::sqrt(variance);

    if (stddev == 0.0) {
        return 0.0;
    }

    // 计算当前间隔
    double now_ms_val = nowMs();
    double current_interval = now_ms_val - state.last_heartbeat_time_ms;
    if (current_interval <= 0.0) {
        return 0.0;
    }

    // Phi = -log10(P(X > current_interval))
    // P(X > current_interval) = 1 - Φ((current_interval - mean) / stddev)
    double z = (current_interval - mean) / stddev;
    double p = 1.0 - normalCDF(z);
    p = std::max(p, 1e-15); // 避免 log10(0)

    return -std::log10(p);
}

std::vector<uint32_t> NodeDiscovery::selectGossipFanout() {
    std::lock_guard<std::mutex> lock(_mtx);

    // 收集存活节点（不调用 aliveNodes() 以避免重入锁）
    std::vector<uint32_t> alive;
    alive.reserve(_node_statuses.size());
    for (const auto& [nid, status] : _node_statuses) {
        if (status == NodeStatus::Alive && nid != _local_node_id) {
            alive.push_back(nid);
        }
    }

    std::vector<uint32_t> result;
    if (alive.empty() || _config.gossip_fanout == 0) {
        return result;
    }

    size_t fanout = std::min(_config.gossip_fanout, alive.size());
    std::shuffle(alive.begin(), alive.end(), _rng);
    result.assign(alive.begin(), alive.begin() + static_cast<ptrdiff_t>(fanout));

    _stats.gossip_messages++;
    return result;
}

float NodeDiscovery::estimateLoadFactor() const {
    return _estimated_load;
}

bool NodeDiscovery::isSeedNode(uint32_t node_id) const {
    std::lock_guard<std::mutex> lock(_mtx);

    for (const auto& seed : _seed_nodes) {
        if (seed.node_id == node_id) {
            return true;
        }
    }
    return false;
}

} // namespace distributed
} // namespace ct