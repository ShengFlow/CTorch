#include "Distributed/FaultTolerance.h"

#include <cstring>
#include <numeric>

namespace ct {
namespace distributed {

// ======================= CRDTSnapshot 序列化 / 反序列化 =======================

std::vector<uint8_t> CRDTSnapshot::serialize() const {
    std::vector<uint8_t> buf;

    // helper: append raw bytes
    auto append = [&](const void* data, size_t bytes) {
        const auto* p = static_cast<const uint8_t*>(data);
        buf.insert(buf.end(), p, p + bytes);
    };

    // snapshot_id
    append(&snapshot_id, sizeof(snapshot_id));

    // timestamp → nanoseconds since epoch
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  timestamp.time_since_epoch()).count();
    append(&ns, sizeof(ns));

    // momentum
    uint64_t sz = momentum.size();
    append(&sz, sizeof(sz));
    if (sz > 0) append(momentum.data(), sz * sizeof(float));

    // version_vector
    sz = version_vector.size();
    append(&sz, sizeof(sz));
    if (sz > 0) append(version_vector.data(), sz * sizeof(uint64_t));

    // global_step
    append(&global_step, sizeof(global_step));

    // local_step
    append(&local_step, sizeof(local_step));

    // grad_counter
    sz = grad_counter.size();
    append(&sz, sizeof(sz));
    if (sz > 0) append(grad_counter.data(), sz * sizeof(uint64_t));

    // serialized_params
    sz = serialized_params.size();
    append(&sz, sizeof(sz));
    if (sz > 0) append(serialized_params.data(), sz);

    return buf;
}

CRDTSnapshot CRDTSnapshot::deserialize(const std::vector<uint8_t>& data) {
    CRDTSnapshot snap;
    size_t offset = 0;

    auto read = [&](void* dst, size_t bytes) -> bool {
        if (offset + bytes > data.size()) return false;
        std::memcpy(dst, data.data() + offset, bytes);
        offset += bytes;
        return true;
    };

    // snapshot_id
    read(&snap.snapshot_id, sizeof(snap.snapshot_id));

    // timestamp
    int64_t ns = 0;
    read(&ns, sizeof(ns));
    snap.timestamp = std::chrono::steady_clock::time_point(
        std::chrono::nanoseconds(ns));

    // momentum
    uint64_t sz = 0;
    read(&sz, sizeof(sz));
    snap.momentum.resize(sz);
    if (sz > 0) read(snap.momentum.data(), sz * sizeof(float));

    // version_vector
    read(&sz, sizeof(sz));
    snap.version_vector.resize(sz);
    if (sz > 0) read(snap.version_vector.data(), sz * sizeof(uint64_t));

    // global_step
    read(&snap.global_step, sizeof(snap.global_step));

    // local_step
    read(&snap.local_step, sizeof(snap.local_step));

    // grad_counter
    read(&sz, sizeof(sz));
    snap.grad_counter.resize(sz);
    if (sz > 0) read(snap.grad_counter.data(), sz * sizeof(uint64_t));

    // serialized_params
    read(&sz, sizeof(sz));
    snap.serialized_params.resize(sz);
    if (sz > 0) read(snap.serialized_params.data(), sz);

    return snap;
}

// ======================= 构造 / 析构 =======================

FaultTolerance::FaultTolerance(uint32_t local_node_id,
                               FaultToleranceConfig config)
    : _local_node_id(local_node_id)
    , _config(config)
    , _next_snapshot_id(1)
    , _last_snapshot_time(std::chrono::steady_clock::now())
{
}

// ======================= 快照管理 =======================

uint64_t FaultTolerance::takeSnapshot(const std::vector<Tensor*>& params) {
    std::lock_guard<std::mutex> lock(_mtx);

    CRDTSnapshot snap;
    snap.snapshot_id = _next_snapshot_id++;
    snap.timestamp = std::chrono::steady_clock::now();

    if (_state_provider) {
        // 如果注册了状态提供回调，直接使用
        snap = _state_provider();
        snap.snapshot_id = _next_snapshot_id - 1;  // 保留我们分配的 ID
        snap.timestamp = std::chrono::steady_clock::now();
    } else if (!params.empty()) {
        // 从参数构建快照：序列化所有 Tensor 的 float 数据
        std::vector<uint8_t> serialized;
        size_t num_tensors = params.size();
        serialized.resize(sizeof(num_tensors));
        std::memcpy(serialized.data(), &num_tensors, sizeof(num_tensors));

        for (const auto* t : params) {
            if (!t) continue;
            // shape
            auto shape = t->shape();
            uint64_t ndim = shape.size();
            size_t off = serialized.size();
            serialized.resize(off + sizeof(ndim) + ndim * sizeof(size_t) + sizeof(uint64_t));
            std::memcpy(serialized.data() + off, &ndim, sizeof(ndim));
            off += sizeof(ndim);
            std::memcpy(serialized.data() + off, shape.data(), ndim * sizeof(size_t));
            off += ndim * sizeof(size_t);

            // raw float data
            uint64_t numel = static_cast<uint64_t>(t->numel());
            std::memcpy(serialized.data() + off, &numel, sizeof(numel));
            off += sizeof(numel);
            if (numel > 0) {
                serialized.resize(off + numel * sizeof(float));
                std::memcpy(serialized.data() + off, t->data_read<float>(), numel * sizeof(float));
            }
        }
        snap.serialized_params = std::move(serialized);
    }

    _snapshots.push_back(std::move(snap));

    pruneSnapshots();

    // 更新统计
    _stats.total_snapshots++;
    if (!_snapshots.empty()) {
        _stats.avg_snapshot_size_bytes = _snapshots.back().serialized_params.size();
    }

    _last_snapshot_time = std::chrono::steady_clock::now();

    return _snapshots.back().snapshot_id;
}

CRDTSnapshot FaultTolerance::latestSnapshot() const {
    std::lock_guard<std::mutex> lock(_mtx);
    if (_snapshots.empty()) return CRDTSnapshot{};
    return _snapshots.back();
}

CRDTSnapshot FaultTolerance::getSnapshot(uint64_t snapshot_id) const {
    std::lock_guard<std::mutex> lock(_mtx);
    for (const auto& s : _snapshots) {
        if (s.snapshot_id == snapshot_id) return s;
    }
    return CRDTSnapshot{};
}

std::vector<CRDTSnapshot> FaultTolerance::allSnapshots() const {
    std::lock_guard<std::mutex> lock(_mtx);
    return _snapshots;
}

bool FaultTolerance::needsSnapshot() const {
    std::lock_guard<std::mutex> lock(_mtx);
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<float>(now - _last_snapshot_time).count();
    return elapsed >= _config.snapshot_interval_s;
}

// ======================= 分区检测 =======================

std::vector<uint32_t> FaultTolerance::detectPartitions(
    const std::vector<uint32_t>& alive_nodes) {
    std::lock_guard<std::mutex> lock(_mtx);
    std::vector<uint32_t> newly_confirmed;
    auto now = std::chrono::steady_clock::now();

    for (auto node_id : alive_nodes) {
        auto it = _partitions.find(node_id);
        if (it == _partitions.end()) {
            // 首次见到该节点，创建 Connected 记录
            PartitionEntry entry;
            entry.node_id = node_id;
            entry.status = PartitionStatus::Connected;
            entry.detected_at = now;
            entry.missed_heartbeats = 0;
            _partitions[node_id] = entry;
            continue;
        }

        auto& entry = it->second;

        // 已恢复的节点不再检测
        if (entry.status == PartitionStatus::Recovered) continue;

        // 计算自上次心跳以来的时间
        std::chrono::steady_clock::time_point last_contact = entry.last_known_state.timestamp;
        // 如果 last_known_state 是默认构造的（epoch），使用 detected_at 作为参考
        if (last_contact.time_since_epoch().count() == 0) {
            last_contact = entry.detected_at;
        }

        float elapsed = std::chrono::duration<float>(now - last_contact).count();

        if (elapsed < _config.partition_threshold_s) {
            // 还在阈值内，正常
            continue;
        }

        entry.missed_heartbeats++;

        if (entry.status == PartitionStatus::Connected) {
            // 首次超时 → Suspected
            entry.status = PartitionStatus::Suspected;
            entry.detected_at = now;
        } else if (entry.status == PartitionStatus::Suspected) {
            // 持续超时 → Confirmed
            entry.status = PartitionStatus::Confirmed;
            entry.detected_at = now;
            newly_confirmed.push_back(node_id);
            _stats.partitions_detected++;
        }
    }

    return newly_confirmed;
}

void FaultTolerance::recordPeerState(uint32_t node_id,
                                      const CRDTSnapshot& snapshot) {
    std::lock_guard<std::mutex> lock(_mtx);
    auto it = _partitions.find(node_id);
    if (it == _partitions.end()) {
        PartitionEntry entry;
        entry.node_id = node_id;
        entry.status = PartitionStatus::Connected;
        entry.detected_at = std::chrono::steady_clock::now();
        entry.last_known_state = snapshot;
        entry.missed_heartbeats = 0;
        _partitions[node_id] = entry;
    } else {
        it->second.last_known_state = snapshot;
        // 收到状态更新说明连接正常，重置状态
        if (it->second.status == PartitionStatus::Suspected ||
            it->second.status == PartitionStatus::Confirmed) {
            it->second.status = PartitionStatus::Connected;
            it->second.missed_heartbeats = 0;
        }
    }
}

RecoveryPlan FaultTolerance::markRecovery(uint32_t node_id) {
    std::lock_guard<std::mutex> lock(_mtx);

    // 获取本地最新快照
    CRDTSnapshot local_snap;
    if (!_snapshots.empty()) {
        local_snap = _snapshots.back();
    }

    // 查找分区记录
    auto it = _partitions.find(node_id);
    if (it == _partitions.end()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::TENSOR_STATE,
            "FaultTolerance::markRecovery: no partition entry for node " +
            std::to_string(node_id));
    }

    CRDTSnapshot remote_snap = it->second.last_known_state;

    // 执行 CRDT 合并
    CRDTSnapshot merged = mergeCRDT(local_snap, remote_snap);

    // 构建恢复计划
    RecoveryPlan plan;
    plan.target_node_id = node_id;
    plan.local_snapshot = local_snap;
    plan.remote_snapshot = remote_snap;
    plan.merged_version = merged.version_vector;
    plan.merged_global_step = merged.global_step;
    plan.requires_full_sync = needsFullSync(local_snap, remote_snap);

    // 计算需要同步的参数字节数
    if (plan.requires_full_sync) {
        plan.param_bytes_to_sync = local_snap.serialized_params.size()
                                 + remote_snap.serialized_params.size();
    } else {
        // 增量同步：只同步差异部分（简化处理：取两者中较大的）
        plan.param_bytes_to_sync = std::max(
            local_snap.serialized_params.size(),
            remote_snap.serialized_params.size());
    }

    // 标记恢复
    it->second.status = PartitionStatus::Recovered;
    it->second.recovered_at = std::chrono::steady_clock::now();

    _stats.partitions_recovered++;

    return plan;
}

// ======================= 恢复 =======================

RecoveryPlan FaultTolerance::recover(uint32_t node_id) {
    // 获取本地快照（在 markRecovery 内部也会加锁，所以这里不需要提前加锁）
    return markRecovery(node_id);
}

CRDTSnapshot FaultTolerance::mergeCRDT(
    const CRDTSnapshot& local, const CRDTSnapshot& remote) {
    CRDTSnapshot merged;

    // 版本向量：逐元素取 max
    size_t max_vv = std::max(local.version_vector.size(), remote.version_vector.size());
    merged.version_vector.resize(max_vv, 0);
    for (size_t i = 0; i < local.version_vector.size(); ++i) {
        merged.version_vector[i] = std::max(merged.version_vector[i], local.version_vector[i]);
    }
    for (size_t i = 0; i < remote.version_vector.size(); ++i) {
        merged.version_vector[i] = std::max(merged.version_vector[i], remote.version_vector[i]);
    }

    // 全局步数：取 max
    merged.global_step = std::max(local.global_step, remote.global_step);

    // 本地步数：取 max（LWW）
    merged.local_step = std::max(local.local_step, remote.local_step);

    // 梯度计数器 (G-Counter)：逐元素取 max
    size_t max_gc = std::max(local.grad_counter.size(), remote.grad_counter.size());
    merged.grad_counter.resize(max_gc, 0);
    for (size_t i = 0; i < local.grad_counter.size(); ++i) {
        merged.grad_counter[i] = std::max(merged.grad_counter[i], local.grad_counter[i]);
    }
    for (size_t i = 0; i < remote.grad_counter.size(); ++i) {
        merged.grad_counter[i] = std::max(merged.grad_counter[i], remote.grad_counter[i]);
    }

    // 动量 (LWW)：取 local_step 较大的那一方的动量
    if (local.local_step >= remote.local_step) {
        merged.momentum = local.momentum;
    } else {
        merged.momentum = remote.momentum;
    }

    // 合并后的快照使用当前时间
    merged.timestamp = std::chrono::steady_clock::now();

    return merged;
}

bool FaultTolerance::needsFullSync(
    const CRDTSnapshot& local, const CRDTSnapshot& remote) {
    size_t max_dims = std::max(local.version_vector.size(), remote.version_vector.size());

    for (size_t i = 0; i < max_dims; ++i) {
        uint64_t lv = (i < local.version_vector.size()) ? local.version_vector[i] : 0;
        uint64_t rv = (i < remote.version_vector.size()) ? remote.version_vector[i] : 0;
        // 使用 int64_t 避免无符号溢出
        int64_t diff = static_cast<int64_t>(lv) - static_cast<int64_t>(rv);
        if (diff < 0) diff = -diff;
        if (diff > 3) return true;
    }
    return false;
}

// ======================= 内部工具 =======================

void FaultTolerance::pruneSnapshots() {
    while (_snapshots.size() > _config.max_snapshots) {
        _snapshots.erase(_snapshots.begin());
    }
}

} // namespace distributed
} // namespace ct