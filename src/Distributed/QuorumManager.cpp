/**
 * @file QuorumManager.cpp
 * @brief Quorum NRW 管理器实现 — 后端感知 Quorum 管理
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 * @version v0.1 (Gen 2 BANT)
 */

#include "Distributed/QuorumManager.h"

#include <cmath>

namespace ct {
namespace distributed {

// ======================= 构造 / 析构 =======================

QuorumManager::QuorumManager(QuorumConfig config)
    : _config(config)
    , _next_request_id(1)
{
}

// ======================= 请求生命周期 =======================

uint64_t QuorumManager::createRequest(size_t total_replicas,
                                       size_t write_quorum,
                                       size_t backend_coverage_quorum) {
    std::lock_guard<std::mutex> lock(_mtx);

    // 确定写 Quorum
    size_t actual_write_quorum = write_quorum;
    if (actual_write_quorum == 0) {
        if (_config.enable_adaptive_quorum) {
            actual_write_quorum = computeAdaptiveQuorum(total_replicas);
        } else {
            actual_write_quorum = _config.default_write_quorum;
        }
    }

    // 确定后端覆盖 Quorum
    size_t actual_backend_coverage = backend_coverage_quorum;
    if (actual_backend_coverage == 0) {
        actual_backend_coverage = _config.default_backend_coverage;
    }

    // 计算截止时间
    auto now = std::chrono::steady_clock::now();
    auto deadline = now + std::chrono::milliseconds(
        static_cast<int64_t>(_config.quorum_timeout_ms));

    uint64_t request_id = _next_request_id++;

    QuorumRequest request;
    request.request_id = request_id;
    request.write_quorum = actual_write_quorum;
    request.read_quorum = _config.default_read_quorum;
    request.total_replicas = total_replicas;
    request.backend_coverage_quorum = actual_backend_coverage;
    request.current_acks = 0;
    request.current_backend_coverage = 0;
    request.created_at = now;
    request.deadline = deadline;
    request.status = QuorumStatus::Pending;

    _requests.emplace(request_id, std::move(request));
    _stats.total_requests++;

    return request_id;
}

QuorumStatus QuorumManager::recordAck(uint64_t request_id, uint32_t node_id,
                                       DeviceType backend_type) {
    std::lock_guard<std::mutex> lock(_mtx);

    auto it = _requests.find(request_id);
    if (it == _requests.end()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::TENSOR_STATE,
            "QuorumManager: request not found (request_id=" + std::to_string(request_id) + ")");
    }

    QuorumRequest& req = it->second;

    // 如果已经是终态，直接返回当前状态
    if (req.status == QuorumStatus::Achieved ||
        req.status == QuorumStatus::Timeout ||
        req.status == QuorumStatus::Failed) {
        return req.status;
    }

    // 更新确认信息
    req.current_acks++;
    req.covered_backends.insert(backend_type);
    req.current_backend_coverage = req.covered_backends.size();

    // 检查超时
    if (isTimedOut(req)) {
        req.status = QuorumStatus::Timeout;
        _stats.timeout_count++;
        return req.status;
    }

    // 检查是否达到 Quorum
    if (hasWriteQuorum(req.current_acks, req.write_quorum) &&
        hasBackendCoverage(req.covered_backends, req.backend_coverage_quorum)) {
        req.status = QuorumStatus::Achieved;

        // 更新统计信息
        auto now = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(
            now - req.created_at).count();

        _stats.achieved_count++;
        _stats.avg_achievement_time_ms =
            (_stats.avg_achievement_time_ms * (_stats.achieved_count - 1) + elapsed_ms)
            / _stats.achieved_count;
        _stats.avg_ack_count =
            (_stats.avg_ack_count * (_stats.achieved_count - 1) + req.current_acks)
            / _stats.achieved_count;

        if (req.current_backend_coverage > 1) {
            _stats.backend_coverage_triggers++;
        }
    }

    return req.status;
}

QuorumStatus QuorumManager::checkStatus(uint64_t request_id) const {
    std::lock_guard<std::mutex> lock(_mtx);

    auto it = _requests.find(request_id);
    if (it == _requests.end()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::TENSOR_STATE,
            "QuorumManager: request not found (request_id=" + std::to_string(request_id) + ")");
    }

    const QuorumRequest& req = it->second;

    // 检查超时
    if (req.status == QuorumStatus::Pending && isTimedOut(req)) {
        return QuorumStatus::Timeout;
    }

    return req.status;
}

std::shared_ptr<const QuorumRequest> QuorumManager::getRequest(uint64_t request_id) const {
    std::lock_guard<std::mutex> lock(_mtx);

    auto it = _requests.find(request_id);
    if (it == _requests.end()) {
        return nullptr;
    }

    return std::make_shared<const QuorumRequest>(it->second);
}

void QuorumManager::removeRequest(uint64_t request_id) {
    std::lock_guard<std::mutex> lock(_mtx);

    _requests.erase(request_id);
}

size_t QuorumManager::cleanupTimedOut() {
    std::lock_guard<std::mutex> lock(_mtx);

    size_t count = 0;
    for (auto it = _requests.begin(); it != _requests.end(); ) {
        if (isTimedOut(it->second)) {
            it->second.status = QuorumStatus::Timeout;
            _stats.timeout_count++;
            it = _requests.erase(it);
            ++count;
        } else {
            ++it;
        }
    }

    return count;
}

// ======================= Quorum 决策 =======================

bool QuorumManager::hasQuorum(uint64_t request_id) const {
    std::lock_guard<std::mutex> lock(_mtx);

    auto it = _requests.find(request_id);
    if (it == _requests.end()) {
        return false;
    }

    const QuorumRequest& req = it->second;
    return hasWriteQuorum(req.current_acks, req.write_quorum) &&
           hasBackendCoverage(req.covered_backends, req.backend_coverage_quorum);
}

size_t QuorumManager::computeAdaptiveQuorum(size_t total_replicas) const {
    if (total_replicas == 0) {
        return 1;
    }

    size_t quorum = static_cast<size_t>(
        std::ceil(static_cast<double>(total_replicas) * _config.adaptive_quorum_factor));

    // 保证 W >= 1 且 W <= N
    quorum = std::max(size_t{1}, quorum);
    quorum = std::min(quorum, total_replicas);

    return quorum;
}

// ======================= 内部辅助方法 =======================

bool QuorumManager::isTimedOut(const QuorumRequest& request) const {
    return std::chrono::steady_clock::now() >= request.deadline;
}

} // namespace distributed
} // namespace ct