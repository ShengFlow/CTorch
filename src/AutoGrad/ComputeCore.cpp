/**
 *@file ComputeCore.cpp
 *@author Beapoe
 *@brief 计算核心
 *@date 2026/2/22
 **/

#include <utility>
#include <queue>
#include <algorithm>

#include "../../include/AutoGrad/ComputeCore.h"
#include "../../include/AutoGrad/Node.h"
#include "../../include/CtorchError.h"
#include "../../include/ThreadPool.h"
#include "../../include/AutoGrad.h"

GradBucket &GradBucket::getInstance() {
    static GradBucket instance;
    return instance;
}

ssize_t GradBucket::find(const std::shared_ptr<Node> &target) {
    auto it = std::find_if(_packs.begin(), _packs.end(), [target](const GradPack &pack) {
        return pack._targetNode == target;
    });
    if (it != _packs.end())
        return std::distance(_packs.begin(), it);
    return -1;
}

void GradBucket::add(std::vector<GradPack>&& newPacks) {
    std::lock_guard<std::mutex> lock(_mtx);
    for (auto&& pack : newPacks) {
        const ssize_t idx = find(pack._targetNode);
        if (idx != -1) {
            if (pack._idx >= 0 && static_cast<size_t>(pack._idx) < pack._grad.size()) {
                if (static_cast<size_t>(pack._idx) < _packs[idx]._grad.size()) {
                    _packs[idx]._grad[pack._idx] = _packs[idx]._grad[pack._idx] + pack._grad[pack._idx];
                } else {
                    if (static_cast<size_t>(pack._idx) >= _packs[idx]._grad.size()) {
                        _packs[idx]._grad.resize(pack._idx + 1);
                    }
                    _packs[idx]._grad[pack._idx] = std::move(pack._grad[pack._idx]);
                }
            }
        } else {
            if (pack._idx >= 0) {
                if (static_cast<size_t>(pack._idx) >= pack._grad.size())
                    pack._grad.resize(pack._idx + 1);
            }
            pack._idx = -1;
            _packs.push_back(std::move(pack));
        }
    }
}

void GradBucket::remove(const std::shared_ptr<Node> &target) {
    std::lock_guard lock(_mtx);
    const ssize_t idx = find(target);
    if (idx != -1)
        _packs.erase(_packs.begin() + idx);
}

bool GradBucket::tryGetGrad(const std::shared_ptr<Node> &target, std::vector<Tensor> &out_grads) {
    std::lock_guard lock(_mtx);
    const ssize_t idx = find(target);
    if (idx != -1) {
        out_grads = std::move(_packs[idx]._grad);
        _packs.erase(_packs.begin() + idx);  // 获取后立即移除
        return true;
    }
    return false;
}

bool GradBucket::empty() {
    std::lock_guard lock(_mtx);
    return _packs.empty();
}

void GradBucket::clear() {
    std::lock_guard lock(_mtx);
    _packs.clear();
}

std::vector<Tensor> GradBucket::operator[](const std::shared_ptr<Node> &target) {
    std::lock_guard<std::mutex> lock(_mtx);
    ssize_t idx = find(target);
    if (idx != -1) {
        auto result = std::move(_packs[idx]._grad);
        _packs.erase(_packs.begin() + idx);
        return result;
    }
    CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN,
                        "Trying to visit an nonexistential grad pack");
    return {};
}

std::shared_ptr<Node> ComputeCore::tryPopReadyNode() {
    std::lock_guard lock(_mtx);
    if (_readyNodes.empty())
        return nullptr;
    auto node = std::move(_readyNodes.front());
    _readyNodes.pop();
    return node;
}

ComputeCore &ComputeCore::getInstance() {
    static ComputeCore instance;
    return instance;
}

void ComputeCore::addReadyNode(std::shared_ptr<Node> node) {
    std::lock_guard lock(_mtx);
    _readyNodes.push(std::move(node));
    _cv.notify_one();
}

void ComputeCore::backward(std::shared_ptr<Node> root, bool retainGraph) {
    // 保存当前的EnableGrad值
    bool original_enable_grad = AutoGrad::EnableGrad;
    // 在backward执行期间，禁用计算图记录
    AutoGrad::EnableGrad = false;

    GradBucket &bucket = GradBucket::getInstance();

    // 清除梯度桶
    bucket.clear();

    // 构造初始 grad：标量 1.0 按 root 的输出形状广播。
    // 这样所有反向节点收到的 downStreamGrad 都与输出同形，可以直接走 element-wise/matmul 路径。
    const std::vector<size_t>& root_shape = root->getResultShape();
    Tensor grad_tensor(1.0f);
    if (!root_shape.empty() && (root_shape.size() > 1 || (root_shape.size() == 1 && root_shape[0] != 1))) {
        // root 不是 0D 标量，把初始 grad 广播到 root 的形状
        Tensor broadcasted(ShapeTag{}, root_shape, grad_tensor.dtype(), grad_tensor.device());
        const float scalar = grad_tensor.item<float>();
        const size_t total = broadcasted.numel();
        float* p = broadcasted.data<float>();
        for (size_t i = 0; i < total; ++i) p[i] = scalar;
        grad_tensor = broadcasted;
    }
    CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - grad_tensor dim: " + std::to_string(grad_tensor.dim()));
    GradPack primary = {root, std::vector({grad_tensor}), -1};
    CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Adding primary grad pack");
    bucket.add(std::vector({std::move(primary)}));

    CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Adding root to ready nodes");
    addReadyNode(root);

    CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Processing ready nodes");
    while (true) {
        std::shared_ptr<Node> node = tryPopReadyNode();
        if (!node) {
            // 检查是否还有未处理的节点
            // 如果梯度桶不为空且readyNodes为空，可能存在死锁
            if (!bucket.empty()) {
                CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Grad bucket not empty but no ready nodes");
                // 这里可以添加额外的处理逻辑
            }
            break;
        }

        CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Processing node");

        // 获取节点的梯度
        std::vector<Tensor> grads;
        if (!bucket.tryGetGrad(node, grads)) {
            CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - No grad found for node, skipping");
            continue;
        }

        // 执行反向传播
        CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Processing node backward");
        std::vector<GradPack> result = node->backward(grads);

        // 处理反向传播的结果
        if (!result.empty()) {
            CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Adding " + std::to_string(result.size()) + " grad packs");

            // 先检查并添加上游节点到ready队列（在move之前）
            for (const auto &pack : result) {
                if (pack._targetNode && pack._targetNode->decrease()) {
                    CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Adding upstream node to ready queue");
                    addReadyNode(pack._targetNode);
                }
            }

            bucket.add(std::move(result));
        }

        // 移除处理过的节点的梯度
        if (!retainGraph) {
            bucket.remove(node);
        }
    }

    CTORCH_TRACE(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Done processing ready nodes");
    
    if (retainGraph) {
        std::unordered_set<Node *> restored;
        root->restoreRecursive(restored);
    }

    // 恢复原始的EnableGrad值
    AutoGrad::EnableGrad = original_enable_grad;
}
