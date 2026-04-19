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

void GradBucket::add(const std::vector<GradPack> &newPacks) {
    std::lock_guard<std::mutex> lock(_mtx);
    for (auto pack : newPacks) {
        const ssize_t idx = find(pack._targetNode);
        if (idx != -1) {
            if (pack._idx >= 0 && static_cast<size_t>(pack._idx) < pack._grad.size()) {
                if (static_cast<size_t>(pack._idx) < _packs[idx]._grad.size()) {
                    _packs[idx]._grad[pack._idx] = _packs[idx]._grad[pack._idx] + pack._grad[pack._idx];
                } else {
                    if (static_cast<size_t>(pack._idx) >= _packs[idx]._grad.size()) {
                        _packs[idx]._grad.resize(pack._idx + 1);
                    }
                    _packs[idx]._grad[pack._idx] = pack._grad[pack._idx];
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
        out_grads = _packs[idx]._grad;
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
    std::lock_guard lock(_mtx);
    ssize_t idx = find(target);
    if (idx != -1)
        return _packs[idx]._grad;
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

void ComputeCore::scheduleNode(const std::vector<GradPack> &newPacks) {
    for (const auto &pack : newPacks) {
        for (const auto &upstream : pack._targetNode->getUpStreamNodes()) {
            if (upstream && upstream->decrease()) {
                addReadyNode(upstream);
            }
        }
    }
}

void ComputeCore::scheduleNode(std::shared_ptr<Node> root) { addReadyNode(std::move(root)); }

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

    // 创建一个标量张量，值为1.0
    Tensor grad_tensor(1.0f);
    CtorchError::trace(ErrorPlatform::kAutoDiff, "ComputeCore::backward - grad_tensor dim: " + std::to_string(grad_tensor.dim()));
    GradPack primary = {root, std::vector({grad_tensor}), -1};
    CtorchError::trace(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Adding primary grad pack");
    bucket.add(std::vector({primary}));
    
    CtorchError::trace(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Adding root to ready nodes");
    addReadyNode(root);

    CtorchError::trace(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Processing ready nodes");
    while (true) {
        std::shared_ptr<Node> node = tryPopReadyNode();
        if (!node) {
            // 检查是否还有未处理的节点
            // 如果梯度桶不为空且readyNodes为空，可能存在死锁
            if (!bucket.empty()) {
                CtorchError::trace(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Grad bucket not empty but no ready nodes");
                // 这里可以添加额外的处理逻辑
            }
            break;
        }
        
        CtorchError::trace(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Processing node");
        
        // 获取节点的梯度
        std::vector<Tensor> grads;
        if (!bucket.tryGetGrad(node, grads)) {
            CtorchError::trace(ErrorPlatform::kAutoDiff, "ComputeCore::backward - No grad found for node, skipping");
            continue;
        }

        // 执行反向传播
        CtorchError::trace(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Processing node backward");
        const std::vector<GradPack> result = node->backward(grads);
        
        // 处理反向传播的结果
        if (!result.empty()) {
            CtorchError::trace(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Adding " + std::to_string(result.size()) + " grad packs");
            bucket.add(result);
            
            // 检查并添加上游节点到ready队列
            for (const auto &pack : result) {
                // pack._targetNode就是上游节点
                if (pack._targetNode && pack._targetNode->decrease()) {
                    CtorchError::trace(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Adding upstream node to ready queue");
                    addReadyNode(pack._targetNode);
                }
            }
        }
        
        // 移除处理过的节点的梯度
        if (!retainGraph) {
            bucket.remove(node);
        }
    }
    
    CtorchError::trace(ErrorPlatform::kAutoDiff, "ComputeCore::backward - Done processing ready nodes");
    
    if (retainGraph) {
        std::unordered_set<Node *> restored;
        root->restoreRecursive(restored);
    }

    // 恢复原始的EnableGrad值
    AutoGrad::EnableGrad = original_enable_grad;
}
