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

size_t GradBucket::find(const std::shared_ptr<Node> &target) {
    auto it = std::find_if(_packs.begin(), _packs.end(), [target](const GradPack &pack) {
        return pack._targetNode == target && pack._idx == -1;
    });
    if (it != _packs.end())
        return std::distance(_packs.begin(), it);
    return -1;
}

void GradBucket::add(const std::vector<GradPack> &newPacks) {
    std::lock_guard lock(_mtx);
    for (auto pack : newPacks) {
        const size_t idx = find(pack._targetNode);
        if (idx != -1) {
            if (pack._idx >= _packs[idx]._grad.size())
                _packs[idx]._grad.resize(pack._idx + 1);
            _packs[idx]._grad[pack._idx] = _packs[idx]._grad[pack._idx] + pack._grad[0];
        } else {
            if (pack._idx >= pack._grad.size())
                pack._grad.resize(pack._idx + 1);
            if (pack._idx != 0) {
                pack._grad[pack._idx] = std::move(pack._grad[0]);
                pack._grad[0]         = Tensor({0.0});
            }
            pack._idx = -1;
            _packs.push_back(std::move(pack));
        }
    }
}

void GradBucket::remove(const std::shared_ptr<Node> &target) {
    std::lock_guard lock(_mtx);
    const size_t idx = find(target);
    if (idx != -1)
        _packs.erase(_packs.begin() + idx);
    else
        CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN,
                            "Trying to erase an unexistent gradPack");
}

bool GradBucket::empty() {
    std::lock_guard lock(_mtx);
    return _packs.empty();
}

std::vector<Tensor> GradBucket::operator[](const std::shared_ptr<Node> &target) {
    std::lock_guard lock(_mtx);
    size_t idx = find(target);
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

    std::atomic<bool> finished{false};

    GradBucket &bucket = GradBucket::getInstance();
    ThreadPool &pool   = ThreadPool::getInstance();
    auto core = [this, &bucket, &finished, retainGraph](const std::shared_ptr<Node> &root) {
        const std::vector<Tensor> grads    = bucket[root];
        const std::vector<GradPack> result = root->backward(grads);
        if (!result.empty()) {
            bucket.add(result);
            scheduleNode(result);
        }
        if (!retainGraph)
            bucket.remove(root);
        if (bucket.empty())
            finished = true;
    };

    GradPack primary = {root, std::vector({Tensor({1.0})}),0};
    bucket.add(std::vector({primary}));
    scheduleNode(root);


    while (!finished.load()) {
        if (auto node = tryPopReadyNode()) {
            if (node->requireAccelerate())
                pool.addTask(core, node);
            else
                core(node);
        }
        // 这样？std::this_thread::yield();
    }
    if (retainGraph) {
        std::unordered_set<Node *> restored;
        root->restoreRecursive(restored);
    }

    // 恢复原始的EnableGrad值
    AutoGrad::EnableGrad = original_enable_grad;
}
