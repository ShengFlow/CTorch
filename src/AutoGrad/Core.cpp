/**
 *@file Core.cpp
 *@author Beapoe
 *@brief 自动微分系统核心实现
 *@date 2026/2/18
 **/

#include "../include/AutoGrad/Core.h"
#include "../include/Ctorch_Error.h"
#include <future>
AutoGradCore::AutoGradCore()
    : Id2Node(std::unordered_map<size_t, std::shared_ptr<Node>>()),
      Node2Id(std::unordered_map<std::shared_ptr<Node>, size_t>()),
      GradFn2ID(std::unordered_map<GradFn *, size_t>()),
      ID2GradFn(std::unordered_map<size_t, GradFn *>()) {}

void AutoGradCore::registerNode(const std::shared_ptr<Node> &node) {
    std::lock_guard lock(_mutex);
    Id2Node[node->getID()] = node;
    Node2Id[node]          = node->getID();
    if (node->getGradFn()) {
        GradFn2ID[node->getGradFn()] = node->getID();
        ID2GradFn[node->getID()]     = node->getGradFn();
    }
}

std::shared_ptr<Node> AutoGradCore::getNode(size_t id) {
    std::lock_guard lock(_mutex);
    const auto it = Id2Node.find(id);
    return (it != Id2Node.end()) ? it->second : nullptr;
}

void AutoGradCore::clear() {
    std::lock_guard lock(_mutex);
    Id2Node.clear();
    GradFn2ID.clear();
}

void AutoGradCore::reset() {
    std::lock_guard lock(_mutex);
    std::vector<size_t> nonLeafIDs;
    std::vector<GradFn *> nonLeafFns;

    // 收集非叶子节点
    for (const auto &[id, node] : Id2Node) {
        if (!node->isLeaf()) {
            nonLeafIDs.push_back(id);
            if (node->getGradFn())
                nonLeafFns.push_back(node->getGradFn());
        }
    }

    // ID2Node和GradFn2ID中删除
    for (auto id : nonLeafIDs)
        Id2Node.erase(id);
    for (auto fn : nonLeafFns)
        GradFn2ID.erase(fn);

    _retainGraph = true;
}

void AutoGradCore::makeLeaf(Tensor &tensor, bool requireGrad) {
    size_t id = tensor.getRelatedNode()->_id;
    if (id == 0) {
        // 0是错误的张量的ID
        Ctorch_Error::warn(ErrorPlatform::kCPU, ErrorType::TENSOR_STATE,
                           "Trying to make leaf node with a wrong tensor");
    } else {
        {
            std::lock_guard lock(_mutex);
            auto it = Id2Node.find(id);
            if (it != Id2Node.end()) {
                it->second->setRequireGrad(requireGrad);
                if (requireGrad && !it->second->getGrad())
                    it->second->setGrad(
                        Tensor(ShapeTag{}, tensor.shape(), tensor.dtype(), tensor.device(), true));
            } else {
                // 新的节点自动加入
                auto tensorPtr = std::make_shared<Tensor>(tensor);
                auto node      = std::make_shared<Node>(tensorPtr, requireGrad, true, nullptr);
                Id2Node[id]    = node;
            }
        }
    }
}
