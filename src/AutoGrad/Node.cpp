/**
 *@file Node.cpp
 *@author Beapoe
 *@brief 节点实现
 *@date 2026/2/17
 **/

#include "../../include/AutoGrad/Node.h"
#include "../../include/CtorchError.h"
#include "../include/Tensor.h"

void Node::increase() {
    _count.fetch_add(1,std::memory_order_acq_rel);
    _dependencies++;
}

bool Node::decrease() {
    size_t old = _count.load(std::memory_order_acquire);
    while (old > 0) {
        if (_count.compare_exchange_strong(old, old - 1, std::memory_order_acq_rel)) {
            return old == 1;
        }
    }
    CtorchError::error(ErrorPlatform::kAutoDiff,ErrorType::UNKNOWN,"Dependency count is negative");
    return false;
}


void Node::restore() { _count.store(_dependencies,std::memory_order_relaxed); }

size_t Node::getDependencies() const { return _dependencies; }

void Node::setDependencies(size_t dependencies) {
    _dependencies = dependencies;
}

void Node::setCount(const size_t count) { _count = count; }

Node::Node(const std::vector<std::shared_ptr<Node>> &upStreamNodes,
           const std::vector<Tensor> &inputs)
               :_upStreamNodes(upStreamNodes),_inputs(inputs) {}

Node::Node(std::vector<std::shared_ptr<Node>>&& upStreamNodes,
           std::vector<Tensor>&& inputs)
               :_upStreamNodes(std::move(upStreamNodes)),_inputs(std::move(inputs)) {}

Node::Node(const std::vector<std::shared_ptr<Node>> &upStreamNodes, const std::vector<Tensor> &inputs,
           const std::weak_ptr<Tensor> &result)
               :_upStreamNodes(upStreamNodes),_inputs(inputs),_result(result)
{
    if (auto t = result.lock()) {
        _resultShape = t->sizes();
    }
}

Node::Node(std::vector<std::shared_ptr<Node>>&& upStreamNodes,
           std::vector<Tensor>&& inputs,
           const std::weak_ptr<Tensor>& result)
               :_upStreamNodes(std::move(upStreamNodes)),_inputs(std::move(inputs)),_result(result)
{
    if (auto t = result.lock()) {
        _resultShape = t->sizes();
    }
}

Node::Node(const std::weak_ptr<Tensor> &result)
    :_upStreamNodes(std::vector<std::shared_ptr<Node>>()),_inputs(std::vector<Tensor>()),_result(result)
{
    if (auto t = result.lock()) {
        _resultShape = t->sizes();
    }
}


std::vector<std::shared_ptr<Node>> Node::getUpStreamNodes() const {return _upStreamNodes; }

bool Node::requireAccelerate() const { return _requireAccelerate; }

void Node::set_requireAccelerate(bool requireAccelerate) {_requireAccelerate = requireAccelerate;}

void Node::restoreRecursive(std::unordered_set<Node *>& visited) {
    if (!visited.count(this)) {
        visited.insert(this);
        restore();
        for (auto &node : _upStreamNodes)
            if (node) node->restoreRecursive(visited);
    }
}

void Node::clearRecursive(std::unordered_set<Node *>& visited) {
    if (!visited.count(this)) {
        visited.insert(this);
        clearResultOwner();
        auto upstream_copy = _upStreamNodes;
        auto result_copy = _result.lock();
        for (auto &node : upstream_copy)
            if (node) node->clearRecursive(visited);
        _upStreamNodes.clear();
        _inputs.clear();
        if (result_copy) {
            result_copy->detach_autograd();
        }
    }
}
