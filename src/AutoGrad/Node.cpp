/**
 *@file Node.cpp
 *@author Beapoe
 *@brief 节点实现
 *@date 2026/2/17
 **/

#include "../../include/AutoGrad/Node.h"
#include "../../include/Ctorch_Error.h"
void Node::increase() {
    _count.fetch_add(1,std::memory_order_acq_rel);
    _dependencies++;
}

bool Node::decrease() {
	const size_t old = _count.fetch_sub(1,std::memory_order_acq_rel);
	if(old == 0) {
	    Ctorch_Error::error(ErrorPlatform::kAutoDiff,ErrorType::UNKNOWN,"Dependency count is negative");
	    return false;
	}
    return old == 1;
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

Node::Node(const std::vector<std::shared_ptr<Node>> &upStreamNodes, const std::vector<Tensor> &inputs,
           const std::weak_ptr<Tensor> &result)
               :_upStreamNodes(upStreamNodes),_inputs(inputs),_result(result)
{}

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
