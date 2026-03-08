/**
 *@file Node.cpp
 *@author Beapoe
 *@brief 节点实现
 *@date 2026/2/17
 **/

#include "../include/AutoGrad/Node.h"
#include "AutoGrad/Node.h"
void Node::increase() {
    _count += 1;
    if (_dependencies < _count) _dependencies += 1;
}

void Node::decrease() {

}


void Node::restore() { _count = _dependencies; }


size_t Node::getDependencies() const { return _dependencies; }

void Node::setDependencies(size_t dependencies) {
    _dependencies = dependencies;
}

size_t Node::getCount() const { return _count; }

void Node::setCount(const size_t count) { _count = count; }

Node::Node(const std::vector<std::weak_ptr<Node>> &upStreamNodes,
           const std::vector<Tensor> &inputs)
               :_upStreamNodes(upStreamNodes),_inputs(inputs) {}

Node::Node(const std::vector<std::weak_ptr<Node>> &upStreamNodes, const std::vector<Tensor> &inputs,
           const std::weak_ptr<Tensor> &result)
               :_upStreamNodes(upStreamNodes),_inputs(inputs),_result(result)
{}