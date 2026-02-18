/**
 *@file Node.cpp
 *@author Beapoe
 *@brief 节点实现
 *@date 2026/2/17
 **/

#include "../include/AutoGrad/Node.h"
#include "../include/AutoGrad/IDDistributor.h"

Node::Node(std::shared_ptr<Tensor> val, bool requireGrad, bool isLeaf, std::shared_ptr<GradFn> fn)
    : _id(IDDistributor::allocateID()), _val(std::move(val)), _fn(std::move(fn)),
      _requireGrad(requireGrad), _isLeaf(isLeaf) {}

size_t Node::getID() const { return _id; }

bool Node::getRequireGrad() const { return _requireGrad; }

Tensor *Node::getGrad() const { return _grad.get(); }

GradFn *Node::getGradFn() const { return _fn.get(); }

bool Node::isLeaf() const { return _isLeaf; }

void Node::setRequireGrad(const bool state) { _requireGrad = state; }

void Node::setGrad(const Tensor &grad) { _grad = std::make_unique<Tensor>(grad); }

void Node::setGrad(std::unique_ptr<Tensor> grad) { _grad = std::move(grad); }

void Node::setGradFn(std::shared_ptr<GradFn> fn) { _fn = std::move(fn); }

void Node::setLeafState(bool state) { _isLeaf = state; }
