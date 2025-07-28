//
// Created by Beapoe on 25-7-27.
//
module;

#include "../../src/dev/Tensor.h"
#include <vector>

module nn;

// Module
Tensor Module::operator()(Tensor &input) { return forward(input); }

void Module::setTrain(bool train, bool recur) {
    if (recur) {
        auto root      = this;
        auto recursive = [&train, &recursive, &recur](const Module *root) -> void {
            std::vector<Module *> nodes = root->_children;
            for (size_t i = 0; i < nodes.size(); ++i) {
                if (nodes[i]->_children.size() > 0)
                    recursive(nodes[i]);
                else
                    nodes[i]->setTrain(train, recur);
            }
        }(root);
    } else
        _train = train;
}

void Module::addChild(Module *child) {
    _children.push_back(child);
}

void Module::addChildren(std::vector<Module*> children) {
    _children.reserve(_children.size()+children.size());
    _children.insert(_children.end(), children.begin(), children.end());
}

void Module::zero_grad() const {
    for (neuron *n : _neurons)
        n->ctx.zero_grad(n->ctx.rootPtr());
}

