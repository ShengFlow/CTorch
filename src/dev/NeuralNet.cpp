//
// Created by Beapoe on 25-7-27.
//
module;

#include "../../src/dev/Tensor.h"
#include <vector>

module nn;

// Parameter
 Parameter::Parameter(Tensor data, bool requiresGrad = true) {
     _data = data;
     _data.requires_grad(requiresGrad);
     _initialized = true;
 }

bool Parameter::isInitialized() const {
     return _initialized;
 }

Tensor Parameter::data() const {
     return _data;
 }

// Module
Tensor Module::operator()(Tensor &input) { return forward(input); }

void Module::train(bool recur) {
    if (recur) {
        auto root      = this;
        auto recursive = [&train, &recursive, &recur](const Module *root) -> void {
            std::unordered_map<std::string,Module*> children = root->_children;
            for (auto& [_,child]:children) {
                if (child->_children.size() > 0)
                    recursive(child);
                else
                    child->_train = true;
            }
        }(root);
    } else
        _train = true;
}

void Module::eval(bool recur) {
    if (recur) {
        auto root      = this;
        auto recursive = [&train, &recursive, &recur](const Module *root) -> void {
            std::unordered_map<std::string,Module*> children = root->_children;
            for (auto& [_,child]:children) {
                if (child->_children.size() > 0)
                    recursive(child);
                else
                    child->_train = false;
            }
        }(root);
    } else
        _train = false;
}

void Module::addChild(std::string name,Module *child) {
    _children.emplace(name,child);
}

void Module::addChildren(std::unordered_map<std::string,Module*> children) {
    _children.reserve(_children.size()+children.size());
    for (auto it = children.begin();it!=children.end();) {
        auto node = children.extract(it++);
        if (!_children.insert(std::move(node)).inserted) {
            children.insert(std::move(node));
            for (auto rit = it;rit!=children.end();) {
                auto prev = _children.extract(rit->first);
                if (!prev.empty()) children.insert(std::move(prev));
                ++rit;
            }
            throw std::runtime_error("Duplicate key found:"+node.key());
        }
    }
}

std::unordered_map<std::string,Module*> Module::children() const{
    return _children;
}

std::vector<Module*> Module::childrenRecur(Module* root) const {
    std::vector<Module*> result;
    auto recursive = [&result, &recursive](const Module *root) -> void {
        std::unordered_map<std::string, Module *> children = root->_children;
        for (auto &[_, child] : children) {
            if (child->_children.size() > 0)
                recursive(child);
            else
                result.push_back(child);
        }
    }(root);
    return result;
}

void Module::registerParameter(std::string name, Parameter *parameter) {
     if (parameter->isInitialized()) _parameters.emplace(name,parameter);
     else throw std::runtime_error("Parameter '"+name+"' is not initialized");
 }

void Module::registerParameters(std::unordered_map<std::string,Parameter*> parameters) {
     _parameters.reserve(_parameters.size()+parameters.size());
     for (auto it = parameters.begin();it != parameters.end();) {
         if (it->second->isInitialized()) {
             auto node = parameters.extract(it++);
             if (!_parameters.insert(std::move(node)).inserted) {
                 for (auto rit = it;rit!=parameters.end();) {
                     auto prev = parameters.extract(it->first);
                     if (!prev.empty()) _parameters.insert(std::move(prev));
                     ++rit;
                 }
                 throw std::runtime_error("Duplicate key found:"+node.key());
             }
         }else throw std::runtime_error("Parameter '"+it->first+"' is not initialized");
     }
 }

Parameter Module::parameter(std::string name) const {
     return *_parameters.at(name);
 }

std::vector<Parameter*> Module::parameters(std::initializer_list<std::string> names) const {
     std::vector<Parameter*> result;
     for (std::string name:names) result.push_back(_parameters.at(name));
     return result;
 }

void Module::zero_grad() const {
    for (neuron *n : _neurons)
        n->ctx.zero_grad(n->ctx.rootPtr());
}

