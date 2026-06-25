/**
*@file AddNode.cpp
 *@author Beapoe
 *@brief 加法节点实现
 *@date 2026/2/21
 **/

#include "AutoGrad/Nodes/AddNode.h"

AddNode::AddNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs) 
    : Node(upStreamNodes, inputs) {}

AddNode::AddNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

AddNode::AddNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result) 
    : Node(upStreamNodes, inputs, result) {}

AddNode::AddNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs, const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> AddNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;
    if (downStreamGrads.empty()) {
        return ret;
    }
    const Tensor& grad = downStreamGrads[0];
    for (size_t i = 0; i < _upStreamNodes.size(); ++i) {
        ret.push_back(GradPack{
            _upStreamNodes[i],
            std::vector({grad}),
            static_cast<int>(i)
        });
    }
    return ret;
}