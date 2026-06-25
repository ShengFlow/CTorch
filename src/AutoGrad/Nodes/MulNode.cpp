/**
*@file MulNode.h
 *@author Beapoe
 *@brief 乘法节点定义
 *@date 2026/2/17
 **/

#include "AutoGrad/Nodes/MulNode.h"

MulNode::MulNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs) 
    : Node(upStreamNodes, inputs) {}

MulNode::MulNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

MulNode::MulNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result) 
    : Node(upStreamNodes, inputs, result) {}

MulNode::MulNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs, const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> MulNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;
    Tensor grad_a = downStreamGrads[0] * _inputs[1];
    Tensor grad_b = downStreamGrads[0] * _inputs[0];
    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({grad_a}),
        0
    });
    ret.push_back(GradPack{
        _upStreamNodes[1],
        std::vector({grad_b}),
        1
    });
   return ret;
}
