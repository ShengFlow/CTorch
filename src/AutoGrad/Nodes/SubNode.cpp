/**
*@file SubNode.cpp
 *@author Beapoe
 *@brief 减法节点定义
 *@date 2026/2/17
 **/

#include "AutoGrad/Nodes/SubNode.h"

SubNode::SubNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs)
    : Node(upStreamNodes, inputs) {}

SubNode::SubNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

SubNode::SubNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result)
    : Node(upStreamNodes, inputs, result) {}

SubNode::SubNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs, const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> SubNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;
    // 对于 c = a - b，导数是：∂c/∂a = 1，∂c/∂b = -1
    // 所以 grad_a = grad，grad_b = -grad
    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({downStreamGrads[0]}),
        0
    });
    ret.push_back(GradPack{
        _upStreamNodes[1],
        std::vector({-downStreamGrads[0]}),
        1
    });
    return ret;
}
