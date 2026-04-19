/**
*@file SoftmaxNode.cpp
 *@author Beapoe
 *@brief Softmax节点实现
 *@date 2026/4/5
 **/

#include "AutoGrad/Nodes/SoftmaxNode.h"

SoftmaxNode::SoftmaxNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs) 
    : Node(upStreamNodes, inputs) {}

SoftmaxNode::SoftmaxNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result) 
    : Node(upStreamNodes, inputs, result) {}

std::vector<GradPack> SoftmaxNode::backward(std::vector<Tensor> downStreamGrads) {
    std::vector<GradPack> ret;
    
    if (_inputs.size() != 1) {
        CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN, "SoftmaxNode: 输入数量错误");
        return ret;
    }
    
    // 检查downStreamGrads大小
    if (downStreamGrads.empty()) {
        CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN, "SoftmaxNode: 下游梯度为空");
        return ret;
    }
    
    const Tensor& x = _inputs[0];
    const Tensor& grad = downStreamGrads[0];
    
    Tensor softmax_x = x.softmax(1);
    
    Tensor grad_softmax = grad * softmax_x;
    Tensor sum_grad_softmax = grad_softmax.sum(1, true);
    
    Tensor grad_x = softmax_x * (grad - sum_grad_softmax);
    
    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({grad_x}),
        0
    });
    
    return ret;
}