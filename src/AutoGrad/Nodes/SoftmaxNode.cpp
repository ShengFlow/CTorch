/**
*@file SoftmaxNode.cpp
 *@author Beapoe
 *@brief Softmax节点实现
 *@date 2026/4/5
 **/

#include "AutoGrad/Nodes/SoftmaxNode.h"

SoftmaxNode::SoftmaxNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs) 
    : Node(upStreamNodes, inputs) {}

SoftmaxNode::SoftmaxNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

SoftmaxNode::SoftmaxNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result, int dim)
    : Node(upStreamNodes, inputs, result), _dim(dim) {}

SoftmaxNode::SoftmaxNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs, const std::weak_ptr<Tensor>& result, int dim)
    : Node(std::move(upStreamNodes), std::move(inputs), result), _dim(dim) {}

std::vector<GradPack> SoftmaxNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;
    
    if (_inputs.size() != 1) {
        CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN, "SoftmaxNode: 输入数量错误");
        return ret;
    }
    
    if (downStreamGrads.empty()) {
        CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN, "SoftmaxNode: 下游梯度为空");
        return ret;
    }
    
    const Tensor& grad = downStreamGrads[0];

    Tensor softmax_x;
    auto cached_result = getResult();
    if (cached_result) {
        softmax_x = *cached_result;
    } else {
        const Tensor& x = _inputs[0];
        softmax_x = x.softmax(_dim);
    }

    Tensor grad_softmax = grad * softmax_x;
    Tensor sum_grad_softmax = grad_softmax.sum(_dim, true);

    Tensor grad_x = softmax_x * (grad - sum_grad_softmax);
    
    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({grad_x}),
        0
    });
    
    return ret;
}