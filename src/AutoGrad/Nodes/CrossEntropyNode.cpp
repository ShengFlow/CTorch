/**
*@file CrossEntropyNode.cpp
 *@author Beapoe
 *@brief 交叉熵损失节点实现
 *@date 2026/4/5
 **/

#include "AutoGrad/Nodes/CrossEntropyNode.h"

CrossEntropyNode::CrossEntropyNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs) 
    : Node(upStreamNodes, inputs) {set_requireAccelerate(true);}

CrossEntropyNode::CrossEntropyNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {set_requireAccelerate(true);}

CrossEntropyNode::CrossEntropyNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs, const std::weak_ptr<Tensor>& result) 
    : Node(upStreamNodes, inputs, result) {set_requireAccelerate(true);}

CrossEntropyNode::CrossEntropyNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs, const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {set_requireAccelerate(true);}

std::vector<GradPack> CrossEntropyNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;

    if (_inputs.size() != 2) {
        CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN, "CrossEntropyNode: 输入数量错误");
        return ret;
    }

    const Tensor& logits = _inputs[0];
    const Tensor& target = _inputs[1];

    if (downStreamGrads.empty()) {
        CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN, "CrossEntropyNode: downStreamGrads is empty");
        return ret;
    }

    const Tensor& grad = downStreamGrads[0];

    Tensor softmax_logits = logits.softmax(1);
    Tensor diff = softmax_logits - target;

    Tensor grad_logits;
    if (grad.dim() == 0) {
        grad_logits = grad * diff;
    } else {
        grad_logits = grad * diff;
    }

    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({grad_logits}),
        0
    });

    return ret;
}