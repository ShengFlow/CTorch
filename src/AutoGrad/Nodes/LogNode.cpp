#include "AutoGrad/Nodes/LogNode.h"

LogNode::LogNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs)
    : Node(upStreamNodes, inputs) {}

LogNode::LogNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

LogNode::LogNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
            const std::weak_ptr<Tensor>& result)
    : Node(upStreamNodes, inputs, result) {}

LogNode::LogNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
            const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> LogNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;

    if (_inputs.size() != 1) {
        return ret;
    }
    if (downStreamGrads.empty()) {
        return ret;
    }

    const Tensor& x = _inputs[0];
    const Tensor& grad_out = downStreamGrads[0];
    
    Tensor grad_x = grad_out / x;

    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({grad_x}),
        0
    });

    return ret;
}