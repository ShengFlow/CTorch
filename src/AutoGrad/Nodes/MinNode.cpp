#include "AutoGrad/Nodes/MinNode.h"

MinNode::MinNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs)
    : Node(upStreamNodes, inputs) {}

MinNode::MinNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

MinNode::MinNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
            const std::weak_ptr<Tensor>& result)
    : Node(upStreamNodes, inputs, result) {}

MinNode::MinNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
            const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> MinNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;

    if (_inputs.size() != 2) {
        return ret;
    }
    if (downStreamGrads.empty()) {
        return ret;
    }

    const Tensor& a = _inputs[0];
    const Tensor& b = _inputs[1];
    const Tensor& grad_out = downStreamGrads[0];
    
    Tensor grad_a = (a <= b) * grad_out;
    Tensor grad_b = (a > b) * grad_out;

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