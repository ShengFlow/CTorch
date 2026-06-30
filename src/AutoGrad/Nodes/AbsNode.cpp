#include "AutoGrad/Nodes/AbsNode.h"

AbsNode::AbsNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs)
    : Node(upStreamNodes, inputs) {}

AbsNode::AbsNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

AbsNode::AbsNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
            const std::weak_ptr<Tensor>& result)
    : Node(upStreamNodes, inputs, result) {}

AbsNode::AbsNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
            const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> AbsNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;

    if (_inputs.size() != 1) {
        return ret;
    }
    if (downStreamGrads.empty()) {
        return ret;
    }

    const Tensor& x = _inputs[0];
    const Tensor& grad_out = downStreamGrads[0];
    
    Tensor grad_x = (x > 0) * grad_out - (x < 0) * grad_out;

    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({grad_x}),
        0
    });

    return ret;
}