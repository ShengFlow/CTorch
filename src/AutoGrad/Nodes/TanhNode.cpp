#include "AutoGrad/Nodes/TanhNode.h"

TanhNode::TanhNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs)
    : Node(upStreamNodes, inputs) {}

TanhNode::TanhNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

TanhNode::TanhNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result)
    : Node(upStreamNodes, inputs, result) {}

TanhNode::TanhNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs, const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> TanhNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;

    if (_inputs.empty() || downStreamGrads.empty()) {
        return ret;
    }

    auto result = getResult();
    if (result) {
        Tensor y = *result;
        Tensor grad = (1 - y * y) * downStreamGrads[0];
        ret.push_back(GradPack{
            _upStreamNodes[0],
            std::vector({grad}),
            0
        });
    } else {
        Tensor tanh_input = _inputs[0].tanh();
        Tensor grad = (1 - tanh_input * tanh_input) * downStreamGrads[0];
        ret.push_back(GradPack{
            _upStreamNodes[0],
            std::vector({grad}),
            0
        });
    }
    
    return ret;
}