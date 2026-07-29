#include "AutoGrad/Nodes/SigmoidNode.h"
#include "../../../include/Tensor.h"

SigmoidNode::SigmoidNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs) 
    : Node(upStreamNodes, inputs) {}

SigmoidNode::SigmoidNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

SigmoidNode::SigmoidNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result) 
    : Node(upStreamNodes, inputs, result) {}

SigmoidNode::SigmoidNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs, const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> SigmoidNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;

    if (_inputs.empty() || downStreamGrads.empty()) {
        return ret;
    }

    auto result = getResult();
    if (result) {
        Tensor y = *result;
        Tensor grad = y * (1 - y) * downStreamGrads[0];
        ret.push_back(GradPack{
            _upStreamNodes[0],
            std::vector({grad}),
            0
        });
    } else {
        Tensor sigmoid_input = _inputs[0].sigmoid();
        Tensor grad = sigmoid_input * (1 - sigmoid_input) * downStreamGrads[0];
        ret.push_back(GradPack{
            _upStreamNodes[0],
            std::vector({grad}),
            0
        });
    }
    
    return ret;
}