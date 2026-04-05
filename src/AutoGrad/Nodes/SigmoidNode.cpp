#include "AutoGrad/Nodes/SigmoidNode.h"
#include "../../../include/Tensor.h"

SigmoidNode::SigmoidNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs) 
    : Node(upStreamNodes, inputs) {}

SigmoidNode::SigmoidNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result) 
    : Node(upStreamNodes, inputs, result) {}

std::vector<GradPack> SigmoidNode::backward(std::vector<Tensor> downStreamGrads) {
    std::vector<GradPack> ret;
    
    // 对于Sigmoid算子，导数是sigmoid(input) * (1 - sigmoid(input))
    Tensor sigmoid_input = _inputs[0].sigmoid();
    Tensor grad = sigmoid_input * (1 - sigmoid_input) * downStreamGrads[0];
    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({grad}),
        0
    });
    
    return ret;
}