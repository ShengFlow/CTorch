#include "AutoGrad/Nodes/CosNode.h"
#include "../../../include/Tensor.h"

CosNode::CosNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs) 
    : Node(upStreamNodes, inputs) {}

CosNode::CosNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result) 
    : Node(upStreamNodes, inputs, result) {}

std::vector<GradPack> CosNode::backward(std::vector<Tensor> downStreamGrads) {
    std::vector<GradPack> ret;
    
    // 对于Cos算子，导数是-sin(input)
    Tensor grad = -_inputs[0].sin() * downStreamGrads[0];
    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({grad}),
        0
    });
    
    return ret;
}