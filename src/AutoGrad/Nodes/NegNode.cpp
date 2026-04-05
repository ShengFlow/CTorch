#include "AutoGrad/Nodes/NegNode.h"

NegNode::NegNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs) 
    : Node(upStreamNodes, inputs) {}

NegNode::NegNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result) 
    : Node(upStreamNodes, inputs, result) {}

std::vector<GradPack> NegNode::backward(const std::vector<Tensor> &downStreamGrads) {
    std::vector<GradPack> ret;
    
    // 对于Neg算子，导数是-1
    Tensor grad = -downStreamGrads[0];
    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({grad}),
        0
    });
    
    return ret;
}