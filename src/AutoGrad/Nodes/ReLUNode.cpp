#include "AutoGrad/Nodes/ReLUNode.h"

ReLUNode::ReLUNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs) 
    : Node(upStreamNodes, inputs) {}

ReLUNode::ReLUNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result) 
    : Node(upStreamNodes, inputs, result) {}

std::vector<GradPack> ReLUNode::backward(std::vector<Tensor> downStreamGrads) {
    std::vector<GradPack> ret;
    
    // 对于ReLU算子，导数是：当输入大于0时为1，否则为0
    Tensor mask = _inputs[0] > 0;
    Tensor grad = downStreamGrads[0] * mask;
    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({grad}),
        0
    });
    
    return ret;
}