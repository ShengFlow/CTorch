#include "AutoGrad/Nodes/DivNode.h"

DivNode::DivNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs) 
    : Node(upStreamNodes, inputs) {}

DivNode::DivNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

DivNode::DivNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result) 
    : Node(upStreamNodes, inputs, result) {}

DivNode::DivNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs, const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> DivNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;
    
    // 对于分子的梯度：1/分母 * 下游梯度
    Tensor grad1 = downStreamGrads[0] / _inputs[1];
    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({grad1}),
        0
    });
    
    // 对于分母的梯度：-分子/(分母^2) * 下游梯度
    Tensor grad2 = -(_inputs[0] / (_inputs[1] * _inputs[1])) * downStreamGrads[0];
    ret.push_back(GradPack{
        _upStreamNodes[1],
        std::vector({grad2}),
        1
    });
    
    return ret;
}