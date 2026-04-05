/**
*@file MulNode.h
 *@author Beapoe
 *@brief 乘法节点定义
 *@date 2026/2/17
 **/

#include "AutoGrad/Nodes/MulNode.h"

MulNode::MulNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs) 
    : Node(upStreamNodes, inputs) {}

MulNode::MulNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result) 
    : Node(upStreamNodes, inputs, result) {}

std::vector<GradPack> MulNode::backward(const std::vector<Tensor> &downStreamGrads) {
    std::vector<GradPack> ret;
    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({_inputs[0] * downStreamGrads[1]}),
        0
    });
    ret.push_back(GradPack{
        _upStreamNodes[1],
        std::vector({_inputs[1] * downStreamGrads[0]}),
        1
    });
   return ret;
}
