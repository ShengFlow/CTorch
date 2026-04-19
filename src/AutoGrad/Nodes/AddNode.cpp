/**
*@file AddNode.cpp
 *@author Beapoe
 *@brief 加法节点实现
 *@date 2026/2/21
 **/

#include "AutoGrad/Nodes/AddNode.h"

AddNode::AddNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs) 
    : Node(upStreamNodes, inputs) {}

AddNode::AddNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result) 
    : Node(upStreamNodes, inputs, result) {}

std::vector<GradPack> AddNode::backward(std::vector<Tensor> downStreamGrads) {
    std::vector<GradPack> ret;
    for (size_t i = 0; i < _upStreamNodes.size(); ++i) {
        if (i < downStreamGrads.size()) {
            ret.push_back(GradPack{
                _upStreamNodes[i],
                std::vector({downStreamGrads[i]}),
                static_cast<int>(i)
            });
        }
    }
    return ret;
}