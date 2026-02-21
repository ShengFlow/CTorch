/**
*@file AddNode.cpp
 *@author Beapoe
 *@brief 加法节点实现
 *@date 2026/2/21
 **/

#include "AutoGrad/Nodes/AddNode.h"

std::vector<GradPack> AddNode::backward(const std::vector<Tensor> &downStreamGrads) {
    std::vector<GradPack> ret;
    for (size_t i{0};i<_upStreamNodes.size();i++)
        ret.push_back(GradPack{
            _upStreamNodes[i],
            downStreamGrads[i]
        });
    return ret;
}