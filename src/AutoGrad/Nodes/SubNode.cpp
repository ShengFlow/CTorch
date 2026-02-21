/**
*@file SubNode.cpp
 *@author Beapoe
 *@brief 减法节点定义
 *@date 2026/2/17
 **/

#include "AutoGrad/Nodes/SubNode.h"

std::vector<GradPack> SubNode::backward(const std::vector<Tensor> &downStreamGrads) {
    std::vector<GradPack> ret;
    ret.push_back(GradPack{
        _upStreamNodes[0],
        downStreamGrads[0]
    });
    ret.push_back(GradPack{
        _upStreamNodes[1],
        -downStreamGrads[1]
    });
    return ret;
}
