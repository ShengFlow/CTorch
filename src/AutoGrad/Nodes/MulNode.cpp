/**
*@file MulNode.h
 *@author Beapoe
 *@brief 乘法节点定义
 *@date 2026/2/17
 **/

#include "AutoGrad/Nodes/MulNode.h"

std::vector<GradPack> MulNode::backward(const std::vector<Tensor> &downStreamGrads) {
    std::vector<GradPack> ret;
    ret.push_back(GradPack{
        _upStreamNodes[0],
        _inputs[1] * downStreamGrads[0]
    });
    ret.push_back(GradPack{
        _upStreamNodes[1],
        _inputs[0] * downStreamGrads[0]
    });
   return ret;
}
