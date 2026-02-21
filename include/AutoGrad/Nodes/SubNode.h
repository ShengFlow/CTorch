/**
*@file SubNode.h
 *@author Beapoe
 *@brief 减法节点定义
 *@date 2026/2/21
 **/

#ifndef CTORCH_SUBNODE_H
#define CTORCH_SUBNODE_H

#include "AutoGrad/Node.h"

class SubNode final: public Node {
public:
    std::vector<GradPack> backward(const std::vector<Tensor> &downStreamGrads) override;
};

#endif // CTORCH_SUBNODE_H
