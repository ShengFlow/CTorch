/**
*@file MulNode.h
 *@author Beapoe
 *@brief 乘法节点定义
 *@date 2026/2/17
 **/

#ifndef CTORCH_MULNODE_H
#define CTORCH_MULNODE_H

#include "AutoGrad/Node.h"

class MulNode final:public Node {
public:
    std::vector<GradPack> backward(const std::vector<Tensor> &downStreamGrads) override;
};

#endif // CTORCH_MULNODE_H
