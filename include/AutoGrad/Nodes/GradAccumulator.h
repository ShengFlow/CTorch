/**
 *@file GradAccumulator.h
 *@brief 梯度累加器(叶子节点)
 *@author Beapoe
 *@date 2026/4/3
 **/

#ifndef CTORCH_GRADACCUMULATOR_H
#define CTORCH_GRADACCUMULATOR_H
#include "AutoGrad/Node.h"

class GradAccumulator : public Node {
public:
    std::vector<GradPack> backward(const std::vector<Tensor> &downStreamGrads) override;
};

#endif // CTORCH_GRADACCUMULATOR_H
