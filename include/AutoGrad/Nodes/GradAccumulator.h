/**
 *@file GradAccumulator.h
 *@brief 梯度累加器(叶子节点)
 *@author Beapoe
 *@date 2026/4/3
 **/

#ifndef CTORCH_GRADACCUMULATOR_H
#define CTORCH_GRADACCUMULATOR_H
#include "../Node.h"

class GradAccumulator final: public Node {
public:
    GradAccumulator(const Tensor& tensor);
    std::vector<GradPack> backward(std::vector<Tensor> downStreamGrads) override;
private:
    const Tensor* _tensor;
};

#endif // CTORCH_GRADACCUMULATOR_H
