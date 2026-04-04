/**
 *@file AutoGrad.h
 *@author Beapoe
 *@brief 自动微分类接口
 *@date 2026/4/4
 **/

#ifndef CTORCH_AUTOGRAD_H
#define CTORCH_AUTOGRAD_H

#include "AutoGrad/DataCore.h"
#include "AutoGrad/ComputeCore.h"

namespace AutoGrad {
template <typename T>
void registerNode(std::vector<Tensor> inputs, std::weak_ptr<Tensor> result) {
    DataCore::registerNode<T>(inputs,result);
}

void backward(std::shared_ptr<Node> root, bool retainGraph);
};

#endif // CTORCH_AUTOGRAD_H
