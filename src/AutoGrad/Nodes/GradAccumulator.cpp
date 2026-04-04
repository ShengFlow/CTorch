/**
 *@file GradAccumulator.cpp
 *@brief 梯度累加器(叶子节点)
 *@author Beapoe
 *@date 2026/4/3
 **/

#include "../include/AutoGrad/Nodes/GradAccumulator.h"

std::vector<GradPack> GradAccumulator::backward(const std::vector<Tensor> &downStreamGrads) {
    if (_result.lock()) {
        std::shared_ptr<Tensor> grad = std::make_shared<Tensor>(downStreamGrads[0]);
        for (int i = 1;i<downStreamGrads.size()-1;i++) *grad = *grad + downStreamGrads[i];
        _result.lock()->setGrad(grad);
    }
    return {};
}
