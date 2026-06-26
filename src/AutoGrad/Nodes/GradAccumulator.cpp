/**
 *@file GradAccumulator.cpp
 *@brief 梯度累加器(叶子节点)
 *@author Beapoe
 *@date 2026/4/3
 **/

#include "../include/AutoGrad/Nodes/GradAccumulator.h"
#include "../include/Tensor.h"

GradAccumulator::GradAccumulator(std::weak_ptr<Tensor> tensor) : _tensor(std::move(tensor)) {
    _upStreamNodes = std::vector<std::shared_ptr<Node>>();
    _inputs = std::vector<Tensor>();
    if (auto t = _tensor.lock()) {
        CTORCH_TRACE(ErrorPlatform::kAutoDiff, "GradAccumulator::GradAccumulator - Created for tensor with requires_grad: " + std::to_string(t->requires_grad()));
    }
}

std::vector<GradPack> GradAccumulator::backward(const std::vector<Tensor>& downStreamGrads) {
    CTORCH_TRACE(ErrorPlatform::kAutoDiff, "GradAccumulator::backward - Called with downStreamGrads size: " + std::to_string(downStreamGrads.size()));
    if (downStreamGrads.empty()) {
        CTORCH_TRACE(ErrorPlatform::kAutoDiff, "GradAccumulator::backward - No gradients received");
        return {};
    }

    if (auto tensor = _tensor.lock()) {
        CTORCH_TRACE(ErrorPlatform::kAutoDiff, "GradAccumulator::backward - Setting grad to tensor");

        std::shared_ptr<Tensor> grad;
        for (size_t i = 0; i < downStreamGrads.size(); i++) {
            if (downStreamGrads[i].numel() > 0 && downStreamGrads[i].storage().data<float>() != nullptr) {
                grad = std::make_shared<Tensor>(downStreamGrads[i]);
                break;
            }
        }

        if (!grad) {
            grad = std::make_shared<Tensor>(ShapeTag{}, tensor->shape(), tensor->dtype(), tensor->device());
            grad->zero();
        }

        tensor->setGrad(grad);
        CTORCH_TRACE(ErrorPlatform::kAutoDiff, "GradAccumulator::backward - Grad set successfully");
    } else {
        CTORCH_TRACE(ErrorPlatform::kAutoDiff, "GradAccumulator::backward - _tensor has been destroyed");
    }
    return {};
}
