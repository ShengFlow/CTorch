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
        CTORCH_TRACE(ErrorPlatform::kAutoDiff, "GradAccumulator::backward - Accumulating grads to tensor");

        Tensor accumulated;
        for (const auto& g : downStreamGrads) {
            if (g.numel() > 0 && g.storage().data<float>() != nullptr) {
                if (accumulated.storage().size() == 0) {
                    accumulated = g;
                } else {
                    accumulated = accumulated + g;
                }
            }
        }

        if (accumulated.numel() == 0) {
            accumulated = Tensor(ShapeTag{}, tensor->shape(), tensor->dtype(), tensor->device());
            accumulated.zero();
        }

        auto existing_grad = tensor->grad();
        if (existing_grad.numel() > 0 && existing_grad.storage().data<float>() != nullptr) {
            accumulated = accumulated + existing_grad;
        }

        tensor->setGrad(std::make_shared<Tensor>(accumulated));
        CTORCH_TRACE(ErrorPlatform::kAutoDiff, "GradAccumulator::backward - Grad accumulated successfully");
    } else {
        CTORCH_TRACE(ErrorPlatform::kAutoDiff, "GradAccumulator::backward - _tensor has been destroyed");
    }
    return {};
}
