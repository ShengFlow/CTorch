/**
 *@file GradAccumulator.cpp
 *@brief 梯度累加器(叶子节点)
 *@author Beapoe
 *@date 2026/4/3
 **/

#include "../include/AutoGrad/Nodes/GradAccumulator.h"
#include "../include/Tensor.h"

GradAccumulator::GradAccumulator(const Tensor& tensor) : _tensor(&tensor) {
    _upStreamNodes = std::vector<std::shared_ptr<Node>>();
    _inputs = std::vector<Tensor>();
    CtorchError::trace(ErrorPlatform::kAutoDiff, "GradAccumulator::GradAccumulator - Created for tensor with requires_grad: " + std::to_string(tensor.requires_grad()));
}

std::vector<GradPack> GradAccumulator::backward(std::vector<Tensor> downStreamGrads) {
    CtorchError::trace(ErrorPlatform::kAutoDiff, "GradAccumulator::backward - Called with downStreamGrads size: " + std::to_string(downStreamGrads.size()));
    if (downStreamGrads.empty()) {
        CtorchError::trace(ErrorPlatform::kAutoDiff, "GradAccumulator::backward - No gradients received");
        return {};
    }

    if (_tensor) {
        CtorchError::trace(ErrorPlatform::kAutoDiff, "GradAccumulator::backward - Setting grad to tensor");
        Tensor* nonConstTensor = const_cast<Tensor*>(_tensor);

        std::shared_ptr<Tensor> grad;
        for (size_t i = 0; i < downStreamGrads.size(); i++) {
            if (downStreamGrads[i].numel() > 0 && downStreamGrads[i].storage().data<float>() != nullptr) {
                grad = std::make_shared<Tensor>(downStreamGrads[i]);
                break;
            }
        }

        if (!grad) {
            grad = std::make_shared<Tensor>(ShapeTag{}, _tensor->shape(), _tensor->dtype(), _tensor->device());
            grad->zero();
        }

        nonConstTensor->setGrad(grad);
        CtorchError::trace(ErrorPlatform::kAutoDiff, "GradAccumulator::backward - Grad set successfully");
    } else {
        CtorchError::trace(ErrorPlatform::kAutoDiff, "GradAccumulator::backward - _tensor is null");
    }
    return {};
}
