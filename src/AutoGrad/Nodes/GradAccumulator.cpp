/**
 *@file GradAccumulator.cpp
 *@brief 梯度累加器(叶子节点)
 *@author Beapoe
 *@date 2026/4/3
 **/

#include "../include/AutoGrad/Nodes/GradAccumulator.h"
#include "../include/Tensor.h"
#ifdef __OBJC__
#include "../../src/kernels/kernels.h"
#endif

extern "C" void MPS_flush_wait(bool wait);

GradAccumulator::GradAccumulator(std::weak_ptr<Tensor> tensor) : _tensor(std::move(tensor)) {
    _upStreamNodes = std::vector<std::shared_ptr<Node>>();
    _inputs = std::vector<Tensor>();
    if (auto t = _tensor.lock()) {
        CTORCH_TRACE(ErrorPlatform::kAutoDiff, "GradAccumulator::GradAccumulator - Created for tensor with requires_grad: " + std::to_string(t->requires_grad()));
    }
}

std::vector<GradPack> GradAccumulator::backward(const std::vector<Tensor>& downStreamGrads) {
    if (downStreamGrads.empty()) {
        return {};
    }

    if (auto tensor = _tensor.lock()) {
        if (tensor->device() == DeviceType::kMPS) {
            MPS_flush_wait(true);
            Tensor accumulated = downStreamGrads[0];

            for (size_t i = 1; i < downStreamGrads.size(); ++i) {
                if (downStreamGrads[i].numel() > 0) {
                    Tensor add_result = accumulated + downStreamGrads[i];
                    accumulated = std::move(add_result);
                }
            }

            auto existing_grad = tensor->grad();
            if (existing_grad.numel() > 0 && existing_grad.storage().data<float>() != nullptr) {
                Tensor add_result = accumulated + existing_grad;
                accumulated = std::move(add_result);
            }

            tensor->setGrad(std::make_shared<Tensor>(std::move(accumulated)));
        } else {
            // 用调度器走 SIMD/AMX 加法，替代标量循环累加
            size_t start_idx = 0;
            while (start_idx < downStreamGrads.size() && downStreamGrads[start_idx].numel() == 0) {
                ++start_idx;
            }

            Tensor accumulated;
            if (start_idx < downStreamGrads.size()) {
                accumulated = downStreamGrads[start_idx];
                for (size_t i = start_idx + 1; i < downStreamGrads.size(); ++i) {
                    if (downStreamGrads[i].numel() > 0) {
                        accumulated = accumulated + downStreamGrads[i];
                    }
                }
            } else {
                accumulated = Tensor(ShapeTag{}, tensor->shape(), tensor->dtype(), tensor->device());
                accumulated.zero();
            }

            auto existing_grad = tensor->grad();
            if (existing_grad.numel() > 0 && existing_grad.storage().data<float>() != nullptr) {
                accumulated = accumulated + existing_grad;
            }

            tensor->setGrad(std::make_shared<Tensor>(std::move(accumulated)));
        }
    }
    return {};
}
