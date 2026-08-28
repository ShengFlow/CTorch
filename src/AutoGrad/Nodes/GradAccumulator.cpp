/**
 *@file GradAccumulator.cpp
 *@brief 梯度累加器(叶子节点)
 *@author Beapoe
 *@date 2026/4/3
 **/

#include "../include/AutoGrad/Nodes/GradAccumulator.h"
#include "../include/Tensor.h"
#include "../../../src/kernels/kernels.h"

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
#ifdef __APPLE__
            MPS_flush_wait(true);
#endif
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
            Tensor accumulated;
            // 单梯度快速路径：绝大多数场景（SGD、单消费）只有一个下游梯度
            if (downStreamGrads.size() == 1 && downStreamGrads[0].numel() > 0) {
                accumulated = downStreamGrads[0];
            } else {
                size_t start_idx = 0;
                while (start_idx < downStreamGrads.size() && downStreamGrads[start_idx].numel() == 0) {
                    ++start_idx;
                }
                if (start_idx < downStreamGrads.size()) {
                    accumulated = downStreamGrads[start_idx];
                    for (size_t i = start_idx + 1; i < downStreamGrads.size(); ++i) {
                        if (downStreamGrads[i].numel() > 0) {
                            accumulated = Add_SIMD_kernel(accumulated, downStreamGrads[i]);
                        }
                    }
                } else {
                    accumulated = Tensor(ShapeTag{}, tensor->shape(), tensor->dtype(), tensor->device());
                    accumulated.zero();
                }
            }

            // 仅当确有已有梯度时累加：grad_ptr() 探测避免 grad() 返回整 Tensor 拷贝
            // [Eager/C3 优化 2026-08-27] 直接调 Add_SIMD_kernel，绕开 operator+/dispatch。
            //   此前用 `accumulated + tensor->grad()` 会走 C3HotPathManager::recordCall + tryExecute
            //   热路径调度（14000 次/5ep ≈ 107ms），而这是参数梯度累加、本就无需 C3 追踪/融合。
            //   Add_SIMD_kernel 是纯 CPU 计算、形状相同走 SIMD，语义等价、零调度开销。
            if (tensor->grad_ptr() != nullptr) {
                accumulated = Add_SIMD_kernel(accumulated, tensor->grad());
            }

            tensor->setGrad(std::make_shared<Tensor>(std::move(accumulated)));
        }
    }
    return {};
}
