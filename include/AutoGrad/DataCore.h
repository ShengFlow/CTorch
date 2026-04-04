/**
 *@file DataCore.h
 *@brief 自动微分系统核心
 *@author Beapoe
 *@date 2026/2/18
 **/

#ifndef CTORCH_CORE_H
#define CTORCH_CORE_H
#include "../include/AutoGrad/Node.h"
#include "Nodes/GradAccumulator.h"
#include "Arena.h"

class DataCore {
    DataCore() = default;

  public:
    template <typename T>
    static void registerNode(std::vector<Tensor> inputs, std::weak_ptr<Tensor> result) {
        bool toContinue = false;
        for (auto& input:inputs) if (input.requires_grad()) toContinue = true;
        if (toContinue) {
            Arena &arena = Arena::getInstance();
            std::vector<std::shared_ptr<Node>> upStreamNodes;
            upStreamNodes.reserve(inputs.size());
            for (const auto &input : inputs) {
                if (input.requires_grad()) upStreamNodes.push_back(input.getRelatedNode());
                else upStreamNodes.push_back(nullptr);
            }
            const auto node = arena.invoke<T>(upStreamNodes,inputs, result);
            if (result.lock()) {
                result.lock()->setRelatedNode(node);
                for (auto& upStream:upStreamNodes) if (upStream != nullptr) upStream->increase();
            }
            else Ctorch_Error::error(ErrorPlatform::kAutoDiff,ErrorType::UNKNOWN,"Tensor was destroyed but called.");
        }
    }
};

#endif // CTORCH_CORE_H