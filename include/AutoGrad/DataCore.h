/**
 *@file DataCore.h
 *@brief 自动微分系统核心
 *@author Beapoe
 *@date 2026/2/18
 **/

#ifndef CTORCH_CORE_H
#define CTORCH_CORE_H
#include "../Arena.h"
#include "Node.h"
#include "Nodes/GradAccumulator.h"

class DataCore {
    DataCore() = default;

  public:
    template <typename T>
    static void registerNode(const Tensor& a, const Tensor& b, std::weak_ptr<Tensor> result) {
        bool toContinue = false;
        if (a.requires_grad() || b.requires_grad()) {
            toContinue = true;
        }
        if (toContinue) {
            Arena &arena = Arena::getInstance();
            std::vector<std::shared_ptr<Node>> upStreamNodes;
            upStreamNodes.reserve(2);

            // 处理第一个输入张量
            if (a.requires_grad()) {
                // 使用 const_cast 获取非 const 引用，以便调用非 const 的方法
                Tensor& nonConstA = const_cast<Tensor&>(a);
                if (nonConstA.getRelatedNode() == nullptr) {
                    nonConstA.setRelatedNode(arena.invoke<GradAccumulator>(a));
                }
                upStreamNodes.push_back(nonConstA.getRelatedNode());
            } else {
                upStreamNodes.push_back(nullptr);
            }

            // 处理第二个输入张量
            if (b.requires_grad()) {
                // 使用 const_cast 获取非 const 引用，以便调用非 const 的方法
                Tensor& nonConstB = const_cast<Tensor&>(b);
                if (nonConstB.getRelatedNode() == nullptr) {
                    nonConstB.setRelatedNode(arena.invoke<GradAccumulator>(b));
                }
                upStreamNodes.push_back(nonConstB.getRelatedNode());
            } else {
                upStreamNodes.push_back(nullptr);
            }

            // 创建输入向量，直接使用原始的 a 和 b 张量
            std::vector<Tensor> inputs = {a, b};

            // 创建操作节点，传递输入张量的引用
            const auto node = arena.invoke<T>(upStreamNodes, inputs, result);
            if (result.lock()) {
                result.lock()->setRelatedNode(node);
                for (auto& upStream : upStreamNodes) {
                    if (upStream != nullptr) {
                        upStream->increase();
                    }
                }
            } else {
                CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN, "Tensor was destroyed but called.");
            }
        }
    }
};

#endif // CTORCH_CORE_H