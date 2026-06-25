/**
 * @file DataCore.h
 * @brief 自动微分系统数据核心
 * @author Beapoe
 * @date 2026/2/18
 */

#ifndef CTORCH_CORE_H
#define CTORCH_CORE_H
#include "../Arena.h"
#include "Node.h"
#include "Nodes/GradAccumulator.h"

/**
 * @class DataCore
 * @brief 自动微分系统的数据核心类
 * @details 负责管理计算图节点的注册和构建，是自动微分系统的核心组件之一。
 *          提供静态方法用于注册单输入和双输入操作节点。
 */
class DataCore {
    /** @brief 私有构造函数，防止外部实例化 */
    DataCore() = default;

  public:
    /**
     * @brief 注册单输入操作节点
     * @tparam T 节点类型（如 ReLUNode、NegNode 等）
     * @param input 输入张量
     * @param result 输出张量的弱引用
     */
    template <typename T>
    static void registerNode(const Tensor& input, std::weak_ptr<Tensor> result) {
        if (!input.requires_grad()) return;

        Arena &arena = Arena::getInstance();
        std::vector<std::shared_ptr<Node>> upStreamNodes;

        Tensor& nonConstInput = const_cast<Tensor&>(input);
        if (nonConstInput.getRelatedNode() == nullptr) {
            nonConstInput.setRelatedNode(arena.invoke<GradAccumulator>(input));
        }
        upStreamNodes.push_back(nonConstInput.getRelatedNode());

        std::vector<Tensor> inputs = {input};

        const auto node = arena.invoke<T>(std::move(upStreamNodes), std::move(inputs), result);
        if (result.lock()) {
            result.lock()->setRelatedNode(node);
            auto& nodeRef = *node;
            for (auto& upStream : nodeRef.getUpStreamNodes()) {
                if (upStream != nullptr) {
                    upStream->increase();
                }
            }
        } else {
            CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN, "Tensor was destroyed but called.");
        }
    }

    /**
     * @brief 注册双输入操作节点
     * @tparam T 节点类型（如 AddNode、MulNode、MatMulNode 等）
     * @param a 第一个输入张量
     * @param b 第二个输入张量
     * @param result 输出张量的弱引用
     */
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

            if (a.requires_grad()) {
                Tensor& nonConstA = const_cast<Tensor&>(a);
                if (nonConstA.getRelatedNode() == nullptr) {
                    nonConstA.setRelatedNode(arena.invoke<GradAccumulator>(a));
                }
                upStreamNodes.push_back(nonConstA.getRelatedNode());
            } else {
                upStreamNodes.push_back(nullptr);
            }

            if (b.requires_grad()) {
                Tensor& nonConstB = const_cast<Tensor&>(b);
                if (nonConstB.getRelatedNode() == nullptr) {
                    nonConstB.setRelatedNode(arena.invoke<GradAccumulator>(b));
                }
                upStreamNodes.push_back(nonConstB.getRelatedNode());
            } else {
                upStreamNodes.push_back(nullptr);
            }

            std::vector<Tensor> inputs = {a, b};

            const auto node = arena.invoke<T>(std::move(upStreamNodes), std::move(inputs), result);
            if (result.lock()) {
                result.lock()->setRelatedNode(node);
                auto& nodeRef = *node;
                for (auto& upStream : nodeRef.getUpStreamNodes()) {
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