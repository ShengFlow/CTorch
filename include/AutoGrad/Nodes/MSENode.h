/**
 * @file MSENode.h
 * @author 苏璃珞
 * @brief 均方误差 (MSE) 损失函数节点
 *
 * @details 数学原理:
 *   前向: L = (1 / N) * sum((a - b)^2)
 *   反向: ∂L/∂a = (2 / N) * (a - b) * grad
 *         ∂L/∂b = -(2 / N) * (a - b) * grad
 *
 * @date 2026/10/13
 */

#ifndef CTORCH_MSENODE_H
#define CTORCH_MSENODE_H

#include "AutoGrad/Node.h"

class MSENode final : public Node {
public:
    MSENode() = default;

    MSENode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,
            const std::vector<Tensor>& inputs);

    MSENode(std::vector<std::shared_ptr<Node>>&& upStreamNodes,
            std::vector<Tensor>&& inputs);

    MSENode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,
            const std::vector<Tensor>& inputs,
            const std::weak_ptr<Tensor>& result);

    MSENode(std::vector<std::shared_ptr<Node>>&& upStreamNodes,
            std::vector<Tensor>&& inputs,
            const std::weak_ptr<Tensor>& result);

    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif // CTORCH_MSENODE_H
