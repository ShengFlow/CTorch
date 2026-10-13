/**
 * @file MAENode.h
 * @author 苏璃珞
 * @brief 平均绝对误差 (MAE) 损失函数节点
 *
 * @details 数学原理:
 *   前向: L = (1 / N) * sum(|a - b|)
 *   反向: ∂L/∂a = (1 / N) * sgn(a - b) * grad
 *         ∂L/∂b = -(1 / N) * sgn(a - b) * grad
 *
 * @date 2026/10/13
 */

#ifndef CTORCH_MAENODE_H
#define CTORCH_MAENODE_H

#include "AutoGrad/Node.h"

class MAENode final : public Node {
public:
    MAENode() = default;

    MAENode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,
            const std::vector<Tensor>& inputs);

    MAENode(std::vector<std::shared_ptr<Node>>&& upStreamNodes,
            std::vector<Tensor>&& inputs);

    MAENode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,
            const std::vector<Tensor>& inputs,
            const std::weak_ptr<Tensor>& result);

    MAENode(std::vector<std::shared_ptr<Node>>&& upStreamNodes,
            std::vector<Tensor>&& inputs,
            const std::weak_ptr<Tensor>& result);

    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif // CTORCH_MAENODE_H
