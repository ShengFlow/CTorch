/**
 * @file SumNode.h
 * @brief 全 reduce sum 节点: c = Σ a
 *
 * 反向传播:
 *   ∂L/∂a = dL/dc 广播到 a 的形状(每个元素都是 dL/dc)
 *
 * 背景: Tensor::sum() 此前经 flat.dot(ones) 挂 DotNode, 但 DotNode 从未实现,
 *       导致 sum 作 loss 时 backward 静默不填梯度。本节点补全该反向链路。
 * @date 2026/09/06
 **/

#ifndef CTORCH_SUMNODE_H
#define CTORCH_SUMNODE_H

#include "AutoGrad/Node.h"

class SumNode final : public Node {
public:
    SumNode() = default;

    SumNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs);

    SumNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);

    SumNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
            const std::weak_ptr<Tensor>& result);

    SumNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
            const std::weak_ptr<Tensor>& result);

    /// 反向: grad_a = downStreamGrad(标量) 广播到输入形状
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif  // CTORCH_SUMNODE_H
