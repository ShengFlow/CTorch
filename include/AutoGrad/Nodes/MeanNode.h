/**
 * @file MeanNode.h
 * @brief 全 reduce mean 节点: c = (1/n) Σ a
 *
 * 反向传播:
 *   ∂L/∂a = dL/dc / n 广播到 a 的形状(每个元素都是 dL/dc / n)
 *
 * 背景: Tensor::mean() 此前为裸循环求和、完全未挂节点, backward 静默断链。
 *       本节点补全该反向链路。
 * @date 2026/09/06
 **/

#ifndef CTORCH_MEANNODE_H
#define CTORCH_MEANNODE_H

#include "AutoGrad/Node.h"

class MeanNode final : public Node {
public:
    MeanNode() = default;

    MeanNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs);

    MeanNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);

    MeanNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
             const std::weak_ptr<Tensor>& result);

    MeanNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
             const std::weak_ptr<Tensor>& result);

    /// 反向: grad_a = (downStreamGrad / n) 广播到输入形状
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif  // CTORCH_MEANNODE_H
