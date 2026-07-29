/**
 * @file SubNode.h
 * @author Beapoe
 * @brief 减法节点定义
 *
 * @details 数学原理:
 *
 * 前向传播 (Forward):
 *   给定两个输入张量 a 和 b，减法操作定义为:
 *     c = a ⊖ b
 *   即 c[i] = a[i] - b[i] (逐元素相减)
 *
 * 反向传播 (Backward):
 *   根据链式法则，对于损失函数 L 对 a 和 b 的梯度:
 *     ∂L/∂a = ∂L/∂c · ∂c/∂a
 *     ∂L/∂b = ∂L/∂c · ∂c/∂b
 *
 *   由于 ∂c/∂a = 1, ∂c/∂b = -1:
 *     ∂L/∂a = ∂L/∂c · 1 = ∂L/∂c
 *     ∂L/∂b = ∂L/∂c · (-1) = -∂L/∂c
 *
 * 注意: 对于广播情况，梯度需要沿广播轴求和
 *
 * @date 2026/2/21
 **/

#ifndef CTORCH_SUBNODE_H
#define CTORCH_SUBNODE_H

#include "AutoGrad/Node.h"

/**
 * @class SubNode
 * @brief 减法运算节点，实现 a - b 的前向和反向传播
 *
 * @note 该节点支持广播机制，当输入形状不同时会自动广播
 */
class SubNode final : public Node {
public:
    SubNode() = default;

    /**
     * @brief 构造减法节点
     * @param upStreamNodes 上游节点列表 (长度为2)
     * @param inputs 输入张量列表 [a, b]
     */
    SubNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs);

    /**
     * @brief 移动构造减法节点
     * @param upStreamNodes 上游节点列表 (右值引用)
     * @param inputs 输入张量列表 (右值引用)
     */
    SubNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);

    /**
     * @brief 构造减法节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    SubNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 移动构造减法节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    SubNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 反向传播
     * @param downStreamGrads 下游梯度 [∂L/∂c]
     * @return 梯度包列表 [GradPack(∂L/∂a), GradPack(∂L/∂b)]
     *
     * 数学公式:
     *   grad_a = downStreamGrad
     *   grad_b = -downStreamGrad
     */
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif  // CTORCH_SUBNODE_H
