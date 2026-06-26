/**
 * @file MulNode.h
 * @author Beapoe
 * @brief 乘法节点定义
 *
 * @details 数学原理:
 *
 * 前向传播 (Forward):
 *   给定两个输入张量 a 和 b，逐元素乘法定义为:
 *     c = a ⊗ b
 *   即 c[i] = a[i] * b[i]
 *
 * 反向传播 (Backward):
 *   根据链式法则和乘积法则，对于损失函数 L:
 *
 *   对于输入 a 的梯度:
 *     ∂L/∂a = ∂L/∂c · ∂c/∂a = ∂L/∂c · b
 *
 *   对于输入 b 的梯度:
 *     ∂L/∂b = ∂L/∂c · ∂c/∂b = ∂L/∂c · a
 *
 *   总结:
 *     grad_a = downStreamGrad * b
 *     grad_b = downStreamGrad * a
 *
 * 注意: 对于广播情况，梯度需要沿广播轴求和
 *
 * @date 2026/2/17
 **/

#ifndef CTORCH_MULNODE_H
#define CTORCH_MULNODE_H

#include "AutoGrad/Node.h"

/**
 * @class MulNode
 * @brief 逐元素乘法运算节点，实现 a * b 的前向和反向传播
 *
 * @note 该节点支持广播机制，当输入形状不同时会自动广播
 */
class MulNode final : public Node {
public:
    MulNode() = default;

    /**
     * @brief 构造乘法节点
     * @param upStreamNodes 上游节点列表 (长度为2)
     * @param inputs 输入张量列表 [a, b]
     */
    MulNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs);

    /**
     * @brief 移动构造乘法节点
     * @param upStreamNodes 上游节点列表 (右值引用)
     * @param inputs 输入张量列表 (右值引用)
     */
    MulNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);

    /**
     * @brief 构造乘法节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    MulNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 移动构造乘法节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    MulNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 反向传播
     * @param downStreamGrads 下游梯度 [∂L/∂c]
     * @return 梯度包列表 [GradPack(∂L/∂a), GradPack(∂L/∂b)]
     *
     * 数学公式:
     *   grad_a = downStreamGrad * b
     *   grad_b = downStreamGrad * a
     */
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif  // CTORCH_MULNODE_H
