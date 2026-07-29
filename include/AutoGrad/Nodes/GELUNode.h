/**
 * @file GELUNode.h
 * @brief GELU 节点定义
 *
 * @details 数学原理:
 *
 * 前向传播 (Forward):
 *   给定输入张量 a，GELU 操作定义为:
 *     c = GELU(a)
 *   即 c[i] = 0.5 * a[i] * (1 + tanh(sqrt(2/π) * (a[i] + 0.044715 * a[i]^3)))
 *
 * 反向传播 (Backward):
 *   令 u = a + 0.044715 * a^3
 *   令 v = sqrt(2/π) * u
 *   ∂c/∂a = 0.5 * (1 + tanh(v))
 *         + 0.5 * a * (1 - tanh^2(v)) * sqrt(2/π) * (1 + 0.134145 * a^2)
 *
 *   因此:
 *     grad_a = downStreamGrad * ∂c/∂a
 *
 * @date 2026/7/28
 **/

#ifndef CTORCH_GELUNODE_H
#define CTORCH_GELUNODE_H

#include "AutoGrad/Node.h"

/**
 * @class GELUNode
 * @brief GELU 激活函数节点，实现 GELU(a) 的前向和反向传播
 */
class GELUNode final : public Node {
public:
    GELUNode() = default;

    /**
     * @brief 构造 GELU 节点
     * @param upStreamNodes 上游节点列表 (长度为1)
     * @param inputs 输入张量列表 [a]
     */
    GELUNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs);

    /**
     * @brief 移动构造 GELU 节点
     * @param upStreamNodes 上游节点列表 (右值引用)
     * @param inputs 输入张量列表 (右值引用)
     */
    GELUNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);

    /**
     * @brief 构造 GELU 节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    GELUNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 移动构造 GELU 节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    GELUNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 反向传播
     * @param downStreamGrads 下游梯度 [∂L/∂c]
     * @return 梯度包列表 [GradPack(∂L/∂a)]
     */
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif  // CTORCH_GELUNODE_H
