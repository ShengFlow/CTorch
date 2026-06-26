/**
 * @file NegNode.h
 * @author Beapoe
 * @brief 负号节点定义
 *
 * @details 数学原理:
 *
 * 前向传播 (Forward):
 *   给定输入张量 a，负号操作定义为:
 *     c = -a
 *   即 c[i] = -a[i]
 *
 * 反向传播 (Backward):
 *   根据链式法则，对于损失函数 L:
 *     ∂L/∂a = ∂L/∂c · ∂c/∂a
 *
 *   由于 ∂c/∂a = -1:
 *     ∂L/∂a = ∂L/∂c · (-1) = -∂L/∂c
 *
 * 总结:
 *   grad_a = -downStreamGrad
 *
 * @date 2026/2/17
 **/

#ifndef CTORCH_NEGNODE_H
#define CTORCH_NEGNODE_H

#include "AutoGrad/Node.h"

/**
 * @class NegNode
 * @brief 负号运算节点，实现 -a 的前向和反向传播
 */
class NegNode final : public Node {
public:
    NegNode() = default;

    /**
     * @brief 构造负号节点
     * @param upStreamNodes 上游节点列表 (长度为1)
     * @param inputs 输入张量列表 [a]
     */
    NegNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs);

    /**
     * @brief 移动构造负号节点
     * @param upStreamNodes 上游节点列表 (右值引用)
     * @param inputs 输入张量列表 (右值引用)
     */
    NegNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);

    /**
     * @brief 构造负号节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    NegNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 移动构造负号节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    NegNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 反向传播
     * @param downStreamGrads 下游梯度 [∂L/∂c]
     * @return 梯度包列表 [GradPack(∂L/∂a)]
     *
     * 数学公式:
     *   grad_a = -downStreamGrad
     */
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif  // CTORCH_NEGNODE_H
