/**
 * @file SinNode.h
 * @author Beapoe
 * @brief 正弦节点定义
 *
 * @details 数学原理:
 *
 * 前向传播 (Forward):
 *   给定输入张量 a，正弦操作定义为:
 *     c = sin(a)
 *   即 c[i] = sin(a[i])
 *
 * 反向传播 (Backward):
 *   根据链式法则，对于损失函数 L:
 *     ∂L/∂a = ∂L/∂c · ∂c/∂a
 *
 *   正弦函数的导数为余弦:
 *     ∂c/∂a = cos(a)
 *
 *   因此:
 *     ∂L/∂a = ∂L/∂c · cos(a)
 *
 * 总结:
 *   grad_a = downStreamGrad * cos(a)
 *
 * @date 2026/2/17
 **/

#ifndef CTORCH_SINNODE_H
#define CTORCH_SINNODE_H

#include "AutoGrad/Node.h"

/**
 * @class SinNode
 * @brief 正弦运算节点，实现 sin(a) 的前向和反向传播
 */
class SinNode final : public Node {
public:
    SinNode() = default;

    /**
     * @brief 构造正弦节点
     * @param upStreamNodes 上游节点列表 (长度为1)
     * @param inputs 输入张量列表 [a]
     */
    SinNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs);

    /**
     * @brief 移动构造正弦节点
     * @param upStreamNodes 上游节点列表 (右值引用)
     * @param inputs 输入张量列表 (右值引用)
     */
    SinNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);

    /**
     * @brief 构造正弦节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    SinNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 移动构造正弦节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    SinNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 反向传播
     * @param downStreamGrads 下游梯度 [∂L/∂c]
     * @return 梯度包列表 [GradPack(∂L/∂a)]
     *
     * 数学公式:
     *   grad_a = downStreamGrad * cos(a)
     */
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif  // CTORCH_SINNODE_H
