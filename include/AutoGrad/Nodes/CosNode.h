/**
 * @file CosNode.h
 * @author Beapoe
 * @brief 余弦节点定义
 *
 * @details 数学原理:
 *
 * 前向传播 (Forward):
 *   给定输入张量 a，余弦操作定义为:
 *     c = cos(a)
 *   即 c[i] = cos(a[i])
 *
 * 反向传播 (Backward):
 *   根据链式法则，对于损失函数 L:
 *     ∂L/∂a = ∂L/∂c · ∂c/∂a
 *
 *   余弦函数的导数为负正弦:
 *     ∂c/∂a = -sin(a)
 *
 *   因此:
 *     ∂L/∂a = ∂L/∂c · (-sin(a)) = -∂L/∂c · sin(a)
 *
 * 总结:
 *   grad_a = -downStreamGrad * sin(a)
 *
 * @date 2026/2/17
 **/

#ifndef CTORCH_COSNODE_H
#define CTORCH_COSNODE_H

#include "AutoGrad/Node.h"

/**
 * @class CosNode
 * @brief 余弦运算节点，实现 cos(a) 的前向和反向传播
 */
class CosNode final : public Node {
public:
    CosNode() = default;

    /**
     * @brief 构造余弦节点
     * @param upStreamNodes 上游节点列表 (长度为1)
     * @param inputs 输入张量列表 [a]
     */
    CosNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs);

    /**
     * @brief 移动构造余弦节点
     * @param upStreamNodes 上游节点列表 (右值引用)
     * @param inputs 输入张量列表 (右值引用)
     */
    CosNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);

    /**
     * @brief 构造余弦节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    CosNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 移动构造余弦节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    CosNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 反向传播
     * @param downStreamGrads 下游梯度 [∂L/∂c]
     * @return 梯度包列表 [GradPack(∂L/∂a)]
     *
     * 数学公式:
     *   grad_a = -downStreamGrad * sin(a)
     */
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif  // CTORCH_COSNODE_H
