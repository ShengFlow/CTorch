/**
 * @file DivNode.h
 * @author Beapoe
 * @brief 除法节点定义
 *
 * @details 数学原理:
 *
 * 前向传播 (Forward):
 *   给定两个输入张量 a 和 b，除法操作定义为:
 *     c = a ⊘ b
 *   即 c[i] = a[i] / b[i]
 *
 * 反向传播 (Backward):
 *   根据链式法则和商的导数法则，对于损失函数 L:
 *
 *   对于输入 a 的梯度 (分子导数):
 *     ∂L/∂a = ∂L/∂c · ∂c/∂a = ∂L/∂c · (1/b)
 *
 *   对于输入 b 的梯度 (分母导数，使用商的导数法则):
 *     ∂L/∂b = ∂L/∂c · ∂c/∂b = ∂L/∂c · (-a/b²)
 *
 *   总结:
 *     grad_a = downStreamGrad / b
 *     grad_b = -downStreamGrad * a / (b * b)
 *
 * 注意:
 *   - 分母 b 不能为零，否则会触发除零错误
 *   - 对于广播情况，梯度需要沿广播轴求和
 *
 * @date 2026/2/17
 **/

#ifndef CTORCH_DIVNODE_H
#define CTORCH_DIVNODE_H

#include "AutoGrad/Node.h"

/**
 * @class DivNode
 * @brief 逐元素除法运算节点，实现 a / b 的前向和反向传播
 *
 * @warning 该节点会检查分母是否为零，零分母会触发错误
 * @note 该节点支持广播机制，当输入形状不同时会自动广播
 */
class DivNode final : public Node {
public:
    DivNode() = default;

    /**
     * @brief 构造除法节点
     * @param upStreamNodes 上游节点列表 (长度为2)
     * @param inputs 输入张量列表 [a, b]
     */
    DivNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs);

    /**
     * @brief 移动构造除法节点
     * @param upStreamNodes 上游节点列表 (右值引用)
     * @param inputs 输入张量列表 (右值引用)
     */
    DivNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);

    /**
     * @brief 构造除法节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    DivNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 移动构造除法节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    DivNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 反向传播
     * @param downStreamGrads 下游梯度 [∂L/∂c]
     * @return 梯度包列表 [GradPack(∂L/∂a), GradPack(∂L/∂b)]
     *
     * 数学公式:
     *   grad_a = downStreamGrad / b
     *   grad_b = -downStreamGrad * a / (b * b)
     *
     * @throws 如果分母 b 包含零值，抛出错误
     */
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif  // CTORCH_DIVNODE_H
