/**
 * @file SigmoidNode.h
 * @author Beapoe
 * @brief Sigmoid 节点定义
 *
 * @details 数学原理:
 *
 * 前向传播 (Forward):
 *   给定输入张量 a，Sigmoid 操作定义为:
 *     c = sigmoid(a)
 *   即 c[i] = 1 / (1 + e^(-a[i]))
 *
 * 反向传播 (Backward):
 *   根据链式法则，对于损失函数 L:
 *     ∂L/∂a = ∂L/∂c · ∂c/∂a
 *
 *   Sigmoid 函数的导数为:
 *     ∂c/∂a = sigmoid(a) · (1 - sigmoid(a)) = c · (1 - c)
 *
 *   因此:
 *     ∂L/∂a = ∂L/∂c · y · (1 - y)  (其中 y = sigmoid(a))
 *
 * 总结:
 *   grad_a = downStreamGrad * y * (1 - y)
 *          = downStreamGrad * sigmoid(a) * (1 - sigmoid(a))
 *
 * 数值稳定性:
 *   - 使用缓存的前向传播结果 y 计算，避免重复计算
 *   - sigmoid 函数的输出范围是 (0, 1)
 *
 * @date 2026/2/17
 **/

#ifndef CTORCH_SIGMOIDNODE_H
#define CTORCH_SIGMOIDNODE_H

#include "AutoGrad/Node.h"

/**
 * @class SigmoidNode
 * @brief Sigmoid 激活函数节点，实现 sigmoid(a) 的前向和反向传播
 */
class SigmoidNode final : public Node {
public:
    SigmoidNode() = default;

    /**
     * @brief 构造 Sigmoid 节点
     * @param upStreamNodes 上游节点列表 (长度为1)
     * @param inputs 输入张量列表 [a]
     */
    SigmoidNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs);

    /**
     * @brief 移动构造 Sigmoid 节点
     * @param upStreamNodes 上游节点列表 (右值引用)
     * @param inputs 输入张量列表 (右值引用)
     */
    SigmoidNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);

    /**
     * @brief 构造 Sigmoid 节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    SigmoidNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 移动构造 Sigmoid 节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    SigmoidNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 反向传播
     * @param downStreamGrads 下游梯度 [∂L/∂c]
     * @return 梯度包列表 [GradPack(∂L/∂a)]
     *
     * 数学公式:
     *   grad_a = downStreamGrad * y * (1 - y)
     *          其中 y = sigmoid(a) 是前向传播的输出
     */
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif  // CTORCH_SIGMOIDNODE_H
