/**
 * @file SoftmaxNode.h
 * @author Beapoe
 * @brief Softmax 节点实现
 *
 * @details 数学原理:
 *
 * 前向传播 (Forward):
 *   给定输入向量 x = [x₁, x₂, ..., xₙ]，沿指定维度 dim 计算 Softmax:
 *     softmax(x)[i] = exp(x[i]) / Σ(exp(x[j])) for j in same dimension
 *
 *   即:
 *     y[i] = e^(x[i]) / Σⱼ e^(x[j])
 *
 *   Softmax 将任意实数向量转换为概率分布，所有输出元素和为 1
 *
 * 反向传播 (Backward):
 *   对于损失函数 L，Softmax 的雅可比矩阵是对角矩阵减去外积:
 *     ∂y[i]/∂x[j] = y[i] * (δ[i,j] - y[j])
 *
 *   其中 δ[i,j] 是克罗内克 delta (当 i=j 时为 1，否则为 0)
 *
 *   最终梯度:
 *     ∂L/∂x = y * (∂L/∂y - Σⱼ(y[j] * ∂L/∂y[j]))
 *
 *   简化计算:
 *     grad_x = softmax_grad = grad * softmax - grad * softmax * softmax.T()
 *              = grad * softmax * (1 - softmax.sum(dim, keepdim=True))
 *
 * 数值稳定性优化:
 *   为防止 exp() 溢出，通常从每个元素减去最大值:
 *     softmax(x) = exp(x - max(x)) / Σ exp(x - max(x))
 *
 *   本实现使用缓存的前向传播结果来提高性能和数值稳定性
 *
 * @date 2026/4/5
 **/

#ifndef CTORCH_SOFTMAXNODE_H
#define CTORCH_SOFTMAXNODE_H

#include "AutoGrad/Node.h"

/**
 * @class SoftmaxNode
 * @brief Softmax 激活函数节点，实现 softmax(x) 的前向和反向传播
 *
 * @note 默认沿最后一维计算 Softmax
 */
class SoftmaxNode final : public Node {
private:
    /** @brief Softmax 计算的维度 */
    int _dim = -1;

public:
    SoftmaxNode() = default;

    /**
     * @brief 构造 Softmax 节点
     * @param upStreamNodes 上游节点列表 (长度为1)
     * @param inputs 输入张量列表 [x]
     */
    SoftmaxNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs);

    /**
     * @brief 移动构造 Softmax 节点
     * @param upStreamNodes 上游节点列表 (右值引用)
     * @param inputs 输入张量列表 (右值引用)
     */
    SoftmaxNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);

    /**
     * @brief 构造 Softmax 节点 (带输出张量和维度)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     * @param dim Softmax 计算的维度
     */
    SoftmaxNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
            const std::weak_ptr<Tensor>& result, int dim = -1);

    /**
     * @brief 移动构造 Softmax 节点 (带输出张量和维度)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     * @param dim Softmax 计算的维度
     */
    SoftmaxNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
            const std::weak_ptr<Tensor>& result, int dim = -1);

    /**
     * @brief 反向传播
     * @param downStreamGrads 下游梯度 [∂L/∂y]
     * @return 梯度包列表 [GradPack(∂L/∂x)]
     *
     * 数学公式:
     *   grad_x = grad * y - (grad * y).sum(dim) * y
     *          = grad * y * (1 - y.sum(dim))
     *
     * @note 使用缓存的 softmax 结果 y 来计算梯度
     */
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif  // CTORCH_SOFTMAXNODE_H
