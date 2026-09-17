/**
 * @file SqrtNode.h
 * @author 苏璃珞
 * @brief 平方根节点定义
 *
 * @details 数学原理:
 *
 * 前向传播 (Forward):
 *   y = sqrt(x)
 *
 * 反向传播 (Backward):
 *   dy/dx = 1 / (2·sqrt(x)) = 1 / (2y)
 *   grad_x = grad_out / (2·y)
 *
 *   用**前向结果 y** 而非重新对 x 求根：既省一次开方，也保证前向/反向用的是
 *   同一个值（重新求根在浮点上可能产生 1 ULP 差异）。
 *
 * @warning x = 0 处导数发散（grad → ∞）。这里不做截断，与 PyTorch 的 sqrt 一致：
 *          调用方若在可能取零的位置使用 sqrt（例如向量模长、四元数归一化），
 *          应当自行加上微小量（如 sqrt(x + eps)）以保证梯度有界。
 *
 * @date 2026/9/17
 **/

#ifndef CTORCH_SQRTNODE_H
#define CTORCH_SQRTNODE_H

#include "AutoGrad/Node.h"

/**
 * @class SqrtNode
 * @brief 平方根运算节点，实现 sqrt(x) 的反向传播
 */
class SqrtNode final : public Node {
public:
    SqrtNode() = default;

    SqrtNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,
             const std::vector<Tensor>& inputs);

    SqrtNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes,
             std::vector<Tensor>&& inputs);

    SqrtNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,
             const std::vector<Tensor>& inputs, const std::weak_ptr<Tensor>& result);

    SqrtNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes,
             std::vector<Tensor>&& inputs, const std::weak_ptr<Tensor>& result);

    /**
     * @brief 反向传播
     * @param downStreamGrads 下游梯度 [∂L/∂y]
     * @return 梯度包列表 [GradPack(∂L/∂x)]
     *
     * 数学公式:
     *   grad_x = grad_out / (2·y)
     */
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif  // CTORCH_SQRTNODE_H
