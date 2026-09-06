/**
 * @file SiLUNode.h
 * @brief SiLU 算子 Autograd Node
 * @details silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *          缓存中间值 sigmoid(x) 用于反向传播, 避免重复计算
 * @author Mavis (PEL25 §6 SwiGLU 算子开发协议 Stage 1)
 * @date 2026-09-05
 *
 * 模板: 照 SigmoidNode 模式 (4 构造 + 1 backward 虚函数)
 *       SiLUNode 缓存 sigmoid(x) 中间值, 类似 SigmoidNode 缓存 y = sigmoid(a)
 */

#ifndef CTORCH_SILUNODE_H
#define CTORCH_SILUNODE_H

#include "AutoGrad/Node.h"

/**
 * @class SiLUNode
 * @brief SiLU 激活函数节点, 实现 silu(a) 的前向和反向传播
 *
 * 前向: c = silu(a) = a * sigmoid(a)
 * 反向: ∂c/∂a = sigmoid(a) + a * sigmoid(a) * (1 - sigmoid(a))
 *                = sigmoid(a) * (1 + a * (1 - sigmoid(a)))
 *       grad_a = downStreamGrad * ∂c/∂a
 */
class SiLUNode final : public Node {
public:
    SiLUNode() = default;

    SiLUNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs);

    SiLUNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);

    SiLUNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
            const std::weak_ptr<Tensor>& result);

    SiLUNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 反向传播
     * @param downStreamGrads 下游梯度 [∂L/∂c]
     * @return 梯度包列表 [GradPack(∂L/∂a)]
     */
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif  // CTORCH_SILUNODE_H
