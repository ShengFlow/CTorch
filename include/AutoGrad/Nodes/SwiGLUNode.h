/**
 * @file SwiGLUNode.h
 * @brief SwiGLU 算子 Autograd Node (双输入)
 * @details swiglu(x, gate) = silu(x) * gate = (x * sigmoid(x)) * gate
 *          反向: ∂L/∂x = ∂L/∂y * gate * silu_derivative(x)
 *                ∂L/∂gate = ∂L/∂y * silu(x)
 * @author Mavis (PEL25 §6 SwiGLU 算子开发协议 Stage 1)
 * @date 2026-09-05
 */

#ifndef CTORCH_SWIGLUNODE_H
#define CTORCH_SWIGLUNODE_H

#include "AutoGrad/Node.h"

/**
 * @class SwiGLUNode
 * @brief SwiGLU 节点, 实现 swiglu(x, gate) = silu(x) * gate
 *
 * 双输入 Node, upstream = [x_node, gate_node]
 */
class SwiGLUNode final : public Node {
public:
    SwiGLUNode() = default;

    SwiGLUNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs);

    SwiGLUNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);

    SwiGLUNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
              const std::weak_ptr<Tensor>& result);

    SwiGLUNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
              const std::weak_ptr<Tensor>& result);

    /**
     * @brief 反向传播
     * @param downStreamGrads [∂L/∂y]
     * @return [GradPack(∂L/∂x), GradPack(∂L/∂gate)]
     */
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif  // CTORCH_SWIGLUNODE_H
