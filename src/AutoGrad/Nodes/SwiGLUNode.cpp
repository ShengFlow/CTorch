/**
 * @file SwiGLUNode.cpp
 * @brief SwiGLU Autograd Node 实现 — 双输入反向 (∂L/∂x + ∂L/∂gate)
 * @author Mavis (PEL25 §6 SwiGLU 算子开发协议 Stage 1)
 * @date 2026-09-05
 */

#include "AutoGrad/Nodes/SwiGLUNode.h"
#include "../../../include/Tensor.h"

SwiGLUNode::SwiGLUNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs)
    : Node(upStreamNodes, inputs) {
    // 期望 inputs = [x, gate]
}

SwiGLUNode::SwiGLUNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

SwiGLUNode::SwiGLUNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
                       const std::weak_ptr<Tensor>& result)
    : Node(upStreamNodes, inputs, result) {}

SwiGLUNode::SwiGLUNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
                       const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> SwiGLUNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;
    if (_inputs.size() < 2 || downStreamGrads.empty()) return ret;
    if (_upStreamNodes.size() < 2) return ret;

    Tensor x = _inputs[0];
    Tensor gate = _inputs[1];
    Tensor grad_y = downStreamGrads[0];

    // ∂silu/∂x = sigmoid + x * sigmoid * (1 - sigmoid)
    Tensor sigmoid_x = x.sigmoid();
    Tensor one = sigmoid_x * 0.0f + 1.0f;
    Tensor silu_d = sigmoid_x + x * sigmoid_x * (one - sigmoid_x);

    // ∂L/∂x = ∂L/∂y * gate * silu_derivative(x)
    Tensor grad_x = grad_y * gate * silu_d;
    // ∂L/∂gate = ∂L/∂y * silu(x) = ∂L/∂y * x * sigmoid(x)
    Tensor grad_gate = grad_y * (x * sigmoid_x);

    ret.push_back(GradPack{_upStreamNodes[0], std::vector({grad_x}), 0});
    ret.push_back(GradPack{_upStreamNodes[1], std::vector({grad_gate}), 1});
    return ret;
}
