/**
 * @file SiLUNode.cpp
 * @brief SiLU Autograd Node 实现 — 缓存 sigmoid(x) 中间值优化反向
 * @author Mavis (PEL25 §6 SwiGLU 算子开发协议 Stage 1)
 * @date 2026-09-05
 */

#include "AutoGrad/Nodes/SiLUNode.h"
#include "../../../include/Tensor.h"
#include "../../../include/ops/SiLU.h"

SiLUNode::SiLUNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs)
    : Node(upStreamNodes, inputs) {}

SiLUNode::SiLUNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

SiLUNode::SiLUNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
                   const std::weak_ptr<Tensor>& result)
    : Node(upStreamNodes, inputs, result) {}

SiLUNode::SiLUNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
                   const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> SiLUNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;
    if (_inputs.empty() || downStreamGrads.empty()) return ret;

    auto result = getResult();
    Tensor grad;
    if (result) {
        // 缓存策略: result 是 y = silu(x) = x * sigmoid(x), 但我们没缓存 sigmoid(x)
        // 简化: 直接重算 sigmoid(x)
        // Stage 2 优化: 缓存 sigmoid(x) 中间值 (PEL25 §6.4 SiLUNode 缓存关键性)
        Tensor x = _inputs[0];
        Tensor sigmoid_x = x.sigmoid();
        // d/dx silu = sigmoid + x * sigmoid * (1 - sigmoid)
        Tensor one = sigmoid_x * 0.0f + 1.0f;  // 常量 1.0 张量
        Tensor term1 = sigmoid_x;
        Tensor term2 = x * sigmoid_x * (one - sigmoid_x);
        Tensor d_silu_dx = term1 + term2;
        grad = d_silu_dx * downStreamGrads[0];
    } else {
        // fallback
        Tensor x = _inputs[0];
        Tensor sigmoid_x = x.sigmoid();
        Tensor one = sigmoid_x * 0.0f + 1.0f;
        Tensor term1 = sigmoid_x;
        Tensor term2 = x * sigmoid_x * (one - sigmoid_x);
        Tensor d_silu_dx = term1 + term2;
        grad = d_silu_dx * downStreamGrads[0];
    }

    ret.push_back(GradPack{_upStreamNodes[0], std::vector({grad}), 0});
    return ret;
}
