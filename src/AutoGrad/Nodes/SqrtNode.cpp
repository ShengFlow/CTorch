/**
 * @file SqrtNode.cpp
 * @author 苏璃珞
 * @brief 平方根节点实现
 * @date 2026/9/17
 **/

#include "AutoGrad/Nodes/SqrtNode.h"

SqrtNode::SqrtNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,
                   const std::vector<Tensor>& inputs)
    : Node(upStreamNodes, inputs) {}

SqrtNode::SqrtNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes,
                   std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

SqrtNode::SqrtNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,
                   const std::vector<Tensor>& inputs,
                   const std::weak_ptr<Tensor>& result)
    : Node(upStreamNodes, inputs, result) {}

SqrtNode::SqrtNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes,
                   std::vector<Tensor>&& inputs, const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> SqrtNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;

    if (_inputs.size() != 1 || _upStreamNodes.empty()) {
        return ret;
    }
    if (downStreamGrads.empty()) {
        return ret;
    }

    const Tensor& grad_out = downStreamGrads[0];
    auto result = getResult();
    if (!result) {
        return ret;
    }

    // dy/dx = 1/(2·sqrt(x)) = 1/(2y)，直接用前向结果 y：
    // 既省一次开方，也保证前向与反向取到同一个值。
    //
    // x = 0 时分母为零、梯度发散，此处不截断（与 PyTorch 的 sqrt 一致）。
    // 调用方若在可能取零处使用 sqrt（模长、归一化等），应自行加 eps。
    const Tensor two_y = (*result) * 2.0f;
    const Tensor grad_x = grad_out / two_y;

    ret.push_back(GradPack{_upStreamNodes[0], std::vector({grad_x}), 0});
    return ret;
}
