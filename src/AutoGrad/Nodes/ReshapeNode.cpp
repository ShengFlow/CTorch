/**
 * @file ReshapeNode.cpp
 * @author 苏璃珞
 * @brief 形状重塑节点实现
 * @date 2026/9/17
 **/

#include "AutoGrad/Nodes/ReshapeNode.h"

ReshapeNode::ReshapeNode(std::vector<std::size_t> input_shape,
                         std::vector<std::shared_ptr<Node>>&& upStreamNodes,
                         std::vector<Tensor>&& inputs,
                         const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result),
      _input_shape(std::move(input_shape)) {}

std::vector<GradPack> ReshapeNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;
    if (downStreamGrads.empty() || _upStreamNodes.empty() || _input_shape.empty()) {
        return ret;
    }

    // 下游梯度可能来自更上游留下的非连续视图（例如切片），先连续化再还原形状 ——
    // reshapeNoGrad 明确拒绝非连续输入，直接传下去会抛异常。
    // 这里处于反向过程中，不建图，故做一次拷贝是安全且必要的。
    const Tensor grad = downStreamGrads[0].is_contiguous() ? downStreamGrads[0]
                                                           : downStreamGrads[0].contiguous();

    // 重塑是重解释布局而非选取子集：元素一一对应，反向只需按输入形状还原。
    // 必须用 reshapeNoGrad —— 若走 reshape() 会在反向过程中再次建图，
    // 导致每次 backward 都往图上接一段，图逐轮膨胀。
    Tensor grad_input = grad.reshapeNoGrad(_input_shape);

    ret.push_back(GradPack{_upStreamNodes[0], std::vector({grad_input}), 0});
    return ret;
}
