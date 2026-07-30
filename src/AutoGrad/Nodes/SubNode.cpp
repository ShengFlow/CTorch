/**
*@file SubNode.cpp
 *@author Beapoe
 *@brief 减法节点定义
 *@date 2026/2/17
 **/

#include "AutoGrad/Nodes/SubNode.h"
#include "AutoGrad/Nodes/BroadcastUtils.h"
#include "Tensor.h"

using ctorch::autograd::compute_broadcast_reduce_dims;

SubNode::SubNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs)
    : Node(upStreamNodes, inputs) {}

SubNode::SubNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

SubNode::SubNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result)
    : Node(upStreamNodes, inputs, result) {}

SubNode::SubNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs, const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> SubNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;
    if (downStreamGrads.empty()) {
        return ret;
    }
    const Tensor& grad = downStreamGrads[0];

    // 对于 c = a - b，导数是：∂c/∂a = 1，∂c/∂b = -1
    // 所以 grad_a = grad，grad_b = -grad。
    // 处理广播：下游梯度 shape 可能大于输入 shape，需要按广播规则求和。
    for (size_t i = 0; i < _inputs.size(); ++i) {
        const Tensor& input = _inputs[i];
        Tensor grad_input = (i == 0) ? grad : -grad;

        std::vector<int> reduce_dims = compute_broadcast_reduce_dims(input.sizes(), grad.sizes());
        if (!reduce_dims.empty()) {
            grad_input = grad_input.sum(reduce_dims);
        }

        if (grad_input.sizes() != input.sizes()) {
            grad_input = grad_input.reshape(input.sizes());
        }

        ret.push_back(GradPack{
            _upStreamNodes[i],
            std::vector({grad_input}),
            static_cast<int>(i)
        });
    }
    return ret;
}
