/**
*@file AddNode.cpp
 *@author Beapoe
 *@brief 加法节点实现
 *@date 2026/2/21
 **/

#include "AutoGrad/Nodes/AddNode.h"
#include "AutoGrad/Nodes/BroadcastUtils.h"
#include "Tensor.h"
#include "Ctools.h"

using ctorch::autograd::compute_broadcast_reduce_dims;

AddNode::AddNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs) 
    : Node(upStreamNodes, inputs) {}

AddNode::AddNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

AddNode::AddNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result) 
    : Node(upStreamNodes, inputs, result) {}

AddNode::AddNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs, const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> AddNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;
    if (downStreamGrads.empty()) {
        return ret;
    }
    const Tensor& grad = downStreamGrads[0];
    
    for (size_t i = 0; i < _inputs.size(); ++i) {
        const Tensor& input = _inputs[i];
        Tensor grad_input = grad;

        // 处理广播：将 grad 在 input shape 为 1 而 grad shape 大于 1 的维度上求和。
        // 对齐到最右边，避免依赖 dim() 相等（标量 Tensor 的 shape 可能是 {1}）。
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