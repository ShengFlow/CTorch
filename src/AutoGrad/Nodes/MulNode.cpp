/**
*@file MulNode.h
 *@author Beapoe
 *@brief 乘法节点定义
 *@date 2026/2/17
 **/

#include "AutoGrad/Nodes/MulNode.h"
#include "AutoGrad/Nodes/BroadcastUtils.h"

using ctorch::autograd::compute_broadcast_reduce_dims;

MulNode::MulNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs) 
    : Node(upStreamNodes, inputs) {}

MulNode::MulNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

MulNode::MulNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result) 
    : Node(upStreamNodes, inputs, result) {}

MulNode::MulNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs, const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> MulNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;

    if (_inputs.size() != 2) {
        CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN, "MulNode: 输入数量错误");
        return ret;
    }
    if (downStreamGrads.empty()) {
        return ret;
    }

    const Tensor& grad = downStreamGrads[0];
    
    for (size_t i = 0; i < _inputs.size(); ++i) {
        const Tensor& input = _inputs[i];
        const Tensor& other_input = _inputs[1 - i];
        Tensor grad_input = grad * other_input;

        // 处理广播：将 grad_input 在 input shape 为 1 而 grad shape 大于 1 的维度上求和。
        // 对齐到最右边，避免依赖 dim() 相等（标量 Tensor 的 shape 可能是 {1}）。
        std::vector<int> reduce_dims = compute_broadcast_reduce_dims(input.sizes(), grad_input.sizes());
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
