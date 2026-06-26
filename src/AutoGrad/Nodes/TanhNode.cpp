#include "AutoGrad/Nodes/TanhNode.h"
#include "AutoGrad/Nodes/TanhNode.h"
#include <cmath>

TanhNode::TanhNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs)
    : Node(upStreamNodes, inputs) {}

TanhNode::TanhNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

TanhNode::TanhNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result)
    : Node(upStreamNodes, inputs, result) {}

TanhNode::TanhNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs, const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> TanhNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;
    
    // 对于Tanh算子，导数是1 - tanh^2(input)
    // 手动计算tanh，避免调用tanh()方法可能带来的问题
    const float* input_data = _inputs[0].data<float>();
    size_t count = _inputs[0].numel();
    
    Tensor grad(ShapeTag{}, _inputs[0].sizes(), _inputs[0].dtype(), _inputs[0].device());
    float* grad_data = grad.data<float>();
    const float* down_data = downStreamGrads[0].data<float>();
    
    for (size_t i = 0; i < count; ++i) {
        float tanh_val = std::tanh(input_data[i]);
        grad_data[i] = (1.0f - tanh_val * tanh_val) * down_data[i];
    }
    
    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({grad}),
        0
    });
    
    return ret;
}