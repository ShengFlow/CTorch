#include "AutoGrad/Nodes/ReLUNode.h"

ReLUNode::ReLUNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs) 
    : Node(upStreamNodes, inputs) {}

ReLUNode::ReLUNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

ReLUNode::ReLUNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result) 
    : Node(upStreamNodes, inputs, result) {}

ReLUNode::ReLUNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs, const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> ReLUNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;
    
    const Tensor& x = _inputs[0];
    const Tensor& grad_out = downStreamGrads[0];
    Tensor grad_x(ShapeTag{}, x.sizes(), x.dtype(), x.device());
    const float* x_p = x.data<float>();
    const float* gout_p = grad_out.data<float>();
    float* gx_p = grad_x.data<float>();
    size_t n = x.numel();
    for (size_t i = 0; i < n; ++i) {
        gx_p[i] = x_p[i] > 0.0f ? gout_p[i] : 0.0f;
    }
    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({grad_x}),
        0
    });
    
    return ret;
}