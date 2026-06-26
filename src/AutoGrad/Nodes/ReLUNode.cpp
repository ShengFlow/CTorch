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

    if (_inputs.size() != 1) {
        CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN, "ReLUNode: 输入数量错误");
        return ret;
    }
    if (downStreamGrads.empty()) {
        return ret;
    }
    
    const Tensor& x = _inputs[0];
    const Tensor& grad_out = downStreamGrads[0];
    Tensor grad_x(ShapeTag{}, x.sizes(), x.dtype(), x.device());
    size_t n = x.numel();

    auto relu_grad = [&](auto x_p, auto gout_p, auto gx_p) {
        for (size_t i = 0; i < n; ++i) {
            gx_p[i] = x_p[i] > 0 ? gout_p[i] : 0;
        }
    };

    switch (x.dtype()) {
        case DType::kFloat: relu_grad(x.data<float>(), grad_out.data<float>(), grad_x.data<float>()); break;
        case DType::kDouble: relu_grad(x.data<double>(), grad_out.data<double>(), grad_x.data<double>()); break;
        case DType::kInt: relu_grad(x.data<int32_t>(), grad_out.data<int32_t>(), grad_x.data<int32_t>()); break;
        case DType::kLong: relu_grad(x.data<int64_t>(), grad_out.data<int64_t>(), grad_x.data<int64_t>()); break;
        default: break;
    }

    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({grad_x}),
        0
    });
    
    return ret;
}