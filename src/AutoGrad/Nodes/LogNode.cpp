#include "AutoGrad/Nodes/LogNode.h"

LogNode::LogNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs)
    : Node(upStreamNodes, inputs) {}

LogNode::LogNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

LogNode::LogNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
            const std::weak_ptr<Tensor>& result)
    : Node(upStreamNodes, inputs, result) {}

LogNode::LogNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
            const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> LogNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;

    if (_inputs.size() != 1) {
        return ret;
    }
    if (downStreamGrads.empty()) {
        return ret;
    }

    const Tensor& x = _inputs[0];
    const Tensor& grad_out = downStreamGrads[0];
    
    std::cout << "LogNode::backward - x shape: [";
    for (size_t i = 0; i < x.sizes().size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << x.sizes()[i];
    }
    std::cout << "], x numel: " << x.numel() << ", x storage size: " << x.storage().size() << std::endl;
    
    std::cout << "LogNode::backward - grad_out shape: [";
    for (size_t i = 0; i < grad_out.sizes().size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << grad_out.sizes()[i];
    }
    std::cout << "], grad_out numel: " << grad_out.numel() << ", grad_out storage size: " << grad_out.storage().size() << std::endl;
    
    std::cout << "LogNode::backward - before division" << std::endl;
    Tensor grad_x = grad_out / x;
    std::cout << "LogNode::backward - after division, grad_x shape: [";
    for (size_t i = 0; i < grad_x.sizes().size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << grad_x.sizes()[i];
    }
    std::cout << "], grad_x numel: " << grad_x.numel() << ", grad_x storage size: " << grad_x.storage().size() << std::endl;

    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({grad_x}),
        0
    });

    return ret;
}