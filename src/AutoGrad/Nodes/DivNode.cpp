#include "AutoGrad/Nodes/DivNode.h"

DivNode::DivNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs) 
    : Node(upStreamNodes, inputs) {}

DivNode::DivNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

DivNode::DivNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result) 
    : Node(upStreamNodes, inputs, result) {}

DivNode::DivNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs, const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> DivNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;

    if (_inputs.size() != 2) {
        CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN, "DivNode: 输入数量错误");
        return ret;
    }
    if (downStreamGrads.empty()) {
        return ret;
    }
    
    const Tensor& denominator = _inputs[1];
    const float* denom_data = denominator.data<float>();
    for (size_t i = 0; i < denominator.numel(); ++i) {
        if (denom_data[i] == 0.0f) {
            CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN, "DivNode: 分母为零");
            return ret;
        }
    }

    Tensor grad1 = downStreamGrads[0] / denominator;
    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({grad1}),
        0
    });
    
    Tensor grad2 = -(_inputs[0] / (denominator * denominator)) * downStreamGrads[0];
    ret.push_back(GradPack{
        _upStreamNodes[1],
        std::vector({grad2}),
        1
    });
    
    return ret;
}