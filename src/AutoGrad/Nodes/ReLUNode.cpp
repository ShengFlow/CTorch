#include "AutoGrad/Nodes/ReLUNode.h"
#include "CoreDefs.h"
#include "../../../src/kernels/kernels.h"

ReLUNode::ReLUNode(const std::vector<std::shared_ptr<Node>> &upStreamNodes,
                   const std::vector<Tensor> &inputs)
    : Node(upStreamNodes, inputs) {}

ReLUNode::ReLUNode(std::vector<std::shared_ptr<Node>> &&upStreamNodes, std::vector<Tensor> &&inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

ReLUNode::ReLUNode(const std::vector<std::shared_ptr<Node>> &upStreamNodes,
                   const std::vector<Tensor> &inputs, const std::weak_ptr<Tensor> &result)
    : Node(upStreamNodes, inputs, result) {}

ReLUNode::ReLUNode(std::vector<std::shared_ptr<Node>> &&upStreamNodes, std::vector<Tensor> &&inputs,
                   const std::weak_ptr<Tensor> &result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> ReLUNode::backward(const std::vector<Tensor> &downStreamGrads) {
    std::vector<GradPack> ret;

    if (_inputs.size() != 1) {
        CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN, "ReLUNode: 输入数量错误");
        return ret;
    }
    if (downStreamGrads.empty()) {
        return ret;
    }

    const Tensor &x        = _inputs[0];
    const Tensor &grad_out = downStreamGrads[0];

    Tensor mask   = (x > 0);
    Tensor grad_x = mask * grad_out;

    // MPS：确保反向中的元素级 kernel 写回完成后再把梯度传递出去，避免后续深拷贝读到旧值
    if (x.device() == DeviceType::kMPS) {
        MPS_flush_wait(true);
    }

    ret.push_back(GradPack{_upStreamNodes[0], std::vector({grad_x}), 0});

    return ret;
}