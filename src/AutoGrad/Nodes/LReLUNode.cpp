#include "AutoGrad/Nodes/LReLUNode.h"
#include "../../../src/kernels/kernels.h"

LReLUNode::LReLUNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs) 
    : Node(upStreamNodes, inputs) {}

LReLUNode::LReLUNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

LReLUNode::LReLUNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result) 
    : Node(upStreamNodes, inputs, result) {}

LReLUNode::LReLUNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs, const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> LReLUNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;

    if (_inputs.size() != 1) {
        CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN, "LReLUNode: 输入数量错误");
        return ret;
    }
    if (downStreamGrads.empty()) {
        return ret;
    }
    
    const Tensor& x = _inputs[0];
    const Tensor& grad_out = downStreamGrads[0];
    Tensor grad_x;

    switch (x.device()) {
        case DeviceType::kMPS:
            grad_x = LReLU_Grad_MPS_kernel(x, grad_out);
            break;
        case DeviceType::kSIMD:
            grad_x = LReLU_Grad_SIMD_kernel(x, grad_out);
            break;
        case DeviceType::kCPU:
            grad_x = LReLU_Grad_BASIC_kernel(x, grad_out);
            break;
        case DeviceType::kAMX:
            // AMX 不适合逐元素 unary 激活函数，降级到 SIMD
            grad_x = LReLU_Grad_SIMD_kernel(x, grad_out);
            break;
        default:
            CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::DEVICE_COMPAT,
                               "LReLUNode::backward: 不支持的设备类型");
            return ret;
    }

#ifdef __APPLE__
    if (x.device() == DeviceType::kMPS) {
        MPS_flush_wait(true);
    }
#endif

    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({grad_x}),
        0
    });
    
    return ret;
}
