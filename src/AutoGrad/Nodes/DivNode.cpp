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
    
    const Tensor& numerator = _inputs[0];
    const Tensor& denominator = _inputs[1];
    const float* denom_data = denominator.data<float>();
    for (size_t i = 0; i < denominator.numel(); ++i) {
        if (denom_data[i] == 0.0f) {
            CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN, "DivNode: 分母为零");
            return ret;
        }
    }

    const Tensor& grad = downStreamGrads[0];
    
    Tensor grad1 = grad / denominator;
    if (numerator.dim() < grad1.dim()) {
        std::vector<size_t> input_shape = numerator.sizes();
        std::vector<size_t> grad_shape = grad1.sizes();
        
        std::vector<int> reduce_dims;
        size_t dim_diff = grad1.dim() - numerator.dim();
        
        for (size_t d = 0; d < grad1.dim(); ++d) {
            size_t input_dim_size;
            if (d < dim_diff) {
                input_dim_size = 1;
            } else {
                input_dim_size = input_shape[d - dim_diff];
            }
            size_t grad_dim_size = grad_shape[d];
            
            if (input_dim_size == 1 && grad_dim_size > 1) {
                reduce_dims.push_back(static_cast<int>(d));
            }
        }
        
        if (!reduce_dims.empty()) {
            grad1 = grad1.sum(reduce_dims);
        }
    }
    if (grad1.dim() > numerator.dim()) {
        grad1 = grad1.reshape(numerator.sizes());
    }
    
    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({grad1}),
        0
    });
    
    Tensor grad2 = -(_inputs[0] / (denominator * denominator)) * grad;
    if (denominator.dim() < grad2.dim()) {
        std::vector<size_t> input_shape = denominator.sizes();
        std::vector<size_t> grad_shape = grad2.sizes();
        
        std::vector<int> reduce_dims;
        size_t dim_diff = grad2.dim() - denominator.dim();
        
        for (size_t d = 0; d < grad2.dim(); ++d) {
            size_t input_dim_size;
            if (d < dim_diff) {
                input_dim_size = 1;
            } else {
                input_dim_size = input_shape[d - dim_diff];
            }
            size_t grad_dim_size = grad_shape[d];
            
            if (input_dim_size == 1 && grad_dim_size > 1) {
                reduce_dims.push_back(static_cast<int>(d));
            }
        }
        
        if (!reduce_dims.empty()) {
            grad2 = grad2.sum(reduce_dims);
        }
    }
    if (grad2.dim() > denominator.dim()) {
        grad2 = grad2.reshape(denominator.sizes());
    }
    
    ret.push_back(GradPack{
        _upStreamNodes[1],
        std::vector({grad2}),
        1
    });
    
    return ret;
}