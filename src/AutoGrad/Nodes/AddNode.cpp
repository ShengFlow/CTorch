/**
*@file AddNode.cpp
 *@author Beapoe
 *@brief 加法节点实现
 *@date 2026/2/21
 **/

#include "AutoGrad/Nodes/AddNode.h"
#include "Tensor.h"
#include "Ctools.h"

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
        
        if (input.dim() < grad.dim()) {
            std::vector<size_t> input_shape = input.sizes();
            std::vector<size_t> grad_shape = grad.sizes();
            
            std::vector<int> reduce_dims;
            size_t dim_diff = grad.dim() - input.dim();
            
            for (size_t d = 0; d < grad.dim(); ++d) {
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
                grad_input = grad.sum(reduce_dims);
            }
        }
        
        if (grad_input.dim() > input.dim()) {
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