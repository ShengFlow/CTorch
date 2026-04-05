/**
*@file SoftmaxNode.cpp
 *@author Beapoe
 *@brief Softmax节点实现
 *@date 2026/4/5
 **/

#include "AutoGrad/Nodes/SoftmaxNode.h"

SoftmaxNode::SoftmaxNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs) 
    : Node(upStreamNodes, inputs) {}

SoftmaxNode::SoftmaxNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result) 
    : Node(upStreamNodes, inputs, result) {}

std::vector<GradPack> SoftmaxNode::backward(std::vector<Tensor> downStreamGrads) {
    std::vector<GradPack> ret;
    
    // 对于softmax，导数是：
    // softmax(x) * (downStreamGrads - sum(downStreamGrads * softmax(x), dim))
    
    // 检查输入数量
    if (_inputs.size() != 1) {
        Ctorch_Error::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN, "SoftmaxNode: 输入数量错误");
        return ret;
    }
    
    const Tensor& x = _inputs[0]; // 输入
    const Tensor& grad = downStreamGrads[0]; // 下游梯度
    
    // 计算softmax(x)
    Tensor softmax_x = x.softmax(1);
    
    // 计算sum(downStreamGrads * softmax(x), dim=1)
    Tensor sum_grad_softmax = (grad * softmax_x).sum(1, true);
    
    // 计算梯度：softmax(x) * (downStreamGrads - sum_grad_softmax)
    Tensor grad_x = softmax_x * (grad - sum_grad_softmax);
    
    // 添加到返回值
    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({grad_x}),
        0
    });
    
    return ret;
}