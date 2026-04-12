/**
*@file CrossEntropyNode.cpp
 *@author Beapoe
 *@brief 交叉熵损失节点实现
 *@date 2026/4/5
 **/

#include "AutoGrad/Nodes/CrossEntropyNode.h"

CrossEntropyNode::CrossEntropyNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs) 
    : Node(upStreamNodes, inputs) {set_requireAccelerate(true);}

CrossEntropyNode::CrossEntropyNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs, const std::weak_ptr<Tensor>& result) 
    : Node(upStreamNodes, inputs, result) {set_requireAccelerate(true);}

std::vector<GradPack> CrossEntropyNode::backward(std::vector<Tensor> downStreamGrads) {
    std::vector<GradPack> ret;

    // 对于交叉熵损失，导数是：
    // 对于第一个输入（logits），导数是 softmax(logits) - target
    // 对于第二个输入（target），导数是 -log(softmax(logits))
    
    // 检查输入数量
    if (_inputs.size() != 2) {
        CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN, "CrossEntropyNode: 输入数量错误");
        return ret;
    }
    
    const Tensor& logits = _inputs[0]; // 第一个输入：logits
    const Tensor& target = _inputs[1]; // 第二个输入：target
    const Tensor& grad = downStreamGrads[0]; // 下游梯度
    
    // 计算softmax(logits)
    Tensor softmax_logits = logits.softmax(1);
    
    // 计算第一个输入的梯度：grad * (softmax_logits - target)
    Tensor grad_logits = grad * (softmax_logits - target);
    
    // 添加到返回值
    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({grad_logits}),
        0
    });
    
    // 第二个输入的梯度通常不需要计算，因为target是标签，不是可训练参数
    
    return ret;
}