/**
 *@file SoftmaxNode.h
 *@author Beapoe
 *@brief Softmax节点实现
 *@date 2026/4/5
 **/

#ifndef CTORCH_SOFTMAXNODE_H
#define CTORCH_SOFTMAXNODE_H

#include "AutoGrad/Node.h"

class SoftmaxNode final:public Node {
private:
    int _dim = -1;
public:
    SoftmaxNode() = default;
    SoftmaxNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs);
    SoftmaxNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);
    SoftmaxNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result, int dim = -1);
    SoftmaxNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs, const std::weak_ptr<Tensor>& result, int dim = -1);
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif // CTORCH_SOFTMAXNODE_H