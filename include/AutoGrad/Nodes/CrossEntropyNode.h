/**
 *@file CrossEntropyNode.h
 *@author Beapoe
 *@brief 交叉熵损失节点实现
 *@date 2026/4/5
 **/

#ifndef CTORCH_CROSSENTROPYNODE_H
#define CTORCH_CROSSENTROPYNODE_H

#include "AutoGrad/Node.h"

class CrossEntropyNode final:public Node {
public:
    CrossEntropyNode() = default;
    CrossEntropyNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs);
    CrossEntropyNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result);
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif // CTORCH_CROSSENTROPYNODE_H