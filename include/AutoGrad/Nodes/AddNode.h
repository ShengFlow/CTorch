/**
 *@file AddNode.h
 *@author Beapoe
 *@brief 加法节点实现
 *@date 2026/2/21
 **/

#ifndef CTORCH_ADDNODE_H
#define CTORCH_ADDNODE_H

#include "AutoGrad/Node.h"

class AddNode final:public Node {
public:
    AddNode() = default;
    AddNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs);
    AddNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result);
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif // CTORCH_ADDNODE_H