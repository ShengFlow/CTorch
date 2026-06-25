/**
* @file SubNode.h
 * @author Beapoe
 * @brief 减法节点定义
 * @date 2026/2/21
 **/

#ifndef CTORCH_SUBNODE_H
#define CTORCH_SUBNODE_H

#include "AutoGrad/Node.h"

class SubNode final: public Node {
public:
    SubNode() = default;
    SubNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs);
    SubNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);
    SubNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result);
    SubNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs, const std::weak_ptr<Tensor>& result);
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif // CTORCH_SUBNODE_H
