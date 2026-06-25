/**
 * @file MatMulNode.h
 * @author Beapoe
 * @brief 矩阵乘法节点实现
 * @date 2026/4/5
 **/

#ifndef CTORCH_MATMULNODE_H
#define CTORCH_MATMULNODE_H

#include "AutoGrad/Node.h"

class MatMulNode final:public Node {
public:
    MatMulNode() = default;
    MatMulNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs);
    MatMulNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);
    MatMulNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result);
    MatMulNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs, const std::weak_ptr<Tensor>& result);
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif // CTORCH_MATMULNODE_H