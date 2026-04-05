#ifndef CTORCH_SINNODE_H
#define CTORCH_SINNODE_H

#include "AutoGrad/Node.h"

class SinNode final:public Node {
public:
    SinNode() = default;
    SinNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs);
    SinNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result);
    std::vector<GradPack> backward(std::vector<Tensor> downStreamGrads) override;
};

#endif // CTORCH_SINNODE_H