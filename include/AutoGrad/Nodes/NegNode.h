#ifndef CTORCH_NEGNODE_H
#define CTORCH_NEGNODE_H

#include "AutoGrad/Node.h"

class NegNode final:public Node {
public:
    NegNode() = default;
    NegNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs);
    NegNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result);
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif // CTORCH_NEGNODE_H