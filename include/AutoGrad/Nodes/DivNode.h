#ifndef CTORCH_DIVNODE_H
#define CTORCH_DIVNODE_H

#include "AutoGrad/Node.h"

class DivNode final:public Node {
public:
    DivNode() = default;
    DivNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs);
    DivNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result);
    std::vector<GradPack> backward(std::vector<Tensor> downStreamGrads) override;
};

#endif // CTORCH_DIVNODE_H