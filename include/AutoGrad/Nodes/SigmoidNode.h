#ifndef CTORCH_SIGMOIDNODE_H
#define CTORCH_SIGMOIDNODE_H

#include "AutoGrad/Node.h"

class SigmoidNode final:public Node {
public:
    SigmoidNode() = default;
    SigmoidNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs);
    SigmoidNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result);
    std::vector<GradPack> backward(std::vector<Tensor> downStreamGrads) override;
};

#endif // CTORCH_SIGMOIDNODE_H