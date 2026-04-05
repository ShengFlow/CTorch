#ifndef CTORCH_COSNODE_H
#define CTORCH_COSNODE_H

#include "AutoGrad/Node.h"

class CosNode final:public Node {
public:
    CosNode() = default;
    CosNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs);
    CosNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result);
    std::vector<GradPack> backward(std::vector<Tensor> downStreamGrads) override;
};

#endif // CTORCH_COSNODE_H