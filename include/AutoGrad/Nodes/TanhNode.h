#ifndef CTORCH_TANHNODE_H
#define CTORCH_TANHNODE_H

#include "AutoGrad/Node.h"

class TanhNode final:public Node {
public:
    TanhNode() = default;
    TanhNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs);
    TanhNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result);
    std::vector<GradPack> backward(std::vector<Tensor> downStreamGrads) override;
};

#endif // CTORCH_TANHNODE_H