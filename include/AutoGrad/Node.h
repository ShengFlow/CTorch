/**
 *@file Node.h
 *@author Beapoe
 *@brief 节点定义
 *@date 2026/2/17
 **/

#ifndef CTORCH_NODE_H
#define CTORCH_NODE_H

#include "Tensor.h"
#include <memory>

class ComputeCore;
class Node;

struct GradPack {
    std::weak_ptr<Node> _targetNode;
    Tensor _grad;
};

class Node {
protected:
    std::vector<std::weak_ptr<Node>> _upStreamNodes;
    std::vector<Tensor> _inputs;
    std::weak_ptr<Tensor> _result;
    size_t _dependencies{0};
    size_t _count{0};
public:
    Node(const std::vector<std::weak_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs);

    Node(const std::vector<std::weak_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result);

    virtual ~Node() = default;

    void increase();

    void decrease();

    void restore();

    [[nodiscard]] size_t getDependencies() const;

    void setDependencies(size_t dependencies);

    [[nodiscard]] size_t getCount() const;

    void setCount(size_t count);

    virtual std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) = 0;
};

#endif // CTORCH_NODE_H
