/**
 *@file Node.h
 *@author Beapoe
 *@brief 节点定义
 *@date 2026/2/17
 **/

#ifndef CTORCH_NODE_H
#define CTORCH_NODE_H
#include <atomic>
#include <unordered_set>

#include "Tensor.h"
class Node;
#include <vector>
#include <memory>

struct GradPack{
    std::shared_ptr<Node> _targetNode;
    std::vector<Tensor> _grad;
    int _idx{0};
};


class Node {
protected:
    std::vector<std::shared_ptr<Node>> _upStreamNodes;
    std::vector<Tensor> _inputs;
    std::weak_ptr<Tensor> _result;
    size_t _dependencies{0};
    std::atomic<size_t> _count{0};
    bool _requireAccelerate{false};
public:
    Node() = default;

    Node(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs);

    Node(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result);

    explicit Node(const std::weak_ptr<Tensor> &result);

    virtual ~Node() = default;

    void increase();

    bool decrease();

    void restore();

    [[nodiscard]] size_t getDependencies() const;

    void setDependencies(size_t dependencies);

    void setCount(size_t count);
	
    [[nodiscard]] std::vector<std::shared_ptr<Node>> getUpStreamNodes() const;

    [[nodiscard]] bool requireAccelerate() const;

    void set_requireAccelerate(bool requireAccelerate);

    virtual std::vector<GradPack> backward(std::vector<Tensor> downStreamGrads) = 0;

    void restoreRecursive(std::unordered_set<Node*>& visited);
};

#endif // CTORCH_NODE_H
