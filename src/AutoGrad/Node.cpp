/**
 *@file Node.cpp
 *@author Beapoe
 *@brief 节点实现
 *@date 2026/2/17
 **/

#include "../../include/AutoGrad/Node.h"
#include "../../include/CtorchError.h"
#include "../include/Tensor.h"

void Node::increase() {
    _count.fetch_add(1,std::memory_order_acq_rel);
    _dependencies++;
    // [2026-09-07] 稳定 fanout: 记录被下游注册引用次数(构建期一次, 不随 backward 递减)。
    // MIMO 反向融合等用 getDownstreamCount()==1 做"单消费者"守卫, 防多消费者共享中间丢梯度。
    _downstreamCount.fetch_add(1, std::memory_order_acq_rel);
}

size_t Node::getDownstreamCount() const { return _downstreamCount.load(std::memory_order_acquire); }

bool Node::decrease() {
    size_t old = _count.load(std::memory_order_acquire);
    while (old > 0) {
        if (_count.compare_exchange_strong(old, old - 1, std::memory_order_acq_rel)) {
            return old == 1;
        }
    }
    CtorchError::error(ErrorPlatform::kAutoDiff,ErrorType::UNKNOWN,"Dependency count is negative");
    return false;
}


void Node::restore() { _count.store(_dependencies,std::memory_order_relaxed); }

size_t Node::getDependencies() const { return _dependencies; }

void Node::setDependencies(size_t dependencies) {
    _dependencies = dependencies;
}

void Node::setCount(const size_t count) { _count = count; }

size_t Node::getCount() const { return _count.load(std::memory_order_acquire); }

Node::Node(const std::vector<std::shared_ptr<Node>> &upStreamNodes,
           const std::vector<Tensor> &inputs)
               :_upStreamNodes(upStreamNodes),_inputs(inputs), _dependencies(upStreamNodes.size()) {
    // 初始依赖计数等于上游节点数，保证 Leaf 节点也能正确入队
}

Node::Node(std::vector<std::shared_ptr<Node>>&& upStreamNodes,
           std::vector<Tensor>&& inputs)
               :_upStreamNodes(std::move(upStreamNodes)),_inputs(std::move(inputs)),
                _dependencies(_upStreamNodes.size()) {
    // 初始依赖计数等于上游节点数，保证 Leaf 节点也能正确入队。
    // 注意：必须在 move 完成后从被赋值成员读取 size，源参数已被 move 置空。
}

Node::Node(const std::vector<std::shared_ptr<Node>> &upStreamNodes, const std::vector<Tensor> &inputs,
           const std::weak_ptr<Tensor> &result)
               :_upStreamNodes(upStreamNodes),_inputs(inputs),_result(result), _dependencies(upStreamNodes.size())
{
    if (auto t = result.lock()) {
        _resultShape = t->sizes();
    }
}

Node::Node(std::vector<std::shared_ptr<Node>>&& upStreamNodes,
           std::vector<Tensor>&& inputs,
           const std::weak_ptr<Tensor>& result)
               :_upStreamNodes(std::move(upStreamNodes)),_inputs(std::move(inputs)),_result(result),
                _dependencies(_upStreamNodes.size())
{
    // 注意：必须在 move 完成后从被赋值成员读取 size，源参数已被 move 置空。
    if (auto t = result.lock()) {
        _resultShape = t->sizes();
    }
}

Node::Node(const std::weak_ptr<Tensor> &result)
    :_upStreamNodes(std::vector<std::shared_ptr<Node>>()),_inputs(std::vector<Tensor>()),_result(result), _dependencies(0)
{
    // Leaf 节点：无上游节点，依赖计数为 0，由注册时 increase() 设置
    if (auto t = result.lock()) {
        _resultShape = t->sizes();
    }
}


const std::vector<std::shared_ptr<Node>>& Node::getUpStreamNodes() const { return _upStreamNodes; }

bool Node::requireAccelerate() const { return _requireAccelerate; }

void Node::set_requireAccelerate(bool requireAccelerate) {_requireAccelerate = requireAccelerate;}

// [Fix 2026-09-17] restore / clear 由递归改为显式栈迭代。
//
// 递归实现的调用深度等于计算图的**最长链长度**。CTorch 原有训练图宽而浅
// （MNIST 的 FC、LLaMA 的 FFN 深度十余层），因此长期未暴露；而深链图 —— 例如
// 可微仿真把一条轨迹的每个时间步展开成数十个算子 —— 深度可达上万乃至十万层，
// 直接把线程栈耗尽：实测 10 万层稳定 SIGSEGV（主线程 8 MB 栈，每帧约百字节）。
//
// 传入 self（而非隐式的 this）是因为迭代遍历需要在容器里持有 shared_ptr 来
// 维持节点在清理过程中的存活；裸 this 无法承担这一职责。
void Node::restoreGraph(const std::shared_ptr<Node> &self,
                        std::unordered_set<Node *> &visited) {
    std::vector<std::shared_ptr<Node>> pending;
    pending.push_back(self);
    while (!pending.empty()) {
        std::shared_ptr<Node> n = std::move(pending.back());
        pending.pop_back();
        if (!n || visited.count(n.get()) != 0) {
            continue;
        }
        visited.insert(n.get());
        n->restore();
        for (const auto &up : n->getUpStreamNodes()) {
            if (up) {
                pending.push_back(up);
            }
        }
    }
}

void Node::clearGraph(const std::shared_ptr<Node> &self,
                      std::unordered_set<Node *> &visited) {
    std::vector<std::shared_ptr<Node>> order;   // 父先于子；同时持有强引用
    std::vector<std::shared_ptr<Node>> pending;
    pending.push_back(self);
    while (!pending.empty()) {
        std::shared_ptr<Node> n = std::move(pending.back());
        pending.pop_back();
        if (!n || visited.count(n.get()) != 0) {
            continue;
        }
        visited.insert(n.get());
        order.push_back(n);
        for (const auto &up : n->getUpStreamNodes()) {
            if (up) {
                pending.push_back(up);
            }
        }
    }
    // 逆序 = 子先于父，等价于原递归实现的后序清理
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        Node *n = it->get();
        n->clearResultOwner();
        auto result_copy = n->_result.lock();
        n->_upStreamNodes.clear();
        n->_inputs.clear();
        if (result_copy) {
            result_copy->detach_autograd();
        }
    }
}
