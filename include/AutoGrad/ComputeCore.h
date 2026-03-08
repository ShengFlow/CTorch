/**
*@file ComputeCore.h
 *@author Beapoe
 *@brief 计算核心
 *@date 2026/2/22
 **/

#ifndef CTORCH_COMPUTECORE_H
#define CTORCH_COMPUTECORE_H

#include <memory>
#include <vector>
#include <queue>
#include <future>
#include <condition_variable>
#include <optional>
#include "AutoGrad/Node.h"

// struct ReadyNode {
//     std::shared_ptr<Node> _node;
//     std::vector<Tensor> _downStreamGrads;
// };
//
// template <typename T>
// class ThreadSafeQueue {
//     std::queue<T> _data;
//     std::mutex _mtx;
//     std::condition_variable _cv;
//     bool _stop{false};
// public:
//     void push(const T&& elem);
//
//     std::optional<T> pop() const;
// };

class GradBucket {
    std::vector<GradPack> _packs;
};

class ComputeCore {
    static std::vector<std::unique_ptr<Node>> _nodes;
public:
    static void addReadyNode(std::unique_ptr<Node>& node){_nodes.push_back(std::move(node));}
};

#endif // CTORCH_COMPUTECORE_H
