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
#include <mutex>

#include "Node.h"

// 前向声明
// class Node;
// class Tensor;
// struct GradPack;

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
    std::mutex _mtx;

    ssize_t find(const std::shared_ptr<Node>& target);

    GradBucket() = default;
  public:
    static GradBucket& getInstance();

    void add(const std::vector<GradPack>& newPacks);

    void remove(const std::shared_ptr<Node>& target);

    [[nodiscard]] bool empty();

    void clear();

    std::vector<Tensor> operator[](const std::shared_ptr<Node>& target);

    bool tryGetGrad(const std::shared_ptr<Node>& target, std::vector<Tensor>& out_grads);
};

class ComputeCore {
    std::queue<std::shared_ptr<Node>> _readyNodes;

    std::mutex _mtx;
    std::condition_variable _cv;

    ComputeCore() = default;

    std::shared_ptr<Node> tryPopReadyNode();

    void scheduleNode(const std::vector<GradPack>& newPacks);

    void scheduleNode(std::shared_ptr<Node> root);

  public:
    static ComputeCore &getInstance();

    void addReadyNode(std::shared_ptr<Node> node);

    void backward(std::shared_ptr<Node> root,bool retainGraph = false);
};

#endif // CTORCH_COMPUTECORE_H
