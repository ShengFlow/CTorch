/**
 * @file AutogradMeta.h
 * @brief 自动微分元数据结构体
 * @author GhostFace, Beapoe
 * @date 2026/4/10
 * @details 封装 Tensor 中与自动微分相关的所有状态，实现职责分离。
 */
#ifndef AUTOGRAD_META_H
#define AUTOGRAD_META_H

#include <memory>

class Tensor;
class Node;

/**
 * @struct AutogradMeta
 * @brief 自动微分元数据
 * @details 封装 Tensor 中与自动微分相关的成员，包括：
 *          - _self: 自引用 weak_ptr，用于支持外部获取 weak_ptr
 *          - _node: 与该张量相关的计算图节点
 *          - _requires_grad: 是否参与自动微分计算
 *          - _grad: 梯度张量
 */
struct AutogradMeta {
    std::shared_ptr<Tensor> _self;
    mutable std::shared_ptr<Node> _node;
    bool _requires_grad = false;
    std::shared_ptr<Tensor> _grad;
};

#endif // AUTOGRAD_META_H
