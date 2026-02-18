/**
 *@file Node.h
 *@author Beapoe
 *@brief 节点定义
 *@date 2026/2/17
 **/

#ifndef CTORCH_NODE_H
#define CTORCH_NODE_H

#include "../Tensor.h"
#include "GradFn.h"
#include <memory>

/**
 * @class Node
 * @brief 节点类
 */
class Node {
    /**
     * @var _id 节点ID
     * @var _val 前向计算结果
     * @var _requireGrad 是否保留梯度
     * @var _isLeaf 是否是叶子节点
     */
    size_t _id{0};
    std::shared_ptr<Tensor> _val;
    std::unique_ptr<Tensor> _grad;
    std::shared_ptr<GradFn> _fn;
    bool _requireGrad;
    bool _isLeaf;

  public:
    /**
     * @brief 默认有参构造
     * @param val 前向计算结果
     * @param requireGrad 是否计算梯度
     * @param isLeaf 是否是叶子节点
     * @param fn 反向传播函数
     */
    Node(std::shared_ptr<Tensor> val, bool requireGrad, bool isLeaf,
         std::shared_ptr<GradFn> fn = nullptr);

    // 禁止拷贝
    Node(const Node &)            = delete;
    Node &operator=(const Node &) = delete;

    /**
     * @brief 节点ID获取函数
     * @return 返回该节点的ID
     */
    size_t getID() const;

    /**
     * @brief 节点梯度要求状态获取函数
     * @return 返回该节点对于梯度的要求状态
     **/
    bool getRequireGrad() const;

    /**
     * @brief Node梯度获取函数
     * @return 返回Node对应的梯度
     */
    Tensor *getGrad() const;

    /**
     * @brief Node反向传播函数获取函数
     * @return 返回反向传播函数
     */
    GradFn *getGradFn() const;

    /**
     * @brief 叶子节点判断函数
     * @return 判断该节点是否为叶子节点
     */
    bool isLeaf() const;

    /**
     * @brief 设置节点梯度要求状态函数
     * @param state 节点梯度要求状态被设置状态
     */
    void setRequireGrad(bool state);

    /**
     * @brief 直接梯度张量-Node梯度张量设置函数
     * @param grad 梯度张量直接赋值
     */
    void setGrad(const Tensor &grad);

    /**
     * @brief 梯度张量独占指针-Node梯度张量赋值函数
     * @param grad 梯度张量独占指针
     */
    void setGrad(std::unique_ptr<Tensor> grad);

    /**
     * @brief 设置Node的反向传播函数
     * @param fn 被设置的反向传播函数
     */
    void setGradFn(std::shared_ptr<GradFn> fn);

    /**
     * @brief 叶子节点设置函数
     * @param state 叶子节点状态
     */
    void setLeafState(bool state);
};

#endif // CTORCH_NODE_H
