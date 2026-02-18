/**
 *@file Core.h
 *@brief 自动微分系统核心
 *@author Beapoe
 *@date 2026/2/18
 **/

#ifndef CTORCH_CORE_H
#define CTORCH_CORE_H
#include "../include/AutoGrad/Node.h"

#include <memory>

/**
 * @class AutoGradCore
 * @brief 自动微分系统的核心类
 */
class AutoGradCore {
  public:
    /**
     * @brief 单例模式获取
     * @return 返回当前线程唯一实例
     */
    static AutoGradCore &getInstance() {
        static thread_local AutoGradCore instance;
        return instance;
    }

    /**
     * @brief 注册节点
     * @param node 被注册节点
     */
    void registerNode(const std::shared_ptr<Node> &node);

    /**
     * @brief 通过ID获取节点
     * @param id 被获取节点ID
     * @return 被获取节点的SharedPtr
     */
    std::shared_ptr<Node> getNode(size_t id);

    /**
     * @brief 反向传播函数入口
     * @param root 反向传播初始节点
     * @param gradInput 反向传播初始节点的梯度
     * @param retainGraph 是否保留计算图
     */
    void backward(const Tensor &root, const Tensor &gradInput, bool retainGraph);

    /**
     * @brief 彻底清除当前线程的所有结点
     */
    void clear();

    /**
     * @brief 重置当前线程节点，保留叶子节点
     */
    void reset();

    /**
     * @brief 注册叶子节点
     * @param tensor 被注册张量
     * @param requireGrad 是否需要梯度
     */
    void makeLeaf(Tensor &tensor, bool requireGrad);

  private:
    /**
     * @var Id2Node ID到Node的映射
     * @var GradFn2Node GradFn*到ID的映射
     * @var Node2Id Node到ID的映射
     * @var Node2GradFn ID到GradFn的映射
     * @var retainGraph 是否保存计算图
     **/
    std::unordered_map<size_t, std::shared_ptr<Node>> Id2Node;
    std::unordered_map<std::shared_ptr<Node>, size_t> Node2Id;
    std::unordered_map<GradFn *, size_t> GradFn2ID;
    std::unordered_map<size_t, GradFn *> ID2GradFn;
    bool _retainGraph{true};
    std::mutex _mutex;

    /**
     * @brief 私有化构造函数
     */
    AutoGradCore();
};

#endif // CTORCH_CORE_H
