/**
 * @file ComputeCore.h
 * @author Beapoe
 * @brief 自动微分系统计算核心
 * @date 2026/2/22
 */

#ifndef CTORCH_COMPUTECORE_H
#define CTORCH_COMPUTECORE_H

#include <memory>
#include <vector>
#include <queue>
#include <condition_variable>
#include <mutex>

#include "Node.h"

/**
 * @class GradBucket
 * @brief 梯度桶类，用于存储和管理反向传播过程中的梯度
 * @details 在反向传播时，上游节点的梯度会被暂存在梯度桶中，
 *          当所有下游梯度都到达后，才会触发该节点的 backward 计算。
 */
class GradBucket {
    /** @brief 梯度包列表 */
    std::vector<GradPack> _packs;
    /** @brief 互斥锁，保证线程安全 */
    std::mutex _mtx;

    /**
     * @brief 查找指定节点的梯度包索引
     * @param target 目标节点
     * @return 梯度包索引，未找到返回-1
     */
    ssize_t find(const std::shared_ptr<Node>& target);

    /** @brief 私有构造函数，防止外部实例化 */
    GradBucket() = default;
  public:
    /**
     * @brief 获取单例实例
     * @return GradBucket的引用
     */
    static GradBucket& getInstance();

    /**
     * @brief 添加梯度包列表
     * @param newPacks 新的梯度包列表（移动语义）
     */
    void add(std::vector<GradPack>&& newPacks);

    /**
     * @brief 移除指定节点的梯度包
     * @param target 目标节点
     */
    void remove(const std::shared_ptr<Node>& target);

    /** @brief 检查梯度桶是否为空 */
    [[nodiscard]] bool empty();

    /** @brief 清空梯度桶 */
    void clear();

    /**
     * @brief 获取指定节点的梯度列表
     * @param target 目标节点
     * @return 梯度张量列表
     */
    std::vector<Tensor> operator[](const std::shared_ptr<Node>& target);

    /**
     * @brief 尝试获取指定节点的梯度
     * @param target 目标节点
     * @param out_grads 输出参数，存储获取到的梯度
     * @return 成功获取返回true，否则返回false
     */
    bool tryGetGrad(const std::shared_ptr<Node>& target, std::vector<Tensor>& out_grads);
};

/**
 * @class ComputeCore
 * @brief 自动微分系统的计算核心类
 * @details 负责执行反向传播算法，管理计算图节点的执行顺序，
 *          实现高效的拓扑排序和梯度传播。
 */
class ComputeCore {
    /** @brief 就绪节点队列，等待执行反向传播的节点 */
    std::queue<std::shared_ptr<Node>> _readyNodes;

    /** @brief 互斥锁，保证线程安全 */
    std::mutex _mtx;
    /** @brief 条件变量，用于线程同步 */
    std::condition_variable _cv;

    /** @brief 私有构造函数，防止外部实例化 */
    ComputeCore() = default;

    /**
     * @brief 尝试从就绪队列中取出一个节点
     * @return 就绪节点，如果队列为空返回nullptr
     */
    std::shared_ptr<Node> tryPopReadyNode();

  public:
    /**
     * @brief 获取单例实例
     * @return ComputeCore的引用
     */
    static ComputeCore &getInstance();

    /**
     * @brief 将节点添加到就绪队列
     * @param node 要添加的节点
     */
    void addReadyNode(std::shared_ptr<Node> node);

    /**
     * @brief 执行反向传播
     * @param root 根节点（损失函数节点）
     * @param retainGraph 是否保留计算图，默认false（释放计算图）
     */
    void backward(std::shared_ptr<Node> root, bool retainGraph = false);
};

#endif // CTORCH_COMPUTECORE_H
