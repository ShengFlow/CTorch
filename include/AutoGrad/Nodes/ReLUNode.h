/**
 * @file ReLUNode.h
 * @author Beapoe
 * @brief ReLU 激活函数节点定义
 *
 * @details 数学原理:
 *
 * 前向传播 (Forward):
 *   给定输入张量 a，ReLU (Rectified Linear Unit) 操作定义为:
 *     c = max(0, a)
 *   即 c[i] = max(0, a[i]) = { a[i] if a[i] > 0, 0 otherwise }
 *
 * 反向传播 (Backward):
 *   根据链式法则，对于损失函数 L:
 *     ∂L/∂a = ∂L/∂c · ∂c/∂a
 *
 *   ReLU 函数的导数为分段函数:
 *     ∂c/∂a = { 1 if a > 0, 0 if a <= 0 }
 *
 *   因此:
 *     ∂L/∂a = ∂L/∂c · 导数
 *           = { ∂L/∂c if a > 0, 0 otherwise }
 *
 * 总结:
 *   grad_a[i] = downStreamGrad[i] * (a[i] > 0 ? 1 : 0)
 *
 * 特点:
 *   - 计算简单高效
 *   - 缓解梯度消失问题
 *   - 引入了稀疏性
 *   - 不是处处可微 (在 0 处)
 *
 * @date 2026/2/17
 **/

#ifndef CTORCH_RELUNODE_H
#define CTORCH_RELUNODE_H

#include "AutoGrad/Node.h"

/**
 * @class ReLUNode
 * @brief ReLU (Rectified Linear Unit) 激活函数节点，实现 max(0, a) 的前向和反向传播
 */
class ReLUNode final : public Node {
public:
    ReLUNode() = default;

    /**
     * @brief 构造 ReLU 节点
     * @param upStreamNodes 上游节点列表 (长度为1)
     * @param inputs 输入张量列表 [a]
     */
    ReLUNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs);

    /**
     * @brief 移动构造 ReLU 节点
     * @param upStreamNodes 上游节点列表 (右值引用)
     * @param inputs 输入张量列表 (右值引用)
     */
    ReLUNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);

    /**
     * @brief 构造 ReLU 节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    ReLUNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 移动构造 ReLU 节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    ReLUNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 反向传播
     * @param downStreamGrads 下游梯度 [∂L/∂c]
     * @return 梯度包列表 [GradPack(∂L/∂a)]
     *
     * 数学公式:
     *   grad_a[i] = downStreamGrad[i] if a[i] > 0 else 0
     */
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif  // CTORCH_RELUNODE_H
