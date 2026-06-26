/**
 * @file TanhNode.h
 * @author Beapoe
 * @brief 双曲正切节点定义
 *
 * @details 数学原理:
 *
 * 前向传播 (Forward):
 *   给定输入张量 a，双曲正切操作定义为:
 *     c = tanh(a)
 *   即 c[i] = tanh(a[i]) = (e^a - e^-a) / (e^a + e^-a)
 *
 * 反向传播 (Backward):
 *   根据链式法则，对于损失函数 L:
 *     ∂L/∂a = ∂L/∂c · ∂c/∂a
 *
 *   双曲正切的导数为:
 *     ∂c/∂a = sech²(a) = 1 - tanh²(a)
 *
 *   因此:
 *     ∂L/∂a = ∂L/∂c · (1 - tanh²(a))
 *
 * 总结:
 *   grad_a = downStreamGrad * (1 - tanh(a)²)
 *          = downStreamGrad * (1 - y²)  (其中 y = tanh(a))
 *
 * 数值稳定性:
 *   - tanh 的输出范围是 (-1, 1)，具有良好的数值稳定性
 *   - 使用缓存的前向传播结果 y 计算，避免重复计算
 *
 * @date 2026/2/17
 **/

#ifndef CTORCH_TANHNODE_H
#define CTORCH_TANHNODE_H

#include "AutoGrad/Node.h"

/**
 * @class TanhNode
 * @brief 双曲正切运算节点，实现 tanh(a) 的前向和反向传播
 */
class TanhNode final : public Node {
public:
    TanhNode() = default;

    /**
     * @brief 构造双曲正切节点
     * @param upStreamNodes 上游节点列表 (长度为1)
     * @param inputs 输入张量列表 [a]
     */
    TanhNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs);

    /**
     * @brief 移动构造双曲正切节点
     * @param upStreamNodes 上游节点列表 (右值引用)
     * @param inputs 输入张量列表 (右值引用)
     */
    TanhNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);

    /**
     * @brief 构造双曲正切节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    TanhNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 移动构造双曲正切节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    TanhNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 反向传播
     * @param downStreamGrads 下游梯度 [∂L/∂c]
     * @return 梯度包列表 [GradPack(∂L/∂a)]
     *
     * 数学公式:
     *   grad_a = downStreamGrad * (1 - y²)  (其中 y = tanh(a) 是前向传播的输出)
     */
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif  // CTORCH_TANHNODE_H
