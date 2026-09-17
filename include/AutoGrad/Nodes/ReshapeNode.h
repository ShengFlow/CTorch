/**
 * @file ReshapeNode.h
 * @author 苏璃珞
 * @brief 形状重塑节点定义
 *
 * @details 数学原理:
 *
 * 前向传播 (Forward):
 *   在元素线性顺序不变的前提下改变形状:
 *     c = reshape(a, new_shape)     要求 numel(c) == numel(a)
 *
 * 反向传播 (Backward):
 *   重塑不改变元素的线性顺序，因此其伴随就是重塑回输入形状:
 *     grad_a = reshape(grad_c, a.shape)
 *
 *   与切片（投影，反向需补零）和转置（双射，自逆）都不同：重塑是**重解释布局**，
 *   元素一一对应，反向只需按逆形状还原。
 *
 * @note 前向不经过调度器。与转置、切片同理，重塑只改 shape / strides，
 *       是纯元数据操作、不需要 kernel；本节点只承担反向传播所需的图连接。
 *       这样也避免了为它新增 op 枚举项（那会触及 op 顺序与 kCount 静态断言）。
 *
 * @warning 反向实现要求**输入与下游梯度都连续**。`reshapeNoGrad` 已对非连续
 *          输入做显式拒绝（抛异常），因此能走到本节点即意味着输入连续；
 *          但下游梯度仍可能是更上游留下的非连续视图，故反向前仍做一次连续化。
 *
 * @date 2026/9/17
 **/

#ifndef CTORCH_RESHAPENODE_H
#define CTORCH_RESHAPENODE_H

#include "AutoGrad/Node.h"

#include <cstddef>
#include <vector>

/**
 * @class ReshapeNode
 * @brief 形状重塑节点，实现 reshape(new_shape) 的反向传播
 */
class ReshapeNode final : public Node {
public:
    ReshapeNode() = default;

    /**
     * @brief 构造重塑节点
     * @param input_shape 输入张量形状（反向时按此还原）
     * @param upStreamNodes 上游节点列表（长度为 1）
     * @param inputs 输入张量列表 [a]
     * @param result 输出张量的弱引用
     */
    ReshapeNode(std::vector<std::size_t> input_shape,
                std::vector<std::shared_ptr<Node>>&& upStreamNodes,
                std::vector<Tensor>&& inputs,
                const std::weak_ptr<Tensor>& result);

    /**
     * @brief 反向传播
     * @param downStreamGrads 下游梯度 [∂L/∂c]
     * @return 梯度包列表 [GradPack(∂L/∂a)]，形状与输入一致
     *
     * 数学公式:
     *   grad_a = reshape(grad_c, shape(a))
     */
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;

private:
    std::vector<std::size_t> _input_shape;
};

#endif  // CTORCH_RESHAPENODE_H
