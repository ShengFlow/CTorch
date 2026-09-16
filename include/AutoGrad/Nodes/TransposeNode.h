/**
 * @file TransposeNode.h
 * @author 苏璃珞
 * @brief 转置节点定义
 *
 * @details 数学原理:
 *
 * 前向传播 (Forward):
 *   给定输入张量 a，沿 dim0 / dim1 交换两个维度:
 *     c = transpose(a, dim0, dim1)
 *
 * 反向传播 (Backward):
 *   转置是自身的逆运算（transpose ∘ transpose = identity），故:
 *     grad_a = transpose(downStreamGrad, dim0, dim1)
 *
 * @note 前向不经过调度器。转置只交换 shape / strides，是纯元数据操作，
 *       不需要 kernel；本节点只承担反向传播所需的图连接。
 *       这样也避免了为它新增 op 枚举项（那会触及 op 顺序与 kCount 静态断言）。
 *
 * @date 2026/9/16
 **/

#ifndef CTORCH_TRANSPOSENODE_H
#define CTORCH_TRANSPOSENODE_H

#include "AutoGrad/Node.h"

/**
 * @class TransposeNode
 * @brief 转置运算节点，实现 transpose(a, dim0, dim1) 的反向传播
 */
class TransposeNode final : public Node {
public:
    TransposeNode() = default;

    /**
     * @brief 构造转置节点
     * @param dim0 参与交换的第一个维度
     * @param dim1 参与交换的第二个维度
     * @param upStreamNodes 上游节点列表（长度为 1）
     * @param inputs 输入张量列表 [a]
     * @param result 输出张量的弱引用
     */
    TransposeNode(int dim0, int dim1,
                  std::vector<std::shared_ptr<Node>>&& upStreamNodes,
                  std::vector<Tensor>&& inputs,
                  const std::weak_ptr<Tensor>& result);

    /**
     * @brief 反向传播
     * @param downStreamGrads 下游梯度 [∂L/∂c]
     * @return 梯度包列表 [GradPack(∂L/∂a)]
     *
     * 数学公式:
     *   grad_a = transpose(downStreamGrad, dim0, dim1)
     */
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;

private:
    int _dim0 = 0;
    int _dim1 = 1;
};

#endif  // CTORCH_TRANSPOSENODE_H
