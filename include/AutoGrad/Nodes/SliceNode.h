/**
 * @file SliceNode.h
 * @author 苏璃珞
 * @brief 切片节点定义
 *
 * @details 数学原理:
 *
 * 前向传播 (Forward):
 *   沿 dim 维截取 [start, start + size) 区间:
 *     c = input[.., start : start + size, ..]   (dim 维)
 *
 * 反向传播 (Backward):
 *   切片是「选取子集」操作，其转置（伴随）是「按同一位置散射、其余置零」:
 *     grad_a = zeros_like(a);  grad_a[.., start : start + size, ..] = grad_c
 *
 *   这区别于转置（双射，自逆）与重塑（重解释布局，按逆形状还原）：
 *   切片是投影（非满射），反向必须显式补零，否则梯度形状与输入不符。
 *
 * @note 前向不经过调度器。与转置同理，切片只改 shape 与 storage_offset、
 *       **不重算 strides**，是纯元数据操作，不需要 kernel；本节点只承担反向
 *       传播所需的图连接。这样也避免了为它新增 op 枚举项（那会触及 op 顺序与
 *       kCount 静态断言两条红线）。
 *
 * @note 由于 strides 保持不变，本节点对非连续输入同样正确 —— 无论是前向切片
 *       还是反向散射，都按「外层 x 区间 x 内层」的块结构拷贝，不假设紧凑布局。
 *       注意这与 ReshapeNode 不同：后者必须重算 strides，因而要求输入连续。
 *
 * @date 2026/9/17
 **/

#ifndef CTORCH_SLICENODE_H
#define CTORCH_SLICENODE_H

#include "AutoGrad/Node.h"

#include <cstddef>

/**
 * @class SliceNode
 * @brief 切片运算节点，实现 slice(dim, start, size) 的反向传播
 */
class SliceNode final : public Node {
public:
    SliceNode() = default;

    /**
     * @brief 构造切片节点
     * @param dim   切片维度
     * @param start 起始下标
     * @param size  截取长度
     * @param upStreamNodes 上游节点列表（长度为 1）
     * @param inputs 输入张量列表 [a]
     * @param result 输出张量的弱引用
     */
    SliceNode(int dim, std::size_t start, std::size_t size,
              std::vector<std::shared_ptr<Node>>&& upStreamNodes,
              std::vector<Tensor>&& inputs,
              const std::weak_ptr<Tensor>& result);

    /**
     * @brief 反向传播
     * @param downStreamGrads 下游梯度 [∂L/∂c]
     * @return 梯度包列表 [GradPack(∂L/∂a)]，形状与输入一致
     *
     * 数学公式:
     *   grad_a = zeros_like(a); grad_a[dim, start : start + size] = grad_c
     */
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;

private:
    int _dim = 0;
    std::size_t _start = 0;
    std::size_t _size = 0;
};

#endif  // CTORCH_SLICENODE_H
