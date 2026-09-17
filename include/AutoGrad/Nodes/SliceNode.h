/**
 * @file SliceNode.h
 * @author 苏璃珞
 * @brief 切片节点定义
 *
 * @details 数学原理:
 *
 * 前向传播 (Forward):
 *   沿第 0 维截取 [start, start + size) 区间:
 *     c = input[start : start + size]
 *
 * 反向传播 (Backward):
 *   切片是「选取子集」操作，其转置（伴随）是「按同一位置散射、其余置零」:
 *     grad_a = zeros_like(a);  grad_a[start : start + size] = grad_c
 *
 *   这区别于转置：转置是自身的逆（双射），而切片是投影（非满射），
 *   反向必须显式补零，否则梯度形状与输入不符。
 *
 * @note 前向不经过调度器。与转置同理，切片只改 shape / strides / storage_offset，
 *       是纯元数据操作，不需要 kernel；本节点只承担反向传播所需的图连接。
 *       这样也避免了为它新增 op 枚举项（那会触及 op 顺序与 kCount 静态断言）。
 *
 * @warning 反向实现要求输入张量连续。目前 `slice_dim0` 的调用方均为连续张量
 *          （新构造或已 contiguous 化）；若非连续输入触发本节点，反向会先做
 *          连续化再散射，代价是一次拷贝，正确性不受影响。
 *
 * @date 2026/9/17
 **/

#ifndef CTORCH_SLICENODE_H
#define CTORCH_SLICENODE_H

#include "AutoGrad/Node.h"

#include <cstddef>

/**
 * @class SliceNode
 * @brief 切片运算节点，实现 slice_dim0(start, size) 的反向传播
 */
class SliceNode final : public Node {
public:
    SliceNode() = default;

    /**
     * @brief 构造切片节点
     * @param start 起始下标（沿第 0 维）
     * @param size  截取长度
     * @param upStreamNodes 上游节点列表（长度为 1）
     * @param inputs 输入张量列表 [a]
     * @param result 输出张量的弱引用
     */
    SliceNode(std::size_t start, std::size_t size,
              std::vector<std::shared_ptr<Node>>&& upStreamNodes,
              std::vector<Tensor>&& inputs,
              const std::weak_ptr<Tensor>& result);

    /**
     * @brief 反向传播
     * @param downStreamGrads 下游梯度 [∂L/∂c]，形状为 (size, ...)
     * @return 梯度包列表 [GradPack(∂L/∂a)]，形状与输入一致
     *
     * 数学公式:
     *   grad_a = zeros_like(a); grad_a[start : start + size] = grad_c
     */
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;

private:
    std::size_t _start = 0;
    std::size_t _size = 0;
};

#endif  // CTORCH_SLICENODE_H
