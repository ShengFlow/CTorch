/**
 * @file DimReduceNode.h
 * @brief 沿某维 reduce 节点(sum(dim)/mean(dim) 共用, scale=1 或 1/dim_size)
 *
 * 反向传播: 输出 grad 沿 reduce 维广播回输入形状。
 *   sum(dim): ∂L/∂a = dL/dc 沿 dim 广播 (scale=1)
 *   mean(dim): ∂L/∂a = dL/dc / dim_size 沿 dim 广播 (scale=1/dim_size)
 *
 * 背景: Tensor::sum(dim)/mean(dim) 此前为裸标量循环且完全未挂节点, backward 断链;
 *       mean(dim) 甚至只有声明无实现(链接错误)。本节点 + SIMD 前向/广播一并补全。
 * @date 2026/09/06
 **/

#ifndef CTORCH_DIMREDUCENODE_H
#define CTORCH_DIMREDUCENODE_H

#include "AutoGrad/Node.h"

class DimReduceNode final : public Node {
public:
    DimReduceNode() = default;

    DimReduceNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs);

    DimReduceNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);

    DimReduceNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
                  const std::weak_ptr<Tensor>& result);

    DimReduceNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
                  const std::weak_ptr<Tensor>& result);

    /// 设置 reduce 元数据(由 Tensor::sum(dim)/mean(dim) 在注册后调用)
    /// dim: 被 reduce 的维; keepdim: 是否保留; dim_size: 该维长度;
    /// stride_dim: 该维步长(contiguous 最内维=1); pre/post: 前后维元素数;
    /// pre_stride = stride_dim * dim_size; scale: sum=1, mean=1/dim_size。
    void setReduceMeta(int dim, bool keepdim, size_t dim_size, size_t stride_dim,
                       size_t pre, size_t post, size_t pre_stride, float scale);

    /// 反向: grad_out 沿 dim 广播回输入形状(SIMD 填充, stride==1 走 4-wide)
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;

private:
    bool meta_set_ = false;
    int dim_ = -1;
    bool keepdim_ = false;
    size_t dim_size_ = 0;
    size_t stride_dim_ = 0;
    size_t pre_ = 0;
    size_t post_ = 0;
    size_t pre_stride_ = 0;
    float scale_ = 1.0f;
};

#endif  // CTORCH_DIMREDUCENODE_H
