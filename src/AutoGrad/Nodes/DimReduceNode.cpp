/**
 * @file DimReduceNode.cpp
 * @brief 沿某维 reduce 节点实现
 * @date 2026/09/06
 **/

#include "AutoGrad/Nodes/DimReduceNode.h"
#include "Tensor.h"
#include "Ctools.h"
#include "kernels/CPU-SIMD/ReduceSIMD.h"

using ctorch::kernels::simd::fill_f32;

DimReduceNode::DimReduceNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,
                             const std::vector<Tensor>& inputs)
    : Node(upStreamNodes, inputs) {}

DimReduceNode::DimReduceNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes,
                             std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

DimReduceNode::DimReduceNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,
                             const std::vector<Tensor>& inputs,
                             const std::weak_ptr<Tensor>& result)
    : Node(upStreamNodes, inputs, result) {}

DimReduceNode::DimReduceNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes,
                             std::vector<Tensor>&& inputs,
                             const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

void DimReduceNode::setReduceMeta(int dim, bool keepdim, size_t dim_size, size_t stride_dim,
                                  size_t pre, size_t post, size_t pre_stride, float scale) {
    dim_ = dim;
    keepdim_ = keepdim;
    dim_size_ = dim_size;
    stride_dim_ = stride_dim;
    pre_ = pre;
    post_ = post;
    pre_stride_ = pre_stride;
    scale_ = scale;
    meta_set_ = true;
}

std::vector<GradPack> DimReduceNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;
    if (downStreamGrads.empty() || _inputs.empty() || !meta_set_) {
        return ret;
    }
    const Tensor& input = _inputs[0];
    const float* g = downStreamGrads[0].data_read<float>();
    if (!g) {
        return ret;
    }

    Tensor grad_input(ShapeTag{}, input.sizes(), input.dtype(), input.device());
    float* gp = grad_input.data_write<float>();

    if (stride_dim_ == 1) {
        // 最内维归约(contiguous): 每个 pre 行的 dim_size 连续元素 = 同一个 grad 值
        for (size_t r = 0; r < pre_; ++r) {
            fill_f32(gp + r * dim_size_, dim_size_, g[r] * scale_);
        }
    } else {
        // 非最内维: 标量 scatter (grad[pre*post] 的每个值沿 dim 复制 dim_size 份)
        for (size_t i = 0; i < pre_; ++i) {
            for (size_t k = 0; k < post_; ++k) {
                float v = g[i * post_ + k] * scale_;
                for (size_t j = 0; j < dim_size_; ++j) {
                    gp[i * pre_stride_ + j * stride_dim_ + k] = v;
                }
            }
        }
    }

    ret.push_back(GradPack{_upStreamNodes[0], std::vector({grad_input}), 0});
    return ret;
}
