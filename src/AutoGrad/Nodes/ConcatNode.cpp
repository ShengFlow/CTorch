/**
 * @file ConcatNode.cpp
 * @author 苏璃珞
 * @brief 拼接节点实现
 * @date 2026/9/17
 **/

#include "AutoGrad/Nodes/ConcatNode.h"

ConcatNode::ConcatNode(int dim, std::size_t a_dim_len,
                       std::vector<std::shared_ptr<Node>>&& upStreamNodes,
                       std::vector<Tensor>&& inputs,
                       const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result),
      _dim(dim),
      _a_dim_len(a_dim_len) {}

std::vector<GradPack> ConcatNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;
    if (downStreamGrads.empty() || _upStreamNodes.size() < 2 || _inputs.size() < 2) {
        return ret;
    }

    const auto& a_shape = _inputs[0].sizes();
    const auto& b_shape = _inputs[1].sizes();
    if (a_shape.empty() || b_shape.empty()) {
        return ret;
    }
    if (_dim < 0 || _dim >= static_cast<int>(a_shape.size())) {
        return ret;
    }

    // 拼接是「两路合并」，其伴随（转置）是把下游梯度沿 dim 切回两段 ——
    // 与切片互为伴随：concatenate 的前向首尾相接，反向则正好是被拼接的两段。
    // 因此这里直接复用 sliceNoGrad，无需重写块拷贝逻辑。
    //
    // 切片视图与 grad_out 共享存储，而 grad_out 的生命周期由调用方（反向遍历）
    // 持有；为避免梯度在累积期间被上游复用存储影响，这里物化一份独立副本。
    const Tensor grad_out = downStreamGrads[0].is_contiguous() ? downStreamGrads[0]
                                                               : downStreamGrads[0].contiguous();

    if (_a_dim_len > static_cast<std::size_t>(grad_out.sizes()[static_cast<std::size_t>(_dim)])) {
        return ret;
    }
    const std::size_t b_len = grad_out.sizes()[static_cast<std::size_t>(_dim)] - _a_dim_len;

    Tensor grad_a = grad_out.sliceNoGrad(_dim, 0, _a_dim_len).contiguous();
    Tensor grad_b = grad_out.sliceNoGrad(_dim, _a_dim_len, b_len).contiguous();
    grad_a.requires_grad(false);
    grad_b.requires_grad(false);

    // 上游可能不需要梯度（upStreamNodes 对应项为 nullptr），GradPack 原样携带，
    // 由反向遍历端跳过 —— 与 AddNode 的处理一致。
    ret.push_back(GradPack{_upStreamNodes[0], std::vector({grad_a}), 0});
    ret.push_back(GradPack{_upStreamNodes[1], std::vector({grad_b}), 1});
    return ret;
}
