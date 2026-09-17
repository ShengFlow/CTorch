/**
 * @file SliceNode.cpp
 * @author 苏璃珞
 * @brief 切片节点实现
 * @date 2026/9/17
 **/

#include "AutoGrad/Nodes/SliceNode.h"

#include <algorithm>

SliceNode::SliceNode(int dim, std::size_t start, std::size_t size,
                     std::vector<std::shared_ptr<Node>>&& upStreamNodes,
                     std::vector<Tensor>&& inputs,
                     const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result),
      _dim(dim),
      _start(start),
      _size(size) {}

std::vector<GradPack> SliceNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;
    if (downStreamGrads.empty() || _upStreamNodes.empty() || _inputs.empty()) {
        return ret;
    }

    const Tensor& input = _inputs[0];
    const auto& shape = input.sizes();
    if (shape.empty() || _dim < 0 || _dim >= static_cast<int>(shape.size())) {
        return ret;
    }

    const auto dim = static_cast<std::size_t>(_dim);
    if (_start + _size > shape[dim]) {
        return ret;
    }

    // 沿 dim 维切片时，张量可视为「外层 x 该维 x 内层」的三段块结构：
    //   outer = Π shape[0..dim)       —— 该维之前的维度乘积
    //   inner = Π shape[dim+1..end)   —— 该维之后的维度乘积
    // 切出的是每个外层块中连续 inner * size 个元素（步长正是该维的 stride）。
    std::size_t outer = 1;
    for (std::size_t i = 0; i < dim; ++i) {
        outer *= shape[i];
    }
    std::size_t inner = 1;
    for (std::size_t i = dim + 1; i < shape.size(); ++i) {
        inner *= shape[i];
    }

    const std::size_t dim_len = shape[dim];
    const std::size_t block = inner * _size;      // 每个外层块中拷贝的元素数
    const std::size_t src_stride = inner * dim_len; // 每个外层块在输入中的跨度
    const std::size_t dst_stride = inner * _size;   // 每个外层块在输出中的跨度

    // 输入梯度与输入同形状，先整体置零 —— 切片是投影而非双射，
    // 未被选中的位置梯度恒为 0，必须显式写出。
    const std::size_t total = input.numel();
    Tensor grad_in(ShapeTag{}, input.sizes(), DType::kFloat, input.device(), false);
    float* dst = grad_in.data_write<float>();
    if (dst == nullptr || total == 0) {
        return ret;
    }
    std::fill(dst, dst + total, 0.0f);

    // 下游梯度可能来自更上游留下的非连续视图，先连续化再按块拷贝。
    // 反向过程中不建图，故直接拷一次是安全且必要的。
    const Tensor grad_out = downStreamGrads[0].is_contiguous() ? downStreamGrads[0]
                                                               : downStreamGrads[0].contiguous();
    const float* src = grad_out.data<float>();
    if (src == nullptr) {
        return ret;
    }

    for (std::size_t o = 0; o < outer; ++o) {
        std::copy(src + o * dst_stride, src + o * dst_stride + block,
                  dst + o * src_stride + _start * inner);
    }

    ret.push_back(GradPack{_upStreamNodes[0], std::vector({grad_in}), 0});
    return ret;
}
