/**
 * @file SliceNode.cpp
 * @author 苏璃珞
 * @brief 切片节点实现
 * @date 2026/9/17
 **/

#include "AutoGrad/Nodes/SliceNode.h"

#include <algorithm>

SliceNode::SliceNode(std::size_t start, std::size_t size,
                     std::vector<std::shared_ptr<Node>>&& upStreamNodes,
                     std::vector<Tensor>&& inputs,
                     const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result),
      _start(start),
      _size(size) {}

std::vector<GradPack> SliceNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;
    if (downStreamGrads.empty() || _upStreamNodes.empty() || _inputs.empty()) {
        return ret;
    }

    const Tensor& input = _inputs[0];
    const auto& in_shape = input.sizes();
    if (in_shape.empty()) {
        return ret;
    }

    // 沿第 0 维的「行」为单位散射：每行含 numel / shape[0] 个元素。
    // 切片只改变第 0 维的长度，行内布局不变，因此可以按整行连续拷贝。
    const std::size_t rows = in_shape[0];
    const std::size_t total = input.numel();
    const std::size_t row_elems = (rows > 0) ? (total / rows) : 0;
    const std::size_t copy_elems = _size * row_elems;

    // 输入梯度与输入同形状，先整体置零 —— 切片是投影而非双射，
    // 未被选中的位置梯度恒为 0，必须显式写出。
    Tensor grad_in(ShapeTag{}, input.sizes(), DType::kFloat, input.device(), false);
    float* dst = grad_in.data_write<float>();
    if (dst == nullptr || total == 0) {
        return ret;
    }
    std::fill(dst, dst + total, 0.0f);

    // 下游梯度可能来自更上游的切片而呈非连续视图，先连续化再按行拷贝。
    // 这里处于反向过程中，不建图，故直接做一次拷贝是安全且必要的。
    const Tensor grad_out =
        downStreamGrads[0].is_contiguous() ? downStreamGrads[0] : downStreamGrads[0].contiguous();
    const float* src = grad_out.data<float>();
    if (src == nullptr) {
        return ret;
    }

    // 边界保护：避免越界写（正常路径下调用方已校验 start + size <= shape[0]）
    if (_start + _size > rows || copy_elems > total) {
        return ret;
    }
    if (copy_elems > 0) {
        std::copy(src, src + copy_elems, dst + _start * row_elems);
    }

    ret.push_back(GradPack{_upStreamNodes[0], std::vector({grad_in}), 0});
    return ret;
}
