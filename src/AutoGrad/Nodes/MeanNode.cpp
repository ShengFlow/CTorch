/**
 * @file MeanNode.cpp
 * @brief 全 reduce mean 节点实现
 * @date 2026/09/06
 **/

#include "AutoGrad/Nodes/MeanNode.h"
#include "Tensor.h"
#include "Ctools.h"
#include "kernels/CPU-SIMD/ReduceSIMD.h"

using ctorch::kernels::simd::fill_f32;

MeanNode::MeanNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs)
    : Node(upStreamNodes, inputs) {}

MeanNode::MeanNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

MeanNode::MeanNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
                   const std::weak_ptr<Tensor>& result)
    : Node(upStreamNodes, inputs, result) {}

MeanNode::MeanNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
                   const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> MeanNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;
    if (downStreamGrads.empty() || _inputs.empty()) {
        return ret;
    }
    const Tensor& input = _inputs[0];
    const float* gdata = downStreamGrads[0].data_read<float>();
    if (!gdata || input.numel() == 0) {
        return ret;
    }
    float g = gdata[0] / static_cast<float>(input.numel());  // dL/dc / n

    // 全 reduce 反向: 把 g/n 广播(填满)到输入形状, NEON 4-wide 填充
    Tensor grad_input(ShapeTag{}, input.sizes(), input.dtype(), input.device());
    fill_f32(grad_input.data_write<float>(), input.numel(), g);

    ret.push_back(GradPack{_upStreamNodes[0], std::vector({grad_input}), 0});
    return ret;
}
