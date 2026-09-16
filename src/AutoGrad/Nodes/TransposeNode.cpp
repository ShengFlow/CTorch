/**
 * @file TransposeNode.cpp
 * @author 苏璃珞
 * @brief 转置节点实现
 * @date 2026/9/16
 **/

#include "AutoGrad/Nodes/TransposeNode.h"

TransposeNode::TransposeNode(int dim0, int dim1,
                             std::vector<std::shared_ptr<Node>>&& upStreamNodes,
                             std::vector<Tensor>&& inputs,
                             const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result), _dim0(dim0), _dim1(dim1) {}

std::vector<GradPack> TransposeNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;
    if (downStreamGrads.empty() || _upStreamNodes.empty()) {
        return ret;
    }

    // 转置自逆：对下游梯度沿同样的两个维度再转置一次即得输入梯度。
    //
    // 这里必须用 transposeNoGrad（纯元数据视图，不注册节点）——若在反向过程中
    // 走 transpose() 会再次建图，导致每次 backward 都往图上接一段，图逐轮膨胀。
    Tensor grad = downStreamGrads[0].transposeNoGrad(_dim0, _dim1);

    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({grad}),
        0
    });
    return ret;
}
