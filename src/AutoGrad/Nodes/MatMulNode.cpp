/**
*@file MatMulNode.cpp
 *@author Beapoe
 *@brief 矩阵乘法节点实现
 *@date 2026/4/5
 **/

#include "AutoGrad/Nodes/MatMulNode.h"
#include "Ctools.h"
#include "CtorchError.h"
#include "Tensor.h"

MatMulNode::MatMulNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs) 
    : Node(upStreamNodes, inputs) {set_requireAccelerate(true);}

MatMulNode::MatMulNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {set_requireAccelerate(true);}

MatMulNode::MatMulNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,const std::vector<Tensor>& inputs,const std::weak_ptr<Tensor>& result) 
    : Node(upStreamNodes, inputs, result) {set_requireAccelerate(true);}

MatMulNode::MatMulNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs, const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {set_requireAccelerate(true);}

std::vector<GradPack> MatMulNode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;

    // 对于矩阵乘法，导数是：
    // 对于第一个输入，导数是 downStreamGrads[0] * other^T
    // 对于第二个输入，导数是 this^T * downStreamGrads[0]

    // 检查输入数量
    if (_inputs.size() != 2) {
        CtorchError::error(ErrorPlatform::kAutoDiff, ErrorType::UNKNOWN, "MatMulNode: 输入数量错误");
        return ret;
    }
    if (downStreamGrads.empty()) {
        return ret;
    }

    const Tensor& A = _inputs[0]; // 第一个输入
    const Tensor& B = _inputs[1]; // 第二个输入
    Tensor grad = downStreamGrads[0]; // 下游梯度

    // 若 grad 是标量（root 的初始 grad），先广播到与输出同形再继续 matmul。
    // 这避免了 0D 张量进入 MatMul kernel（kernel 仅支持 ≥2D）。
    if (grad.dim() == 0) {
        const auto& result_shape = getResultShape();
        if (!result_shape.empty()) {
            Tensor broadcasted(ShapeTag{}, result_shape, grad.dtype(), grad.device());
            const float scalar = grad.item<float>();
            const size_t total = broadcasted.numel();
            float* p = broadcasted.data_write<float>();
            for (size_t i = 0; i < total; ++i) p[i] = scalar;
            grad = broadcasted;
        }
    }

    // 计算第一个输入的梯度：grad * B^T
    Tensor grad_A = grad.matmul(B.transpose(0, 1));

    // 计算第二个输入的梯度：A^T * grad
    Tensor grad_B = A.transpose(0, 1).matmul(grad);

    // 添加到返回值
    ret.push_back(GradPack{
        _upStreamNodes[0],
        std::vector({grad_A}),
        0
    });

    ret.push_back(GradPack{
        _upStreamNodes[1],
        std::vector({grad_B}),
        1
    });

    return ret;
}