/**
 * @file MAENode.cpp
 * @author 苏璃珞
 * @brief 平均绝对误差 (MAE) 损失函数节点实现
 * @date 2026/10/13
 */

#include "AutoGrad/Nodes/MAENode.h"
#include "AutoGrad/Nodes/BroadcastUtils.h"
#include "Tensor.h"

using ctorch::autograd::compute_broadcast_reduce_dims;

MAENode::MAENode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,
                 const std::vector<Tensor>& inputs)
    : Node(upStreamNodes, inputs) {}

MAENode::MAENode(std::vector<std::shared_ptr<Node>>&& upStreamNodes,
                 std::vector<Tensor>&& inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

MAENode::MAENode(const std::vector<std::shared_ptr<Node>>& upStreamNodes,
                 const std::vector<Tensor>& inputs,
                 const std::weak_ptr<Tensor>& result)
    : Node(upStreamNodes, inputs, result) {}

MAENode::MAENode(std::vector<std::shared_ptr<Node>>&& upStreamNodes,
                 std::vector<Tensor>&& inputs,
                 const std::weak_ptr<Tensor>& result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> MAENode::backward(const std::vector<Tensor>& downStreamGrads) {
    std::vector<GradPack> ret;
    if (downStreamGrads.empty() || _inputs.size() < 2) {
        return ret;
    }
    const Tensor& grad = downStreamGrads[0];
    const Tensor a_contig = _inputs[0].is_contiguous() ? _inputs[0] : _inputs[0].contiguous();
    const Tensor b_contig = _inputs[1].is_contiguous() ? _inputs[1] : _inputs[1].contiguous();

    size_t count = a_contig.numel();
    float scale = (count > 0) ? (1.0f / static_cast<float>(count)) : 1.0f;

    Tensor sgn(ShapeTag{}, a_contig.shape(), a_contig.dtype(), a_contig.device());
    const float* da = a_contig.data_read<float>();
    const float* db = b_contig.data_read<float>();
    float* ds = sgn.data_write<float>();

    if (da && db && ds) {
        for (size_t i = 0; i < count; ++i) {
            float d = da[i] - db[i];
            float s = (d > 0.0f) ? 1.0f : ((d < 0.0f) ? -1.0f : 0.0f);
            ds[i] = s * scale;
        }
    }

    Tensor base_grad = sgn * grad;

    for (size_t i = 0; i < _inputs.size(); ++i) {
        const Tensor& input = _inputs[i];
        Tensor grad_input = (i == 0) ? base_grad : -base_grad;

        std::vector<int> reduce_dims = compute_broadcast_reduce_dims(input.sizes(), grad_input.sizes());
        if (!reduce_dims.empty()) {
            grad_input = grad_input.sum(reduce_dims);
        }

        if (grad_input.sizes() != input.sizes()) {
            grad_input = grad_input.reshapeNoGrad(input.sizes());
        }

        ret.push_back(GradPack{
            _upStreamNodes[i],
            std::vector({grad_input}),
            static_cast<int>(i)
        });
    }
    return ret;
}
