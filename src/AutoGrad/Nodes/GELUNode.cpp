#include "AutoGrad/Nodes/GELUNode.h"
#include "../../../include/Tensor.h"
#include "../../../src/kernels/kernels.h"
#include <cmath>

namespace {
constexpr float kSqrt2OverPi    = 0.7978845608f;
constexpr float kGeluCoeff      = 0.044715f;
constexpr float kGeluCoeffDeriv = 0.134145f; // 3 * 0.044715

inline float gelu_derivative_scalar(float x) {
    float v      = kSqrt2OverPi * (x + kGeluCoeff * x * x * x);
    float tanh_v = std::tanh(v);
    float term1  = 0.5f * (1.0f + tanh_v);
    float term2 =
        0.5f * x * (1.0f - tanh_v * tanh_v) * kSqrt2OverPi * (1.0f + kGeluCoeffDeriv * x * x);
    return term1 + term2;
}
} // namespace

GELUNode::GELUNode(const std::vector<std::shared_ptr<Node>> &upStreamNodes,
                   const std::vector<Tensor> &inputs)
    : Node(upStreamNodes, inputs) {}

GELUNode::GELUNode(std::vector<std::shared_ptr<Node>> &&upStreamNodes, std::vector<Tensor> &&inputs)
    : Node(std::move(upStreamNodes), std::move(inputs)) {}

GELUNode::GELUNode(const std::vector<std::shared_ptr<Node>> &upStreamNodes,
                   const std::vector<Tensor> &inputs, const std::weak_ptr<Tensor> &result)
    : Node(upStreamNodes, inputs, result) {}

GELUNode::GELUNode(std::vector<std::shared_ptr<Node>> &&upStreamNodes, std::vector<Tensor> &&inputs,
                   const std::weak_ptr<Tensor> &result)
    : Node(std::move(upStreamNodes), std::move(inputs), result) {}

std::vector<GradPack> GELUNode::backward(const std::vector<Tensor> &downStreamGrads) {
    std::vector<GradPack> ret;

    if (_inputs.empty() || downStreamGrads.empty()) {
        return ret;
    }

    const Tensor &x        = _inputs[0];
    const Tensor &grad_out = downStreamGrads[0];

    // (SYNC) MPS 路径：确保 accumulator 中所有元素级 kernel 写回完成后再读取 buffer
#ifdef __APPLE__
    if (x.device() == DeviceType::kMPS) {
        MPS_flush_wait(true);
    }
#endif

    Tensor grad_x(ShapeTag{}, x.sizes(), x.dtype(), x.device());

    size_t count               = x.numel();
    const float *x_data        = x.data_read<float>();
    const float *grad_out_data = grad_out.data_read<float>();
    float *grad_x_data         = grad_x.data_write<float>();

    for (size_t i = 0; i < count; ++i) {
        grad_x_data[i] = grad_out_data[i] * gelu_derivative_scalar(x_data[i]);
    }

    // (SYNC) CPU 写入 MPS buffer 后通知 Metal，确保后续 GPU kernel 读取到最新数据
    if (grad_x.device() == DeviceType::kMPS) {
        MPS_markBufferModified(static_cast<void *>(grad_x_data), count * sizeof(float));
    }

    ret.push_back(GradPack{_upStreamNodes[0], std::vector({grad_x}), 0});

    return ret;
}
