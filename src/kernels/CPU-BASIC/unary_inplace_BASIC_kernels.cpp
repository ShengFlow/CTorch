/**
 * @file unary_inplace_BASIC_kernels.cpp
 * @brief CPU-BASIC 单输入原地算子实现（P1-3）
 * @details 为 relu_/leaky_relu_/gelu_/sigmoid_/tanh_/neg_/sin_/cos_/log_/exp_/abs_ 等
 *          公开 in-place API 提供 CPU fallback。所有实现均要求输入张量位于 CPU。
 */

#include "./../kernels.h"
#include "./../../../include/CtorchError.h"
#include "./../../../include/Tensor.h"
#include "./../../../include/CoreDefs.h"
#include <cmath>
#include <string>

namespace {

inline const char* kErrorPrefix = "CPU-BASIC in-place";

inline void check_cpu(const Tensor& a, const char* name) {
    if (a.device() != DeviceType::kCPU) [[unlikely]] {
        std::string msg = std::string(kErrorPrefix) + " " + name + ": 仅在CPU支持";
        CtorchError::throwException(DeviceTypeToErrorPlatform(a.device()),
                                    ErrorType::DEVICE_COMPAT, msg);
    }
}

template <typename F>
inline void for_each_inplace(Tensor& a, const char* name, F&& func) {
    check_cpu(a, name);
    size_t count = a.numel();
    if (count == 0) return;
    float* data = a.data_write<float>();
    if (!data) return;

    if (a.is_contiguous()) {
        for (size_t i = 0; i < count; ++i) {
            data[i] = func(data[i]);
        }
    } else {
        const auto& shape = a.shape();
        const auto& strides = a.strides();
        std::vector<size_t> coord(shape.size(), 0);
        for (size_t lin = 0; lin < count; ++lin) {
            size_t off = 0;
            for (size_t d = 0; d < coord.size(); ++d) off += coord[d] * strides[d];
            data[off] = func(data[off]);
            for (int d = static_cast<int>(coord.size()) - 1; d >= 0; --d) {
                if (++coord[d] < shape[d]) break;
                coord[d] = 0;
            }
        }
    }
}

} // namespace

void Neg_BASIC_inplace(Tensor& a) {
    for_each_inplace(a, "Neg", [](float x) { return -x; });
}

void Cos_BASIC_inplace(Tensor& a) {
    for_each_inplace(a, "Cos", [](float x) { return std::cos(x); });
}

void Sin_BASIC_inplace(Tensor& a) {
    for_each_inplace(a, "Sin", [](float x) { return std::sin(x); });
}

void ReLU_BASIC_inplace(Tensor& a) {
    for_each_inplace(a, "ReLU", [](float x) { return std::max(0.0f, x); });
}

void Tanh_BASIC_inplace(Tensor& a) {
    for_each_inplace(a, "Tanh", [](float x) { return std::tanh(x); });
}

void Sigmoid_BASIC_inplace(Tensor& a) {
    for_each_inplace(a, "Sigmoid", [](float x) { return 1.0f / (1.0f + std::exp(-x)); });
}

void GELU_BASIC_inplace(Tensor& a) {
    for_each_inplace(a, "GELU", [](float x) {
        float v = 0.7978845608f * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1.0f + std::tanh(v));
    });
}

// [PEL25 Stage 3.1] SiLU inplace: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
void SiLU_BASIC_inplace(Tensor& a) {
    for_each_inplace(a, "SiLU", [](float x) { return x / (1.0f + std::exp(-x)); });
}

void LReLU_BASIC_inplace(Tensor& a) {
    for_each_inplace(a, "LReLU", [](float x) { return x > 0.0f ? x : x * 0.01f; });
}

void Log_BASIC_inplace(Tensor& a) {
    for_each_inplace(a, "Log", [](float x) { return std::log(x); });
}

void Exp_BASIC_inplace(Tensor& a) {
    for_each_inplace(a, "Exp", [](float x) { return std::exp(x); });
}

void Abs_BASIC_inplace(Tensor& a) {
    for_each_inplace(a, "Abs", [](float x) { return std::fabs(x); });
}
