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

} // namespace

void Neg_BASIC_inplace(Tensor& a) {
    check_cpu(a, "Neg");
    size_t count = a.numel();
    float* data = a.data<float>();
    for (size_t i = 0; i < count; ++i) {
        data[i] = -data[i];
    }
}

void Cos_BASIC_inplace(Tensor& a) {
    check_cpu(a, "Cos");
    size_t count = a.numel();
    float* data = a.data<float>();
    for (size_t i = 0; i < count; ++i) {
        data[i] = std::cos(data[i]);
    }
}

void Sin_BASIC_inplace(Tensor& a) {
    check_cpu(a, "Sin");
    size_t count = a.numel();
    float* data = a.data<float>();
    for (size_t i = 0; i < count; ++i) {
        data[i] = std::sin(data[i]);
    }
}

void ReLU_BASIC_inplace(Tensor& a) {
    check_cpu(a, "ReLU");
    size_t count = a.numel();
    float* data = a.data<float>();
    for (size_t i = 0; i < count; ++i) {
        data[i] = std::max(0.0f, data[i]);
    }
}

void Tanh_BASIC_inplace(Tensor& a) {
    check_cpu(a, "Tanh");
    size_t count = a.numel();
    float* data = a.data<float>();
    for (size_t i = 0; i < count; ++i) {
        data[i] = std::tanh(data[i]);
    }
}

void Sigmoid_BASIC_inplace(Tensor& a) {
    check_cpu(a, "Sigmoid");
    size_t count = a.numel();
    float* data = a.data<float>();
    for (size_t i = 0; i < count; ++i) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
}

void GELU_BASIC_inplace(Tensor& a) {
    check_cpu(a, "GELU");
    size_t count = a.numel();
    float* data = a.data<float>();
    for (size_t i = 0; i < count; ++i) {
        float x = data[i];
        float v = 0.7978845608f * (x + 0.044715f * x * x * x);
        data[i] = 0.5f * x * (1.0f + std::tanh(v));
    }
}

void LReLU_BASIC_inplace(Tensor& a) {
    check_cpu(a, "LReLU");
    size_t count = a.numel();
    float* data = a.data<float>();
    for (size_t i = 0; i < count; ++i) {
        data[i] = data[i] > 0.0f ? data[i] : data[i] * 0.01f;
    }
}

void Log_BASIC_inplace(Tensor& a) {
    check_cpu(a, "Log");
    size_t count = a.numel();
    float* data = a.data<float>();
    for (size_t i = 0; i < count; ++i) {
        data[i] = std::log(data[i]);
    }
}

void Exp_BASIC_inplace(Tensor& a) {
    check_cpu(a, "Exp");
    size_t count = a.numel();
    float* data = a.data<float>();
    for (size_t i = 0; i < count; ++i) {
        data[i] = std::exp(data[i]);
    }
}

void Abs_BASIC_inplace(Tensor& a) {
    check_cpu(a, "Abs");
    size_t count = a.numel();
    float* data = a.data<float>();
    for (size_t i = 0; i < count; ++i) {
        data[i] = std::fabs(data[i]);
    }
}
