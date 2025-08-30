//
// Created by Beapoe on 25-7-28.
//

module;

import Tensor_dev;
#include <cmath>
#include <stdexcept>
#include <vector>

module functional;

Tensor cos(Tensor x,AutoGrad& ctx){
    Tensor result(ShapeTag{}, x.shape(), x.dtype(), x.device());

    switch (x.dtype()) {
    case DType::kFloat: {
        const float *src = x.data<float>();
        float *dst       = result.data<float>();
        for (size_t i = 0; i < x.numel(); ++i) {
            dst[i] = std::cos(src[i]);
        }
        break;
    }
    case DType::kDouble: {
        const double *src = x.data<double>();
        double *dst       = result.data<double>();
        for (size_t i = 0; i < x.numel(); ++i) {
            dst[i] = std::cos(src[i]);
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for cos");
    }

    // 记录操作到自动微分计算图
    if (x.grad().data<decltype(Dtype2cpp(x.dtype()))>()) {
        ctx.record_op(std::vector<Tensor*>({&result}), op::Cos, {const_cast<Tensor *>(&x)});
    }

    return result;
}

Tensor sin(Tensor x,AutoGrad ctx) {
    Tensor result(ShapeTag{}, x.shape(), x.dtype(), x.device());

    switch (x.dtype()) {
    case DType::kFloat: {
        const float *src = x.data<float>();
        float *dst       = result.data<float>();
        for (size_t i = 0; i < x.numel(); ++i) {
            dst[i] = std::sin(src[i]);
        }
        break;
    }
    case DType::kDouble: {
        const double *src = x.data<double>();
        double *dst       = result.data<double>();
        for (size_t i = 0; i < x.numel(); ++i) {
            dst[i] = std::sin(src[i]);
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for sin");
    }

    // 记录操作到自动微分计算图
    if (x.grad().data<decltype(Dtype2cpp(x.dtype()))>()) {
        ctx.record_op(std::vector<Tensor*>({&result}), op::Sin, {const_cast<Tensor *>(&x)});
    }

    return result;
}

Tensor relu(Tensor x,AutoGrad ctx) {
    Tensor result(ShapeTag{}, x.shape(), x.dtype(), x.device());

    switch (x.dtype()) {
    case DType::kFloat: {
        const float *src = x.data<float>();
        float *dst       = result.data<float>();
        for (size_t i = 0; i < x.numel(); ++i) {
            dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
        }
        break;
    }
    case DType::kDouble: {
        const double *src = x.data<double>();
        double *dst       = result.data<double>();
        for (size_t i = 0; i < x.numel(); ++i) {
            dst[i] = src[i] > 0.0 ? src[i] : 0.0;
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for ReLU");
    }

    // 记录操作到自动微分计算图
    if (x.grad().data<decltype(Dtype2cpp(x.dtype()))>()) {
        ctx.record_op(std::vector<Tensor*>({&result}), op::ReLU, {const_cast<Tensor *>(&x)});
    }

    return result;
}

Tensor sigmoid(Tensor x,AutoGrad ctx){
    Tensor result(ShapeTag{}, x.shape(), x.dtype(), x.device());

    switch (x.dtype()) {
    case DType::kFloat: {
        const float *src = x.data<float>();
        float *dst       = result.data<float>();
        for (size_t i = 0; i < x.numel(); ++i) {
            dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
        }
        break;
    }
    case DType::kDouble: {
        const double *src = x.data<double>();
        double *dst       = result.data<double>();
        for (size_t i = 0; i < x.numel(); ++i) {
            dst[i] = 1.0 / (1.0 + std::exp(-src[i]));
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for sigmoid");
    }

    // 记录操作到自动微分计算图
    if (x.grad().data<decltype(Dtype2cpp(x.dtype()))>()) {
        ctx.record_op(std::vector<Tensor*>({&result}), op::Sigmoid, {const_cast<Tensor *>(&x)});
    }

    return result;
}

Tensor tanh(Tensor x,AutoGrad ctx) {
    Tensor result(ShapeTag{}, x.shape(), x.dtype(), x.device());

    switch (x.dtype()) {
    case DType::kFloat: {
        const float *src = x.data<float>();
        float *dst       = result.data<float>();
        for (size_t i = 0; i < x.numel(); ++i) {
            float exp_2x = std::exp(2 * src[i]);
            dst[i]       = (exp_2x - 1) / (exp_2x + 1);
        }
        break;
    }
    case DType::kDouble: {
        const double *src = x.data<double>();
        double *dst       = result.data<double>();
        for (size_t i = 0; i < x.numel(); ++i) {
            double exp_2x = std::exp(2 * src[i]);
            dst[i]        = (exp_2x - 1) / (exp_2x + 1);
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for tanh");
    }

    // 记录操作到自动微分计算图
    if (x.grad().data<decltype(Dtype2cpp(x.dtype()))>()) {
        ctx.record_op(std::vector<Tensor*>({&result}), op::Tanh, {const_cast<Tensor *>(&x)});
    }

    return result;
}

Tensor softmax(Tensor x,AutoGrad& ctx,int dim) {
    int actual_dim = dim;
    if (actual_dim < 0) {
        actual_dim = x.dim() + actual_dim; // 负值表示从后往前计数
    }

    if (actual_dim < 0 || actual_dim >= static_cast<int>(x.dim())) {
        throw std::runtime_error("Invalid dimension for softmax");
    }

    Tensor result(ShapeTag{}, x.shape(), x.dtype(), x.device());

    switch (x.dtype()) {
    case DType::kFloat: {
        const float *src = x.data<float>();
        float *dst       = result.data<float>();

        // 计算每个切片的softmax
        size_t slice_size = x.shape()[dim];
        size_t num_slices = x.numel() / slice_size;

        for (size_t s = 0; s < num_slices; ++s) {
            // 找到最大值防止数值溢出
            float max_val = src[s * slice_size];
            for (size_t i = 1; i < slice_size; ++i) {
                if (src[s * slice_size + i] > max_val) {
                    max_val = src[s * slice_size + i];
                }
            }

            // 计算指数和
            float exp_sum = 0.0f;
            for (size_t i = 0; i < slice_size; ++i) {
                float val               = std::exp(src[s * slice_size + i] - max_val);
                dst[s * slice_size + i] = val;
                exp_sum += val;
            }

            // 归一化
            for (size_t i = 0; i < slice_size; ++i) {
                dst[s * slice_size + i] /= exp_sum;
            }
        }
        break;
    }
    case DType::kDouble: {
        const double *src = x.data<double>();
        double *dst       = result.data<double>();

        // 计算每个切片的softmax
        size_t slice_size = x.shape()[dim];
        size_t num_slices = x.numel() / slice_size;

        for (size_t s = 0; s < num_slices; ++s) {
            // 找到最大值防止数值溢出
            double max_val = src[s * slice_size];
            for (size_t i = 1; i < slice_size; ++i) {
                if (src[s * slice_size + i] > max_val) {
                    max_val = src[s * slice_size + i];
                }
            }

            // 计算指数和
            double exp_sum = 0.0;
            for (size_t i = 0; i < slice_size; ++i) {
                double val              = std::exp(src[s * slice_size + i] - max_val);
                dst[s * slice_size + i] = val;
                exp_sum += val;
            }

            // 归一化
            for (size_t i = 0; i < slice_size; ++i) {
                dst[s * slice_size + i] /= exp_sum;
            }
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported dtype for softmax");
    }

    // 记录操作到自动微分计算图
    if (x.grad().data<decltype(Dtype2cpp(x.dtype()))>()) {
        ctx.record_op(std::vector<Tensor*>({&result}), op::Softmax, {const_cast<Tensor *>(&x)});
    }

    return result;
}