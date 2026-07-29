/**
 * @file CrossEntropy_SIMD_kernel.cpp
 * @brief CPU-SIMD 交叉熵算子
 */

#include "../kernels.h"
#include "../../../include/CtorchError.h"
#include "../../../include/Tensor.h"
#include "../../../include/CoreDefs.h"
#include <cmath>
#include <algorithm>

CT_HOT Tensor CrossEntropy_SIMD_kernel(const Tensor& a, const Tensor& b) {
    if (a.device() != DeviceType::kCPU || b.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()),
                         ErrorType::DEVICE_COMPAT,
                         "CPU-SIMD CrossEntropy_Kernel: 仅在CPU支持");
    }
    if (a.dtype() != b.dtype()) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DATATYPE,
                         "CPU-SIMD CrossEntropy_Kernel: 张量数据类型不一致");
    }

    const float* CT_RESTRICT data_a = a.data<float>();
    const float* CT_RESTRICT data_b = b.data<float>();
    float cross_entropy = 0.0f;

    if (a.sizes() == b.sizes()) {
        if (a.sizes().size() == 2) {
            size_t batch_size = a.sizes()[0];
            size_t num_classes = a.sizes()[1];

            #pragma omp parallel for reduction(+:cross_entropy)
            for (size_t i = 0; i < batch_size; ++i) {
                float max_val = data_a[i * num_classes];
                for (size_t j = 1; j < num_classes; ++j) {
                    max_val = std::max(max_val, data_a[i * num_classes + j]);
                }

                float exp_sum = 0.0f;
                #pragma omp simd reduction(+:exp_sum)
                for (size_t j = 0; j < num_classes; ++j) {
                    exp_sum += std::exp(data_a[i * num_classes + j] - max_val);
                }

                float inv_exp_sum = 1.0f / exp_sum;
                #pragma omp simd reduction(+:cross_entropy)
                for (size_t j = 0; j < num_classes; ++j) {
                    float pred = std::exp(data_a[i * num_classes + j] - max_val) * inv_exp_sum;
                    pred = std::max(pred, 1e-10f);
                    cross_entropy -= data_b[i * num_classes + j] * std::log(pred);
                }
            }
        } else if (a.sizes().size() == 1) {
            size_t num_classes = a.sizes()[0];
            float max_val = data_a[0];
            for (size_t j = 1; j < num_classes; ++j) {
                max_val = std::max(max_val, data_a[j]);
            }
            float exp_sum = 0.0f;
            #pragma omp simd reduction(+:exp_sum)
            for (size_t j = 0; j < num_classes; ++j) {
                exp_sum += std::exp(data_a[j] - max_val);
            }
            float inv_exp_sum = 1.0f / exp_sum;
            #pragma omp simd reduction(+:cross_entropy)
            for (size_t j = 0; j < num_classes; ++j) {
                float pred = std::exp(data_a[j] - max_val) * inv_exp_sum;
                pred = std::max(pred, 1e-10f);
                cross_entropy -= data_b[j] * std::log(pred);
            }
        } else {
            CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                             "CPU-SIMD CrossEntropy_Kernel: one-hot 仅支持 1D/2D");
            return Tensor();
        }
    } else if (b.sizes().size() == 1 && a.sizes().size() == 2 && b.sizes()[0] == a.sizes()[0]) {
        size_t batch_size = b.sizes()[0];
        size_t num_classes = a.sizes()[1];

        #pragma omp parallel for reduction(+:cross_entropy)
        for (size_t i = 0; i < batch_size; ++i) {
            int class_idx = static_cast<int>(data_b[i]);
            if (class_idx < 0 || class_idx >= static_cast<int>(num_classes)) {
                CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::TENSOR_STATE,
                                 "CPU-SIMD CrossEntropy_Kernel: 类别索引超出范围");
                continue;
            }
            float max_val = data_a[i * num_classes];
            for (size_t j = 1; j < num_classes; ++j) {
                max_val = std::max(max_val, data_a[i * num_classes + j]);
            }
            float exp_sum = 0.0f;
            #pragma omp simd reduction(+:exp_sum)
            for (size_t j = 0; j < num_classes; ++j) {
                exp_sum += std::exp(data_a[i * num_classes + j] - max_val);
            }
            float pred = std::exp(data_a[i * num_classes + class_idx] - max_val) / exp_sum;
            pred = std::max(pred, 1e-10f);
            cross_entropy -= std::log(pred);
        }
    } else {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                         "CPU-SIMD CrossEntropy_Kernel: 张量形状不兼容");
        return Tensor();
    }

    size_t batch_size = (a.sizes().size() == 2) ? a.sizes()[0] : 1;
    float avg_loss = cross_entropy / static_cast<float>(batch_size);
    return Tensor(avg_loss);
}
