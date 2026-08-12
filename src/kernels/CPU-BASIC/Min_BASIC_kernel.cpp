#include "../../../include/Tensor.h"
#include "../../../include/Ctools.h"
#include "../../../include/CtorchError.h"
#include "../../../include/CoreDefs.h"
#include <algorithm>

CT_HOT Tensor Min_BASIC_kernel(const Tensor& a, const Tensor& b) {
    if (a.device() != DeviceType::kCPU || b.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT, "CPU-BASIC Min_Kernel: 仅在CPU支持");
    }

    if (a.dtype() != b.dtype()) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "CPU-BASIC Min_Kernel: Tensor数据类型不匹配");
    }

    if (a.sizes() != b.sizes()) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DIMENSION, "CPU-BASIC Min_Kernel: Tensor形状不一致");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device(), false);
    const float* CT_RESTRICT a_data = a.data_read<float>();
    const float* CT_RESTRICT b_data = b.data_read<float>();
    float* CT_RESTRICT result_data = result.data_write<float>();

    size_t count = a.numel();
    for (size_t i = 0; i < count; ++i) {
        result_data[i] = std::min(a_data[i], b_data[i]);
    }

    return result;
}
