#include "../../../include/Tensor.h"
#include "../../../include/Ctools.h"
#include "../../../include/CtorchError.h"
#include "../../../include/CoreDefs.h"
#include <cmath>

CT_HOT Tensor Abs_BASIC_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kCPU) [[unlikely]] {
        CtorchError::log(ErrorLevel::ERROR, DeviceTypeToErrorPlatform(a.device()), ErrorType::DEVICE_COMPAT, "CPU-BASIC Abs_Kernel: 仅在CPU支持");
    }

    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
    const float* CT_RESTRICT in = a.data<float>();
    float* CT_RESTRICT out = result.data<float>();

    size_t count = a.numel();
    for (size_t i = 0; i < count; ++i) {
        out[i] = std::abs(in[i]);
    }

    return result;
}
