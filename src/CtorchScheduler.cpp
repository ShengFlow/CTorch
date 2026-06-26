/**
 * @file CtorchScheduler.cpp
 * @brief Ctorch 调度器实现
 * @author GhostFace
 * @date 2026/04/04
 */

#include "./../include/CtorchScheduler.h"

CtorchScheduler::CtorchScheduler() {
    printf(ESC_START COLOR_INFO"[INFO]  " ESC_END "[%s %" PRIu64 "] Ctorch Scheduler Started\n", getFormattedTimeMs().c_str(), getTimestampMs());
    initKernels();
}

void CtorchScheduler::initKernels() {
    auto set_bin = [this](op o, DeviceType d, BinaryKernelFunc f) {
        binary_kernels_[static_cast<size_t>(o)][static_cast<size_t>(d)]
            .store(f, std::memory_order_relaxed);
    };
    auto set_unary = [this](op o, DeviceType d, UnaryKernelFunc f) {
        unary_kernels_[static_cast<size_t>(o)][static_cast<size_t>(d)]
            .store(f, std::memory_order_relaxed);
    };
    auto set_softmax = [this](DeviceType d, Tensor (*f)(const Tensor&, int)) {
        softmax_kernels_[static_cast<size_t>(d)].store(f, std::memory_order_relaxed);
    };

    // CPU kernels
    set_bin(op::Add, DeviceType::kCPU, Add_BASIC_kernel);
    set_bin(op::Sub, DeviceType::kCPU, Sub_BASIC_kernel);
    set_bin(op::Mul, DeviceType::kCPU, Mul_BASIC_kernel);
    set_bin(op::Div, DeviceType::kCPU, Div_BASIC_kernel);
    set_bin(op::MatMul, DeviceType::kCPU, MatMul_BASIC_kernel);
    set_bin(op::Dot, DeviceType::kCPU, Dot_BASIC_kernel);
    set_bin(op::MSE, DeviceType::kCPU, MSE_BASIC_kernel);
    set_bin(op::CE, DeviceType::kCPU, CrossEntropy_BASIC_kernel);
    set_bin(op::MAE, DeviceType::kCPU, MAE_BASIC_kernel);

    set_unary(op::Neg, DeviceType::kCPU, Neg_BASIC_kernel);
    set_unary(op::ReLU, DeviceType::kCPU, ReLU_BASIC_kernel);
    set_unary(op::Cos, DeviceType::kCPU, Cos_BASIC_kernel);
    set_unary(op::Sin, DeviceType::kCPU, Sin_BASIC_kernel);
    set_unary(op::Tanh, DeviceType::kCPU, Tanh_BASIC_kernel);
    set_unary(op::Sigmoid, DeviceType::kCPU, Sigmoid_BASIC_kernel);
    set_unary(op::LReLU, DeviceType::kCPU, nullptr);

    set_softmax(DeviceType::kCPU, Softmax_BASIC_kernel);

    // AMX kernels（目前只有 MatMul 有 AMX 实现）
    set_bin(op::MatMul, DeviceType::kAMX, MatMul_AMX_kernel);
}

BinaryKernelFunc CtorchScheduler::selectBestBinary(
    op op_type, DeviceType dev,
    const std::array<std::array<std::atomic<BinaryKernelFunc>, DEVICE_COUNT>, OP_COUNT>& table)
{
    size_t op_idx = static_cast<size_t>(op_type);

    if (dev == DeviceType::kCPU && isDeviceAvailable(DeviceType::kAMX)) {
        BinaryKernelFunc func = table[op_idx][static_cast<size_t>(DeviceType::kAMX)]
            .load(std::memory_order_acquire);
        if (func != nullptr) return func;
    }

    size_t dev_idx = static_cast<size_t>(dev);
    BinaryKernelFunc func = table[op_idx][dev_idx].load(std::memory_order_acquire);
    if (func != nullptr && isDeviceAvailable(dev)) {
        return func;
    }

    func = table[op_idx][static_cast<size_t>(DeviceType::kCPU)].load(std::memory_order_acquire);
    return func;
}

UnaryKernelFunc CtorchScheduler::selectBestUnary(
    op op_type, DeviceType dev,
    const std::array<std::array<std::atomic<UnaryKernelFunc>, DEVICE_COUNT>, OP_COUNT>& table)
{
    size_t op_idx = static_cast<size_t>(op_type);

    if (dev == DeviceType::kCPU && isDeviceAvailable(DeviceType::kAMX)) {
        UnaryKernelFunc func = table[op_idx][static_cast<size_t>(DeviceType::kAMX)]
            .load(std::memory_order_acquire);
        if (func != nullptr) return func;
    }

    size_t dev_idx = static_cast<size_t>(dev);
    UnaryKernelFunc func = table[op_idx][dev_idx].load(std::memory_order_acquire);
    if (func != nullptr && isDeviceAvailable(dev)) {
        return func;
    }

    func = table[op_idx][static_cast<size_t>(DeviceType::kCPU)].load(std::memory_order_acquire);
    return func;
}

Tensor CtorchScheduler::dispatch(const Tensor& a, const Tensor& b, op op_type) {
    if (a.dtype() != b.dtype()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL,ErrorType::DATATYPE,"Ctorch_Scheduler: Tensor类型不一致");
    }
    if (op_type != op::Add && op_type != op::Mul && op_type != op::Sub && op_type != op::Div && op_type != op::CE && op_type != op::MatMul && a.sizes() != b.sizes()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL,ErrorType::DIMENSION,"Ctorch_Scheduler: Tensor形状不一致");
    }

    DeviceType target_dev = getTargetDevice(a, b);
    BinaryKernelFunc target_kernel = selectBestBinary(op_type, target_dev, binary_kernels_);

    if (target_kernel == nullptr) {
        CtorchError::throwException(ErrorPlatform::kGENERAL,ErrorType::PLATFORM_API,"Ctorch_Scheduler: 没有可用的Kernel");
    }
    return target_kernel(a, b);
}

Tensor CtorchScheduler::dispatch(const Tensor& a, op op_type) {
    DeviceType target_dev = a.device();
    UnaryKernelFunc target_kernel = selectBestUnary(op_type, target_dev, unary_kernels_);

    if (target_kernel == nullptr) {
        CtorchError::throwException(ErrorPlatform::kGENERAL,ErrorType::PLATFORM_API,"Ctorch_Scheduler: 没有可用的Kernel");
    }
    return target_kernel(a);
}

Tensor CtorchScheduler::dispatch_softmax(const Tensor& a, int dim) {
    DeviceType target_dev = a.device();

    if (target_dev == DeviceType::kCPU && isDeviceAvailable(DeviceType::kAMX)) {
        auto amx_kernel = softmax_kernels_[static_cast<size_t>(DeviceType::kAMX)]
            .load(std::memory_order_acquire);
        if (amx_kernel != nullptr) {
            return amx_kernel(a, dim);
        }
    }

    size_t dev_idx = static_cast<size_t>(target_dev);
    auto target_kernel = softmax_kernels_[dev_idx].load(std::memory_order_acquire);

    if (target_kernel == nullptr || !isDeviceAvailable(target_dev)) {
        target_kernel = softmax_kernels_[static_cast<size_t>(DeviceType::kCPU)]
            .load(std::memory_order_acquire);
    }

    if (target_kernel == nullptr) {
        CtorchError::throwException(ErrorPlatform::kGENERAL,ErrorType::PLATFORM_API,"Ctorch_Scheduler: 没有可用的Softmax Kernel");
    }
    return target_kernel(a, dim);
}
