/**
 * @file CtorchScheduler.cpp
 * @brief Ctorch 调度器实现
 * @author GhostFace
 * @date 2026/04/04
 */

#include "./../include/CtorchScheduler.h"
#include "./../include/DeviceAllocator.h"

#ifdef __APPLE__
// 前向声明 Metal 设备创建函数，避免在 .cpp 中直接 #import <Metal/Metal.h>
// 该函数返回 id<MTLDevice>，在 C++ 侧以 void* 比较是否非空即可。
extern "C" void* MTLCreateSystemDefaultDevice(void);
#else
#include <cpuid.h>
#endif

bool CtorchScheduler::isDeviceAvailable(DeviceType dev_type) {
    switch (dev_type) {
        case DeviceType::kCPU:
            return true;
        case DeviceType::kCUDA:
            return false;
        case DeviceType::kMPS:
#ifdef __APPLE__
            return MTLCreateSystemDefaultDevice() != nullptr;
#else
            return false;
#endif
        case DeviceType::kAMX:
#ifdef __APPLE__
            // Apple Silicon 的 AMX 通过 Accelerate 框架抽象暴露，不直接对应 x86 AMX 指令集；
            // 当前 MatMul AMX kernel 在 macOS 上实际调用的是 Accelerate/BLAS，因此标记为可用。
            return true;
#else
            // x86_64 检测 AMX_TILE (CPUID leaf 7, subleaf 0, EDX bit 24)
            unsigned int eax, ebx, ecx, edx;
            if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
                return (edx & (1u << 24)) != 0;
            }
            return false;
#endif
        case DeviceType::kSIMD:
            // SIMD 路径依赖编译器 auto-vectorization，默认在 CPU 上可用。
            return true;
        default:
            return false;
    }
}

CtorchScheduler::CtorchScheduler() {
    printf(ESC_START COLOR_INFO"[INFO]  " ESC_END "[%s %" PRIu64 "] Ctorch Scheduler Started\n", getFormattedTimeMs().c_str(), getTimestampMs());
    
    AllocatorManager::getInstance().registerAllocator(
        DeviceType::kCPU, std::make_unique<CPUAllocator>());
    AllocatorManager::getInstance().registerAllocator(
        DeviceType::kMPS, std::make_unique<MPSAllocator>());
    
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
    auto set_unary_inplace = [this](op o, DeviceType d, UnaryInplaceKernelFunc f) {
        unary_inplace_kernels_[static_cast<size_t>(o)][static_cast<size_t>(d)]
            .store(f, std::memory_order_relaxed);
    };
    auto set_softmax = [this](DeviceType d, Tensor (*f)(const Tensor&, int)) {
        softmax_kernels_[static_cast<size_t>(d)].store(f, std::memory_order_relaxed);
    };

    // CPU kernels (BASIC 作为 fallback)
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
    set_unary(op::GELU, DeviceType::kCPU, GELU_BASIC_kernel);
    set_unary(op::LReLU, DeviceType::kCPU, LReLU_BASIC_kernel);
    set_unary(op::Log, DeviceType::kCPU, Log_BASIC_kernel);
    set_unary(op::Exp, DeviceType::kCPU, Exp_BASIC_kernel);
    set_unary(op::Abs, DeviceType::kCPU, Abs_BASIC_kernel);

    // CPU in-place unary kernels (BASIC fallback)
    set_unary_inplace(op::Neg, DeviceType::kCPU, Neg_BASIC_inplace);
    set_unary_inplace(op::Cos, DeviceType::kCPU, Cos_BASIC_inplace);
    set_unary_inplace(op::Sin, DeviceType::kCPU, Sin_BASIC_inplace);
    set_unary_inplace(op::ReLU, DeviceType::kCPU, ReLU_BASIC_inplace);
    set_unary_inplace(op::Tanh, DeviceType::kCPU, Tanh_BASIC_inplace);
    set_unary_inplace(op::Sigmoid, DeviceType::kCPU, Sigmoid_BASIC_inplace);
    set_unary_inplace(op::GELU, DeviceType::kCPU, GELU_BASIC_inplace);
    set_unary_inplace(op::LReLU, DeviceType::kCPU, LReLU_BASIC_inplace);
    set_unary_inplace(op::Log, DeviceType::kCPU, Log_BASIC_inplace);
    set_unary_inplace(op::Exp, DeviceType::kCPU, Exp_BASIC_inplace);
    set_unary_inplace(op::Abs, DeviceType::kCPU, Abs_BASIC_inplace);

    set_bin(op::Min, DeviceType::kCPU, Min_BASIC_kernel);
    set_bin(op::Max, DeviceType::kCPU, Max_BASIC_kernel);

    set_softmax(DeviceType::kCPU, Softmax_BASIC_kernel);

    // SIMD kernels (优先于 BASIC)
    set_bin(op::Add, DeviceType::kSIMD, Add_SIMD_kernel);
    set_bin(op::Sub, DeviceType::kSIMD, Sub_SIMD_kernel);
    set_bin(op::Mul, DeviceType::kSIMD, Mul_SIMD_kernel);
    set_bin(op::Div, DeviceType::kSIMD, Div_SIMD_kernel);

    set_unary(op::Neg, DeviceType::kSIMD, Neg_SIMD_kernel);
    set_unary(op::ReLU, DeviceType::kSIMD, ReLU_SIMD_kernel);
    set_unary(op::Tanh, DeviceType::kSIMD, Tanh_SIMD_kernel);
    set_unary(op::Sigmoid, DeviceType::kSIMD, Sigmoid_SIMD_kernel);
    set_unary(op::GELU, DeviceType::kSIMD, GELU_SIMD_kernel);
    set_unary(op::Log, DeviceType::kSIMD, Log_SIMD_kernel);
    set_unary(op::Exp, DeviceType::kSIMD, Exp_SIMD_kernel);
    set_unary(op::Abs, DeviceType::kSIMD, Abs_SIMD_kernel);

    set_bin(op::Min, DeviceType::kSIMD, Min_SIMD_kernel);
    set_bin(op::Max, DeviceType::kSIMD, Max_SIMD_kernel);
    set_bin(op::MatMul, DeviceType::kSIMD, MatMul_SIMD_kernel);
    set_bin(op::Dot, DeviceType::kSIMD, Dot_SIMD_kernel);
    set_bin(op::MSE, DeviceType::kSIMD, MSE_SIMD_kernel);
    set_bin(op::CE, DeviceType::kSIMD, CrossEntropy_SIMD_kernel);
    set_bin(op::MAE, DeviceType::kSIMD, MAE_SIMD_kernel);

    set_unary(op::Sin, DeviceType::kSIMD, Sin_SIMD_kernel);
    set_unary(op::Cos, DeviceType::kSIMD, Cos_SIMD_kernel);
    set_unary(op::LReLU, DeviceType::kSIMD, LReLU_SIMD_kernel);

    set_softmax(DeviceType::kSIMD, Softmax_SIMD_kernel);

    // AMX kernels（目前只有 MatMul 有 AMX 实现）
    set_bin(op::MatMul, DeviceType::kAMX, MatMul_AMX_kernel);
    set_unary(op::GELU, DeviceType::kAMX, GELU_AMX_kernel);

    // MPS kernels
    set_bin(op::Add, DeviceType::kMPS, Add_MPS_kernel);
    set_bin(op::Sub, DeviceType::kMPS, Sub_MPS_kernel);
    set_bin(op::Mul, DeviceType::kMPS, Mul_MPS_kernel);
    set_bin(op::Div, DeviceType::kMPS, Div_MPS_kernel);
    set_bin(op::MatMul, DeviceType::kMPS, MatMul_MPS_kernel);
    set_bin(op::Dot, DeviceType::kMPS, Dot_MPS_kernel);
    set_bin(op::MSE, DeviceType::kMPS, MSE_MPS_kernel);
    set_bin(op::CE, DeviceType::kMPS, CrossEntropy_MPS_kernel);
    set_bin(op::MAE, DeviceType::kMPS, MAE_MPS_kernel);
    set_bin(op::Min, DeviceType::kMPS, Min_MPS_kernel);
    set_bin(op::Max, DeviceType::kMPS, Max_MPS_kernel);

    set_unary(op::Neg, DeviceType::kMPS, Neg_MPS_kernel);
    set_unary(op::ReLU, DeviceType::kMPS, ReLU_MPS_kernel);
    set_unary(op::Cos, DeviceType::kMPS, Cos_MPS_kernel);
    set_unary(op::Sin, DeviceType::kMPS, Sin_MPS_kernel);
    set_unary(op::Tanh, DeviceType::kMPS, Tanh_MPS_kernel);
    set_unary(op::Sigmoid, DeviceType::kMPS, Sigmoid_MPS_kernel);
    set_unary(op::GELU, DeviceType::kMPS, GELU_MPS_kernel);
    set_unary(op::LReLU, DeviceType::kMPS, LReLU_MPS_kernel);
    set_unary(op::Log, DeviceType::kMPS, Log_MPS_kernel);
    set_unary(op::Exp, DeviceType::kMPS, Exp_MPS_kernel);
    set_unary(op::Abs, DeviceType::kMPS, Abs_MPS_kernel);

    // MPS in-place unary kernels (P1-3)
    set_unary_inplace(op::Neg, DeviceType::kMPS, Neg_MPS_inplace);
    set_unary_inplace(op::Cos, DeviceType::kMPS, Cos_MPS_inplace);
    set_unary_inplace(op::Sin, DeviceType::kMPS, Sin_MPS_inplace);
    set_unary_inplace(op::ReLU, DeviceType::kMPS, ReLU_MPS_inplace);
    set_unary_inplace(op::Tanh, DeviceType::kMPS, Tanh_MPS_inplace);
    set_unary_inplace(op::Sigmoid, DeviceType::kMPS, Sigmoid_MPS_inplace);
    set_unary_inplace(op::GELU, DeviceType::kMPS, GELU_MPS_inplace);
    set_unary_inplace(op::LReLU, DeviceType::kMPS, LReLU_MPS_inplace);
    set_unary_inplace(op::Log, DeviceType::kMPS, Log_MPS_inplace);
    set_unary_inplace(op::Exp, DeviceType::kMPS, Exp_MPS_inplace);
    set_unary_inplace(op::Abs, DeviceType::kMPS, Abs_MPS_inplace);

    set_softmax(DeviceType::kMPS, Softmax_MPS_kernel);
}

BinaryKernelFunc CtorchScheduler::selectBestBinary(
    op op_type, DeviceType dev,
    const std::array<std::array<std::atomic<BinaryKernelFunc>, DEVICE_COUNT>, OP_COUNT>& table)
{
    size_t op_idx = static_cast<size_t>(op_type);

    // MPS 张量必须走 MPS kernel
    if (dev == DeviceType::kMPS) {
        BinaryKernelFunc func = table[op_idx][static_cast<size_t>(DeviceType::kMPS)]
            .load(std::memory_order_acquire);
        if (func != nullptr && isDeviceAvailable(DeviceType::kMPS)) {
            return func;
        }
        return nullptr;
    }

    // CPU/AMX/SIMD 张量：优先 AMX -> SIMD -> BASIC
    if (isDeviceAvailable(DeviceType::kAMX)) {
        BinaryKernelFunc func = table[op_idx][static_cast<size_t>(DeviceType::kAMX)]
            .load(std::memory_order_acquire);
        if (func != nullptr) return func;
    }

    BinaryKernelFunc func = table[op_idx][static_cast<size_t>(DeviceType::kSIMD)]
        .load(std::memory_order_acquire);
    if (func != nullptr) return func;

    func = table[op_idx][static_cast<size_t>(DeviceType::kCPU)].load(std::memory_order_acquire);
    return func;
}

UnaryKernelFunc CtorchScheduler::selectBestUnary(
    op op_type, DeviceType dev,
    const std::array<std::array<std::atomic<UnaryKernelFunc>, DEVICE_COUNT>, OP_COUNT>& table)
{
    size_t op_idx = static_cast<size_t>(op_type);

    // MPS 张量必须走 MPS kernel
    if (dev == DeviceType::kMPS) {
        UnaryKernelFunc func = table[op_idx][static_cast<size_t>(DeviceType::kMPS)]
            .load(std::memory_order_acquire);
        if (func != nullptr && isDeviceAvailable(DeviceType::kMPS)) {
            return func;
        }
        return nullptr;
    }

    // CPU/AMX/SIMD 张量：优先 AMX -> SIMD -> BASIC
    if (isDeviceAvailable(DeviceType::kAMX)) {
        UnaryKernelFunc func = table[op_idx][static_cast<size_t>(DeviceType::kAMX)]
            .load(std::memory_order_acquire);
        if (func != nullptr) return func;
    }

    UnaryKernelFunc func = table[op_idx][static_cast<size_t>(DeviceType::kSIMD)]
        .load(std::memory_order_acquire);
    if (func != nullptr) return func;

    func = table[op_idx][static_cast<size_t>(DeviceType::kCPU)].load(std::memory_order_acquire);
    return func;
}

UnaryInplaceKernelFunc CtorchScheduler::selectBestUnaryInplace(
    op op_type, DeviceType dev,
    const std::array<std::array<std::atomic<UnaryInplaceKernelFunc>, DEVICE_COUNT>, OP_COUNT>& table)
{
    size_t op_idx = static_cast<size_t>(op_type);

    // MPS 张量必须走 MPS kernel
    if (dev == DeviceType::kMPS) {
        UnaryInplaceKernelFunc func = table[op_idx][static_cast<size_t>(DeviceType::kMPS)]
            .load(std::memory_order_acquire);
        if (func != nullptr && isDeviceAvailable(DeviceType::kMPS)) {
            return func;
        }
        return nullptr;
    }

    // CPU/AMX/SIMD 张量：优先 AMX -> SIMD -> BASIC；当前仅 BASIC 实现 in-place，
    // 因此实际会回退到 CPU BASIC。未来可按算子补充 AMX/SIMD in-place 实现。
    if (isDeviceAvailable(DeviceType::kAMX)) {
        UnaryInplaceKernelFunc func = table[op_idx][static_cast<size_t>(DeviceType::kAMX)]
            .load(std::memory_order_acquire);
        if (func != nullptr) return func;
    }

    UnaryInplaceKernelFunc func = table[op_idx][static_cast<size_t>(DeviceType::kSIMD)]
        .load(std::memory_order_acquire);
    if (func != nullptr) return func;

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

void CtorchScheduler::dispatch_inplace(Tensor& a, op op_type) {
    if (!supports_unary_memory_overlap(a.device(), op_type)) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API,
            "Ctorch_Scheduler: 该算子/设备不支持 in-place");
    }

    DeviceType target_dev = a.device();
    UnaryInplaceKernelFunc target_kernel = selectBestUnaryInplace(op_type, target_dev, unary_inplace_kernels_);

    if (target_kernel == nullptr) {
        CtorchError::throwException(ErrorPlatform::kGENERAL,ErrorType::PLATFORM_API,"Ctorch_Scheduler: 没有可用的 in-place Kernel");
    }
    target_kernel(a);
}

Tensor CtorchScheduler::dispatch_softmax(const Tensor& a, int dim) {
    DeviceType target_dev = a.device();

    if (target_dev == DeviceType::kMPS) {
        auto mps_kernel = softmax_kernels_[static_cast<size_t>(DeviceType::kMPS)]
            .load(std::memory_order_acquire);
        if (mps_kernel != nullptr && isDeviceAvailable(DeviceType::kMPS)) {
            return mps_kernel(a, dim);
        }
        CtorchError::throwException(ErrorPlatform::kGENERAL,ErrorType::PLATFORM_API,"Ctorch_Scheduler: 没有可用的MPS Softmax Kernel");
    }

    // CPU 路径：AMX -> SIMD -> BASIC
    if (isDeviceAvailable(DeviceType::kAMX)) {
        auto amx_kernel = softmax_kernels_[static_cast<size_t>(DeviceType::kAMX)]
            .load(std::memory_order_acquire);
        if (amx_kernel != nullptr) {
            return amx_kernel(a, dim);
        }
    }

    auto simd_kernel = softmax_kernels_[static_cast<size_t>(DeviceType::kSIMD)]
        .load(std::memory_order_acquire);
    if (simd_kernel != nullptr) {
        return simd_kernel(a, dim);
    }

    auto cpu_kernel = softmax_kernels_[static_cast<size_t>(DeviceType::kCPU)]
        .load(std::memory_order_acquire);
    if (cpu_kernel != nullptr) {
        return cpu_kernel(a, dim);
    }

    CtorchError::throwException(ErrorPlatform::kGENERAL,ErrorType::PLATFORM_API,"Ctorch_Scheduler: 没有可用的Softmax Kernel");
    return Tensor();
}
