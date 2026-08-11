/**
 * @file CtorchScheduler.cpp
 * @brief Ctorch 调度器实现
 * @author GhostFace
 * @date 2026/04/04
 */

#include "./../include/CtorchScheduler.h"
#include "./../include/AutoGrad.h"
#include "./../include/DeviceAllocator.h"

#ifdef __APPLE__
// 前向声明 Metal 设备创建函数，避免在 .cpp 中直接 #import <Metal/Metal.h>
// 该函数返回 id<MTLDevice>，在 C++ 侧以 void* 比较是否非空即可。
extern "C" void* MTLCreateSystemDefaultDevice(void);
#else
#include <cpuid.h>
#endif

bool CtorchScheduler::isDeviceAvailable(DeviceType dev_type) {
    // [Dev] v0.5.2+ (2026-08-09): static cache 让 system call (cpuid / MTLCreateSystemDefaultDevice)
    // 只在第一次调用时执行,后续 O(1) 查表。原版每次 dispatch 都调系统调用 0.5-1us,
    // 53848 dispatch/epoch × 1us = 54ms/epoch 净亏。
    // 设备可用性是 startup-time 不变量 (MPS 设备不会运行时插拔,AMX_TILE CPUID 不会运行时变),
    // 缓存安全。
    static const bool kCPU_Available = true;
    static const bool kCUDA_Available = false;
#ifdef __APPLE__
    static const bool kMPS_Available = MTLCreateSystemDefaultDevice() != nullptr;
    static const bool kAMX_Available = true;  // Apple Silicon: AMX 走 Accelerate 框架
#else
    static const bool kMPS_Available = false;
    static const bool kAMX_Available = []() {
        unsigned int eax, ebx, ecx, edx;
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
            return (edx & (1u << 24)) != 0;
        }
        return false;
    }();
#endif
    switch (dev_type) {
        case DeviceType::kCPU:  return kCPU_Available;
        case DeviceType::kCUDA: return kCUDA_Available;
        case DeviceType::kMPS:  return kMPS_Available;
        case DeviceType::kAMX:  return kAMX_Available;
        case DeviceType::kSIMD:
            // SIMD 路径依赖编译器 auto-vectorization，默认在 CPU 上可用。
            return true;
        default:
            return false;
    }
}

CtorchScheduler::CtorchScheduler() {
    printf(ESC_START COLOR_INFO"[INFO]  " ESC_END "[%s %" PRIu64 "] Ctorch Scheduler Started\n", getFormattedTimeMs().c_str(), getTimestampMs());
    
    // Allocator 现在由 AllocatorManager::getAllocator() 延迟初始化，
    // 避免 Storage 构造早于 CtorchScheduler 单例时无法获取 MPS buffer。
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
    // [Fix] v0.5.2 Linux build: AMX 是 macOS 专属 (Accelerate framework), Linux 跳过
    //   DeviceType::kAMX 在 Linux 上不存在, kernel 也不编, 不 register 即可
#ifdef __APPLE__
    set_bin(op::MatMul, DeviceType::kAMX, MatMul_AMX_kernel);
    set_unary(op::GELU, DeviceType::kAMX, GELU_AMX_kernel);
#endif

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

    // ========== 线性代数专用 kernel 注册（2026-08-10）==========
    // Rot (Givens 旋转) — 用于 JacobiSVD 的列对正交化
    // ApplyHk (Householder 反射应用) — 用于 HouseholderQR 的反射器 apply
    // 优先 AMX 路径（Apple Accelerate cblas_srot / cblas_sgemv + cblas_sger）
    // 其它设备暂未实现（cblas_srot/cblas_sger 在非 Apple 平台也能用 cblas 接口，
    // 后续如果上 CUDA/CPU-x86 可在对应 backend 加 SIMD/BASIC 回退）
#ifdef __APPLE__
    register_rot_kernel(DeviceType::kAMX, Rot_AMX_kernel);
#endif
#ifdef __APPLE__
    register_rot_kernel(DeviceType::kMPS, Rot_AMX_kernel);  // MPS 内存是 shared，AMX kernel 可用 (但 MPS 也 macOS only)
    register_rot_kernel(DeviceType::kCPU, Rot_AMX_kernel);  // CPU 也走 AMX（Accelerate 框架）
    register_rot_kernel(DeviceType::kSIMD, Rot_AMX_kernel);
#endif

#ifdef __APPLE__
    register_applyhk_kernel(DeviceType::kAMX, ApplyHk_AMX_kernel);
#endif
#ifdef __APPLE__
    register_applyhk_kernel(DeviceType::kMPS, ApplyHk_AMX_kernel);
    register_applyhk_kernel(DeviceType::kCPU, ApplyHk_AMX_kernel);
    register_applyhk_kernel(DeviceType::kSIMD, ApplyHk_AMX_kernel);
#endif
}

// ============================================================
// 线性代数专用 dispatch（不走 op enum，2026-08-10）
// ============================================================

void CtorchScheduler::dispatch_rot(Tensor& x, Tensor& y, float c, float s) {
    // 按 device 选 kernel：MPS > AMX > SIMD > CPU > CPU_BASIC
    DeviceType dev = x.device();
    RotKernelFunc f = get_rot_kernel(dev);
    if (f == nullptr) {
        // 回退：MPS → AMX → SIMD → CPU
        if (isDeviceAvailable(DeviceType::kAMX)) {
            f = get_rot_kernel(DeviceType::kAMX);
        }
        if (f == nullptr) {
            f = get_rot_kernel(DeviceType::kSIMD);
        }
        if (f == nullptr) {
            f = get_rot_kernel(DeviceType::kCPU);
        }
    }
    if (f == nullptr) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT,
                                     "dispatch_rot: 没有可用的 Rot kernel");
    }
    f(x, y, c, s);
}

void CtorchScheduler::dispatch_applyhk(Tensor& M, const Tensor& v, float tau,
                                        std::size_t k_offset, std::size_t p_cols) {
    DeviceType dev = M.device();
    ApplyHkKernelFunc f = get_applyhk_kernel(dev);
    if (f == nullptr) {
        if (isDeviceAvailable(DeviceType::kAMX)) {
            f = get_applyhk_kernel(DeviceType::kAMX);
        }
        if (f == nullptr) {
            f = get_applyhk_kernel(DeviceType::kSIMD);
        }
        if (f == nullptr) {
            f = get_applyhk_kernel(DeviceType::kCPU);
        }
    }
    if (f == nullptr) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT,
                                     "dispatch_applyhk: 没有可用的 ApplyHk kernel");
    }
    f(M, v, tau, k_offset, p_cols);
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

// ============================================================================
// 【Stub 2026-08-08】Region Fusion 预走接口 tryRegionDispatch()
// ============================================================================
// 当前状态：fix/c3-p0-on-wip 分支把 tryRegionDispatch 的 3 个调用点加到了 CtorchScheduler.h
// (L301, L358, L490, L503 区域) 但 .cpp 实现还没写。这是 link 错误根因。
// DEBT-NEW-7 region fusion:向后匹配(以当前 dispatch 的 op 结尾的 region)
// 总是缓存最近 K 个 dispatch 的 external inputs(按 buildFusedGraph 约定),
// 当 extended trace 末尾形成已知 region 模式时,从缓存里取出所有外部 input invoke kernel。
//
// 当前实现假设:
//   - region 的 op_seq[0] 是 region 入口 op,所有 inputs 是外部
//   - 后续 op 的 inputs[0] 是 chain (来自前一个 op 的输出),其余是外部
//   - 缓存容量 K=8(够 MatMul+Add+ReLU+MatMul+Add+ReLU 这种长链)
// 简化:不预测未来,只匹配"过去+当前=完整 region"。Eager 会执行 past ops,
// kernel 会重新计算(浪费了 past op 的 eager work,但保证正确性)。
#ifndef CT_DISABLE_C3
std::optional<Tensor> CtorchScheduler::tryRegionDispatch(
    op op_type, const Tensor* inputs, size_t num_inputs, DeviceType /*dev*/) {
#ifdef CT_PROFILE_PERF
    auto _t0 = std::chrono::steady_clock::now();
    // RAII-like defer:把 dispatch 耗时统计放这里,确保所有 return 路径都统计到
    struct PerfGuard {
        std::chrono::steady_clock::time_point t0;
        PerfGuard(std::chrono::steady_clock::time_point t) : t0(t) {}
        ~PerfGuard() {
            auto t1 = std::chrono::steady_clock::now();
            ct::c3::C3KernelRegistry::getInstance().recordPerfRegionDispatch(
                (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        }
    } _guard(_t0);
#endif
    // [Dev 2026-08-09 tryRegionDispatch 无候选短路] 第一道: 0 region 时 O(1) 返回
    // 53848 dispatch/epoch 大量 trace 拷贝 + hash + 7 次循环是无谓开销
    // 训练开始时 region 还没注册, 或 evict 后清空, 此分支直接返回
    auto& registry = ct::c3::RegionFusionRegistry::getInstance();
    if (registry.installedCountNoLock() == 0) {
        return std::nullopt;
    }

    // [Dev 2026-08-11 候选短路提前] 第二道: 当前 op 不可能作为任何已注册 region
    // 的末尾 op 时 O(1) 返回. 注意: 这是"末尾"过滤 (当前 dispatch 触发的 op),
    // 不是"任意位置"过滤 (region 可以任意 op 结尾, 但匹配窗口只到当前 op,
    // 所以末尾匹配即可)。
    // 放在 trace 快照之前: mayMatchAsLastOp 只依赖 op_type + 已注册 region,
    // 与 trace 无关。提前过滤可省掉下面 region_trace_mutex_ 锁 + vector 拷贝
    // (训练期大多数 op 都不是任一 region 末尾, 此分支命中率高)。
    if (!registry.mayMatchAsLastOp(op_type)) {
        return std::nullopt;
    }

    // 读取 trace 快照
    std::vector<op> trace_snapshot;
    {
        std::lock_guard<std::mutex> lk(region_trace_mutex_);
        trace_snapshot = region_trace_;
    }

    // 构建 extended trace: [current trace] + [current op]
    std::vector<op> extended = trace_snapshot;
    extended.push_back(op_type);
    auto extended_prefix = ct::c3::RollingHash::computePrefixHashes(extended);
    if (extended_prefix.size() < 2) {
        return std::nullopt;  // 至少需要 2 个 op 才有 region 意义
    }

    size_t current_pos = extended.size() - 1;

    // DEBT-NEW-7:跟 installWithCost 一样,把 shape 混入 hash
    uint64_t shape_hash = 0;
    if (op_type == op::MatMul) {
        if (num_inputs > 0) {
            for (auto s : inputs[0].shape()) {
                shape_hash = shape_hash * 31 + s + 1;
            }
        }
    } else if (prewalk_cache_count_ > 0) {
        // 从新到旧 (逻辑下标递减) 找最近一个 MatMul 的输入 shape
        for (size_t li = prewalk_cache_count_; li-- > 0;) {
            const auto& e = prewalkAt(li);
            if (e.op_type == op::MatMul && !e.original_inputs.empty()) {
                for (auto s : e.original_inputs.front().shape()) {
                    shape_hash = shape_hash * 31 + s + 1;
                }
                break;
            }
        }
    }

    // 手动实现向后匹配:从最长可能长度到最短,寻找 sub_hash = entry.hash 的 region
    ct::c3::RegionEntry* match = nullptr;
    for (size_t len = std::min<size_t>(current_pos + 1, 8); len >= 2 && !match; --len) {
        size_t start = current_pos + 1 - len;
        uint64_t op_hash = ct::c3::RollingHash::getSubHash(extended_prefix, start, current_pos);
        uint64_t full_hash = op_hash ^ (shape_hash << 32);
        auto* candidate = registry.find(full_hash);
        if (candidate && candidate->active && candidate->len == len) {
            match = candidate;
            break;
        }
    }
    if (!match) {
        return std::nullopt;
    }

    // M1 1.3 (2026-08-09): FusionCostModel 接到 tryRegionDispatch
    // 调度前 ROI 评估: 二次验证 match->cost.worthwhile, sanity check + 可观察性
    // 理论上 installWithCost 已 gating (entry.active = worth_it), 此分支理论上不触发,
    // 但某些路径走 install() 绕过 cost gating, 这里是最后一道防线
    // (O(0) 读字段 + 1 atomic, 不会拖慢 hot path)
#ifdef CT_PROFILE_PERF
    if (!match->cost.worthwhile) {
        ct::c3::C3KernelRegistry::getInstance().recordPerfRegionCostRejected();
    }
#endif
    if (!match->cost.worthwhile) {
        return std::nullopt;
    }

    // 匹配成功!需要从 prewalk_cache_ 里取最近 (match.len - 1) 个 dispatch 的 external inputs
    size_t needed = match->len - 1;
    if (prewalk_cache_count_ < needed) {
        return std::nullopt;
    }

    // DEBT-NEW-7 重要:rolling hash 只按 op_seq 算,不同 shape 的同 op_seq 序列
    // 会产生相同 hash(registry 多次 install 会互相覆盖)。所以 match 之后必须
    // 验证 shape,否则会用错 shape 的 kernel 计算。
    if (!match->first_input_shapes.empty() && needed > 0) {
        const auto& first_cached = prewalkAt(prewalk_cache_count_ - needed).original_inputs;
        if (!first_cached.empty()) {
            const auto& expected_shape = match->first_input_shapes.front();
            const auto& actual_shape = first_cached.front().shape();
            if (expected_shape != actual_shape) {
                return std::nullopt;
            }
        }
    }

    // 收集所有 external inputs
    std::vector<Tensor> external_inputs;
    for (size_t i = prewalk_cache_count_ - needed; i < prewalk_cache_count_; ++i) {
        for (const auto& t : prewalkAt(i).original_inputs) {
            external_inputs.push_back(t);
        }
    }
    if (op_type == op::MatMul) {
        for (size_t k = 0; k < num_inputs; ++k) external_inputs.push_back(inputs[k]);
    } else if (num_inputs > 1) {
        external_inputs.push_back(inputs[1]);
    } else {
        // 一元 op:input 是 chain,不取
    }

    std::vector<size_t> out_shape = computeOutputShape(op_type, inputs, num_inputs);
    if (out_shape.empty()) {
        return std::nullopt;
    }

    // 预读取 external inputs 验证 data pointer
    for (const auto& t : external_inputs) {
        if (!t.data_read<float>()) return std::nullopt;
    }

    // invoke kernel(通过 C3KernelRegistry::executeFusedWithInputs,内含 fused_hit 计数)
    try {
        ct::c3::KernelShapeInfo shapes;
        if (!external_inputs.empty()) shapes.lhs_shape = external_inputs.front().shape();
        if (external_inputs.size() > 1) shapes.rhs_shape = external_inputs[1].shape();
        shapes.out_shape = out_shape;
        shapes.fused_pattern = "region-fusion";

        Tensor kernel_result = ct::c3::C3KernelRegistry::getInstance()
            .executeFusedWithInputs(match->kernel, external_inputs, shapes);
        if (kernel_result.storage().empty()) {
            return std::nullopt;
        }
        return kernel_result;
    } catch (...) {
        return std::nullopt;
    }
    // (timing 由 _guard RAII 在函数退出时统计,见函数入口 PerfGuard)
}

std::vector<size_t> CtorchScheduler::computeOutputShape(
    op op_type, const Tensor* inputs, size_t num_inputs) const {
    // DEBT-NEW-7 region fusion 配套：每个 op 的真实输出 shape 计算
    // 用于 prewalk 期间创建 placeholder Tensor(空 storage + LazyMaterializer)
    if (num_inputs == 0) return {};

    // 一元算子：输出 shape = 输入 shape
    // [位掩码 2026-08-11] 一元算子集合 → 单 uint64 位掩码, O(1) 查表,
    // 消除多条件 OR 的分支预测失败惩罚. computeOutputShape 在每次 dispatch
    // 末尾 (prewalk_cache 记录) 调用于热路径, 位掩码化有真实收益.
    // op 连续枚举 (0..kCount-1=27 < 64), 静态断言兜底.
    static_assert(static_cast<size_t>(op::kCount) <= 64,
                  "op::kCount exceeds uint64 bitmask capacity");
    static constexpr uint64_t kUnaryOpMask =
          (1ull << static_cast<size_t>(op::ReLU))
        | (1ull << static_cast<size_t>(op::Tanh))
        | (1ull << static_cast<size_t>(op::Sigmoid))
        | (1ull << static_cast<size_t>(op::Neg))
        | (1ull << static_cast<size_t>(op::Exp))
        | (1ull << static_cast<size_t>(op::Log))
        | (1ull << static_cast<size_t>(op::Abs))
        | (1ull << static_cast<size_t>(op::GELU))
        | (1ull << static_cast<size_t>(op::Softmax));
    auto isUnary = [](op t) {
        return (kUnaryOpMask >> static_cast<size_t>(t)) & 1ull;
    };

    if (isUnary(op_type) || num_inputs == 1) {
        return inputs[0].sizes();
    }

    // 二元算子：分两类
    const auto& a = inputs[0].sizes();
    const auto& b = num_inputs > 1 ? inputs[1].sizes() : a;

    if (op_type == op::MatMul) {
        // [M, K] @ [K, N] → [M, N]（最常见的 2D matmul；高维暂不支持）
        if (a.size() >= 2 && b.size() >= 2) {
            return {a[a.size() - 2], b[b.size() - 1]};
        }
        return a;  // 退化 fallback
    }

    // 元素级（Add/Sub/Mul/Div/CE）：广播到较大 shape
    if (a == b) return a;

    // 简化广播：取两边维度的最大值（与 broadCast 逻辑对齐）
    std::vector<size_t> out(std::max(a.size(), b.size()));
    for (size_t i = 0; i < out.size(); ++i) {
        size_t ad = (i < a.size()) ? a[a.size() - 1 - i] : 1;
        size_t bd = (i < b.size()) ? b[b.size() - 1 - i] : 1;
        out[out.size() - 1 - i] = std::max(ad, bd);
    }
    return out;
}
#endif // CT_DISABLE_C3
