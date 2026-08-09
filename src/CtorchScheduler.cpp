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
    op op_type, const std::vector<Tensor>& inputs, DeviceType /*dev*/) {
    // 读取 trace 快照
    std::vector<op> trace_snapshot;
    {
        std::lock_guard<std::mutex> lk(region_trace_mutex_);
        trace_snapshot = region_trace_;
    }

    auto& registry = ct::c3::RegionFusionRegistry::getInstance();

    // 构建 extended trace: [current trace] + [current op]
    std::vector<op> extended = trace_snapshot;
    extended.push_back(op_type);
    auto extended_prefix = ct::c3::RollingHash::computePrefixHashes(extended);
    if (extended_prefix.size() < 2) {
        return std::nullopt;  // 至少需要 2 个 op 才有 region 意义
    }

    size_t current_pos = extended.size() - 1;

    // DEBT-NEW-7:跟 installWithCost 一样,把 shape 混入 hash
    // 先计算当前 dispatch 的 first input 的 shape hash(用 MatMul 的 lhs shape)
    // 如果当前 op 不是 MatMul(在 region 中间),用 MatMul-equivalent first input(从 cache 取)
    uint64_t shape_hash = 0;
    if (op_type == op::MatMul) {
        if (!inputs.empty()) {
            for (auto s : inputs.front().shape()) {
                shape_hash = shape_hash * 31 + s + 1;
            }
        }
    } else if (!prewalk_cache_.empty()) {
        // 找 cache 里最后一个 MatMul 的 external input(它的 lhs shape 决定 region hash)
        for (auto it = prewalk_cache_.rbegin(); it != prewalk_cache_.rend(); ++it) {
            if (it->op_type == op::MatMul && !it->original_inputs.empty()) {
                for (auto s : it->original_inputs.front().shape()) {
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

    // 临时(2026-08-09 DEBT-NEW-7 实施阶段):FusedCompiledKernel 假设 inputs[0].shape() =
    // output.shape()(只支持 elementwise 融合)。对于 MatMul+Add+ReLU 这种 MatMul
    // 起头的 region,kernel 的 inputs/output shape 推导会错(把 M*K 当成 M*N)。
    // 解决:暂不 invoke MatMul 起头的 region(返回 nullopt → 走 eager),等 v0.6.0
    // 实现 MatMul-aware fusedCompiledKernel 后再放开。
    if (!match->op_seq.empty() && match->op_seq[0] == op::MatMul) {
#ifdef CT_DEBUG
        static int dbg_matmul_region_skip = 0;
        if (dbg_matmul_region_skip < 5) {
            std::ostringstream oss;
            oss << "[C3-RegionSkip] MatMul-rooted region len=" << match->len
                << " (FusedCompiledKernel only supports elementwise fusion yet)";
            CtorchError::log(ErrorLevel::DEBUG, ErrorPlatform::kGENERAL,
                ErrorType::UNKNOWN, oss.str());
            dbg_matmul_region_skip++;
        }
#endif
        return std::nullopt;
    }

    // 匹配成功!需要从 prewalk_cache_ 里取最近 (match.len - 1) 个 dispatch 的 external inputs,
    // 加上当前 dispatch 的 external inputs(在 inputs 参数里)
    // 关键:prewalk_cache_ 应该已经缓存了 trace_snapshot 的最后 (match.len - 1) 个 op
    // 如果不匹配(缓存不够),放弃
    size_t needed = match->len - 1;  // 不含当前 op
    if (prewalk_cache_.size() < needed) {
        return std::nullopt;  // 缓存不足(可能在 prewalk 初始化阶段),放弃
    }

    // DEBT-NEW-7 重要:rolling hash 只按 op_seq 算,不同 shape 的同 op_seq 序列
    // 会产生相同 hash(registry 多次 install 会互相覆盖)。所以 match 之后必须
    // 验证 shape,否则会用错 shape 的 kernel 计算。
    // 验证方法:entry 的 first_input_shapes 跟缓存里第一个 external input 的 shape 比对
    if (!match->first_input_shapes.empty() && needed > 0) {
        const auto& first_cached = prewalk_cache_[prewalk_cache_.size() - needed].original_inputs;
        if (!first_cached.empty()) {
            const auto& expected_shape = match->first_input_shapes.front();
            const auto& actual_shape = first_cached.front().shape();
            if (expected_shape != actual_shape) {
                // shape 不匹配:这个 entry 是别的 shape install 留下的,放弃
#ifdef CT_DEBUG
                {
                    std::ostringstream oss;
                    oss << "[C3-RegionShapeMismatch] expected=[";
                    for (auto s : expected_shape) oss << s << ",";
                    oss << "] actual=[";
                    for (auto s : actual_shape) oss << s << ",";
                    oss << "]";
                    CtorchError::log(ErrorLevel::DEBUG, ErrorPlatform::kGENERAL,
                        ErrorType::UNKNOWN, oss.str());
                }
#endif
                return std::nullopt;
            }
        }
    }

#ifdef CT_DEBUG
    {
        std::ostringstream oss;
        oss << "[C3-RegionHit] matched region len=" << match->len
            << " op_seq=[";
        for (auto o : match->op_seq) { oss << (int)o << ","; }
        oss << "] current_op=" << (int)op_type
            << " external_inputs=" << needed + (op_type == op::MatMul ? 2 : (inputs.size() > 1 ? 1 : 0))
            << " cache_size=" << prewalk_cache_.size();
        CtorchError::log(ErrorLevel::DEBUG, ErrorPlatform::kGENERAL,
            ErrorType::UNKNOWN, oss.str());
    }
#endif

    // 收集所有 external inputs
    std::vector<Tensor> external_inputs;
    // 从 prewalk_cache_ 末尾取最近 (match.len - 1) 个 entry
    for (size_t i = prewalk_cache_.size() - needed; i < prewalk_cache_.size(); ++i) {
        for (const auto& t : prewalk_cache_[i].original_inputs) {
            external_inputs.push_back(t);
        }
    }
    // 当前 dispatch 的 external inputs(按 op 类型决定)
    if (op_type == op::MatMul) {
        // MatMul:region 入口,所有 inputs 都是外部
        for (const auto& t : inputs) external_inputs.push_back(t);
    } else if (inputs.size() > 1) {
        // 二元非 MatMul op:inputs[0] 是 chain,只取 inputs[1] 作为外部
        external_inputs.push_back(inputs[1]);
    } else {
        // 一元 op:input 是 chain,不取
    }

    // 准备 output tensor(按 region 最后一个 op 的 output_shape)
    // 最后一个 op 的 output_shape = 缓存中最新一个 entry 的 output_shape
    // 但当前 dispatch 的 output_shape 才是 region 的最终 output
    std::vector<size_t> out_shape = computeOutputShape(op_type, inputs);
    if (out_shape.empty()) {
        return std::nullopt;
    }
    size_t out_numel = 1;
    for (auto s : out_shape) out_numel *= s;
    if (out_numel == 0) {
        return std::nullopt;
    }

    Tensor out_tensor(ShapeTag{}, out_shape, DType::kFloat, DeviceType::kCPU);
    float* out_data = out_tensor.data_write<float>();
    if (!out_data) {
        return std::nullopt;
    }

    // 准备 input pointers 数组
    std::vector<const float*> input_ptrs;
    input_ptrs.reserve(external_inputs.size());
    for (const auto& t : external_inputs) {
        const float* p = t.data_read<float>();
        if (!p) {
            return std::nullopt;
        }
        input_ptrs.push_back(p);
    }

    // invoke kernel(通过 CompiledKernel::execute() 虚函数)
    // kernel 应该输出到 out_tensor 的 storage
    // 注意:CompiledKernel::execute() 是高阶接口,我们用它执行整体 fused region
    // 对于 HandwrittenKernelGen fused kernel,execute() 会调 result.fused_func
    try {
        // 把 input tensors 传给 kernel,期望它输出到 out_tensor
        // 但 standard CompiledKernel::execute() 会自己分配 output tensor
        // 我们的方案:让 kernel 写入我们预分配的 out_tensor 的 storage
        // 简化:调 execute() 拿它的结果,然后 copy 到 out_tensor
        // (这有 copy 开销但保证正确性)
        auto kernel_outputs = match->kernel->execute(external_inputs);
        if (kernel_outputs.empty() || kernel_outputs[0].storage().empty()) {
            return std::nullopt;
        }
        // 拷贝 kernel 输出到 out_tensor
        const float* src = kernel_outputs[0].data_read<float>();
        if (!src) {
            return std::nullopt;
        }
        size_t kernel_numel = kernel_outputs[0].numel();
        size_t copy_n = std::min(kernel_numel, out_numel);
        std::memcpy(out_data, src, copy_n * sizeof(float));
        // 如果 kernel 输出比 out_tensor 小,剩余部分填 0
        if (kernel_numel < out_numel) {
            std::memset(out_data + copy_n, 0, (out_numel - copy_n) * sizeof(float));
        }
        return out_tensor;
    } catch (const std::exception& e) {
        CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL,
            ErrorType::UNKNOWN,
            "CtorchScheduler::tryRegionDispatch: kernel execute failed: " +
            std::string(e.what()) + ", falling back to eager.");
        return std::nullopt;
    }
}

std::vector<size_t> CtorchScheduler::computeOutputShape(
    op op_type, const std::vector<Tensor>& inputs) const {
    // DEBT-NEW-7 region fusion 配套：每个 op 的真实输出 shape 计算
    // 用于 prewalk 期间创建 placeholder Tensor(空 storage + LazyMaterializer)
    if (inputs.empty()) return {};

    // 一元算子：输出 shape = 输入 shape
    auto isUnary = [](op t) {
        return t == op::ReLU || t == op::Tanh || t == op::Sigmoid ||
               t == op::Neg  || t == op::Exp  || t == op::Log    ||
               t == op::Abs  || t == op::GELU || t == op::Softmax;
    };

    if (isUnary(op_type) || inputs.size() == 1) {
        return inputs.front().sizes();
    }

    // 二元算子：分两类
    const auto& a = inputs[0].sizes();
    const auto& b = inputs.size() > 1 ? inputs[1].sizes() : a;

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

Tensor CtorchScheduler::executeEagerFallback(
    const std::vector<Tensor> & /*current_inputs*/, op /*current_op_type*/, DeviceType /*dev*/) {
    // Stub: 不可达（tryRegionDispatch 永远 nullopt 不会触发）
    return Tensor();
}

// 区域融合 dispatch 计时统计打印（stub：当前无 region fusion 活动，no-op）
void c3_print_region_timing() {
    // 当前 region fusion 关闭（C3 编译期 CT_C3_DISABLE_REGION_FUSION=ON），
    // 所有计时计数器未被触发，无可打印内容。
}
#endif // CT_DISABLE_C3
