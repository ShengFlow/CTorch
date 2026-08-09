/**
 * @file CtorchScheduler.h
 * @brief Ctorch 框架的核心调度器类
 * @details 采用单例模式实现，负责管理所有 kernel 映射关系，根据算子类型和设备类型，
 *          自动查找并调用对应的 kernel，实现 kernel 的统一调度。
 *          职责单一：只做 kernel 查找和调用，不涉及自动微分。
 *          支持热替换：通过原子操作实现 C3 JIT 编译后的 kernel 在线替换。
 * @author GhostFace
 * @date 2025/12/20
 */
#ifndef CTORCH_SCHEDULER_H
#define CTORCH_SCHEDULER_H
#include <atomic>
#include <array>
#include <deque>
#include <mutex>
#include <optional>
#include <vector>
#include "CtorchError.h"
#include "Tensor.h"
#include "./../src/kernels/kernels.h"
#ifndef CT_DISABLE_C3
#include "C3/C3Config.h"
#include "C3/C3KernelRegistry.h"
#include "C3/C3HotPathManager.h"
#include "C3/RegionFusion.h"
#endif

#ifdef C3_DISPATCH_TIMING
#include <chrono>
#endif

#ifndef CT_DISABLE_C3
namespace ct { namespace detail {
/// 实验开关：单 kernel hotpath 注入是否禁用（编译期 CT_C3_DISABLE_SINGLE_KERNEL 宏
/// 或运行时 C3_DISABLE_SINGLE_KERNEL=1 均视为禁用）。禁用后仅保留区域融合。
/// 用于归因 H2 缺陷（单 kernel 渐进注入形成 Eager→JIT 混合轨迹破坏训练一致性）。
inline bool c3SingleKernelDisabled() {
    return !ct::c3::singleKernelInjectionEnabled();
}

/// 反向传播标志：ComputeCore::backward 入口置 true、出口置 false。
/// 用于在 dispatch 中识别反向传播路径（其梯度 MatMul 输入不 requires_grad，
/// 但形状是转置的，走单 kernel 会命中错误形状）。
inline bool& g_in_backward_flag() {
    static thread_local bool in_backward = false;
    return in_backward;
}
inline bool g_in_backward() { return g_in_backward_flag(); }
inline void set_in_backward(bool v) { g_in_backward_flag() = v; }

/// 判断是否处于 autograd 追踪区域（反向传播中，或任一输入 requires_grad）。
/// 方案 A：训练/反向传播期间禁用单 kernel 渐进注入（Eager→JIT 混合轨迹
/// 破坏训练一致性），仅保留区域融合（整体接管序列，不产生逐 op 交替）。
/// 纯推理（EnableGrad=true 且输入不 requires_grad）时保留单 kernel 加速。
inline bool inAutogradScope(bool a_grad, bool b_grad) {
    return g_in_backward() || a_grad || b_grad;
}

/// 诊断开关：C3_DISABLE_OP=<逗号分隔的 op 整数值> 时，跳过这些 op 的单 kernel 注入。
/// 仅用于二分定位"哪个 op 破坏训练轨迹"，排查完移除。
inline bool c3OpDisabled(int op_id) {
    static const std::vector<int> disabled = []() {
        std::vector<int> v;
        const char* env = std::getenv("C3_DISABLE_OP");
        if (env != nullptr) {
            std::istringstream ss(env);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty()) v.push_back(atoi(tok.c_str()));
            }
        }
        return v;
    }();
    for (int d : disabled) if (d == op_id) return true;
    return false;
}
}} // namespace ct::detail
#endif

class CtorchScheduler{

    // ========== [DEPRECATED] Trace-based fusion ==========
    // tryFusedDispatch 已被区域融合 (Region Fusion) 替代。
    // 该实现存在形状匹配过于宽松的 bug（只检查 lhs_shape[0]），
    // 导致融合核输出形状错误，MNIST 准确率从 97.24% 跌至 11.23%。
    // 保留代码作为历史参考，详见 STATUS_CONTEXT.md Bug 5。
    //
    // struct FusedTraceEntry {
    //     op op_type;
    //     std::vector<size_t> input_shapes;
    //     std::vector<size_t> output_shape;
    //     std::vector<Tensor> input_tensors;
    //     DeviceType dev;
    // };
    //
    // static constexpr size_t kFusedTraceMaxLen = 8;
    // std::deque<FusedTraceEntry> fused_trace_;
    // mutable std::mutex fused_trace_mutex_;
    //
    // void recordToFusedTrace(op op_type, const std::vector<size_t>& input_shapes,
    //                        const std::vector<size_t>& output_shape,
    //                        std::vector<Tensor> input_tensors, DeviceType dev);
    //
    // std::optional<Tensor> tryFusedDispatch(op op_type, const std::vector<Tensor>& inputs,
    //                                        const std::vector<size_t>& output_shape,
    //                                        DeviceType dev);
    // =====================================================

    #ifndef CT_DISABLE_C3
    // ======================= 区域融合 trace =======================

    /// 区域融合 trace：记录 dispatch 的 op 类型序列，用于 rolling hash 匹配
    std::vector<op> region_trace_;
    mutable std::mutex region_trace_mutex_;
    std::vector<uint64_t> region_prefix_hashes_;  // 缓存的前缀哈希

    // ======================= 区域融合（Region Fusion） =======================

    enum class PrewalkState {
        kIdle,          // 不在预走模式
        kPrewalking,    // 预走进行中
        kFallback       // 预走失败，回退 eager
    };

    /// 预走缓存项
    struct PrewalkEntry {
        op op_type;
        std::vector<Tensor> original_inputs;  // 原始输入（非占位符）
        std::vector<size_t> output_shape;     // 输出形状
    };

    PrewalkState prewalk_state_ = PrewalkState::kIdle;
    std::vector<PrewalkEntry> prewalk_cache_;
    ct::c3::RegionEntry* matched_region_ = nullptr;
    size_t prewalk_pos_ = 0;  // 当前预走到的位置（在 region 的 op_seq 中）

    /// 缓存上次匹配的 region，避免空闲模式重复计算 hash
    ct::c3::RegionEntry* cached_region_ = nullptr;
    uint64_t cached_hash_ = 0;

    /// 执行区域融合预走检查
    /// @return 非空：区域融合结果（占位符/region 执行结果/回退结果）
    ///         nullopt：继续正常 eager dispatch
    std::optional<Tensor> tryRegionDispatch(op op_type,
                                            const std::vector<Tensor>& inputs,
                                            DeviceType dev);

    /// 计算 op 的输出形状（用于预走占位符）
    std::vector<size_t> computeOutputShape(op op_type,
                                           const std::vector<Tensor>& inputs) const;

    /// 执行 eager fallback：重新执行所有缓存的 op
    Tensor executeEagerFallback(const std::vector<Tensor>& current_inputs,
                                op current_op_type, DeviceType dev);

    /// 为预走占位张量构造惰性物化器（LazyBox）
    /// @param cache 预走缓存（到目标 op 为止的前缀）
    /// @param target_idx 目标 op 在 region op_seq 中的索引（物化到该 op 的输出）
    /// @param dev 目标设备
    /// @return 物化器（携带 eager 前缀重放闭包）；缓存为空时返回 nullptr
    std::shared_ptr<LazyMaterializer> buildLazyMaterializer(
        const std::vector<PrewalkEntry>& cache, size_t target_idx, DeviceType dev);

    /// true prewalk 配套:创建 placeholder Tensor (空 storage + LazyMaterializer)
    /// 闭包 re-run 当前 op(简化:cache 仅用于 future 优化,当前版本不重跑 prefix)
    Tensor createPrewalkPlaceholder(
        op op_type, const std::vector<Tensor>& inputs,
        const std::vector<PrewalkEntry>& cache);
#endif

private:
    CtorchScheduler();
    CtorchScheduler(const CtorchScheduler&);
    CtorchScheduler& operator=(const CtorchScheduler&) = delete;

    static constexpr size_t OP_COUNT = static_cast<size_t>(op::kCount);
    static constexpr size_t DEVICE_COUNT = static_cast<size_t>(DeviceType::kCount);

    // ABI 门控：op::kCount / DeviceType::kCount 改变会改变 kernel 查找表维度。
    // 以下硬编码数字是 ABI 变更的强制检查点。新增枚举值时必须同步更新此处，
    // 否则编译失败，防止调度表维度与注册逻辑错位。
    // 详见 ABI_POLICY.md 第 3.2 节“新增算子 ABI 检查清单”。
    static_assert(static_cast<size_t>(op::kCount) == 28,
                  "op::kCount changed. Update this assert and all backend kernel registrations (see ABI_POLICY.md)");
    static_assert(static_cast<size_t>(DeviceType::kCount) == 7,
                  "DeviceType::kCount changed. Update this assert and all backend kernel registrations (see ABI_POLICY.md)");

    std::array<std::array<std::atomic<BinaryKernelFunc>, DEVICE_COUNT>, OP_COUNT> binary_kernels_{};
    std::array<std::array<std::atomic<UnaryKernelFunc>, DEVICE_COUNT>, OP_COUNT> unary_kernels_{};
    std::array<std::array<std::atomic<UnaryInplaceKernelFunc>, DEVICE_COUNT>, OP_COUNT> unary_inplace_kernels_{};
    std::array<std::atomic<Tensor (*)(const Tensor&, int)>, DEVICE_COUNT> softmax_kernels_{};

    void initKernels();

    static BinaryKernelFunc selectBestBinary(op op_type, DeviceType dev,
        const std::array<std::array<std::atomic<BinaryKernelFunc>, DEVICE_COUNT>, OP_COUNT>& table);
    static UnaryKernelFunc selectBestUnary(op op_type, DeviceType dev,
        const std::array<std::array<std::atomic<UnaryKernelFunc>, DEVICE_COUNT>, OP_COUNT>& table);
    static UnaryInplaceKernelFunc selectBestUnaryInplace(op op_type, DeviceType dev,
        const std::array<std::array<std::atomic<UnaryInplaceKernelFunc>, DEVICE_COUNT>, OP_COUNT>& table);
public:
    static CtorchScheduler& getInstance() {
        static CtorchScheduler instance_;
        return instance_;
    }

    static bool isDeviceAvailable(DeviceType dev_type);

    static DeviceType getTargetDevice(const Tensor& a, const Tensor& b) {
        if (a.device() != b.device()) {
            CtorchError::log(ErrorLevel::ERROR,ErrorPlatform::kGENERAL,ErrorType::DEVICE_COMPAT,"Ctorch_Scheduler: Tensor不在同一平台");
        }
        return a.device();
    }

    // P1-3: MPS in-place unary kernel memory overlap 支持。
    // 仅对逐元素 unary 算子返回 true；Softmax/Sum/Min/Max 等含规约或二元语义的算子返回 false。
    static bool supports_unary_memory_overlap(DeviceType dev, op op_type) {
        (void)dev;
        switch (op_type) {
            case op::Neg:
            case op::Cos:
            case op::Sin:
            case op::ReLU:
            case op::Tanh:
            case op::Sigmoid:
            case op::GELU:
            case op::LReLU:
            case op::Log:
            case op::Exp:
            case op::Abs:
                return true;
            default:
                return false;
        }
    }

    BinaryKernelFunc get_binary_kernel(op op_type, DeviceType dev) const {
        return binary_kernels_[static_cast<size_t>(op_type)][static_cast<size_t>(dev)]
            .load(std::memory_order_acquire);
    }

    UnaryKernelFunc get_unary_kernel(op op_type, DeviceType dev) const {
        return unary_kernels_[static_cast<size_t>(op_type)][static_cast<size_t>(dev)]
            .load(std::memory_order_acquire);
    }

    UnaryInplaceKernelFunc get_unary_inplace_kernel(op op_type, DeviceType dev) const {
        return unary_inplace_kernels_[static_cast<size_t>(op_type)][static_cast<size_t>(dev)]
            .load(std::memory_order_acquire);
    }

    Tensor (*get_softmax_kernel(DeviceType dev) const)(const Tensor&, int) {
        return softmax_kernels_[static_cast<size_t>(dev)].load(std::memory_order_acquire);
    }

    void replace_binary_kernel(op op_type, DeviceType dev, BinaryKernelFunc new_kernel) {
        binary_kernels_[static_cast<size_t>(op_type)][static_cast<size_t>(dev)]
            .store(new_kernel, std::memory_order_release);
    }

    void replace_unary_kernel(op op_type, DeviceType dev, UnaryKernelFunc new_kernel) {
        unary_kernels_[static_cast<size_t>(op_type)][static_cast<size_t>(dev)]
            .store(new_kernel, std::memory_order_release);
    }

    void replace_unary_inplace_kernel(op op_type, DeviceType dev, UnaryInplaceKernelFunc new_kernel) {
        unary_inplace_kernels_[static_cast<size_t>(op_type)][static_cast<size_t>(dev)]
            .store(new_kernel, std::memory_order_release);
    }

    void replace_softmax_kernel(DeviceType dev, Tensor (*new_kernel)(const Tensor&, int)) {
        softmax_kernels_[static_cast<size_t>(dev)].store(new_kernel, std::memory_order_release);
    }

    Tensor dispatch(const Tensor& a, const Tensor& b, op op_type);
    Tensor dispatch(const Tensor& a, op op_type);
    void dispatch_inplace(Tensor& a, op op_type);
    Tensor dispatch_softmax(const Tensor& a, int dim = -1);

    #ifndef CT_DISABLE_C3
    /// 重置区域融合状态（包含 trace 和预走状态，用于测试场景）
    void resetRegionFusion() {
        {
            std::lock_guard<std::mutex> lk(region_trace_mutex_);
            region_trace_.clear();
            region_prefix_hashes_.clear();
        }
        prewalk_state_ = PrewalkState::kIdle;
        prewalk_cache_.clear();
        matched_region_ = nullptr;
        prewalk_pos_ = 0;
        cached_region_ = nullptr;
        cached_hash_ = 0;
    }

    /// 打印区域融合 dispatch 计时统计（调试用）
    friend void c3_print_region_timing();
#endif

    template <op OpType>
    inline Tensor dispatch(const Tensor& a, const Tensor& b) {
#ifndef CT_DISABLE_C3
        // [区域融合] 快速路径：预走模式中跳过 dtype/shape 检查，直接调用 tryRegionDispatch
        if (prewalk_state_ == PrewalkState::kPrewalking && ct::c3::regionFusionEnabled()) {
            auto region_result = tryRegionDispatch(OpType, {a, b}, getTargetDevice(a, b));
            if (region_result.has_value()) {
                return std::move(region_result.value());
            }
        }
#endif
#ifdef C3_DISPATCH_TIMING
        static int64_t g_dtype_ns = 0, g_shape_ns = 0, g_dev_ns = 0, g_vec_ns = 0, g_region_ns = 0;
        static int64_t g_c3_ns = 0, g_fused_ns = 0, g_hotpath_ns = 0, g_select_ns = 0, g_kernel_ns = 0;
        static int64_t g_trace_ns = 0, g_region_trace_ns = 0;
        static int g_count = 0;
        auto t_total = std::chrono::high_resolution_clock::now();
        auto t0 = t_total;
#endif
        if (a.dtype() != b.dtype()) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "Ctorch_Scheduler: Tensor类型不一致");
        }
#ifdef C3_DISPATCH_TIMING
        auto t1 = std::chrono::high_resolution_clock::now();
        g_dtype_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        t0 = t1;
#endif
        if (OpType != op::Add && OpType != op::Mul && OpType != op::Sub && OpType != op::Div && OpType != op::CE && OpType != op::MatMul && a.sizes() != b.sizes()) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION, "Ctorch_Scheduler: Tensor形状不一致");
        }
#ifdef CT_DEBUG
        // 无计数限制的 MatMul 追踪：打印每个 MatMul 的输入形状和 tensor id
        if constexpr (OpType == op::MatMul) {
            fprintf(stderr, "[TRACE-MatMul] a=[");
            for (size_t i = 0; i < a.shape().size(); ++i) { if (i > 0) fprintf(stderr, ","); fprintf(stderr, "%zu", a.shape()[i]); }
            fprintf(stderr, "] b=[");
            for (size_t i = 0; i < b.shape().size(); ++i) { if (i > 0) fprintf(stderr, ","); fprintf(stderr, "%zu", b.shape()[i]); }
            fprintf(stderr, "] tid_a=%zu tid_b=%zu\n", a.id(), b.id());
        }
#endif
#ifdef C3_DISPATCH_TIMING
        t1 = std::chrono::high_resolution_clock::now();
        g_shape_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        t0 = t1;
#endif

        DeviceType target_dev = getTargetDevice(a, b);
#ifdef C3_DISPATCH_TIMING
        t1 = std::chrono::high_resolution_clock::now();
        g_dev_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        t0 = t1;
#endif

        #ifndef CT_DISABLE_C3
        // [区域融合] 预走/匹配检查
        if (ct::c3::regionFusionEnabled()) {
            std::vector<Tensor> region_inputs = {a, b};
#ifdef C3_DISPATCH_TIMING
            t1 = std::chrono::high_resolution_clock::now();
            g_vec_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            t0 = t1;
#endif
            auto region_result = tryRegionDispatch(OpType, region_inputs, target_dev);
#ifdef C3_DISPATCH_TIMING
            t1 = std::chrono::high_resolution_clock::now();
            g_region_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            t0 = t1;
#endif
            if (region_result.has_value()) {
#ifdef C3_DISPATCH_TIMING
                g_count++;
                if (g_count <= 20) {
                    fprintf(stderr, "[DISPATCH TIMING] op=%d dtype=%.0f shape=%.0f dev=%.0f vec=%.0f region=%.0f ns\n",
                            (int)OpType,
                            (double)g_dtype_ns / g_count, (double)g_shape_ns / g_count,
                            (double)g_dev_ns / g_count, (double)g_vec_ns / g_count,
                            (double)g_region_ns / g_count);
                }
#endif
                return std::move(region_result.value());
            }
        }
#endif

#ifndef CT_DISABLE_C3
        // C3 JIT 热替换优先查询：若已安装 C3 kernel，优先使用
        // 实验开关 C3_DISABLE_SINGLE_KERNEL=1 时跳过，仅保留区域融合
        // 方案 A：autograd 追踪区域（输入 requires_grad）也跳过单 kernel 注入，
        // 避免 Eager→JIT 混合轨迹破坏训练一致性。
        // DEBT-NEW-7 H2 fix 计数器:c3_attemptable + in_autograd 时记录 bypass
        const bool c3_attemptable = !ct::detail::c3SingleKernelDisabled() &&
                                    !ct::detail::c3OpDisabled(static_cast<int>(OpType));
        const bool in_autograd = ct::detail::inAutogradScope(a.requires_grad(), b.requires_grad());
        if (c3_attemptable && in_autograd) {
            ct::c3::C3KernelRegistry::getInstance().recordBypass();
        }
        if (c3_attemptable && !in_autograd) {
            auto c3_result = ct::c3::C3KernelRegistry::getInstance().tryExecute(OpType, a, b);
            if (c3_result.has_value()) {
#ifdef CT_DEBUG
                {
                    std::ostringstream oss;
                    oss << "[C3-Dispatch] binary_c3_hit op=" << (int)OpType
                        << " a=[";
                    for (size_t i = 0; i < a.shape().size(); ++i) {
                        if (i > 0) oss << ",";
                        oss << a.shape()[i];
                    }
                    oss << "] b=[";
                    for (size_t i = 0; i < b.shape().size(); ++i) {
                        if (i > 0) oss << ",";
                        oss << b.shape()[i];
                    }
                    oss << "] out=[";
                    auto& out_s = c3_result.value().shape();
                    for (size_t i = 0; i < out_s.size(); ++i) {
                        if (i > 0) oss << ",";
                        oss << out_s[i];
                    }
                    oss << "]";
                    CtorchError::log(ErrorLevel::DEBUG, ErrorPlatform::kGENERAL,
                                     ErrorType::UNKNOWN, oss.str());
                }
#endif
                return c3_result.value();
            }
        }
#endif

        // [DEPRECATED] trace-based 融合已被区域融合替代，此处跳过

        #ifndef CT_DISABLE_C3
        // 记录热路径（触发 C3 自动编译）
        if (target_dev != DeviceType::kMPS) {
            std::vector<size_t> shape_sig = a.shape();
            shape_sig.insert(shape_sig.end(), b.shape().begin(), b.shape().end());
            ct::c3::C3HotPathManager::instance().recordCall(OpType, target_dev, shape_sig);
        }
#endif

        BinaryKernelFunc func = selectBestBinary(OpType, target_dev, binary_kernels_);
        if (func == nullptr) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API, "Ctorch_Scheduler: 没有可用的Kernel");
        }
#ifdef CT_DEBUG
        {
            static int dbg_cnt = 0;
            if (dbg_cnt < 500) {
                std::ostringstream oss;
                oss << "[C3-Dispatch] binary_eager op=" << (int)OpType
                    << " a=[";
                for (size_t i = 0; i < a.shape().size(); ++i) {
                    if (i > 0) oss << ",";
                    oss << a.shape()[i];
                }
                oss << "] b=[";
                for (size_t i = 0; i < b.shape().size(); ++i) {
                    if (i > 0) oss << ",";
                    oss << b.shape()[i];
                }
                oss << "] dev=" << (int)target_dev;
                CtorchError::log(ErrorLevel::DEBUG, ErrorPlatform::kGENERAL, ErrorType::UNKNOWN, oss.str());
            }
            dbg_cnt++;
        }
#endif
        Tensor result = func(a, b);

        // [DEPRECATED] 融合 trace 记录已被移除，由区域融合替代

        #ifndef CT_DISABLE_C3
        // 记录到区域融合 trace + 同步更新 prewalk_cache_
        {
            std::lock_guard<std::mutex> lk(region_trace_mutex_);
            region_trace_.push_back(OpType);
            // 限制 trace 最大长度，防止 O(n) 哈希计算和内存膨胀
            if (region_trace_.size() > 64) {
                region_trace_.erase(region_trace_.begin());
            }
            region_prefix_hashes_ = ct::c3::RollingHash::computePrefixHashes(region_trace_);

            // DEBT-NEW-7:同步更新 prewalk_cache_（region fusion match 用）,
            // 按 buildFusedGraph 约定只缓存每个 dispatch 的 external inputs:
            //   - MatMul: 2 external（区域入口）
            //   - 二元非 MatMul: 1 external（inputs[0] 是 chain）
            //   - 一元: 0 external（input 是 chain）
            PrewalkEntry entry;
            entry.op_type = OpType;
            entry.output_shape = computeOutputShape(OpType, {a, b});
            if constexpr (OpType == op::MatMul) {
                entry.original_inputs = {a, b};
            } else if constexpr (OpType == op::Add || OpType == op::Sub ||
                                 OpType == op::Mul || OpType == op::Div ||
                                 OpType == op::CE) {
                entry.original_inputs = {b};
            } else {
                entry.original_inputs = {};
            }
            prewalk_cache_.push_back(std::move(entry));
            if (prewalk_cache_.size() > 8) {
                prewalk_cache_.erase(prewalk_cache_.begin());
            }
        }
#endif

        return result;
    }

    template <op OpType>
    inline Tensor dispatch(const Tensor& a) {
#ifdef CT_DEBUG
        // 无计数限制的 unary 追踪
        if constexpr (OpType == op::ReLU) {
            fprintf(stderr, "[TRACE-ReLU] in=[");
            for (size_t i = 0; i < a.shape().size(); ++i) { if (i > 0) fprintf(stderr, ","); fprintf(stderr, "%zu", a.shape()[i]); }
            fprintf(stderr, "] tid=%zu\n", a.id());
        }
#endif
#ifndef CT_DISABLE_C3
        // [区域融合] 快速路径：预走模式中跳过检查，直接调用 tryRegionDispatch
        if (prewalk_state_ == PrewalkState::kPrewalking && ct::c3::regionFusionEnabled()) {
            auto region_result = tryRegionDispatch(OpType, {a}, a.device());
            if (region_result.has_value()) {
                return std::move(region_result.value());
            }
        }
#endif

        DeviceType target_dev = a.device();

#ifndef CT_DISABLE_C3
        // [区域融合] 预走/匹配检查
        if (ct::c3::regionFusionEnabled()) {
            std::vector<Tensor> region_inputs = {a};
            auto region_result = tryRegionDispatch(OpType, region_inputs, target_dev);
            if (region_result.has_value()) {
                return std::move(region_result.value());
            }
        }
#endif

#ifndef CT_DISABLE_C3
        // C3 JIT 热替换优先查询：若已安装 C3 kernel，优先使用
        // 实验开关 C3_DISABLE_SINGLE_KERNEL=1 时跳过，仅保留区域融合
        // 方案 A：autograd 追踪区域（输入 requires_grad）也跳过单 kernel 注入，
        // 避免 Eager→JIT 混合轨迹破坏训练一致性。
        // DEBT-NEW-7 H2 fix 计数器:unary 版（b=false 因 unary 无第二输入）
        const bool c3_attemptable_u = !ct::detail::c3SingleKernelDisabled() &&
                                      !ct::detail::c3OpDisabled(static_cast<int>(OpType));
        const bool in_autograd_u = ct::detail::inAutogradScope(a.requires_grad(), false);
        if (c3_attemptable_u && in_autograd_u) {
            ct::c3::C3KernelRegistry::getInstance().recordBypass();
        }
        if (c3_attemptable_u && !in_autograd_u) {
            auto c3_result = ct::c3::C3KernelRegistry::getInstance().tryExecuteUnary(OpType, a);
            if (c3_result.has_value()) {
#ifdef CT_DEBUG
                {
                    std::ostringstream oss;
                    oss << "[C3-Dispatch] unary_c3_hit op=" << (int)OpType
                        << " in=[";
                    for (size_t i = 0; i < a.shape().size(); ++i) {
                        if (i > 0) oss << ",";
                        oss << a.shape()[i];
                    }
                    oss << "] out=[";
                    auto& out_s = c3_result.value().shape();
                    for (size_t i = 0; i < out_s.size(); ++i) {
                        if (i > 0) oss << ",";
                        oss << out_s[i];
                    }
                    oss << "]";
                    CtorchError::log(ErrorLevel::DEBUG, ErrorPlatform::kGENERAL,
                                     ErrorType::UNKNOWN, oss.str());
                }
#endif
                return c3_result.value();
            }
        }
#endif

        // [DEPRECATED] trace-based 融合已被区域融合替代，此处跳过

#ifndef CT_DISABLE_C3
        // 记录热路径
        if (target_dev != DeviceType::kMPS) {
            ct::c3::C3HotPathManager::instance().recordCall(OpType, target_dev, a.shape());
        }
#endif

        UnaryKernelFunc func = selectBestUnary(OpType, target_dev, unary_kernels_);
        if (func == nullptr) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API, "Ctorch_Scheduler: 没有可用的Kernel");
        }
#ifdef CT_DEBUG
        {
            static int dbg_unary_cnt = 0;
            if (dbg_unary_cnt < 500) {
                std::ostringstream oss;
                oss << "[C3-Dispatch] unary_eager op=" << (int)OpType
                    << " in=[";
                for (size_t i = 0; i < a.shape().size(); ++i) {
                    if (i > 0) oss << ",";
                    oss << a.shape()[i];
                }
                oss << "] dev=" << (int)target_dev;
                CtorchError::log(ErrorLevel::DEBUG, ErrorPlatform::kGENERAL,
                                 ErrorType::UNKNOWN, oss.str());
            }
            dbg_unary_cnt++;
        }
#endif
        Tensor result = func(a);

        // [DEPRECATED] 融合 trace 记录已被移除，由区域融合替代

        #ifndef CT_DISABLE_C3
        // 记录到区域融合 trace + 同步更新 prewalk_cache_
        {
            std::lock_guard<std::mutex> lk(region_trace_mutex_);
            region_trace_.push_back(OpType);
            if (region_trace_.size() > 64) {
                region_trace_.erase(region_trace_.begin());
            }
            region_prefix_hashes_ = ct::c3::RollingHash::computePrefixHashes(region_trace_);

            // DEBT-NEW-7:同步更新 prewalk_cache_。一元 op 的 input 是 chain,无 external
            PrewalkEntry entry;
            entry.op_type = OpType;
            entry.output_shape = computeOutputShape(OpType, {a});
            entry.original_inputs = {};  // 一元 op:无 external input
            prewalk_cache_.push_back(std::move(entry));
            if (prewalk_cache_.size() > 8) {
                prewalk_cache_.erase(prewalk_cache_.begin());
            }
        }
#endif

        return result;
    }
};

#ifndef CT_DISABLE_C3
/// 打印区域融合 dispatch 计时统计（调试用）
void c3_print_region_timing();
#endif
#endif //CTORCH_SCHEDULER_H
