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
#include "CtorchError.h"
#include "Tensor.h"
#include "./../src/kernels/kernels.h"
#include "C3/C3KernelRegistry.h"

class CtorchScheduler{
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

    template <op OpType>
    inline Tensor dispatch(const Tensor& a, const Tensor& b) {
        if (a.dtype() != b.dtype()) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "Ctorch_Scheduler: Tensor类型不一致");
        }
        if (OpType != op::Add && OpType != op::Mul && OpType != op::Sub && OpType != op::Div && OpType != op::CE && OpType != op::MatMul && a.sizes() != b.sizes()) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION, "Ctorch_Scheduler: Tensor形状不一致");
        }

        DeviceType target_dev = getTargetDevice(a, b);

        // C3 JIT 热替换优先查询：若已安装 C3 kernel，优先使用
        {
            auto c3_result = ct::c3::C3KernelRegistry::getInstance().tryExecute(OpType, a, b);
            if (c3_result.has_value()) {
                return c3_result.value();
            }
        }

        BinaryKernelFunc func = selectBestBinary(OpType, target_dev, binary_kernels_);
        if (func == nullptr) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API, "Ctorch_Scheduler: 没有可用的Kernel");
        }
        return func(a, b);
    }

    template <op OpType>
    inline Tensor dispatch(const Tensor& a) {
        DeviceType target_dev = a.device();
        UnaryKernelFunc func = selectBestUnary(OpType, target_dev, unary_kernels_);
        if (func == nullptr) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API, "Ctorch_Scheduler: 没有可用的Kernel");
        }
        return func(a);
    }
};
#endif //CTORCH_SCHEDULER_H
