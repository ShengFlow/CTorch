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

class CtorchScheduler{
private:
    CtorchScheduler();
    CtorchScheduler(const CtorchScheduler&);
    CtorchScheduler& operator=(const CtorchScheduler&) = delete;

    static constexpr size_t OP_COUNT = static_cast<size_t>(op::Sum) + 1;
    static constexpr size_t DEVICE_COUNT = static_cast<size_t>(DeviceType::kGENERAL) + 1;

    std::array<std::array<std::atomic<BinaryKernelFunc>, DEVICE_COUNT>, OP_COUNT> binary_kernels_{};
    std::array<std::array<std::atomic<UnaryKernelFunc>, DEVICE_COUNT>, OP_COUNT> unary_kernels_{};
    std::array<std::atomic<Tensor (*)(const Tensor&, int)>, DEVICE_COUNT> softmax_kernels_{};

    void initKernels();

    static BinaryKernelFunc selectBestBinary(op op_type, DeviceType dev,
        const std::array<std::array<std::atomic<BinaryKernelFunc>, DEVICE_COUNT>, OP_COUNT>& table);
    static UnaryKernelFunc selectBestUnary(op op_type, DeviceType dev,
        const std::array<std::array<std::atomic<UnaryKernelFunc>, DEVICE_COUNT>, OP_COUNT>& table);
public:
    static CtorchScheduler& getInstance() {
        static CtorchScheduler instance_;
        return instance_;
    }

    static bool isDeviceAvailable(DeviceType dev_type) {
        switch (dev_type) {
            case DeviceType::kCPU: return true;
            case DeviceType::kCUDA: return false;
            case DeviceType::kMPS: return false;
            case DeviceType::kAMX: return true;
            default: return false;
        }
    }

    static DeviceType getTargetDevice(const Tensor& a, const Tensor& b) {
        if (a.device() != b.device()) {
            CtorchError::log(ErrorLevel::ERROR,ErrorPlatform::kGENERAL,ErrorType::DEVICE_COMPAT,"Ctorch_Scheduler: Tensor不在同一平台");
        }
        return a.device();
    }

    BinaryKernelFunc get_binary_kernel(op op_type, DeviceType dev) const {
        return binary_kernels_[static_cast<size_t>(op_type)][static_cast<size_t>(dev)]
            .load(std::memory_order_acquire);
    }

    UnaryKernelFunc get_unary_kernel(op op_type, DeviceType dev) const {
        return unary_kernels_[static_cast<size_t>(op_type)][static_cast<size_t>(dev)]
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

    void replace_softmax_kernel(DeviceType dev, Tensor (*new_kernel)(const Tensor&, int)) {
        softmax_kernels_[static_cast<size_t>(dev)].store(new_kernel, std::memory_order_release);
    }

    Tensor dispatch(const Tensor& a, const Tensor& b, op op_type);
    Tensor dispatch(const Tensor& a, op op_type);
    Tensor dispatch_softmax(const Tensor& a, int dim = -1);
};
#endif //CTORCH_SCHEDULER_H
