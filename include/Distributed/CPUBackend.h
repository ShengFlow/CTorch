/**
 * @file CPUBackend.h
 * @brief CPU 后端实现 — 包装 CtorchScheduler 的 CPU 调度
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 * @version v0.1 (Gen 2 BANT)
 *
 * CPUBackend 是 DeviceBackend 的 CPU 具体实现。
 * 它不重新实现 kernel，而是将 executeKernel 转发给
 * CtorchScheduler::dispatch()，从而继承 AMX→SIMD→BASIC 的
 * 自动优先级调度和 C3 JIT 热替换能力。
 *
 * allocate/deallocate 委托给 CPUAllocator。
 * serialize/deserialize 委托给 CDTF。
 */

#ifndef CTORCH_DISTRIBUTED_CPU_BACKEND_H
#define CTORCH_DISTRIBUTED_CPU_BACKEND_H

#include "DeviceBackend.h"
#include "CDTF.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace ct {
namespace distributed {

class CPUBackend : public DeviceBackend {
public:
    CPUBackend() = default;
    ~CPUBackend() override = default;

    // ======================= 内存管理 =======================

    void* allocate(size_t bytes) override;

    void deallocate(void* ptr) override;

    // ======================= 算子执行 =======================

    void executeKernel(::op op_type, const Tensor& a,
                       const Tensor& b, Tensor& out) override;

    // ======================= 序列化 =======================

    std::vector<uint8_t> serialize(const Tensor& t) override;

    Tensor deserialize(const std::vector<uint8_t>& data) override;

    // ======================= 能力查询 =======================

    BackendCapability capability() const override;

    DeviceType deviceType() const noexcept override {
        return DeviceType::kCPU;
    }

    const char* name() const noexcept override {
        return "CPU";
    }

    // ======================= 同步 =======================

    void synchronize() override {
        // CPU 操作是同步的，无需额外同步
    }

    void markBufferModified(void* ptr, size_t bytes) override {
        // CPU 内存是 cache-coherent 的，无需通知
        (void)ptr;
        (void)bytes;
    }
};

} // namespace distributed
} // namespace ct

#endif // CTORCH_DISTRIBUTED_CPU_BACKEND_H