/**
 * @file MPSBackend.h
 * @brief MPS 后端实现 — 包装 CtorchScheduler 的 MPS 调度
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 * @version v0.1 (Gen 2 BANT)
 *
 * MPSBackend 是 DeviceBackend 的 MPS（Apple Metal Performance Shaders）
 * 具体实现。与 CPUBackend 类似，executeKernel 转发给
 * CtorchScheduler::dispatch()，由调度器自动选择 MPS kernel。
 *
 * MPS 特有的职责：
 * 1. allocate → MPSAllocator（通过 AllocatorManager）
 * 2. synchronize → MPS_flush_wait(true) 确保 GPU 完成
 * 3. markBufferModified → MPS_markBufferModified() 通知 GPU 数据已更新
 *
 * serialize/deserialize 委托给 CDTF，CDTF 是后端无关的。
 */

#ifndef CTORCH_DISTRIBUTED_MPS_BACKEND_H
#define CTORCH_DISTRIBUTED_MPS_BACKEND_H

#include "DeviceBackend.h"
#include "CDTF.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace ct {
namespace distributed {

class MPSBackend : public DeviceBackend {
public:
    MPSBackend() = default;
    ~MPSBackend() override = default;

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
        return DeviceType::kMPS;
    }

    const char* name() const noexcept override {
        return "MPS";
    }

    // ======================= 同步 =======================

    /**
     * @brief 同步 MPS 后端，等待所有 GPU 命令完成
     *
     * 调用 MPS_flush_wait(true) 刷新并等待 command buffer 完成。
     * 在读取 MPS 张量数据前必须调用此方法。
     */
    void synchronize() override;

    /**
     * @brief 标记 MPS 缓冲区已修改
     *
     * 在 host 端写入 MPS buffer 后，必须调用此方法通知 GPU。
     * 内部调用 MPS_markBufferModified()。
     *
     * @param ptr 缓冲区指针
     * @param bytes 修改的字节数
     */
    void markBufferModified(void* ptr, size_t bytes) override;
};

} // namespace distributed
} // namespace ct

#endif // CTORCH_DISTRIBUTED_MPS_BACKEND_H