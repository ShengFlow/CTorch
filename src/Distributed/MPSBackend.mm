/**
 * @file MPSBackend.mm
 * @brief MPS 后端实现（Obj-C++，需 .mm 扩展名以支持 Metal 调用）
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 *
 * MPSBackend 将 executeKernel 转发给 CtorchScheduler::dispatch()，
 * 由调度器自动选择 MPS kernel。
 *
 * MPS 特有的职责：
 * 1. 内存管理通过 MPSAllocator（AllocatorManager 获取）
 * 2. synchronize() 调用 MPS_flush_wait(true) 等待 GPU 完成
 * 3. markBufferModified() 调用 MPS_markBufferModified() 通知 GPU
 *
 * 序列化委托给 CDTF（后端无关）。
 */

#include "Distributed/MPSBackend.h"
#include "CtorchScheduler.h"
#include "DeviceAllocator.h"

#include <iostream>

// MPS kernel dispatch 中声明的 extern "C" 函数
extern "C" void MPS_flush_wait(bool wait);
extern "C" void MPS_markBufferModified(void* ptr, size_t bytes);

namespace ct {
namespace distributed {

// ======================= 内存管理 =======================

void* MPSBackend::allocate(size_t bytes) {
    auto allocator = AllocatorManager::getInstance().getAllocator(DeviceType::kMPS);
    if (!allocator) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::MEMORY,
            "MPSBackend: Failed to get MPS allocator");
        return nullptr;
    }
    return allocator->allocate(bytes, DeviceType::kMPS);
}

void MPSBackend::deallocate(void* ptr) {
    if (!ptr) return;
    auto allocator = AllocatorManager::getInstance().getAllocator(DeviceType::kMPS);
    if (!allocator) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::MEMORY,
            "MPSBackend: Failed to get MPS allocator for deallocation");
        return;
    }
    allocator->deallocate(ptr, DeviceType::kMPS);
}

// ======================= 算子执行 =======================

void MPSBackend::executeKernel(::op op_type, const Tensor& a,
                                const Tensor& b, Tensor& out) {
    auto& scheduler = CtorchScheduler::getInstance();

    // 判断是一元还是二元算子
    // 注意：不能用 b.numel() == 0 判断，因为默认 Tensor() 的 numel() 是 1
    if (b.storage().size() == 0) {
        // 一元算子
        Tensor result = scheduler.dispatch(a, op_type);
        out = std::move(result);
    } else {
        // 二元算子
        Tensor result = scheduler.dispatch(a, b, op_type);
        out = std::move(result);
    }
}

// ======================= 序列化 =======================

std::vector<uint8_t> MPSBackend::serialize(const Tensor& t) {
    return CDTF::serialize(t, CDTF_FLAG_NONE);
}

Tensor MPSBackend::deserialize(const std::vector<uint8_t>& data) {
    return CDTF::deserialize(data);
}

// ======================= 能力查询 =======================

BackendCapability MPSBackend::capability() const {
    BackendCapability cap;
    cap.device = DeviceType::kMPS;
    cap.compute_throughput = 2.6f;       // ~2.6 TFLOPS (Apple M1 Pro)
    cap.memory_bandwidth = 200;           // ~200 GB/s (M1 Pro)
    cap.unified_memory = true;
    cap.numerical_precision = 1.0f;       // float32 基准精度
    cap.available_memory = 16ULL * 1024 * 1024 * 1024; // 假设 16GB 统一内存
    cap.node_id = 0;
    cap.backend_name = "MPS";
    return cap;
}

// ======================= 同步 =======================

void MPSBackend::synchronize() {
    MPS_flush_wait(true);
}

void MPSBackend::markBufferModified(void* ptr, size_t bytes) {
    MPS_markBufferModified(ptr, bytes);
}

} // namespace distributed
} // namespace ct