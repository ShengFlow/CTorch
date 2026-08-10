/**
 * @file CPUBackend.cpp
 * @brief CPU 后端实现
 * @author CTorch Agent (苏璃珞)
 * @date 2026/08/04
 *
 * CPUBackend 将 executeKernel 转发给 CtorchScheduler::dispatch()，
 * 从而自动继承 AMX→SIMD→BASIC 优先级调度和 C3 JIT 热替换能力。
 *
 * 内存管理委托给 CPUAllocator（通过 AllocatorManager 获取）。
 * 序列化委托给 CDTF。
 */

#include "Distributed/CPUBackend.h"
#include "CtorchScheduler.h"
#include "DeviceAllocator.h"

namespace ct {
namespace distributed {

// ======================= 内存管理 =======================

void* CPUBackend::allocate(size_t bytes) {
    auto allocator = AllocatorManager::getInstance().getAllocator(DeviceType::kCPU);
    if (!allocator) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::MEMORY,
            "CPUBackend: Failed to get CPU allocator");
        return nullptr;
    }
    return allocator->allocate(bytes, DeviceType::kCPU);
}

void CPUBackend::deallocate(void* ptr) {
    if (!ptr) return;
    auto allocator = AllocatorManager::getInstance().getAllocator(DeviceType::kCPU);
    if (!allocator) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::MEMORY,
            "CPUBackend: Failed to get CPU allocator for deallocation");
        return;
    }
    allocator->deallocate(ptr, DeviceType::kCPU);
}

// ======================= 算子执行 =======================

void CPUBackend::executeKernel(::op op_type, const Tensor& a,
                                const Tensor& b, Tensor& out) {
    auto& scheduler = CtorchScheduler::getInstance();

    // 判断是一元还是二元算子
    // 对于一元算子，b 是一个空的默认 Tensor（storage.size() == 0）
    // 注意：不能用 b.numel() == 0 判断，因为默认 Tensor() 的 numel() 是 1
    if (b.storage().size() == 0) {
        // 一元算子：dispatch(a, op_type)
        Tensor result = scheduler.dispatch(a, op_type);
        // 复制到输出
        out = std::move(result);
    } else {
        // 二元算子：dispatch(a, b, op_type)
        Tensor result = scheduler.dispatch(a, b, op_type);
        out = std::move(result);
    }
}

// ======================= 序列化 =======================

std::vector<uint8_t> CPUBackend::serialize(const Tensor& t) {
    return CDTF::serialize(t, CDTF_FLAG_NONE);
}

Tensor CPUBackend::deserialize(const std::vector<uint8_t>& data) {
    return CDTF::deserialize(data);
}

// ======================= 能力查询 =======================

BackendCapability CPUBackend::capability() const {
    BackendCapability cap;
    cap.device = DeviceType::kCPU;
    cap.compute_throughput = 0.5f;       // ~0.5 TFLOPS (Apple M1 基准)
    cap.memory_bandwidth = 50;            // ~50 GB/s
    cap.unified_memory = true;
    cap.numerical_precision = 1.0f;       // float32 基准精度
    cap.available_memory = 16ULL * 1024 * 1024 * 1024; // 假设 16GB
    cap.node_id = 0;
    cap.backend_name = "CPU";
    return cap;
}

} // namespace distributed
} // namespace ct