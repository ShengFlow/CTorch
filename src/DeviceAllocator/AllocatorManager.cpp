#include "../../include/DeviceAllocator.h"

std::shared_ptr<DeviceAllocator> AllocatorManager::getAllocator(DeviceType device) {
    std::lock_guard<std::mutex> lock(_mtx);
    auto it = _allocators.find(device);
    if (it != _allocators.end()) {
        return it->second;
    }

    // 延迟初始化默认 allocator，避免 Storage 构造早于 CtorchScheduler 单例。
    if (device == DeviceType::kCPU) {
        auto alloc = std::make_shared<CPUAllocator>();
        _allocators[device] = alloc;
        return alloc;
    }

#ifdef __APPLE__
    if (device == DeviceType::kMPS) {
        auto alloc = std::make_shared<MPSAllocator>();
        _allocators[device] = alloc;
        return alloc;
    }
#endif

    return nullptr;
}
