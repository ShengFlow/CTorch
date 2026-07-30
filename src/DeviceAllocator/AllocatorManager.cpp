#include "../../include/DeviceAllocator.h"

DeviceAllocator* AllocatorManager::getAllocator(DeviceType device) {
    std::lock_guard<std::mutex> lock(_mtx);
    auto it = _allocators.find(device);
    if (it != _allocators.end()) {
        return it->second.get();
    }

    // 延迟初始化默认 allocator，避免 Storage 构造早于 CtorchScheduler 单例。
    if (device == DeviceType::kCPU) {
        auto alloc = std::make_unique<CPUAllocator>();
        DeviceAllocator* raw = alloc.get();
        _allocators[device] = std::move(alloc);
        return raw;
    }

#ifdef __APPLE__
    if (device == DeviceType::kMPS) {
        auto alloc = std::make_unique<MPSAllocator>();
        DeviceAllocator* raw = alloc.get();
        _allocators[device] = std::move(alloc);
        return raw;
    }
#endif

    return nullptr;
}
