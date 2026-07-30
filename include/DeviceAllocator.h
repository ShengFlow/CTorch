#ifndef CTORCH_DEVICE_ALLOCATOR_H
#define CTORCH_DEVICE_ALLOCATOR_H

#include "Ctools.h"
#include "CtorchError.h"
#include <memory>
#include <unordered_map>
#include <mutex>
#include <cstdlib>
#include <cstring>

class DeviceAllocator {
public:
    virtual ~DeviceAllocator() = default;

    virtual void* allocate(size_t bytes, DeviceType device) = 0;
    virtual void deallocate(void* ptr, DeviceType device) = 0;

    virtual void memcpy(void* dst, const void* src, size_t bytes,
                        DeviceType dst_device, DeviceType src_device) = 0;
};

class CPUAllocator : public DeviceAllocator {
public:
    void* allocate(size_t bytes, DeviceType device) override {
        if (device != DeviceType::kCPU) {
            return nullptr;
        }
        return std::malloc(bytes);
    }

    void deallocate(void* ptr, DeviceType device) override {
        if (device != DeviceType::kCPU || ptr == nullptr) {
            return;
        }
        std::free(ptr);
    }

    void memcpy(void* dst, const void* src, size_t bytes,
                DeviceType dst_device, DeviceType src_device) override {
        if (dst_device != DeviceType::kCPU || src_device != DeviceType::kCPU) {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::PLATFORM_API,
                "CPUAllocator: Cross-device memcpy not supported");
        }
        std::memcpy(dst, src, bytes);
    }
};

class MPSAllocator : public DeviceAllocator {
private:
    void* _impl = nullptr;

public:
    MPSAllocator();
    ~MPSAllocator() override;

    void* allocate(size_t bytes, DeviceType device) override;
    void deallocate(void* ptr, DeviceType device) override;

    void memcpy(void* dst, const void* src, size_t bytes,
                DeviceType dst_device, DeviceType src_device) override;

    void* getImpl() { return _impl; }
};

class AllocatorManager {
private:
    std::unordered_map<DeviceType, std::unique_ptr<DeviceAllocator>> _allocators;
    mutable std::mutex _mtx;

    AllocatorManager() = default;

public:
    static AllocatorManager& getInstance() {
        static AllocatorManager instance;
        return instance;
    }

    AllocatorManager(const AllocatorManager&) = delete;
    AllocatorManager& operator=(const AllocatorManager&) = delete;

    void registerAllocator(DeviceType device, std::unique_ptr<DeviceAllocator> allocator) {
        std::lock_guard<std::mutex> lock(_mtx);
        _allocators[device] = std::move(allocator);
    }

    DeviceAllocator* getAllocator(DeviceType device);
};

#endif // CTORCH_DEVICE_ALLOCATOR_H