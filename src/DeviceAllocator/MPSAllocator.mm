#include "../../include/DeviceAllocator.h"
#include "../../include/MetalDevice.h"
#include <unordered_map>
#include <mutex>

struct MPSAllocatorImpl {
    std::unordered_map<void*, id<MTLBuffer>> bufferMap;
    std::mutex mapMutex;
};

MPSAllocator::MPSAllocator() {
    _impl = new MPSAllocatorImpl();
}

MPSAllocator::~MPSAllocator() {
    delete static_cast<MPSAllocatorImpl*>(_impl);
}

void* MPSAllocator::allocate(size_t bytes, DeviceType device) {
    if (device != DeviceType::kMPS) {
        return nullptr;
    }
    id<MTLBuffer> buffer = MetalDevice::getInstance().allocateBuffer(bytes, MTLResourceStorageModeShared);
    void* ptr = [buffer contents];
    {
        std::lock_guard<std::mutex> lock(static_cast<MPSAllocatorImpl*>(_impl)->mapMutex);
        static_cast<MPSAllocatorImpl*>(_impl)->bufferMap[ptr] = buffer;
    }
    return ptr;
}

void MPSAllocator::deallocate(void* ptr, DeviceType device) {
    if (device != DeviceType::kMPS || ptr == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(static_cast<MPSAllocatorImpl*>(_impl)->mapMutex);
    static_cast<MPSAllocatorImpl*>(_impl)->bufferMap.erase(ptr);
}

void MPSAllocator::memcpy(void* dst, const void* src, size_t bytes,
                          DeviceType dst_device, DeviceType src_device) {
    std::memcpy(dst, src, bytes);
}

extern "C" id<MTLBuffer> MPS_getBuffer(void* ptr) {
    MPSAllocator* allocator = static_cast<MPSAllocator*>(
        AllocatorManager::getInstance().getAllocator(DeviceType::kMPS));
    if (!allocator) {
        return nil;
    }
    MPSAllocatorImpl* impl = static_cast<MPSAllocatorImpl*>(allocator->getImpl());
    std::lock_guard<std::mutex> lock(impl->mapMutex);
    
    auto it = impl->bufferMap.find(ptr);
    if (it != impl->bufferMap.end()) {
        return it->second;
    }
    
    static bool not_found = true;
    if (not_found) {
        not_found = false;
        std::cout << "[DEBUG] MPS_getBuffer: ptr=" << ptr << " not found! map size=" << impl->bufferMap.size() << std::endl;
        for (auto& pair : impl->bufferMap) {
            std::cout << "[DEBUG] MPS_getBuffer: key=" << pair.first << std::endl;
        }
    }
    
    return nil;
}

extern "C" void MPS_markBufferModified(void* ptr, size_t bytes) {
    MPSAllocator* allocator = static_cast<MPSAllocator*>(
        AllocatorManager::getInstance().getAllocator(DeviceType::kMPS));
    if (!allocator) {
        return;
    }
    MPSAllocatorImpl* impl = static_cast<MPSAllocatorImpl*>(allocator->getImpl());
    std::lock_guard<std::mutex> lock(impl->mapMutex);
    auto it = impl->bufferMap.find(ptr);
    if (it != impl->bufferMap.end()) {
        [it->second didModifyRange:NSMakeRange(0, bytes)];
    }
}