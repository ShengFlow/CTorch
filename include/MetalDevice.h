#ifndef CTORCH_METAL_DEVICE_H
#define CTORCH_METAL_DEVICE_H

#include <Metal/Metal.h>
#include <memory>
#include <mutex>

class MetalDevice {
private:
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    mutable std::mutex _mtx;

    MetalDevice();
    ~MetalDevice();

public:
    static MetalDevice& getInstance();

    MetalDevice(const MetalDevice&) = delete;
    MetalDevice& operator=(const MetalDevice&) = delete;

    id<MTLDevice> device() const;
    id<MTLCommandQueue> commandQueue() const;

    id<MTLBuffer> allocateBuffer(size_t size, MTLResourceOptions options = MTLResourceStorageModeShared);
    
    void executeCompute(id<MTLComputePipelineState> pipeline,
                        id<MTLBuffer>* buffers,
                        uint32_t bufferCount,
                        uint32_t width,
                        uint32_t height = 1,
                        uint32_t depth = 1);
};

#endif // CTORCH_METAL_DEVICE_H