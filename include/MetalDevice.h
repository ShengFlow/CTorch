#ifndef CTORCH_METAL_DEVICE_H
#define CTORCH_METAL_DEVICE_H

// [Fix] v0.5.2 Linux build: Metal framework 是 macOS 专属, Linux/DCU 节点编译会失败
// 整体 class 用 #ifdef __APPLE__ 守卫, 防止 Linux 编译时触发 Objective-C 类型 / Metal 头
#ifdef __APPLE__

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

#endif // __APPLE__

#endif // CTORCH_METAL_DEVICE_H