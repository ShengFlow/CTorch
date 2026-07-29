#include "../include/MetalDevice.h"
#include "../include/CtorchError.h"

MetalDevice::MetalDevice() {
    _device = MTLCreateSystemDefaultDevice();
    if (!_device) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API,
            "MetalDevice: 无法创建 Metal 设备");
    }
    
    _commandQueue = [_device newCommandQueue];
    if (!_commandQueue) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API,
            "MetalDevice: 无法创建命令队列");
    }
}

MetalDevice::~MetalDevice() {
    // ARC 自动管理
}

MetalDevice& MetalDevice::getInstance() {
    static MetalDevice instance;
    return instance;
}

id<MTLDevice> MetalDevice::device() const {
    return _device;
}

id<MTLCommandQueue> MetalDevice::commandQueue() const {
    return _commandQueue;
}

id<MTLBuffer> MetalDevice::allocateBuffer(size_t size, MTLResourceOptions options) {
    id<MTLBuffer> buffer = [_device newBufferWithLength:size options:options];
    if (!buffer) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API,
            "MetalDevice: 无法分配 Metal 缓冲区");
    }
    return buffer;
}

void MetalDevice::executeCompute(id<MTLComputePipelineState> pipeline,
                                 id<MTLBuffer>* buffers,
                                 uint32_t bufferCount,
                                 uint32_t width,
                                 uint32_t height,
                                 uint32_t depth) {
    id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipeline];
    
    for (uint32_t i = 0; i < bufferCount; ++i) {
        [encoder setBuffer:buffers[i] offset:0 atIndex:i];
    }
    
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    MTLSize gridSize = MTLSizeMake(width, height, depth);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
}