#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <cmath>

#ifdef __APPLE__
#import <Metal/Metal.h>
#endif

#include "./../../../include/Tensor.h"

static id<MTLDevice> _device = nil;
static id<MTLCommandQueue> _commandQueue = nil;
static id<MTLLibrary> _library = nil;
static std::unordered_map<std::string, id<MTLComputePipelineState>> _pipelines;
static std::mutex _pipelineMutex;

#if __has_feature(objc_arc)
#define CT_MPS_RETAIN(x) (x)
#define CT_MPS_RELEASE(x) ((void)0)
#else
#define CT_MPS_RETAIN(x) [x retain]
#define CT_MPS_RELEASE(x) [x release]
#endif

static id<MTLCommandBuffer> _updateCommandBuffer = nil;
static id<MTLComputeCommandEncoder> _updateEncoder = nil;

class CommandBufferAccumulator {
public:
    ~CommandBufferAccumulator() { flush(true); }
    
    void flush(bool wait = true) {
        if (_encoder != nil) {
            [_encoder endEncoding];
            _encoder = nil;
        }
        if (_commandBuffer != nil) {
            [_commandBuffer commit];
            if (wait) {
                [_commandBuffer waitUntilCompleted];
            }
            _commandBuffer = nil;
        }
    }
    
    id<MTLComputeCommandEncoder> getEncoder() {
        if (_commandBuffer == nil) {
            _commandBuffer = [_commandQueue commandBuffer];
        }
        if (_encoder == nil) {
            _encoder = [_commandBuffer computeCommandEncoder];
        }
        return _encoder;
    }
    
private:
    id<MTLCommandBuffer> _commandBuffer = nil;
    id<MTLComputeCommandEncoder> _encoder = nil;
};

static thread_local CommandBufferAccumulator _accumulator;

void MPS_flush(bool wait) {
    _accumulator.flush(wait);
}

extern "C" void MPS_flush_wait(bool wait) {
    _accumulator.flush(wait);
}

extern "C" void MPS_update_begin() {
    if (_updateCommandBuffer == nil) {
        _updateCommandBuffer = CT_MPS_RETAIN([_commandQueue commandBuffer]);
        _updateEncoder = CT_MPS_RETAIN([_updateCommandBuffer computeCommandEncoder]);
    }
}

extern "C" void MPS_update_end() {
    if (_updateEncoder != nil) {
        [_updateEncoder endEncoding];
        CT_MPS_RELEASE(_updateEncoder);
        _updateEncoder = nil;
    }
    if (_updateCommandBuffer != nil) {
        [_updateCommandBuffer commit];
        [_updateCommandBuffer waitUntilCompleted];
        CT_MPS_RELEASE(_updateCommandBuffer);
        _updateCommandBuffer = nil;
    }
}

static void computeBroadcastStrides(
    const std::vector<size_t>& shape,
    const std::vector<size_t>& target_shape,
    std::vector<size_t>& strides) {
    
    size_t dims = target_shape.size();
    strides.resize(dims);
    
    std::vector<size_t> padded_shape(dims, 1);
    size_t offset = dims - shape.size();
    for (size_t i = 0; i < shape.size(); ++i) {
        padded_shape[offset + i] = shape[i];
    }
    
    strides[dims - 1] = (padded_shape[dims - 1] == 1) ? 0 : 1;
    for (int i = dims - 2; i >= 0; --i) {
        if (padded_shape[i] == 1) {
            strides[i] = 0;
        } else {
            strides[i] = strides[i + 1] * padded_shape[i + 1];
        }
    }
}

static const char* metalSource = R"(
#include <metal_stdlib>
using namespace metal;

kernel void add_kernel(device float* a [[buffer(0)]], device float* b [[buffer(1)]], device float* result [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    result[id] = a[id] + b[id];
}

kernel void sub_kernel(device float* a [[buffer(0)]], device float* b [[buffer(1)]], device float* result [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    result[id] = a[id] - b[id];
}

kernel void mul_kernel(device float* a [[buffer(0)]], device float* b [[buffer(1)]], device float* result [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    result[id] = a[id] * b[id];
}

kernel void div_kernel(device float* a [[buffer(0)]], device float* b [[buffer(1)]], device float* result [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    result[id] = a[id] / b[id];
}

kernel void neg_kernel(device float* a [[buffer(0)]], device float* result [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    result[id] = -a[id];
}

kernel void relu_kernel(device float* a [[buffer(0)]], device float* result [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    result[id] = max(a[id], 0.0f);
}

kernel void tanh_kernel(device float* a [[buffer(0)]], device float* result [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    result[id] = tanh(a[id]);
}

kernel void sigmoid_kernel(device float* a [[buffer(0)]], device float* result [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    result[id] = 1.0f / (1.0f + exp(-a[id]));
}

kernel void gelu_kernel(device float* a [[buffer(0)]], device float* result [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    float x = a[id];
    float v = 0.7978845608f * (x + 0.044715f * x * x * x);
    result[id] = 0.5f * x * (1.0f + tanh(v));
}

kernel void log_kernel(device float* a [[buffer(0)]], device float* result [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    result[id] = log(a[id]);
}

kernel void exp_kernel(device float* a [[buffer(0)]], device float* result [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    result[id] = exp(a[id]);
}

kernel void abs_kernel(device float* a [[buffer(0)]], device float* result [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    result[id] = abs(a[id]);
}

kernel void sin_kernel(device float* a [[buffer(0)]], device float* result [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    result[id] = sin(a[id]);
}

kernel void cos_kernel(device float* a [[buffer(0)]], device float* result [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    result[id] = cos(a[id]);
}

kernel void min_kernel(device float* a [[buffer(0)]], device float* b [[buffer(1)]], device float* result [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    result[id] = min(a[id], b[id]);
}

kernel void max_kernel(device float* a [[buffer(0)]], device float* b [[buffer(1)]], device float* result [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    result[id] = max(a[id], b[id]);
}

kernel void zero_kernel(device float* result [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    result[id] = 0.0f;
}

kernel void sgd_step_kernel(device float* param [[buffer(0)]], device float* grad [[buffer(1)]], 
                            constant float* lr [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    param[id] -= grad[id] * lr[0];
}

kernel void sgd_step_zero_kernel(device float* param [[buffer(0)]], device float* grad [[buffer(1)]], 
                                 constant float* lr [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    param[id] -= grad[id] * lr[0];
    grad[id] = 0.0f;
}

kernel void softmax_kernel(device float* a [[buffer(0)]],
                           device float* result [[buffer(1)]],
                           uint id [[thread_position_in_grid]],
                           constant uint& batch_size [[buffer(2)]],
                           constant uint& hidden_size [[buffer(3)]]) {
    uint batch_idx = id / hidden_size;
    float max_val = a[batch_idx * hidden_size];
    for (uint i = 1; i < hidden_size; ++i) {
        max_val = max(max_val, a[batch_idx * hidden_size + i]);
    }
    float exp_val = exp(a[id] - max_val);
    float sum_exp = 0.0f;
    for (uint i = 0; i < hidden_size; ++i) {
        sum_exp += exp(a[batch_idx * hidden_size + i] - max_val);
    }
    result[id] = exp_val / sum_exp;
}

kernel void cross_entropy_kernel(device float* logits [[buffer(0)]],
                                 device float* targets [[buffer(1)]],
                                 device float* result [[buffer(2)]],
                                 uint id [[thread_position_in_grid]],
                                 constant uint& batch_size [[buffer(3)]],
                                 constant uint& num_classes [[buffer(4)]],
                                 constant uint& is_one_hot [[buffer(5)]]) {
    if (id >= batch_size) return;
    float loss = 0.0f;
    if (is_one_hot) {
        for (uint c = 0; c < num_classes; ++c) {
            float target = targets[id * num_classes + c];
            float logit = logits[id * num_classes + c];
            float max_val = logits[id * num_classes];
            for (uint i = 1; i < num_classes; ++i) {
                max_val = max(max_val, logits[id * num_classes + i]);
            }
            float sum_exp = 0.0f;
            for (uint i = 0; i < num_classes; ++i) {
                sum_exp += exp(logits[id * num_classes + i] - max_val);
            }
            float prob = exp(logit - max_val) / sum_exp;
            loss -= target * log(prob + 1e-10f);
        }
    } else {
        uint target_class = (uint)targets[id];
        float max_val = logits[id * num_classes];
        for (uint i = 1; i < num_classes; ++i) {
            max_val = max(max_val, logits[id * num_classes + i]);
        }
        float sum_exp = 0.0f;
        for (uint i = 0; i < num_classes; ++i) {
            sum_exp += exp(logits[id * num_classes + i] - max_val);
        }
        float prob = exp(logits[id * num_classes + target_class] - max_val) / sum_exp;
        loss = -log(prob + 1e-10f);
    }
    result[id] = loss;
}

kernel void matmul_kernel(device float* a [[buffer(0)]],
                          device float* b [[buffer(1)]],
                          device float* result [[buffer(2)]],
                          uint3 pos [[thread_position_in_grid]],
                          constant uint& m [[buffer(3)]],
                          constant uint& k [[buffer(4)]],
                          constant uint& n [[buffer(5)]],
                          constant uint& a_stride0 [[buffer(6)]],
                          constant uint& a_stride1 [[buffer(7)]],
                          constant uint& b_stride0 [[buffer(8)]],
                          constant uint& b_stride1 [[buffer(9)]],
                          constant uint& result_stride0 [[buffer(10)]],
                          constant uint& result_stride1 [[buffer(11)]]) {
    uint row = pos.y;
    uint col = pos.x;

    if (row >= m || col >= n) return;

    float sum = 0.0f;
    for (uint i = 0; i < k; ++i) {
        sum += a[row * a_stride0 + i * a_stride1] * b[i * b_stride0 + col * b_stride1];
    }
    result[row * result_stride0 + col * result_stride1] = sum;
}

kernel void dot_kernel(device float* a [[buffer(0)]],
                       device float* b [[buffer(1)]],
                       device float* result [[buffer(2)]],
                       uint id [[thread_position_in_grid]],
                       constant uint& n [[buffer(3)]]) {
    if (id != 0) return;
    float sum = 0.0f;
    for (uint i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    result[0] = sum;
}

kernel void mse_kernel(device float* a [[buffer(0)]],
                       device float* b [[buffer(1)]],
                       device float* result [[buffer(2)]],
                       uint id [[thread_position_in_grid]],
                       constant uint& n [[buffer(3)]]) {
    if (id >= n) return;
    float diff = a[id] - b[id];
    result[id] = diff * diff;
}

kernel void mae_kernel(device float* a [[buffer(0)]],
                       device float* b [[buffer(1)]],
                       device float* result [[buffer(2)]],
                       uint id [[thread_position_in_grid]],
                       constant uint& n [[buffer(3)]]) {
    if (id >= n) return;
    result[id] = abs(a[id] - b[id]);
}

kernel void broadcast_add_kernel(device float* a [[buffer(0)]],
                                 device float* b [[buffer(1)]],
                                 device float* result [[buffer(2)]],
                                 uint id [[thread_position_in_grid]],
                                 constant uint& n [[buffer(3)]],
                                 constant uint& a_stride [[buffer(4)]],
                                 constant uint& b_stride [[buffer(5)]]) {
    result[id] = a[id * a_stride] + b[id * b_stride];
}

kernel void broadcast_sub_kernel(device float* a [[buffer(0)]],
                                 device float* b [[buffer(1)]],
                                 device float* result [[buffer(2)]],
                                 uint id [[thread_position_in_grid]],
                                 constant uint& n [[buffer(3)]],
                                 constant uint& a_stride [[buffer(4)]],
                                 constant uint& b_stride [[buffer(5)]]) {
    result[id] = a[id * a_stride] - b[id * b_stride];
}

kernel void broadcast_mul_kernel(device float* a [[buffer(0)]],
                                 device float* b [[buffer(1)]],
                                 device float* result [[buffer(2)]],
                                 uint id [[thread_position_in_grid]],
                                 constant uint& n [[buffer(3)]],
                                 constant uint& a_stride [[buffer(4)]],
                                 constant uint& b_stride [[buffer(5)]]) {
    result[id] = a[id * a_stride] * b[id * b_stride];
}

kernel void broadcast_div_kernel(device float* a [[buffer(0)]],
                                 device float* b [[buffer(1)]],
                                 device float* result [[buffer(2)]],
                                 uint id [[thread_position_in_grid]],
                                 constant uint& n [[buffer(3)]],
                                 constant uint& a_stride [[buffer(4)]],
                                 constant uint& b_stride [[buffer(5)]]) {
    result[id] = a[id * a_stride] / b[id * b_stride];
}
)";

static id<MTLComputePipelineState> getPipeline(const std::string& kernelName) {
    std::lock_guard<std::mutex> lock(_pipelineMutex);
    
    auto it = _pipelines.find(kernelName);
    if (it != _pipelines.end()) {
        return it->second;
    }
    
    if (_library == nil) {
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:metalSource];
        _library = [_device newLibraryWithSource:source options:nil error:&error];
        
        if (!_library) {
            NSLog(@"Error compiling shader library: %@", error);
            return nil;
        }
    }
    
    id<MTLFunction> function = [_library newFunctionWithName:[NSString stringWithUTF8String:kernelName.c_str()]];
    if (!function) {
        NSLog(@"Error loading function: %s", kernelName.c_str());
        return nil;
    }
    
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [_device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        NSLog(@"Error creating pipeline state: %@", error);
        return nil;
    }
    
    _pipelines[kernelName] = pipeline;
    return pipeline;
}

static void initMetal() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        _device = MTLCreateSystemDefaultDevice();
        if (!_device) {
            NSLog(@"Metal device not found");
            return;
        }
        
        _commandQueue = [_device newCommandQueue];
        if (!_commandQueue) {
            NSLog(@"Failed to create command queue");
            return;
        }
    });
}

extern "C" id<MTLBuffer> MPS_getBuffer(void* ptr);

CT_HOT Tensor Add_MPS_kernel(const Tensor& a, const Tensor& b) {
    if (a.device() != DeviceType::kMPS || b.device() != DeviceType::kMPS) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS Add_Kernel: 仅在MPS支持");
    }
    if (a.dtype() != b.dtype()) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "MPS Add_Kernel: Tensor数据类型不匹配");
    }

    if (a.sizes() == b.sizes()) {
        size_t elem_count = a.numel();
        Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
        
        id<MTLBuffer> a_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a.data<float>())));
        id<MTLBuffer> b_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(b.data<float>())));
        id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
        
        if (!a_buffer || !b_buffer || !result_buffer) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Add_Kernel: 无法获取Metal Buffer");
        }
        
        initMetal();
        id<MTLComputePipelineState> pipeline = getPipeline("add_kernel");
        if (!pipeline) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Add_Kernel: 无法获取pipeline");
        }
        
        id<MTLComputeCommandEncoder> encoder = _accumulator.getEncoder();
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        [encoder setBuffer:b_buffer offset:0 atIndex:1];
        [encoder setBuffer:result_buffer offset:0 atIndex:2];
        
        MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
        MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        
        return result;
    }
    
    std::vector<size_t> broadcast_shape;
    int max_dims = std::max(a.dim(), b.dim());
    broadcast_shape.resize(max_dims, 1);
    
    for (int i = 0; i < a.dim(); ++i) {
        broadcast_shape[max_dims - a.dim() + i] = std::max(broadcast_shape[max_dims - a.dim() + i], a.size(i));
    }
    for (int i = 0; i < b.dim(); ++i) {
        broadcast_shape[max_dims - b.dim() + i] = std::max(broadcast_shape[max_dims - b.dim() + i], b.size(i));
    }
    
    size_t elem_count = 1;
    for (size_t s : broadcast_shape) elem_count *= s;
    
    Tensor result(ShapeTag{}, broadcast_shape, a.dtype(), a.device());
    id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
    
    Tensor a_broadcast = a;
    Tensor b_broadcast = b;
    
    if (a.sizes() != broadcast_shape) {
        a_broadcast = Tensor(ShapeTag{}, broadcast_shape, a.dtype(), a.device());
        float* ab_data = a_broadcast.data<float>();
        const float* a_data = a.data<float>();
        std::vector<size_t> a_strides;
        computeBroadcastStrides(a.sizes(), broadcast_shape, a_strides);
        for (size_t i = 0; i < elem_count; ++i) {
            size_t idx = 0;
            size_t temp = i;
            for (int j = max_dims - 1; j >= 0; --j) {
                idx += (temp % broadcast_shape[j]) * a_strides[j];
                temp /= broadcast_shape[j];
            }
            ab_data[i] = a_data[idx];
        }
    }
    
    if (b.sizes() != broadcast_shape) {
        b_broadcast = Tensor(ShapeTag{}, broadcast_shape, b.dtype(), b.device());
        float* bb_data = b_broadcast.data<float>();
        const float* b_data = b.data<float>();
        std::vector<size_t> b_strides;
        computeBroadcastStrides(b.sizes(), broadcast_shape, b_strides);
        for (size_t i = 0; i < elem_count; ++i) {
            size_t idx = 0;
            size_t temp = i;
            for (int j = max_dims - 1; j >= 0; --j) {
                idx += (temp % broadcast_shape[j]) * b_strides[j];
                temp /= broadcast_shape[j];
            }
            bb_data[i] = b_data[idx];
        }
    }
    
    id<MTLBuffer> ab_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a_broadcast.data<float>())));
    id<MTLBuffer> bb_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(b_broadcast.data<float>())));
    
    if (!ab_buffer || !bb_buffer) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Add_Kernel: 无法获取broadcast Metal Buffer");
    }
    
    initMetal();
    id<MTLComputePipelineState> pipeline = getPipeline("add_kernel");
    if (!pipeline) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Add_Kernel: 无法获取pipeline");
    }
    
    id<MTLComputeCommandEncoder> encoder = _accumulator.getEncoder();
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:ab_buffer offset:0 atIndex:0];
    [encoder setBuffer:bb_buffer offset:0 atIndex:1];
    [encoder setBuffer:result_buffer offset:0 atIndex:2];
    
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    return result;
}

CT_HOT Tensor Sub_MPS_kernel(const Tensor& a, const Tensor& b) {
    if (a.device() != DeviceType::kMPS || b.device() != DeviceType::kMPS) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS Sub_Kernel: 仅在MPS支持");
    }
    if (a.dtype() != b.dtype()) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "MPS Sub_Kernel: Tensor数据类型不匹配");
    }

    if (a.sizes() == b.sizes()) {
        size_t elem_count = a.numel();
        Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
        
        id<MTLBuffer> a_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a.data<float>())));
        id<MTLBuffer> b_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(b.data<float>())));
        id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
        
        if (!a_buffer || !b_buffer || !result_buffer) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Sub_Kernel: 无法获取Metal Buffer");
        }
        
        initMetal();
        id<MTLComputePipelineState> pipeline = getPipeline("sub_kernel");
        if (!pipeline) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Sub_Kernel: 无法获取pipeline");
        }
        
        id<MTLComputeCommandEncoder> encoder = _accumulator.getEncoder();
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        [encoder setBuffer:b_buffer offset:0 atIndex:1];
        [encoder setBuffer:result_buffer offset:0 atIndex:2];
        
        MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
        MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        
        return result;
    }
    
    std::vector<size_t> broadcast_shape;
    int max_dims = std::max(a.dim(), b.dim());
    broadcast_shape.resize(max_dims, 1);
    
    for (int i = 0; i < a.dim(); ++i) {
        broadcast_shape[max_dims - a.dim() + i] = std::max(broadcast_shape[max_dims - a.dim() + i], a.size(i));
    }
    for (int i = 0; i < b.dim(); ++i) {
        broadcast_shape[max_dims - b.dim() + i] = std::max(broadcast_shape[max_dims - b.dim() + i], b.size(i));
    }
    
    size_t elem_count = 1;
    for (size_t s : broadcast_shape) elem_count *= s;
    
    Tensor result(ShapeTag{}, broadcast_shape, a.dtype(), a.device());
    id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
    
    Tensor a_broadcast = a;
    Tensor b_broadcast = b;
    
    if (a.sizes() != broadcast_shape) {
        a_broadcast = Tensor(ShapeTag{}, broadcast_shape, a.dtype(), a.device());
        float* ab_data = a_broadcast.data<float>();
        const float* a_data = a.data<float>();
        std::vector<size_t> a_strides;
        computeBroadcastStrides(a.sizes(), broadcast_shape, a_strides);
        for (size_t i = 0; i < elem_count; ++i) {
            size_t idx = 0;
            size_t temp = i;
            for (int j = max_dims - 1; j >= 0; --j) {
                idx += (temp % broadcast_shape[j]) * a_strides[j];
                temp /= broadcast_shape[j];
            }
            ab_data[i] = a_data[idx];
        }
    }
    
    if (b.sizes() != broadcast_shape) {
        b_broadcast = Tensor(ShapeTag{}, broadcast_shape, b.dtype(), b.device());
        float* bb_data = b_broadcast.data<float>();
        const float* b_data = b.data<float>();
        std::vector<size_t> b_strides;
        computeBroadcastStrides(b.sizes(), broadcast_shape, b_strides);
        for (size_t i = 0; i < elem_count; ++i) {
            size_t idx = 0;
            size_t temp = i;
            for (int j = max_dims - 1; j >= 0; --j) {
                idx += (temp % broadcast_shape[j]) * b_strides[j];
                temp /= broadcast_shape[j];
            }
            bb_data[i] = b_data[idx];
        }
    }
    
    id<MTLBuffer> ab_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a_broadcast.data<float>())));
    id<MTLBuffer> bb_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(b_broadcast.data<float>())));
    
    if (!ab_buffer || !bb_buffer) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Sub_Kernel: 无法获取broadcast Metal Buffer");
    }
    
    initMetal();
    id<MTLComputePipelineState> pipeline = getPipeline("sub_kernel");
    if (!pipeline) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Sub_Kernel: 无法获取pipeline");
    }
    
    id<MTLComputeCommandEncoder> encoder = _accumulator.getEncoder();
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:ab_buffer offset:0 atIndex:0];
    [encoder setBuffer:bb_buffer offset:0 atIndex:1];
    [encoder setBuffer:result_buffer offset:0 atIndex:2];
    
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    return result;
}

CT_HOT Tensor Mul_MPS_kernel(const Tensor& a, const Tensor& b) {
    if (a.device() != DeviceType::kMPS || b.device() != DeviceType::kMPS) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS Mul_Kernel: 仅在MPS支持");
    }
    if (a.dtype() != b.dtype()) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "MPS Mul_Kernel: Tensor数据类型不匹配");
    }

    if (a.sizes() == b.sizes()) {
        size_t elem_count = a.numel();
        Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
        
        id<MTLBuffer> a_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a.data<float>())));
        id<MTLBuffer> b_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(b.data<float>())));
        id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
        
        if (!a_buffer || !b_buffer || !result_buffer) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Mul_Kernel: 无法获取Metal Buffer");
        }
        
        initMetal();
        id<MTLComputePipelineState> pipeline = getPipeline("mul_kernel");
        if (!pipeline) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Mul_Kernel: 无法获取pipeline");
        }
        
        id<MTLComputeCommandEncoder> encoder = _accumulator.getEncoder();
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        [encoder setBuffer:b_buffer offset:0 atIndex:1];
        [encoder setBuffer:result_buffer offset:0 atIndex:2];
        
        MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
        MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        
        return result;
    }
    
    std::vector<size_t> broadcast_shape;
    int max_dims = std::max(a.dim(), b.dim());
    broadcast_shape.resize(max_dims, 1);
    
    for (int i = 0; i < a.dim(); ++i) {
        broadcast_shape[max_dims - a.dim() + i] = std::max(broadcast_shape[max_dims - a.dim() + i], a.size(i));
    }
    for (int i = 0; i < b.dim(); ++i) {
        broadcast_shape[max_dims - b.dim() + i] = std::max(broadcast_shape[max_dims - b.dim() + i], b.size(i));
    }
    
    size_t elem_count = 1;
    for (size_t s : broadcast_shape) elem_count *= s;
    
    Tensor result(ShapeTag{}, broadcast_shape, a.dtype(), a.device());
    id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
    
    Tensor a_broadcast = a;
    Tensor b_broadcast = b;
    
    if (a.sizes() != broadcast_shape) {
        a_broadcast = Tensor(ShapeTag{}, broadcast_shape, a.dtype(), a.device());
        float* ab_data = a_broadcast.data<float>();
        const float* a_data = a.data<float>();
        std::vector<size_t> a_strides;
        computeBroadcastStrides(a.sizes(), broadcast_shape, a_strides);
        for (size_t i = 0; i < elem_count; ++i) {
            size_t idx = 0;
            size_t temp = i;
            for (int j = max_dims - 1; j >= 0; --j) {
                idx += (temp % broadcast_shape[j]) * a_strides[j];
                temp /= broadcast_shape[j];
            }
            ab_data[i] = a_data[idx];
        }
    }
    
    if (b.sizes() != broadcast_shape) {
        b_broadcast = Tensor(ShapeTag{}, broadcast_shape, b.dtype(), b.device());
        float* bb_data = b_broadcast.data<float>();
        const float* b_data = b.data<float>();
        std::vector<size_t> b_strides;
        computeBroadcastStrides(b.sizes(), broadcast_shape, b_strides);
        for (size_t i = 0; i < elem_count; ++i) {
            size_t idx = 0;
            size_t temp = i;
            for (int j = max_dims - 1; j >= 0; --j) {
                idx += (temp % broadcast_shape[j]) * b_strides[j];
                temp /= broadcast_shape[j];
            }
            bb_data[i] = b_data[idx];
        }
    }
    
    id<MTLBuffer> ab_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a_broadcast.data<float>())));
    id<MTLBuffer> bb_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(b_broadcast.data<float>())));
    
    if (!ab_buffer || !bb_buffer) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Mul_Kernel: 无法获取broadcast Metal Buffer");
    }
    
    initMetal();
    id<MTLComputePipelineState> pipeline = getPipeline("mul_kernel");
    if (!pipeline) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Mul_Kernel: 无法获取pipeline");
    }
    
    id<MTLComputeCommandEncoder> encoder = _accumulator.getEncoder();
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:ab_buffer offset:0 atIndex:0];
    [encoder setBuffer:bb_buffer offset:0 atIndex:1];
    [encoder setBuffer:result_buffer offset:0 atIndex:2];
    
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    return result;
}

CT_HOT Tensor Div_MPS_kernel(const Tensor& a, const Tensor& b) {
    if (a.device() != DeviceType::kMPS || b.device() != DeviceType::kMPS) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS Div_Kernel: 仅在MPS支持");
    }
    if (a.dtype() != b.dtype()) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "MPS Div_Kernel: Tensor数据类型不匹配");
    }

    if (a.sizes() == b.sizes()) {
        size_t elem_count = a.numel();
        Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
        
        id<MTLBuffer> a_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a.data<float>())));
        id<MTLBuffer> b_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(b.data<float>())));
        id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
        
        if (!a_buffer || !b_buffer || !result_buffer) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Div_Kernel: 无法获取Metal Buffer");
        }
        
        initMetal();
        id<MTLComputePipelineState> pipeline = getPipeline("div_kernel");
        if (!pipeline) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Div_Kernel: 无法获取pipeline");
        }
        
        id<MTLComputeCommandEncoder> encoder = _accumulator.getEncoder();
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        [encoder setBuffer:b_buffer offset:0 atIndex:1];
        [encoder setBuffer:result_buffer offset:0 atIndex:2];
        
        MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
        MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        
        return result;
    }
    
    std::vector<size_t> broadcast_shape;
    int max_dims = std::max(a.dim(), b.dim());
    broadcast_shape.resize(max_dims, 1);
    
    for (int i = 0; i < a.dim(); ++i) {
        broadcast_shape[max_dims - a.dim() + i] = std::max(broadcast_shape[max_dims - a.dim() + i], a.size(i));
    }
    for (int i = 0; i < b.dim(); ++i) {
        broadcast_shape[max_dims - b.dim() + i] = std::max(broadcast_shape[max_dims - b.dim() + i], b.size(i));
    }
    
    size_t elem_count = 1;
    for (size_t s : broadcast_shape) elem_count *= s;
    
    Tensor result(ShapeTag{}, broadcast_shape, a.dtype(), a.device());
    id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
    
    Tensor a_broadcast = a;
    Tensor b_broadcast = b;
    
    if (a.sizes() != broadcast_shape) {
        a_broadcast = Tensor(ShapeTag{}, broadcast_shape, a.dtype(), a.device());
        float* ab_data = a_broadcast.data<float>();
        const float* a_data = a.data<float>();
        std::vector<size_t> a_strides;
        computeBroadcastStrides(a.sizes(), broadcast_shape, a_strides);
        for (size_t i = 0; i < elem_count; ++i) {
            size_t idx = 0;
            size_t temp = i;
            for (int j = max_dims - 1; j >= 0; --j) {
                idx += (temp % broadcast_shape[j]) * a_strides[j];
                temp /= broadcast_shape[j];
            }
            ab_data[i] = a_data[idx];
        }
    }
    
    if (b.sizes() != broadcast_shape) {
        b_broadcast = Tensor(ShapeTag{}, broadcast_shape, b.dtype(), b.device());
        float* bb_data = b_broadcast.data<float>();
        const float* b_data = b.data<float>();
        std::vector<size_t> b_strides;
        computeBroadcastStrides(b.sizes(), broadcast_shape, b_strides);
        for (size_t i = 0; i < elem_count; ++i) {
            size_t idx = 0;
            size_t temp = i;
            for (int j = max_dims - 1; j >= 0; --j) {
                idx += (temp % broadcast_shape[j]) * b_strides[j];
                temp /= broadcast_shape[j];
            }
            bb_data[i] = b_data[idx];
        }
    }
    
    id<MTLBuffer> ab_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a_broadcast.data<float>())));
    id<MTLBuffer> bb_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(b_broadcast.data<float>())));
    
    if (!ab_buffer || !bb_buffer) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Div_Kernel: 无法获取broadcast Metal Buffer");
    }
    
    initMetal();
    id<MTLComputePipelineState> pipeline = getPipeline("div_kernel");
    if (!pipeline) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Div_Kernel: 无法获取pipeline");
    }
    
    id<MTLComputeCommandEncoder> encoder = _accumulator.getEncoder();
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:ab_buffer offset:0 atIndex:0];
    [encoder setBuffer:bb_buffer offset:0 atIndex:1];
    [encoder setBuffer:result_buffer offset:0 atIndex:2];
    
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    return result;
}

CT_HOT Tensor Neg_MPS_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kMPS) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS Neg_Kernel: 仅在MPS支持");
    }
    
    size_t elem_count = a.numel();
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
    
    id<MTLBuffer> a_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a.data<float>())));
    id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
    
    if (!a_buffer || !result_buffer) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Neg_Kernel: 无法获取Metal Buffer");
    }
    
    initMetal();
    id<MTLComputePipelineState> pipeline = getPipeline("neg_kernel");
    if (!pipeline) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Neg_Kernel: 无法获取pipeline");
    }
    
    id<MTLComputeCommandEncoder> encoder = _accumulator.getEncoder();
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a_buffer offset:0 atIndex:0];
    [encoder setBuffer:result_buffer offset:0 atIndex:1];
    
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    return result;
}

CT_HOT Tensor ReLU_MPS_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kMPS) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS ReLU_Kernel: 仅在MPS支持");
    }
    
    size_t elem_count = a.numel();
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
    
    id<MTLBuffer> a_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a.data<float>())));
    id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
    
    if (!a_buffer || !result_buffer) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS ReLU_Kernel: 无法获取Metal Buffer");
    }
    
    initMetal();
    id<MTLComputePipelineState> pipeline = getPipeline("relu_kernel");
    if (!pipeline) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS ReLU_Kernel: 无法获取pipeline");
    }
    
    id<MTLComputeCommandEncoder> encoder = _accumulator.getEncoder();
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a_buffer offset:0 atIndex:0];
    [encoder setBuffer:result_buffer offset:0 atIndex:1];
    
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    return result;
}

CT_HOT Tensor Sin_MPS_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kMPS) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS Sin_Kernel: 仅在MPS支持");
    }
    
    size_t elem_count = a.numel();
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
    
    id<MTLBuffer> a_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a.data<float>())));
    id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
    
    if (!a_buffer || !result_buffer) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Sin_Kernel: 无法获取Metal Buffer");
    }
    
    initMetal();
    id<MTLComputePipelineState> pipeline = getPipeline("sin_kernel");
    if (!pipeline) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Sin_Kernel: 无法获取pipeline");
    }
    
    id<MTLComputeCommandEncoder> encoder = _accumulator.getEncoder();
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a_buffer offset:0 atIndex:0];
    [encoder setBuffer:result_buffer offset:0 atIndex:1];
    
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    return result;
}

CT_HOT Tensor Cos_MPS_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kMPS) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS Cos_Kernel: 仅在MPS支持");
    }
    
    size_t elem_count = a.numel();
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
    
    id<MTLBuffer> a_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a.data<float>())));
    id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
    
    if (!a_buffer || !result_buffer) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Cos_Kernel: 无法获取Metal Buffer");
    }
    
    initMetal();
    id<MTLComputePipelineState> pipeline = getPipeline("cos_kernel");
    if (!pipeline) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Cos_Kernel: 无法获取pipeline");
    }
    
    id<MTLComputeCommandEncoder> encoder = _accumulator.getEncoder();
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a_buffer offset:0 atIndex:0];
    [encoder setBuffer:result_buffer offset:0 atIndex:1];
    
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    return result;
}

CT_HOT Tensor Tanh_MPS_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kMPS) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS Tanh_Kernel: 仅在MPS支持");
    }
    
    size_t elem_count = a.numel();
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
    
    id<MTLBuffer> a_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a.data<float>())));
    id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
    
    if (!a_buffer || !result_buffer) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Tanh_Kernel: 无法获取Metal Buffer");
    }
    
    initMetal();
    id<MTLComputePipelineState> pipeline = getPipeline("tanh_kernel");
    if (!pipeline) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Tanh_Kernel: 无法获取pipeline");
    }
    
    id<MTLComputeCommandEncoder> encoder = _accumulator.getEncoder();
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a_buffer offset:0 atIndex:0];
    [encoder setBuffer:result_buffer offset:0 atIndex:1];
    
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    return result;
}

CT_HOT Tensor Sigmoid_MPS_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kMPS) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS Sigmoid_Kernel: 仅在MPS支持");
    }
    
    size_t elem_count = a.numel();
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
    
    id<MTLBuffer> a_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a.data<float>())));
    id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
    
    if (!a_buffer || !result_buffer) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Sigmoid_Kernel: 无法获取Metal Buffer");
    }
    
    initMetal();
    id<MTLComputePipelineState> pipeline = getPipeline("sigmoid_kernel");
    if (!pipeline) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Sigmoid_Kernel: 无法获取pipeline");
    }
    
    id<MTLComputeCommandEncoder> encoder = _accumulator.getEncoder();
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a_buffer offset:0 atIndex:0];
    [encoder setBuffer:result_buffer offset:0 atIndex:1];
    
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    return result;
}

CT_HOT Tensor GELU_MPS_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kMPS) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS GELU_Kernel: 仅在MPS支持");
    }
    
    size_t elem_count = a.numel();
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
    
    id<MTLBuffer> a_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a.data<float>())));
    id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
    
    if (!a_buffer || !result_buffer) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS GELU_Kernel: 无法获取Metal Buffer");
    }
    
    initMetal();
    id<MTLComputePipelineState> pipeline = getPipeline("gelu_kernel");
    if (!pipeline) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS GELU_Kernel: 无法获取pipeline");
    }
    
    id<MTLComputeCommandEncoder> encoder = _accumulator.getEncoder();
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a_buffer offset:0 atIndex:0];
    [encoder setBuffer:result_buffer offset:0 atIndex:1];
    
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    return result;
}

CT_HOT Tensor MatMul_MPS_kernel(const Tensor& a, const Tensor& b) {
    @autoreleasepool {
        if (a.device() != DeviceType::kMPS || b.device() != DeviceType::kMPS) [[unlikely]] {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS MatMul_Kernel: 仅在MPS支持");
        }
        if (a.dtype() != b.dtype()) [[unlikely]] {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "MPS MatMul_Kernel: Tensor数据类型不匹配");
        }

        if (a.dim() != 2 || b.dim() != 2) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DATATYPE, "MPS MatMul_Kernel: 仅支持2D矩阵");
        }

        size_t M = a.size(0);
        size_t K = a.size(1);
        size_t N = b.size(1);

        if (K != b.size(0)) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DATATYPE, "MPS MatMul_Kernel: 矩阵维度不匹配");
        }

        Tensor result(ShapeTag{}, {M, N}, a.dtype(), a.device());

        id<MTLBuffer> a_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a.data<float>())));
        id<MTLBuffer> b_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(b.data<float>())));
        id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));

        if (!a_buffer || !b_buffer || !result_buffer) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS MatMul_Kernel: 无法获取Metal Buffer");
        }

        initMetal();
        id<MTLComputePipelineState> pipeline = getPipeline("matmul_kernel");
        if (!pipeline) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS MatMul_Kernel: 无法获取pipeline");
        }

        uint32_t m_val = static_cast<uint32_t>(M);
        uint32_t k_val = static_cast<uint32_t>(K);
        uint32_t n_val = static_cast<uint32_t>(N);
        uint32_t a_stride0_val = static_cast<uint32_t>(a.strides()[0]);
        uint32_t a_stride1_val = static_cast<uint32_t>(a.strides()[1]);
        uint32_t b_stride0_val = static_cast<uint32_t>(b.strides()[0]);
        uint32_t b_stride1_val = static_cast<uint32_t>(b.strides()[1]);
        uint32_t result_stride0_val = static_cast<uint32_t>(result.strides()[0]);
        uint32_t result_stride1_val = static_cast<uint32_t>(result.strides()[1]);

        id<MTLBuffer> m_buffer = [_device newBufferWithBytes:&m_val length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> k_buffer = [_device newBufferWithBytes:&k_val length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> n_buffer = [_device newBufferWithBytes:&n_val length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> a_stride0_buffer = [_device newBufferWithBytes:&a_stride0_val length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> a_stride1_buffer = [_device newBufferWithBytes:&a_stride1_val length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_stride0_buffer = [_device newBufferWithBytes:&b_stride0_val length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_stride1_buffer = [_device newBufferWithBytes:&b_stride1_val length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> result_stride0_buffer = [_device newBufferWithBytes:&result_stride0_val length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> result_stride1_buffer = [_device newBufferWithBytes:&result_stride1_val length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        MPS_flush(true);

        id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        [encoder setBuffer:b_buffer offset:0 atIndex:1];
        [encoder setBuffer:result_buffer offset:0 atIndex:2];
        [encoder setBuffer:m_buffer offset:0 atIndex:3];
        [encoder setBuffer:k_buffer offset:0 atIndex:4];
        [encoder setBuffer:n_buffer offset:0 atIndex:5];
        [encoder setBuffer:a_stride0_buffer offset:0 atIndex:6];
        [encoder setBuffer:a_stride1_buffer offset:0 atIndex:7];
        [encoder setBuffer:b_stride0_buffer offset:0 atIndex:8];
        [encoder setBuffer:b_stride1_buffer offset:0 atIndex:9];
        [encoder setBuffer:result_stride0_buffer offset:0 atIndex:10];
        [encoder setBuffer:result_stride1_buffer offset:0 atIndex:11];

        MTLSize threadGroupSize = MTLSizeMake(16, 16, 1);
        MTLSize gridSize = MTLSizeMake((N + 15) / 16 * 16, (M + 15) / 16 * 16, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        return result;
    }
}

CT_HOT Tensor Softmax_MPS_kernel(const Tensor& a, int dim) {
    @autoreleasepool {
        if (a.device() != DeviceType::kMPS) [[unlikely]] {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS Softmax_Kernel: 仅在MPS支持");
        }

        MPS_flush(true);

        const std::vector<size_t>& sizes = a.sizes();
        size_t elem_count = a.numel();
        
        Tensor result(ShapeTag{}, sizes, a.dtype(), a.device());

        id<MTLBuffer> a_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a.data<float>())));
        id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
        
        if (!a_buffer || !result_buffer) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Softmax_Kernel: 无法获取Metal Buffer");
        }

        initMetal();
        id<MTLComputePipelineState> pipeline = getPipeline("softmax_kernel");
        if (!pipeline) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Softmax_Kernel: 无法获取pipeline");
        }

        size_t dim_size = sizes[dim];
        size_t outer_size = 1;
        for (size_t i = 0; i < dim; ++i) outer_size *= sizes[i];
        size_t inner_size = 1;
        for (size_t i = dim + 1; i < sizes.size(); ++i) inner_size *= sizes[i];
        
        uint32_t batch_size_val = static_cast<uint32_t>(outer_size * inner_size);
        uint32_t hidden_size_val = static_cast<uint32_t>(dim_size);
        id<MTLBuffer> batch_size_buffer = [_device newBufferWithBytes:&batch_size_val length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> hidden_size_buffer = [_device newBufferWithBytes:&hidden_size_val length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        [encoder setBuffer:result_buffer offset:0 atIndex:1];
        [encoder setBuffer:batch_size_buffer offset:0 atIndex:2];
        [encoder setBuffer:hidden_size_buffer offset:0 atIndex:3];
        
        MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
        MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        return result;
    }
}

CT_HOT Tensor CrossEntropy_MPS_kernel(const Tensor& input, const Tensor& target) {
    @autoreleasepool {
        if (input.device() != DeviceType::kMPS || target.device() != DeviceType::kMPS) [[unlikely]] {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS CrossEntropy_Kernel: 仅在MPS支持");
        }

        MPS_flush(true);

        bool is_one_hot = (target.dim() == 2 && target.size(1) == input.size(1));
        
        size_t batch_size = input.size(0);
        size_t num_classes = input.size(1);
        
        Tensor result(ShapeTag{}, {batch_size}, input.dtype(), input.device());
        float* result_data = result.data<float>();
        for (size_t i = 0; i < batch_size; ++i) {
            result_data[i] = 0.0f;
        }

        id<MTLBuffer> input_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(input.data<float>())));
        id<MTLBuffer> target_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(target.data<float>())));
        id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
        
        if (!input_buffer || !target_buffer || !result_buffer) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS CrossEntropy_Kernel: 无法获取Metal Buffer");
        }

        initMetal();
        id<MTLComputePipelineState> pipeline = getPipeline("cross_entropy_kernel");
        if (!pipeline) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS CrossEntropy_Kernel: 无法获取pipeline");
        }

        uint32_t batch_size_val = static_cast<uint32_t>(batch_size);
        uint32_t num_classes_val = static_cast<uint32_t>(num_classes);
        uint32_t is_one_hot_val = static_cast<uint32_t>(is_one_hot ? 1 : 0);
        id<MTLBuffer> batch_size_buffer = [_device newBufferWithBytes:&batch_size_val length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> num_classes_buffer = [_device newBufferWithBytes:&num_classes_val length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> is_one_hot_buffer = [_device newBufferWithBytes:&is_one_hot_val length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:target_buffer offset:0 atIndex:1];
        [encoder setBuffer:result_buffer offset:0 atIndex:2];
        [encoder setBuffer:batch_size_buffer offset:0 atIndex:3];
        [encoder setBuffer:num_classes_buffer offset:0 atIndex:4];
        [encoder setBuffer:is_one_hot_buffer offset:0 atIndex:5];
        
        MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
        MTLSize gridSize = MTLSizeMake(batch_size, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        float total_loss = 0.0f;
        for (size_t i = 0; i < batch_size; ++i) {
            total_loss += result_data[i];
        }
        
        Tensor final_result(ShapeTag{}, {1}, input.dtype(), input.device());
        final_result.data<float>()[0] = total_loss / batch_size;
        
        return final_result;
    }
}

CT_HOT Tensor Dot_MPS_kernel(const Tensor& a, const Tensor& b) {
    @autoreleasepool {
        if (a.device() != DeviceType::kMPS || b.device() != DeviceType::kMPS) [[unlikely]] {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS Dot_Kernel: 仅在MPS支持");
        }
        if (a.dtype() != b.dtype()) [[unlikely]] {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "MPS Dot_Kernel: Tensor数据类型不匹配");
        }
        
        MPS_flush(true);
        
        if (a.numel() != b.numel()) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DATATYPE, "MPS Dot_Kernel: Tensor元素数量不匹配");
        }
        
        size_t elem_count = a.numel();
        
        Tensor result(ShapeTag{}, {1}, a.dtype(), a.device());
        float* result_data = result.data<float>();
        result_data[0] = 0.0f;
        
        id<MTLBuffer> a_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a.data<float>())));
        id<MTLBuffer> b_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(b.data<float>())));
        id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
        
        if (!a_buffer || !b_buffer || !result_buffer) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Dot_Kernel: 无法获取Metal Buffer");
        }
        
        initMetal();
        id<MTLComputePipelineState> pipeline = getPipeline("dot_kernel");
        if (!pipeline) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Dot_Kernel: 无法获取pipeline");
        }
        
        uint32_t params[1] = {static_cast<uint32_t>(elem_count)};
        id<MTLBuffer> params_buffer = [_device newBufferWithBytes:params length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        [encoder setBuffer:b_buffer offset:0 atIndex:1];
        [encoder setBuffer:result_buffer offset:0 atIndex:2];
        [encoder setBuffer:params_buffer offset:0 atIndex:3];
        
        MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
        MTLSize gridSize = MTLSizeMake((elem_count + 255) / 256, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        return result;
    }
}

CT_HOT Tensor MSE_MPS_kernel(const Tensor& a, const Tensor& b) {
    @autoreleasepool {
        if (a.device() != DeviceType::kMPS || b.device() != DeviceType::kMPS) [[unlikely]] {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS MSE_Kernel: 仅在MPS支持");
        }
        if (a.dtype() != b.dtype()) [[unlikely]] {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "MPS MSE_Kernel: Tensor数据类型不匹配");
        }
        if (a.sizes() != b.sizes()) [[unlikely]] {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DATATYPE, "MPS MSE_Kernel: Tensor形状不匹配");
        }
        
        MPS_flush(true);
        
        size_t elem_count = a.numel();
        Tensor result(ShapeTag{}, {elem_count}, a.dtype(), a.device());
        float* result_data = result.data<float>();
        for (size_t i = 0; i < elem_count; ++i) {
            result_data[i] = 0.0f;
        }
        
        id<MTLBuffer> a_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a.data<float>())));
        id<MTLBuffer> b_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(b.data<float>())));
        id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
        
        if (!a_buffer || !b_buffer || !result_buffer) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS MSE_Kernel: 无法获取Metal Buffer");
        }
        
        initMetal();
        id<MTLComputePipelineState> pipeline = getPipeline("mse_kernel");
        if (!pipeline) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS MSE_Kernel: 无法获取pipeline");
        }
        
        uint32_t n_val = static_cast<uint32_t>(elem_count);
        id<MTLBuffer> n_buffer = [_device newBufferWithBytes:&n_val length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        [encoder setBuffer:b_buffer offset:0 atIndex:1];
        [encoder setBuffer:result_buffer offset:0 atIndex:2];
        [encoder setBuffer:n_buffer offset:0 atIndex:3];
        
        MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
        MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        float total = 0.0f;
        for (size_t i = 0; i < elem_count; ++i) {
            total += result_data[i];
        }
        Tensor final_result(ShapeTag{}, {1}, a.dtype(), a.device());
        final_result.data<float>()[0] = total / elem_count;
        
        return final_result;
    }
}

CT_HOT Tensor MAE_MPS_kernel(const Tensor& a, const Tensor& b) {
    @autoreleasepool {
        if (a.device() != DeviceType::kMPS || b.device() != DeviceType::kMPS) [[unlikely]] {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS MAE_Kernel: 仅在MPS支持");
        }
        if (a.dtype() != b.dtype()) [[unlikely]] {
            CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "MPS MAE_Kernel: Tensor数据类型不匹配");
        }
        if (a.sizes() != b.sizes()) [[unlikely]] {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DATATYPE, "MPS MAE_Kernel: Tensor形状不匹配");
        }
        
        MPS_flush(true);
        
        size_t elem_count = a.numel();
        Tensor result(ShapeTag{}, {elem_count}, a.dtype(), a.device());
        float* result_data = result.data<float>();
        for (size_t i = 0; i < elem_count; ++i) {
            result_data[i] = 0.0f;
        }
        
        id<MTLBuffer> a_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a.data<float>())));
        id<MTLBuffer> b_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(b.data<float>())));
        id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
        
        if (!a_buffer || !b_buffer || !result_buffer) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS MAE_Kernel: 无法获取Metal Buffer");
        }
        
        initMetal();
        id<MTLComputePipelineState> pipeline = getPipeline("mae_kernel");
        if (!pipeline) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS MAE_Kernel: 无法获取pipeline");
        }
        
        uint32_t n_val = static_cast<uint32_t>(elem_count);
        id<MTLBuffer> n_buffer = [_device newBufferWithBytes:&n_val length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        [encoder setBuffer:b_buffer offset:0 atIndex:1];
        [encoder setBuffer:result_buffer offset:0 atIndex:2];
        [encoder setBuffer:n_buffer offset:0 atIndex:3];
        
        MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
        MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        float total = 0.0f;
        for (size_t i = 0; i < elem_count; ++i) {
            total += result_data[i];
        }
        Tensor final_result(ShapeTag{}, {1}, a.dtype(), a.device());
        final_result.data<float>()[0] = total / elem_count;
        
        return final_result;
    }
}

CT_HOT Tensor Log_MPS_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kMPS) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS Log_Kernel: 仅在MPS支持");
    }
    
    size_t elem_count = a.numel();
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
    
    id<MTLBuffer> a_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a.data<float>())));
    id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
    
    if (!a_buffer || !result_buffer) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Log_Kernel: 无法获取Metal Buffer");
    }
    
    initMetal();
    id<MTLComputePipelineState> pipeline = getPipeline("log_kernel");
    if (!pipeline) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Log_Kernel: 无法获取pipeline");
    }
    
    id<MTLComputeCommandEncoder> encoder = _accumulator.getEncoder();
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a_buffer offset:0 atIndex:0];
    [encoder setBuffer:result_buffer offset:0 atIndex:1];
    
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    return result;
}

CT_HOT Tensor Exp_MPS_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kMPS) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS Exp_Kernel: 仅在MPS支持");
    }
    
    size_t elem_count = a.numel();
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
    
    id<MTLBuffer> a_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a.data<float>())));
    id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
    
    if (!a_buffer || !result_buffer) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Exp_Kernel: 无法获取Metal Buffer");
    }
    
    initMetal();
    id<MTLComputePipelineState> pipeline = getPipeline("exp_kernel");
    if (!pipeline) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Exp_Kernel: 无法获取pipeline");
    }
    
    id<MTLComputeCommandEncoder> encoder = _accumulator.getEncoder();
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a_buffer offset:0 atIndex:0];
    [encoder setBuffer:result_buffer offset:0 atIndex:1];
    
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    return result;
}

CT_HOT Tensor Abs_MPS_kernel(const Tensor& a) {
    if (a.device() != DeviceType::kMPS) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS Abs_Kernel: 仅在MPS支持");
    }
    
    size_t elem_count = a.numel();
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
    
    id<MTLBuffer> a_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a.data<float>())));
    id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
    
    if (!a_buffer || !result_buffer) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Abs_Kernel: 无法获取Metal Buffer");
    }
    
    initMetal();
    id<MTLComputePipelineState> pipeline = getPipeline("abs_kernel");
    if (!pipeline) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Abs_Kernel: 无法获取pipeline");
    }
    
    id<MTLComputeCommandEncoder> encoder = _accumulator.getEncoder();
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a_buffer offset:0 atIndex:0];
    [encoder setBuffer:result_buffer offset:0 atIndex:1];
    
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];

    return result;
}

CT_HOT Tensor Min_MPS_kernel(const Tensor& a, const Tensor& b) {
    if (a.device() != DeviceType::kMPS || b.device() != DeviceType::kMPS) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS Min_Kernel: 仅在MPS支持");
    }
    if (a.dtype() != b.dtype()) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "MPS Min_Kernel: Tensor数据类型不匹配");
    }
    if (a.sizes() != b.sizes()) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DATATYPE, "MPS Min_Kernel: Tensor形状不匹配");
    }
    
    size_t elem_count = a.numel();
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
    
    id<MTLBuffer> a_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a.data<float>())));
    id<MTLBuffer> b_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(b.data<float>())));
    id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
    
    if (!a_buffer || !b_buffer || !result_buffer) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Min_Kernel: 无法获取Metal Buffer");
    }
    
    initMetal();
    id<MTLComputePipelineState> pipeline = getPipeline("min_kernel");
    if (!pipeline) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Min_Kernel: 无法获取pipeline");
    }
    
    id<MTLComputeCommandEncoder> encoder = _accumulator.getEncoder();
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a_buffer offset:0 atIndex:0];
    [encoder setBuffer:b_buffer offset:0 atIndex:1];
    [encoder setBuffer:result_buffer offset:0 atIndex:2];
    
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    return result;
}

CT_HOT Tensor Max_MPS_kernel(const Tensor& a, const Tensor& b) {
    if (a.device() != DeviceType::kMPS || b.device() != DeviceType::kMPS) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS Max_Kernel: 仅在MPS支持");
    }
    if (a.dtype() != b.dtype()) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "MPS Max_Kernel: Tensor数据类型不匹配");
    }
    if (a.sizes() != b.sizes()) [[unlikely]] {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DATATYPE, "MPS Max_Kernel: Tensor形状不匹配");
    }
    
    size_t elem_count = a.numel();
    Tensor result(ShapeTag{}, a.sizes(), a.dtype(), a.device());
    
    id<MTLBuffer> a_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a.data<float>())));
    id<MTLBuffer> b_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(b.data<float>())));
    id<MTLBuffer> result_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(result.data<float>())));
    
    if (!a_buffer || !b_buffer || !result_buffer) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Max_Kernel: 无法获取Metal Buffer");
    }
    
    initMetal();
    id<MTLComputePipelineState> pipeline = getPipeline("max_kernel");
    if (!pipeline) {
        CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Max_Kernel: 无法获取pipeline");
    }
    
    id<MTLComputeCommandEncoder> encoder = _accumulator.getEncoder();
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:a_buffer offset:0 atIndex:0];
    [encoder setBuffer:b_buffer offset:0 atIndex:1];
    [encoder setBuffer:result_buffer offset:0 atIndex:2];
    
    MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
    MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
    
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    
    return result;
}

extern "C" void Zero_MPS_kernel(const Tensor& a) {
    @autoreleasepool {
        if (a.device() != DeviceType::kMPS) [[unlikely]] {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS Zero_Kernel: 仅在MPS支持");
        }
        
        size_t elem_count = a.numel();
        
        id<MTLBuffer> a_buffer = MPS_getBuffer(const_cast<void*>(static_cast<const void*>(a.data<float>())));
        
        if (!a_buffer) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Zero_Kernel: 无法获取Metal Buffer");
        }
        
        initMetal();
        id<MTLComputePipelineState> pipeline = getPipeline("zero_kernel");
        if (!pipeline) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS Zero_Kernel: 无法获取pipeline");
        }
        
        id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:a_buffer offset:0 atIndex:0];
        
        MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
        MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

extern "C" void SGD_Step_MPS_kernel(const Tensor& param, const Tensor& grad, float lr) {
    @autoreleasepool {
        // 注意：SGD 使用独立 command buffer，调用者必须保证在调用前 accumulator 已 flush，
        // 否则 grad buffer 可能尚未完成写入。mnist.cpp 的 update_parameters 已统一处理。
        if (param.device() != DeviceType::kMPS || grad.device() != DeviceType::kMPS) [[unlikely]] {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS SGD_Step_Kernel: 仅在MPS支持");
        }
        
        size_t elem_count = param.numel();
        
        void* param_ptr = const_cast<void*>(static_cast<const void*>(param.data<float>()));
        void* grad_ptr = const_cast<void*>(static_cast<const void*>(grad.data<float>()));
        id<MTLBuffer> param_buffer = MPS_getBuffer(param_ptr);
        id<MTLBuffer> grad_buffer = MPS_getBuffer(grad_ptr);
        
        if (!param_buffer || !grad_buffer) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS SGD_Step_Kernel: 无法获取Metal Buffer");
        }
        
        initMetal();
        id<MTLComputePipelineState> pipeline = getPipeline("sgd_step_kernel");
        
        if (!pipeline) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS SGD_Step_Kernel: 无法获取pipeline");
        }
        
        id<MTLBuffer> lr_buffer = [_device newBufferWithBytes:&lr length:sizeof(float) options:MTLResourceStorageModeShared];
        
        id<MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:param_buffer offset:0 atIndex:0];
        [encoder setBuffer:grad_buffer offset:0 atIndex:1];
        [encoder setBuffer:lr_buffer offset:0 atIndex:2];
        
        MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
        MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [encoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

extern "C" void SGD_Step_Zero_MPS_kernel(const Tensor& param, const Tensor& grad, float lr) {
    @autoreleasepool {
        // 在更新参数的同时把对应梯度清零，省掉一次独立的 Zero_MPS_kernel 提交与等待。
        if (param.device() != DeviceType::kMPS || grad.device() != DeviceType::kMPS) [[unlikely]] {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::DEVICE_COMPAT, "MPS SGD_Step_Zero_Kernel: 仅在MPS支持");
        }
        
        size_t elem_count = param.numel();
        
        void* param_ptr = const_cast<void*>(static_cast<const void*>(param.data<float>()));
        void* grad_ptr = const_cast<void*>(static_cast<const void*>(grad.data<float>()));
        id<MTLBuffer> param_buffer = MPS_getBuffer(param_ptr);
        id<MTLBuffer> grad_buffer = MPS_getBuffer(grad_ptr);
        
        if (!param_buffer || !grad_buffer) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS SGD_Step_Zero_Kernel: 无法获取Metal Buffer");
        }
        
        initMetal();
        id<MTLComputePipelineState> pipeline = getPipeline("sgd_step_zero_kernel");
        
        if (!pipeline) {
            CtorchError::throwException(ErrorPlatform::kMPS, ErrorType::PLATFORM_API, "MPS SGD_Step_Zero_Kernel: 无法获取pipeline");
        }
        
        id<MTLBuffer> lr_buffer = [_device newBufferWithBytes:&lr length:sizeof(float) options:MTLResourceStorageModeShared];
        
        bool own_command_buffer = (_updateCommandBuffer == nil);
        id<MTLCommandBuffer> commandBuffer = own_command_buffer ? [_commandQueue commandBuffer] : _updateCommandBuffer;
        id<MTLComputeCommandEncoder> encoder = own_command_buffer ? [commandBuffer computeCommandEncoder] : _updateEncoder;
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:param_buffer offset:0 atIndex:0];
        [encoder setBuffer:grad_buffer offset:0 atIndex:1];
        [encoder setBuffer:lr_buffer offset:0 atIndex:2];
        
        MTLSize threadGroupSize = MTLSizeMake(256, 1, 1);
        MTLSize gridSize = MTLSizeMake(elem_count, 1, 1);
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        if (own_command_buffer) {
            [encoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        }
    }
}
