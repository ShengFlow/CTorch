/**
 * @file DCUCompiledKernel.cpp
 * @brief DCUCompiledKernel 实现 (v0.5 DCU 接入, 2026-08-10)
 * @details 真实 API 调用待 probe-dcu-dtk24.sh 探针回来后根据 hip/GCVM 头文件调整
 *          当前实装是 stub 骨架, 标 [TODO: probe-adjust] 的地方需要探针后实装
 */
#include "../../include/C3/DCUCompiledKernel.h"
#include "../../include/C3/C3KernelRegistry.h"
#include "../../include/C3/GCVMBridge.h"
#include "../../include/CtorchError.h"

#include <iostream>
#include <cstring>
#include <sstream>

#ifdef WITH_DCU
    #include <hip/hip_runtime.h>
    // TODO: probe-adjust: include <hip/hip_module.h>, <hipblas.h>, <hip/hip_runtime_api.h> per probe
#endif

namespace ct {
namespace c3 {

// ======================= 构造 / 析构 =======================

DCUCompiledKernel::DCUCompiledKernel(std::string code_object,
                                       std::string kernel_name,
                                       const Graph& graph,
                                       int device)
    : code_object_(std::move(code_object))
    , kernel_name_(std::move(kernel_name))
    , device_(device)
{
    // 构造 cache_key: 用 Graph toString hash
    cache_key_ = "dcu_" + graph.toString() + "_" + std::to_string(device);

#ifdef WITH_DCU
    if (!loadHIPModule()) {
        CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL, ErrorType::UNKNOWN,
            "DCUCompiledKernel: failed to load HIP module, execute() will fail");
    }
#endif
}

DCUCompiledKernel::~DCUCompiledKernel() {
#ifdef WITH_DCU
    // 清理 device memory
    if (d_output_buffer_) {
        hipFree(d_output_buffer_);
        d_output_buffer_ = nullptr;
    }
    if (d_input_buffers_) {
        for (size_t i = 0; i < /*num_inputs=*/2; ++i) {  // TODO: probe-adjust: 实际 input 数
            if (d_input_buffers_[i]) hipFree(d_input_buffers_[i]);
        }
        delete[] d_input_buffers_;
        d_input_buffers_ = nullptr;
    }
    // 卸载 module
    if (hip_module_) {
        hipModuleUnload(hip_module_);
        hip_module_ = nullptr;
        hip_function_ = nullptr;
    }
#endif
}

// ======================= loadHIPModule =======================

bool DCUCompiledKernel::loadHIPModule() {
#ifndef WITH_DCU
    return false;
#else
    // TODO: probe-adjust: hipModuleLoadData 真实签名
    // 当前实装: 直接从 memory 加载 (如果 HSACO 完整)
    if (code_object_.empty()) {
        return false;
    }

    hipError_t err = hipModuleLoadData(&hip_module_, code_object_.data());
    if (err != hipSuccess) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT,
            "DCUCompiledKernel: hipModuleLoadData failed: " + std::string(hipGetErrorString(err)));
        return false;
    }

    err = hipModuleGetFunction(&hip_function_, hip_module_, kernel_name_.c_str());
    if (err != hipSuccess) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT,
            "DCUCompiledKernel: hipModuleGetFunction failed: " + std::string(hipGetErrorString(err)));
        hipModuleUnload(hip_module_);
        hip_module_ = nullptr;
        return false;
    }

    return true;
#endif
}

// ======================= allocateDeviceMemory =======================

bool DCUCompiledKernel::allocateDeviceMemory(size_t output_bytes) {
#ifndef WITH_DCU
    (void)output_bytes;
    return false;
#else
    if (d_output_buffer_ && d_output_bytes_ >= output_bytes) {
        return true;  // 复用已有 buffer
    }
    if (d_output_buffer_) {
        hipFree(d_output_buffer_);
        d_output_buffer_ = nullptr;
    }
    d_output_bytes_ = output_bytes;
    hipError_t err = hipMalloc(&d_output_buffer_, output_bytes);
    if (err != hipSuccess) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT,
            "DCUCompiledKernel: hipMalloc output failed: " + std::string(hipGetErrorString(err)));
        return false;
    }
    return true;
#endif
}

// ======================= copyInputsToDevice =======================

bool DCUCompiledKernel::copyInputsToDevice(const std::vector<Tensor>& inputs) {
#ifndef WITH_DCU
    (void)inputs;
    return false;
#else
    // TODO: probe-adjust: 简化实装 - 每次 dispatch 重新 malloc + memcpy
    // 优化版: 用 pool / 缓存
    if (!d_input_buffers_) {
        d_input_buffers_ = new void*[inputs.size()]();  // value-init to nullptr
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& t = inputs[i];
        size_t bytes = t.numel() * sizeof(float);
        if (d_input_buffers_[i]) hipFree(d_input_buffers_[i]);
        hipError_t err = hipMalloc(&d_input_buffers_[i], bytes);
        if (err != hipSuccess) return false;
        err = hipMemcpy(d_input_buffers_[i], t.data_read<float>(), bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess) return false;
    }
    return true;
#endif
}

// ======================= copyOutputToHost =======================

Tensor DCUCompiledKernel::copyOutputToHost(size_t numel, const std::vector<size_t>& shape) {
    Tensor out(ShapeTag{}, shape);
    if (d_output_buffer_) {
#ifdef WITH_DCU
        hipError_t err = hipMemcpy(out.data_write<float>(), d_output_buffer_,
                                    numel * sizeof(float), hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT,
                "DCUCompiledKernel: hipMemcpy DtoH failed: " + std::string(hipGetErrorString(err)));
        }
#endif
    }
    return out;
}

// ======================= launchKernel =======================

bool DCUCompiledKernel::launchKernel(const std::vector<Tensor>& inputs) {
#ifndef WITH_DCU
    (void)inputs;
    return false;
#else
    // TODO: probe-adjust: 真实 kernel signature
    // 假设 c3_kernel 签名: (a, b, out, n, M, K, N) 跟 macOS MLIR 一致
    // 实际参数 layout 取决于具体 graph
    if (!hip_function_) return false;

    // 占位: 假设 MatMul 单 op graph, 1 input + 1 output
    // 真实 launch 需要按 graph kernel signature 设置 args
    void* args[] = {
        d_input_buffers_[0],  // a
        d_input_buffers_[1],  // b
        d_output_buffer_,     // out
        nullptr,                // n (i64)
        nullptr,                // M (i64)
        nullptr,                // K (i64)
        nullptr                 // N (i64)
    };

    // TODO: probe-adjust: grid/block dims 按 actual workload 算
    dim3 grid(1);
    dim3 block(64);

    hipError_t err = hipModuleLaunchKernel(hip_function_,
                                            grid.x, grid.y, grid.z,
                                            block.x, block.y, block.z,
                                            0,  // shared mem
                                            nullptr,  // stream
                                            args,
                                            nullptr);  // extra
    if (err != hipSuccess) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT,
            "DCUCompiledKernel: hipModuleLaunchKernel failed: " + std::string(hipGetErrorString(err)));
        return false;
    }
    return true;
#endif
}

// ======================= execute =======================

std::vector<Tensor> DCUCompiledKernel::execute(const std::vector<Tensor>& inputs) {
    std::vector<Tensor> outputs;

#ifndef WITH_DCU
    (void)inputs;
    CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT,
        "DCUCompiledKernel::execute called but WITH_DCU=OFF (rebuild with -DWITH_DCU=ON)");
    return outputs;
#else
    if (!hip_module_ || !hip_function_) {
        CtorchError::log(ErrorLevel::ERROR, ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT,
            "DCUCompiledKernel::execute: HIP module/function not loaded");
        return outputs;
    }

    if (!copyInputsToDevice(inputs)) {
        return outputs;
    }

    // TODO: probe-adjust: 从 graph 拿 output shape + size
    // 当前占位: 假设 output 跟 input[0] 同 shape
    size_t output_numel = inputs[0].numel();
    std::vector<size_t> output_shape = inputs[0].shape();
    if (!allocateDeviceMemory(output_numel * sizeof(float))) {
        return outputs;
    }

    if (!launchKernel(inputs)) {
        return outputs;
    }

    // 同步
    hipDeviceSynchronize();

    outputs.push_back(copyOutputToHost(output_numel, output_shape));
    return outputs;
#endif
}

// ======================= installIntoRegistry =======================

bool DCUCompiledKernel::installIntoRegistry(op op_type, const KernelShapeInfo& shapes) {
    (void)op_type;
    (void)shapes;
    // TODO: probe-adjust: 完整实装 (跟 C3KernelRegistry 配合)
    // DCU path 多节点 kernel 暂不注册 (走 multi_func 路径, 跟 macOS MLIR 一致)
    return false;
}

}  // namespace c3
}  // namespace ct
