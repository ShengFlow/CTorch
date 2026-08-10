/**
 * @file DCUCompiledKernel.h
 * @brief DCU (Hygon C86 / gfx906) CompiledKernel wrapper (v0.5 DCU 接入, 2026-08-10)
 * @details 把 GCVM 编译的 Code Object (HSACO) 包装成 CompiledKernel,
 *          execute() 走 hipBLAS / 裸 hipModuleLaunchKernel
 *
 * 关键设计:
 *   - 继承 CompiledKernel, 跟现有 MLIR/Handwritten backend 接口统一
 *   - input/output Tensor 走 host-device memcpy (MPS/CPU unified memory 不一样)
 *   - hipModule 持 Code Object lifetime
 *   - 注册到 C3KernelRegistry 跟其他 backend 平行
 *
 * 编译: 仅 WITH_DCU=ON 时编译
 */
#ifndef CTORCH_C3_DCU_COMPILED_KERNEL_H
#define CTORCH_C3_DCU_COMPILED_KERNEL_H

#include "C3/C3Engine.h"  // CompiledKernel base
#include "C3/GCVMBridge.h"  // GCVMCompileResult
#include "Tensor.h"

#include <memory>
#include <string>
#include <vector>

#ifdef WITH_DCU
    // hipBLAS/hip runtime (DCU 节点 DTK 24.04+ 路径)
    #include <hip/hip_runtime.h>
    // 注意: 探针回来后需要 include <hip/hip_module.h> 等具体 API 头
#endif

namespace ct {
namespace c3 {

/// DCU (gfx906) CompiledKernel wrapper
/// 包装 hipModule (HSACO) + kernel 名字, execute() 走 host-device memcpy + hipLaunchKernel
class DCUCompiledKernel : public CompiledKernel {
public:
    /// 构造: 拿 GCVM 编译的 Code Object + kernel 名
    /// @param code_object HSACO bytes (from GCVMCompileResult::code_object)
    /// @param kernel_name kernel symbol 名字
    /// @param graph 原始 Graph (用于 shape info)
    /// @param device DCU device index (默认 0)
    DCUCompiledKernel(std::string code_object,
                      std::string kernel_name,
                      const Graph& graph,
                      int device = 0);

    ~DCUCompiledKernel() override;

    std::vector<Tensor> execute(const std::vector<Tensor>& inputs) override;

    [[nodiscard]] const std::string& cacheKey() const override { return cache_key_; }
    [[nodiscard]] DeviceType targetDevice() const override { return DeviceType::kDCU; }
    [[nodiscard]] size_t workspaceBytes() const override { return workspace_bytes_; }

    bool installIntoRegistry(op op_type, const KernelShapeInfo& shapes) override;

private:
    std::string code_object_;        ///< HSACO bytes (Code Object)
    std::string kernel_name_;        ///< kernel symbol name
    std::string cache_key_;          ///< 缓存 key (用 Graph hash)
    int device_;                     ///< DCU device index
    size_t workspace_bytes_ = 0;     ///< device memory 预分配大小

#ifdef WITH_DCU
    hipModule_t hip_module_ = nullptr;       ///< 加载的 HSACO module
    hipFunction_t hip_function_ = nullptr;   ///< kernel function handle
    // device memory 句柄 (input/output buffers, 复用)
    void** d_input_buffers_ = nullptr;
    void* d_output_buffer_ = nullptr;
    size_t d_output_bytes_ = 0;
#endif

    /// hipModuleLoadData + hipModuleGetFunction
    /// @return true if success
    bool loadHIPModule();

    /// 分配 device memory (output buffer)
    /// @return true if success
    bool allocateDeviceMemory(size_t output_bytes);

    /// host -> device memcpy
    /// @return true if success
    bool copyInputsToDevice(const std::vector<Tensor>& inputs);

    /// device -> host memcpy
    Tensor copyOutputToHost(size_t numel, const std::vector<size_t>& shape);

    /// 启动 kernel (hipModuleLaunchKernel)
    /// @return true if success
    bool launchKernel(const std::vector<Tensor>& inputs);
};

}  // namespace c3
}  // namespace ct

#endif  // CTORCH_C3_DCU_COMPILED_KERNEL_H
