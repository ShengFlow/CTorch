/**
 * @file GeneratedKernel.h
 * @generation SHARED 跨代共享基础设施（编译产物统一接口）
 * @brief JIT 2.0 / 2.x / 3.0 三代后端共用的编译产物结构体。
 * @details 本结构体曾定义在 src/C3/HandwrittenKernelGen.h（JIT-2.0 头文件）内，
 *          因三代共用而迁出至此，避免"共享类型住在旧版头文件"的归属混乱。
 *          - JIT-2.0  填 func / fused_func / multi_func（裸函数指针）
 *          - JIT-2.x  填 func（MLIR 标量单算子编译产物）
 *          - JIT-3.0  填 func_any（LinalgElementwiseKernel 执行器，捕获 shared_ptr）
 *
 * @date 2026/8/16
 */

#ifndef CTORCH_C3_GENERATED_KERNEL_H
#define CTORCH_C3_GENERATED_KERNEL_H

#include <cstddef>
#include <functional>
#include <vector>
#include "../../include/C3/C3KernelRegistry.h"

namespace ct {
namespace c3 {

/**
 * @struct GeneratedKernel
 * @brief 编译产物的统一接口，HandwrittenKernelGen、MLIRKernelGen、Linalg*Gen 三代共用。
 */
struct GeneratedKernel {
    C3KernelFunc func = nullptr;
    FusedKernelFunc fused_func = nullptr; ///< 融合 kernel 函数指针（is_fused=true 时使用）
    MultiNodeKernelFunc multi_func = nullptr; ///< 多节点 kernel 函数指针（is_multi_node=true 时使用）
    /// [2026-08-15] linalg.generic 路线：可选执行器（优先级高于 func）。
    /// 用于单节点逐元素算子接入 LinalgElementwiseKernel——它持有自己的 JIT engine，
    /// 无法表示成裸 C3KernelFunc，故用 std::function 捕获 shared_ptr 保证生命周期。
    using SingleNodeExecutor = void(const float*, const float*, float*,
                                    size_t, size_t, size_t, size_t);
    std::function<SingleNodeExecutor> func_any = nullptr;
    void* handle = nullptr;               ///< 资源句柄（dlopen handle 或 ExecutionEngine 引用）
    std::function<void()> deleter;        ///< 析构回调：释放 handle 指向的资源
    bool is_matmul = false;
    bool is_fused = false;                ///< 是否为融合 kernel
    bool is_multi_node = false;           ///< 是否为多节点 kernel
    size_t num_inputs = 2;                ///< 外部输入数量
    size_t M = 0, K = 0, N = 0;          ///< MatMul 维度
    size_t elem_n = 0;                    ///< 逐元素操作的元素数
    size_t scratch_size = 0;              ///< JIT scratchpad 暂存大小 (in floats)
    /// DEBT-NEW-7 候选 A:融合 kernel 的真实输出 shape(从 FusedNode.out_desc 提取)
    /// 让 FusedCompiledKernel::execute() 能正确分配 output buffer(支持 MatMul-rooted region)
    std::vector<size_t> fused_out_shape;
};

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_GENERATED_KERNEL_H
