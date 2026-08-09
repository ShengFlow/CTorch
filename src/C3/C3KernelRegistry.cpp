/**
 * @file C3KernelRegistry.cpp
 * @brief C3 内核注册表 · 融合/反向 kernel 子系统的实现
 * @details 二元/一元单 kernel 路径（install/tryExecute/tryExecuteUnary）已在
 *          C3KernelRegistry.h 内联实现（header-only），本 .cpp 负责：
 *          1. 融合 kernel (fused_entries_) 的安装/查询/执行
 *          2. 反向 kernel (backward_entries_) 的安装/查询/执行
 *          3. 序列/首 op 模糊匹配 (findFusedKernelFor*)
 *          4. 融合 kernel 包装执行 (executeFusedWithInputs)
 *
 *          **当前状态：stub 阶段**。DEBT-NEW-7 修复集（用户之前设计的 region fusion
 *          全套）需要这些方法作为执行后端，但 region fusion 仍按宏开关
 *          CT_C3_DISABLE_REGION_FUSION 关闭；本文件所有方法在 stub 形态下
 *          返回 nullopt / 空向量 / 抛 not_implemented，保证 build 通过且
 *          c3 单 kernel 路径行为不退化（calls fall back to eager）。
 *
 * @date 2026-08-09
 */

#include "../../include/C3/C3KernelRegistry.h"
#include "../../include/C3/C3Engine.h"  // CompiledKernel 完整定义
#include "../../include/CtorchError.h"

namespace ct {
namespace c3 {

// ======================= 融合 kernel 执行 =======================

// TODO(region-fusion): 用户之前的 region fusion 设计需要这个方法作为 backend，
// 当前 stub 形态返回 nullopt → 调度器回退 eager。完整实现需要：
//  1. 从 CompiledKernel 取出 MLIR ExecutionEngine function pointer
//  2. 按 inputs 数量和 shapes 解析 kernel 签名
//  3. 准备 output buffer（按 shapes.out_shape 分配）
//  4. invoke function pointer
//  5. 包装成 Tensor 返回
std::optional<Tensor> C3KernelRegistry::tryExecuteFused(
    op /*op_type*/, const std::vector<Tensor>& /*inputs*/) {
    return std::nullopt;
}

// TODO(region-fusion): 用于 region fusion 接管整个序列时的执行后端。
// 当前 stub 返回 nullopt 维持 c3 单 kernel fallback 行为。
Tensor C3KernelRegistry::executeFusedWithInputs(
    std::shared_ptr<CompiledKernel> /*kernel*/,
    const std::vector<Tensor>& /*inputs*/,
    const KernelShapeInfo& /*shapes*/) {
    CtorchError::log(ErrorLevel::WARN, ErrorPlatform::kGENERAL,
        ErrorType::UNKNOWN,
        "C3KernelRegistry::executeFusedWithInputs: stub, region fusion disabled. "
        "Falling back to eager dispatch.");
    return Tensor();
}

// ======================= 反向 kernel 执行 =======================

// TODO(c3-backward): 反向 fusion kernel 的实际执行后端。
// 当前 stub 返回 nullopt → C3BackwardCapture::tryExecuteBackward 会回退 eager。
// 完整实现需要：
//  1. 在 backward_entries_ 中查找 backward_key
//  2. 验证 grad.shape() 与注册时记录的 grad_shape 一致
//  3. 验证 forward_inputs 数量与 kernel 签名匹配
//  4. invoke CompiledKernel 的 function pointer
//  5. 包装为 vector<Tensor> 返回（多输出支持）
std::optional<std::vector<Tensor>> C3KernelRegistry::tryExecuteBackward(
    const std::string& /*backward_key*/, const Tensor& /*grad*/,
    const std::vector<Tensor>& /*forward_inputs*/) {
    return std::nullopt;
}

// ======================= 序列/首 op 模糊匹配 =======================

// TODO(region-fusion): 用于按 op 序列模糊匹配已注册的融合 kernel（备选 path A）。
// 当前 stub 返回 nullopt → 调度器继续走精确匹配或 eager。
std::optional<std::pair<std::shared_ptr<CompiledKernel>, KernelShapeInfo>>
C3KernelRegistry::findFusedKernelForSequence(
    const std::vector<op>& /*op_seq*/, DeviceType /*dev*/,
    const std::vector<size_t>& /*first_input_shape*/) {
    return std::nullopt;
}

// TODO(region-fusion): 用于按首 op 匹配融合 kernel。
// 当前 stub 返回 nullopt。
std::optional<std::pair<std::shared_ptr<CompiledKernel>, KernelShapeInfo>>
C3KernelRegistry::findFusedKernelForFirstOp(
    op /*op_type*/, const std::vector<size_t>& /*input_shape*/,
    DeviceType /*dev*/) {
    return std::nullopt;
}

} // namespace c3
} // namespace ct
