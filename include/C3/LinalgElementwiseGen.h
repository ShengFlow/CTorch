/**
 * @file LinalgElementwiseGen.h
 * @generation JIT-2.0 声明式单算子后端（Linalg 逐元素路线）
 * @brief 用 linalg.generic 声明式生成逐元素 kernel（JIT 2.0 路线 A 移植）
 *
 * 背景：主库 `MLIRKernelGen.cpp` 的逐元素算子（Add/Mul/ReLU/Sigmoid/...）当前是
 * 手写 LLVM 指针 IR + if-else 分发（buildElementwiseBinary* / buildReLU / ...）。
 * 本组件用 `linalg.generic` 声明式描述同一批逐元素算子，经标准 lowering pipeline
 * （linalg → loops → scf → cf → llvm）编译为 JIT kernel，为后续「统一替换手写分支」
 * （STATUS_CONTEXT 4.7-2）提供正确性与性能依据。
 *
 * 调用 ABI：memref<?xf32> 签名 + `ExecutionEngine::invokePacked` 展开调用
 * （与 exp_linalg_elementwise PoC 一致）。外部只需提供裸数据指针 + 元素个数。
 *
 * @date 2026/08/15
 */

#ifndef CTORCH_C3_LINALG_ELEMENTWISE_GEN_H
#define CTORCH_C3_LINALG_ELEMENTWISE_GEN_H

#include <cstddef>
#include <memory>
#include <vector>

namespace ct {
namespace c3 {

/// 支持用 linalg.generic 声明的逐元素算子（与主库 Node 类型一一对应）
enum class ElementwiseOp {
    ReLU,    ///< out = max(in, 0)
    Sigmoid, ///< out = 1 / (1 + exp(-in))
    Tanh,    ///< out = tanh(in)
    Exp,     ///< out = exp(in)
    Log,     ///< out = log(in)
    Add,     ///< out = a + b
    Sub,     ///< out = a - b
    Mul,     ///< out = a * b
};

/// 一元算子集合
bool isUnaryElementwiseOp(ElementwiseOp op);
/// 算子名称（日志 / 缓存 key）
const char* elementwiseOpName(ElementwiseOp op);
/// 输入个数（一元=1，二元=2）
size_t elementwiseOpNumInputs(ElementwiseOp op);

/// 二元第二输入的广播模数 rhs_mod 语义（一元算子忽略）：
///   0  同尺寸（无广播，linalg indexing map `d0 -> d0`）
///   1  标量广播（rhs size=1，linalg indexing map `d0 -> 0`）
///   k>1  周期广播（rhs 为 1D vector size=k，linalg indexing map `d0 -> (d0 mod k)`）
/// 其余多维/不支持广播回退手写路径。
enum RhsBroadcastMode : int {
    RhsNoBroadcast = 0,   ///< 同尺寸
    RhsScalarBroadcast = 1, ///< 标量（size=1）
};

/// 编译并持有单个 linalg.generic 逐元素 kernel（memref 签名 + invokePacked ABI）
/// 线程安全：编译完成后 execute 可并发调用（JIT 函数无共享状态）。
class LinalgElementwiseKernel {
public:
    /// 编译 kernel（每次构造即完成一次 JIT 编译）；失败抛 std::runtime_error
    /// @param rhs_mod 二元算子第二输入广播模数（见 RhsBroadcastMode），一元忽略。
    explicit LinalgElementwiseKernel(ElementwiseOp op, int opt_level = 3,
                                     int rhs_mod = RhsNoBroadcast);
    ~LinalgElementwiseKernel();
    LinalgElementwiseKernel(const LinalgElementwiseKernel&) = delete;
    LinalgElementwiseKernel& operator=(const LinalgElementwiseKernel&) = delete;
    LinalgElementwiseKernel(LinalgElementwiseKernel&&) noexcept;
    LinalgElementwiseKernel& operator=(LinalgElementwiseKernel&&) noexcept;

    ElementwiseOp op() const { return op_; }
    size_t numInputs() const { return num_inputs_; }
    /// 二元第二输入广播模数（0=同尺寸，1=标量，k>1=周期广播）
    int rhsMod() const { return rhs_mod_; }

    /// 执行 kernel：in_ptrs 按输入顺序（只读），out_ptr 输出，n 元素个数。
    /// 标量广播时 in_ptrs[1] 只需指向 1 个元素；周期广播时 in_ptrs[1] 至少 k 个元素。
    /// 注意：输入输出 buffer 均按 1D 连续访问，元素类型 float32。
    void execute(const float* const* in_ptrs, float* out_ptr, size_t n) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    ElementwiseOp op_;
    size_t num_inputs_;
    int rhs_mod_ = RhsNoBroadcast;
};

/// 共享 kernel 缓存工厂：同一 (op, opt_level, rhs_mod) 只 JIT 编译一次，之后复用。
/// 线程安全：构造互斥 + execute 并发安全。编译失败抛 std::runtime_error。
/// 逃生开关：`C3_LINALG_CACHE=0` 关闭缓存（每次全新编译，便于对比）。
std::shared_ptr<LinalgElementwiseKernel> getCachedLinalgKernel(
    ElementwiseOp op, int opt_level = 3, int rhs_mod = RhsNoBroadcast);

/// 便捷函数：编译并执行一次（测试 / 小规模场景用）。
/// inputs 按算子输入顺序传数据；输出自动分配并返回。
/// 各输入必须至少有 n 个元素；n 超出实际长度是调用方错误。
std::vector<float> runLinalgElementwise(ElementwiseOp op,
                                        const std::vector<std::vector<float>>& inputs,
                                        size_t n);

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_LINALG_ELEMENTWISE_GEN_H
