/**
 * @file SiLU.h
 * @brief SiLU (Sigmoid Linear Unit) 算子 — Eager CPU 入口
 * @details 数学定义: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *          LLaMA / PaLM / SwiGLU 等现代 LLM 的核心激活函数
 * @author Mavis (PEL25 §6 SwiGLU 算子开发协议 Stage 1)
 * @date 2026-09-05
 *
 * Stage 1 范围 (PEL25 §6.7 ANTI_PATTERN_BLOCK 原则: 一次只改一个算子):
 *   - Eager CPU 路径 (BASIC + SIMD AVX2)
 *   - 不改 op 枚举 (28 → 不动, 避免触发 CtorchScheduler.h:229-230 静态断言)
 *   - 不改 isRegionCandidateOp 白名单
 *   - 不改 ElementwiseOp 枚举 (LinalgElementwiseGen.h:29-38)
 *   - 不接 C3 dispatch
 *
 * Stage 2 范围 (PEL25 §7 ABI HITL 决策门 #6-10, 需用户逐项批准):
 *   - op 枚举扩展 (op::kCount 28 → 30)
 *   - isRegionCandidateOp 白名单扩展 (SiLU + SwiGLU)
 *   - ElementwiseOp 枚举扩展
 *   - LinalgElementwiseGen.cpp switch/case 同步
 *   - MLIR Op TableGen 草案 (C3_SiLUOp)
 *   - C3Dialect.cpp 手写 build
 *   - C3BackwardCapture.cpp SiLU/SwiGLU 逆向
 *   - C3KernelRegistry 接入 + Region Fusion 接入
 */

#ifndef CTORCH_OPS_SILU_H
#define CTORCH_OPS_SILU_H

#include "Tensor.h"

namespace ct::ops {

/**
 * @brief SiLU 标量参考实现 (用于测试对照)
 * @details silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *          数值稳定性: 对极大负数, sigmoid(x) → 0, silu → 0; 不溢出
 *          对极大正数, sigmoid(x) → 1, silu → x; 不溢出
 */
inline float silu_scalar(float x) {
    return x / (1.0f + std::exp(-x));
}

/**
 * @brief SiLU 标量导数参考实现 (用于测试对照)
 * @details d/dx silu(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
 *                       = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
 *                       = silu(x) / x + x * sigmoid(x) * (1 - sigmoid(x))  // 当 x != 0
 *                       = sigmoid(x) + x * sigmoid(x) - x * sigmoid(x)^2
 * @return 标量 x 处的 SiLU 导数
 */
inline float silu_derivative_scalar(float x) {
    float s = 1.0f / (1.0f + std::exp(-x));  // sigmoid(x)
    return s + x * s * (1.0f - s);
}

/**
 * @brief SiLU forward: 元素级 x * sigmoid(x)
 * @param input 输入 Tensor (任意 shape, 必须 contiguous)
 * @return 输出 Tensor, shape 跟输入一致
 * @note Eager CPU 路径, 不走 C3 dispatch
 */
Tensor silu_forward(const Tensor& input);

/**
 * @brief SiLU backward: 元素级 sigmoid(x) * (1 + x * (1 - sigmoid(x)))
 * @param grad_output 上游梯度 (跟 input 同 shape)
 * @param input 原始输入 (用于计算 sigmoid)
 * @return 元素级梯度, shape 跟 input 一致
 * @note 配套 SiLUNode 使用, 独立函数方便测试
 */
Tensor silu_backward(const Tensor& grad_output, const Tensor& input);

}  // namespace ct::ops

#endif  // CTORCH_OPS_SILU_H
