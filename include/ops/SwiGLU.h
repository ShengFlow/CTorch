/**
 * @file SwiGLU.h
 * @brief SwiGLU 算子 — Eager CPU 入口
 * @details 数学定义: swiglu(x, gate) = silu(x) * gate
 *                          = (x * sigmoid(x)) * gate
 *          LLaMA / PaLM FFN 核心组件 (CGO 2027 论文战略价值)
 * @author Mavis (PEL25 §6 SwiGLU 算子开发协议 Stage 1)
 * @date 2026-09-05
 *
 * Stage 1 范围: Eager CPU 路径 (BASIC + SIMD AVX2)
 * Stage 2: 接 C3 JIT + Region Fusion (PEL25 §7 ABI HITL 决策门)
 */

#ifndef CTORCH_OPS_SWIGLU_H
#define CTORCH_OPS_SWIGLU_H

#include "Tensor.h"
#include "ops/SiLU.h"

namespace ct::ops {

/**
 * @brief SwiGLU 标量参考实现 (用于测试对照)
 * @details swiglu(x, gate) = silu(x) * gate
 */
inline float swiglu_scalar(float x, float gate) {
    return silu_scalar(x) * gate;
}

/**
 * @brief SwiGLU backward 参考实现 (双输入梯度)
 * @details ∂L/∂x = ∂L/∂y * gate * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
 *                  = ∂L/∂y * gate * silu_derivative(x)
 *          ∂L/∂gate = ∂L/∂y * silu(x) = ∂L/∂y * x * sigmoid(x)
 * @return pair<grad_x, grad_gate>
 */
inline std::pair<float, float> swiglu_backward_scalar(float grad_y, float x, float gate) {
    float s = 1.0f / (1.0f + std::exp(-x));  // sigmoid(x)
    float silu_d = s + x * s * (1.0f - s);    // d silu / dx
    float grad_x = grad_y * gate * silu_d;
    float grad_gate = grad_y * (x * s);        // silu(x) = x * s
    return {grad_x, grad_gate};
}

/**
 * @brief SwiGLU forward: 元素级 silu(x) * gate
 * @param x 输入 Tensor (主输入, 任意 shape, 必须 contiguous)
 * @param gate 输入 Tensor (门控信号, shape 必须跟 x 一致)
 * @return 输出 Tensor, shape 跟 x 一致
 * @note Eager CPU 路径, 不走 C3 dispatch
 * @note 形状必须匹配 (会校验, 不做 broadcasting, Stage 1 范围)
 */
Tensor swiglu_forward(const Tensor& x, const Tensor& gate);

/**
 * @brief SwiGLU backward: 双输入梯度
 * @param grad_output 上游梯度 (跟 x 同 shape)
 * @param x 原始主输入
 * @param gate 原始门控输入
 * @return pair<grad_x, grad_gate>, shape 都跟 x 一致
 */
std::pair<Tensor, Tensor> swiglu_backward(const Tensor& grad_output, const Tensor& x, const Tensor& gate);

}  // namespace ct::ops

#endif  // CTORCH_OPS_SWIGLU_H
