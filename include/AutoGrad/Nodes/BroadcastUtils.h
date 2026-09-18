/**
 * @file BroadcastUtils.h
 * @brief 自动微分节点中处理广播（broadcast）的公共工具函数
 * @author CTorch
 * @date 2026/07/30
 *
 * @details 提供按 NumPy/CTorch 广播规则计算梯度归约维度的工具函数，
 *          供 AddNode、SubNode、MulNode、DivNode 等逐元素二元运算节点复用。
 */

#ifndef CTORCH_BROADCAST_UTILS_H
#define CTORCH_BROADCAST_UTILS_H

#include <cstddef>
#include <vector>

#include "CtorchError.h"

namespace ctorch {
namespace autograd {

/**
 * @brief 计算 broadcast 梯度归约维度。
 *
 * 对有效广播对 (input_shape, grad_shape)，按最右对齐（NumPy 风格）比较：
 * 当 input 维度为 1 而 grad 对应维度大于 1 时，该维度在反向传播中需要被求和规约。
 *
 * @param input_shape 输入张量的形状
 * @param grad_shape  下游梯度张量的形状（通常大于等于 input_shape 的维度）
 * @return 需要归约的维度索引列表（按升序排列）
 */
inline std::vector<int> compute_broadcast_reduce_dims(
    const std::vector<size_t>& input_shape,
    const std::vector<size_t>& grad_shape) {
    // 前置条件：合法的 PyTorch/NumPy 广播要求 input 维度数不超过 grad 维度数。
    //
    // [Fix 2026-09-17] 例外：input 多出来的**前导维度若全为 1**，等价于标量，
    // 不构成非法广播对。这类情形来自 CTorch 内部两种标量形状的混用 ——
    // `sum()` / `dot()` 等全归约返回的是 **0 维张量**（`{}`），而标量运算路径
    // 常构造 **`{1}`**；两者语义相同但维度数不同。混用时（例如 `标量 * 0.5`
    // 之后再与 0 维张量相加）就会走到这里，若直接判非法会抛异常中断反向。
    //
    // 这类输入不需要归约（多余维度长度为 1，梯度与输入本就一一对应），
    // 返回空列表即可；调用方随后会按 input_shape 做 reshape 对齐。
    if (input_shape.size() > grad_shape.size()) {
        for (std::size_t d = 0; d + grad_shape.size() < input_shape.size(); ++d) {
            if (input_shape[d] != 1) {
                CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION,
                    "BroadcastUtils: input 维度数大于 grad 维度数，不是合法广播对");
            }
        }
        return {};
    }

    std::vector<int> reduce_dims;
    size_t in_dims = input_shape.size();
    size_t g_dims = grad_shape.size();

    // 对齐到最右边，避免依赖 dim() 相等（标量 Tensor 的 shape 可能是 {1}）。
    for (size_t d = 0; d < g_dims; ++d) {
        size_t grad_dim_size = grad_shape[g_dims - 1 - d];
        size_t input_dim_size = (d < in_dims) ? input_shape[in_dims - 1 - d] : 1;
        if (input_dim_size == 1 && grad_dim_size > 1) {
            reduce_dims.push_back(static_cast<int>(g_dims - 1 - d));
        }
    }
    return reduce_dims;
}

} // namespace autograd
} // namespace ctorch

#endif // CTORCH_BROADCAST_UTILS_H
