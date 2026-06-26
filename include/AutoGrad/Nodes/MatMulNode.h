/**
 * @file MatMulNode.h
 * @author Beapoe
 * @brief 矩阵乘法节点实现
 *
 * @details 数学原理:
 *
 * 前向传播 (Forward):
 *   给定两个矩阵 A (m×n) 和 B (n×k)，矩阵乘法定义为:
 *     C = A × B
 *   其中 C 是 m×k 的矩阵，满足:
 *     C[i][j] = Σ(A[i][k] * B[k][j]) for k in [0, n-1]
 *
 * 反向传播 (Backward):
 *   设 Y = A × B，对于损失函数 L:
 *
 *   对于输入 A 的梯度:
 *     ∂L/∂A = ∂L/∂Y × Bᵀ
 *
 *   对于输入 B 的梯度:
 *     ∂L/∂B = Aᵀ × ∂L/∂Y
 *
 * 总结:
 *   grad_A = downStreamGrad @ B.T()
 *   grad_B = A.T() @ downStreamGrad
 *
 * 其中 @ 表示矩阵乘法，T() 表示转置
 *
 * 限制:
 *   - 仅支持 2D 矩阵乘法
 *   - 输入形状必须满足矩阵乘法的维度要求
 *
 * @date 2026/4/5
 **/

#ifndef CTORCH_MATMULNODE_H
#define CTORCH_MATMULNODE_H

#include "AutoGrad/Node.h"

/**
 * @class MatMulNode
 * @brief 矩阵乘法节点，实现 A @ B 的前向和反向传播
 *
 * @warning 仅支持 2D 矩阵乘法
 */
class MatMulNode final : public Node {
public:
    MatMulNode() = default;

    /**
     * @brief 构造矩阵乘法节点
     * @param upStreamNodes 上游节点列表 (长度为2)
     * @param inputs 输入张量列表 [A, B]
     */
    MatMulNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs);

    /**
     * @brief 移动构造矩阵乘法节点
     * @param upStreamNodes 上游节点列表 (右值引用)
     * @param inputs 输入张量列表 (右值引用)
     */
    MatMulNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);

    /**
     * @brief 构造矩阵乘法节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    MatMulNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 移动构造矩阵乘法节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    MatMulNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 反向传播
     * @param downStreamGrads 下游梯度 [∂L/∂C]
     * @return 梯度包列表 [GradPack(∂L/∂A), GradPack(∂L/∂B)]
     *
     * 数学公式:
     *   grad_A = downStreamGrad @ B.T()
     *   grad_B = A.T() @ downStreamGrad
     */
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif  // CTORCH_MATMULNODE_H
