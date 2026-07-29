/**
 * @file CrossEntropyNode.h
 * @author Beapoe
 * @brief 交叉熵损失节点实现
 *
 * @details 数学原理:
 *
 * 前向传播 (Forward):
 *   给定 logits 向量 z = [z₁, z₂, ..., zₙ] 和目标概率/标签 y = [y₁, y₂, ..., yₙ]:
 *
 *   交叉熵损失定义为:
 *     L = -Σᵢ yᵢ · log(softmax(z)ᵢ)
 *
 *   展开为:
 *     L = -Σᵢ yᵢ · zᵢ + Σᵢ yᵢ · log(Σⱼ exp(zⱼ))
 *
 *   其中 softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
 *
 * 反向传播 (Backward):
 *   对于损失函数 L 对 logits z 的梯度:
 *
 *   ∂L/∂zᵢ = -yᵢ + yᵢ · exp(zᵢ) / Σⱼ exp(zⱼ)
 *           = softmax(z)ᵢ - yᵢ
 *
 *   简化为:
 *     ∂L/∂z = softmax(z) - y
 *
 * 总结:
 *   grad_logits = softmax(logits) - target
 *
 * 数值稳定性:
 *   - 使用 Log-Softmax 的数值稳定实现
 *   - Softmax 确保概率在 (0, 1) 范围内
 *   - 形状验证确保输入维度正确
 *
 * @date 2026/4/5
 **/

#ifndef CTORCH_CROSSENTROPYNODE_H
#define CTORCH_CROSSENTROPYNODE_H

#include "AutoGrad/Node.h"

/**
 * @class CrossEntropyNode
 * @brief 交叉熵损失节点，实现分类任务的损失函数
 *
 * @note 输入 logits 应为 2D 张量 (batch_size, num_classes)
 * @note 目标 target 可以是 2D 概率分布或 1D 类别标签
 */
class CrossEntropyNode final : public Node {
public:
    CrossEntropyNode() = default;

    /**
     * @brief 构造交叉熵损失节点
     * @param upStreamNodes 上游节点列表 (长度为2)
     * @param inputs 输入张量列表 [logits, target]
     */
    CrossEntropyNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs);

    /**
     * @brief 移动构造交叉熵损失节点
     * @param upStreamNodes 上游节点列表 (右值引用)
     * @param inputs 输入张量列表 (右值引用)
     */
    CrossEntropyNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs);

    /**
     * @brief 构造交叉熵损失节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    CrossEntropyNode(const std::vector<std::shared_ptr<Node>>& upStreamNodes, const std::vector<Tensor>& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 移动构造交叉熵损失节点 (带输出张量)
     * @param upStreamNodes 上游节点列表
     * @param inputs 输入张量列表
     * @param result 输出张量的弱引用
     */
    CrossEntropyNode(std::vector<std::shared_ptr<Node>>&& upStreamNodes, std::vector<Tensor>&& inputs,
            const std::weak_ptr<Tensor>& result);

    /**
     * @brief 反向传播
     * @param downStreamGrads 下游梯度 [∂L/∂loss]
     * @return 梯度包列表 [GradPack(∂L/∂logits)]
     *
     * 数学公式:
     *   grad_logits = softmax(logits) - target
     */
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;
};

#endif  // CTORCH_CROSSENTROPYNODE_H
