/**
 * @file ConcatNode.h
 * @author 苏璃珞
 * @brief 拼接节点定义
 *
 * @details 数学原理:
 *
 * 前向传播 (Forward):
 *   沿 dim 维把两个张量首尾相接:
 *     c = concat(a, b, dim)   要求除 dim 外各维相同
 *     c.shape[dim] = a.shape[dim] + b.shape[dim]
 *
 * 反向传播 (Backward):
 *   拼接是「两路合并」，其伴随是把梯度沿 dim 切回两段:
 *     grad_a = grad_c.slice(dim, 0,               a.shape[dim])
 *     grad_b = grad_c.slice(dim, a.shape[dim],    b.shape[dim])
 *
 *   与切片恰好互为伴随：切片的前向是取子区间，反向是散射补零；
 *   拼接的前向是首尾相接，反向是切回子区间。因此本节点的反向直接复用 slice。
 *
 * @note 前向不走调度器。原因有二：
 *       1. 拼接需要 dim 参数，而调度器的双输入 kernel 签名固定为 (a, b)，无法携带；
 *       2. 数据搬运只在 Tensor 方法内部用指针完成，无需 kernel。
 *       本节点只承担反向传播所需的图连接，因而也不必新增 op 枚举项。
 *
 * @warning 前向要求两个输入都**连续**。非连续输入会先做一次 contiguous()
 *          （见 Tensor::concat），否则按块拷贝会读到错乱的布局。
 *
 * @date 2026/9/17
 **/

#ifndef CTORCH_CONCATNODE_H
#define CTORCH_CONCATNODE_H

#include "AutoGrad/Node.h"

#include <cstddef>

/**
 * @class ConcatNode
 * @brief 拼接运算节点，实现 concat(a, b, dim) 的反向传播
 */
class ConcatNode final : public Node {
public:
    ConcatNode() = default;

    /**
     * @brief 构造拼接节点
     * @param dim        拼接维度
     * @param a_dim_len  第一个输入在 dim 维上的长度（用于切分梯度）
     * @param upStreamNodes 上游节点列表（长度为 2）
     * @param inputs 输入张量列表 [a, b]
     * @param result 输出张量的弱引用
     */
    ConcatNode(int dim, std::size_t a_dim_len,
               std::vector<std::shared_ptr<Node>>&& upStreamNodes,
               std::vector<Tensor>&& inputs,
               const std::weak_ptr<Tensor>& result);

    /**
     * @brief 反向传播
     * @param downStreamGrads 下游梯度 [∂L/∂c]
     * @return 梯度包列表 [GradPack(∂L/∂a), GradPack(∂L/∂b)]
     */
    std::vector<GradPack> backward(const std::vector<Tensor>& downStreamGrads) override;

private:
    int _dim = 0;
    std::size_t _a_dim_len = 0;
};

#endif  // CTORCH_CONCATNODE_H
