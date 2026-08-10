/**
 * @file FusionCostModel.h
 * @brief 区域融合的读写次数成本模型
 * @details 基于"数据访问次数"（data_read/data_write）量化算子融合的收益。
 *
 *          动机：Eager 热点采样显示，数据搬运（data_read/data_write）是主要瓶颈，
 *          而非计算本身。融合的核心收益在于消除中间张量的 materialization：
 *          - 分开执行时，每个中间节点输出写一次、被下一节点读一次（2 次访问）
 *          - 融合执行时，中间结果不落主存，仅外部输入读 + 最终输出写
 *
 *          成本模型据此估计：
 *          - eager_accesses : 分开执行的总数据访问次数
 *          - saved_accesses : 融合节省的数据访问次数
 *          - gain_ratio     : 节省比例（saved / eager）
 *          - worthwhile     : 是否值得融合（节省比例超过阈值）
 *
 *          特例：计算密集型节点（如 MatMul）的输出仍需写回主存，
 *          融合并不省其写+读，故不计入 saved_accesses。
 * @date 2026/08/07
 */
#ifndef CTORCH_C3_FUSION_COST_MODEL_H
#define CTORCH_C3_FUSION_COST_MODEL_H

#include <cstddef>
#include <vector>

#include "Ctools.h"

namespace ct {
namespace c3 {

/// 融合收益估计结果
struct FusionCost {
    size_t eager_accesses = 0; ///< 分开执行的总数据访问次数（读+写）
    size_t fused_accesses = 0; ///< 融合执行的总数据访问次数
    size_t saved_accesses = 0; ///< 融合节省的数据访问次数
    double gain_ratio = 0.0;   ///< 节省比例（saved / eager），0~1
    bool worthwhile = false;   ///< 是否值得融合（节省比例 >= 阈值）
};

/**
 * @class FusionCostModel
 * @brief 纯函数式读写次数成本模型，无状态，可直接单元测试
 */
class FusionCostModel {
public:
    /// 默认最小收益比阈值：节省比例低于该值则视为不值得融合
    static constexpr double kDefaultMinGainRatio = 0.20;

    /// 获取当前阈值
    static double minGainRatio() { return kMinGainRatio; }
    /// 设置阈值（全局）
    static void setMinGainRatio(double ratio) { kMinGainRatio = ratio; }

    /**
     * @brief 判断节点是否为计算密集型（输出仍需写回主存，融合不省其写+读）
     * @param op_type 算子类型
     * @return true 表示计算密集型（如 MatMul）
     */
    static bool isComputeIntensive(op op_type) {
        return op_type == op::MatMul;
    }

    /**
     * @brief 估计融合相对分开执行的数据访问收益
     * @param op_seq     算子序列（按执行顺序）
     * @param out_numels 每个算子的输出元素数（长度必须等于 op_seq.size()）
     * @return FusionCost 收益估计
     * @note out_numels 与 op_seq 长度不一致时按较短者处理，避免越界。
     */
    static FusionCost estimate(const std::vector<op>& op_seq,
                               const std::vector<size_t>& out_numels) {
        FusionCost cost;
        const size_t n = op_seq.size() < out_numels.size()
                             ? op_seq.size()
                             : out_numels.size();
        if (n == 0) return cost;

        // eager 分开执行：每个节点写输出 + 每个中间节点被读一次
        size_t write_all = 0, read_intermediate = 0;
        for (size_t i = 0; i < n; ++i) {
            write_all += out_numels[i];
            if (i + 1 < n) read_intermediate += out_numels[i]; // 非末节点被读
        }
        cost.eager_accesses = write_all + read_intermediate;

        // 融合节省：每个非计算密集型中间节点的 (1 写 + 1 读)
        for (size_t i = 0; i + 1 < n; ++i) {
            if (!isComputeIntensive(op_seq[i])) {
                cost.saved_accesses += out_numels[i] * 2;
            }
        }

        cost.fused_accesses = cost.eager_accesses > cost.saved_accesses
                                  ? cost.eager_accesses - cost.saved_accesses
                                  : 0;

        if (cost.eager_accesses > 0) {
            cost.gain_ratio =
                static_cast<double>(cost.saved_accesses) / cost.eager_accesses;
        }
        cost.worthwhile =
            cost.saved_accesses > 0 && cost.gain_ratio >= kMinGainRatio;

        return cost;
    }

private:
    static double kMinGainRatio;
};

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_FUSION_COST_MODEL_H
