/**
 * @file RegionFusion.h
 * @brief 区域融合注册表
 * @details 基于 Rolling Hash 的 region kernel 注册与查找。
 *          支持 O(1) 哈希匹配和最长匹配优先策略。
 * @date 2026/08/05
 */
#ifndef CTORCH_C3_REGION_FUSION_H
#define CTORCH_C3_REGION_FUSION_H

#include "RollingHash.h"
#include "C3Engine.h"
#include "C3KernelRegistry.h"
#include "FusionCostModel.h"
#include "../../include/Tensor.h"
#include "../../include/Ctools.h"

#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <cstdint>
#include <atomic>

namespace ct {
namespace c3 {

/// 单个 region 入口描述符
struct RegionEntry {
    uint64_t hash;                          ///< Rolling Hash of op sequence
    std::vector<op> op_seq;                 ///< 算子序列
    std::vector<size_t> input_shapes;       ///< 外部输入形状（扁平化）
    /// 首个 op 的输入形状（向量列表，顺序对应首个 op 的各输入）。
    /// 用于预走匹配时的形状校验：仅当当前 op 的输入形状与注册时的首个 op 输入形状
    /// 完全一致才视为匹配，避免反向传播（形状不同）错误匹配前向注册的区域，
    /// 防止形状不匹配的 fused kernel 执行（与 tryFusedDispatch 同源缺陷）。
    /// 为空表示不校验形状（向后兼容）。
    std::vector<std::vector<size_t>> first_input_shapes;
    std::shared_ptr<CompiledKernel> kernel;  ///< 编译好的 region kernel
    FusedKernelFunc fused_func_ptr = nullptr; ///< 直接函数指针（避免 CompiledKernel 虚函数调用开销）
    size_t num_external_inputs = 0;         ///< 外部输入数量
    size_t len = 0;                         ///< op 序列长度
    bool active = false;
    FusionCost cost;                        ///< 融合读写次数成本估计（决定是否值得融合）
};

/// 区域融合注册表（线程安全）
class RegionFusionRegistry {
public:
    static RegionFusionRegistry& getInstance();

    /// 注册一个 region kernel
    void install(uint64_t hash,
                 const std::vector<op>& op_seq,
                 std::shared_ptr<CompiledKernel> kernel,
                 const std::vector<size_t>& input_shapes);

    /// 从编译好的 kernel 自动安装（由 C3HotPathManager 编译完成后调用）
    /// @param op_seq 算子序列
    /// @param kernel 编译好的 CompiledKernel
    void installFromCompiledKernel(const std::vector<op>& op_seq,
                                    std::shared_ptr<CompiledKernel> kernel);

    /// 从编译好的 kernel 安装，并附加读写次数成本模型评估
    /// @param op_seq     算子序列（按执行顺序）
    /// @param kernel     编译好的 CompiledKernel
    /// @param out_numels 每个算子的输出元素数（长度等于 op_seq.size()）
    /// @param first_input_shapes 首个 op 的输入形状（用于预走匹配时形状校验，可空）
    /// @note 仅当成本模型判定融合值得（saved_accesses > 0 且收益比例达阈值）时才激活。
    ///       若输入形状缺失（out_numels 为空），保守按不激活处理，避免低收益融合。
    void installWithCost(const std::vector<op>& op_seq,
                          std::shared_ptr<CompiledKernel> kernel,
                          const std::vector<size_t>& out_numels,
                          const std::vector<std::vector<size_t>>& first_input_shapes = {});

    /// 根据哈希查找匹配的 region 入口
    RegionEntry* find(uint64_t hash);

    /// 从当前位置向后尝试匹配（最长匹配优先）
    /// @param prefix_hashes 前缀哈希数组
    /// @param current_pos 当前位置（0-based，候选 region 的第一个 op 位置）
    /// @param input_shapes 当前 op 的输入形状
    /// @return 匹配到的 entry 或 nullptr
    RegionEntry* matchFromPosition(
        const std::vector<uint64_t>& prefix_hashes,
        size_t current_pos,
        const std::vector<size_t>& input_shapes);

    size_t entryCount() const;

    /// [Dev 2026-08-09 tryRegionDispatch 无候选短路] 无锁 O(1) 查询已注册 region 数量
    /// @details tryRegionDispatch 入口第一道短路: 0 region 时直接返回 nullopt
    ///          省掉 trace 拷贝 + extended hash + shape hash + 7 次循环
    ///          用 atomic 避免每次都加锁
    size_t installedCountNoLock() const {
        return installed_count_.load(std::memory_order_acquire);
    }

    /// [Dev 2026-08-11 tryRegionDispatch 位掩码] 候选过滤
    /// @details tryRegionDispatch 入口第二道短路: 当前 op 不可能作为任何已注册
    ///          region 的末尾 op 时直接返回 false
    ///          实现: 查 installed_last_ops_ 位掩码 (atomic load + 位测试, O(1) 无锁)
    /// @param last_op 候选 region 的最后一个 op (即当前 dispatch 的 op)
    /// @return 是否有任何已注册 region 以 last_op 结尾
    bool mayMatchAsLastOp(op last_op) const;

    void clear();
    void uninstallAll();

private:
    RegionFusionRegistry() = default;
    RegionFusionRegistry(const RegionFusionRegistry&) = delete;
    RegionFusionRegistry& operator=(const RegionFusionRegistry&) = delete;

    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, RegionEntry> entries_;

    /// [Dev 2026-08-09 tryRegionDispatch 无候选短路] 已注册 region 数量 (无锁查询)
    mutable std::atomic<size_t> installed_count_{0};

    /// [Dev 2026-08-11 tryRegionDispatch 位掩码] 已注册 region 末尾 op 位掩码
    /// @details tryRegionDispatch 入口查 installed_last_ops_ 判断当前 op 是否可能
    ///          作为 region 末尾. install 时置位, uninstallAll/clear 时清空.
    ///          用 atomic<uint64_t> 位掩码: mayMatchAsLastOp 变成 O(1) 无锁
    ///          (atomic load + 位测试), 消除热路径每次 dispatch 的 last_op_mutex_ 锁.
    ///          op 是连续枚举 (0..kCount-1), 单 uint64 足够 (kCount=28 < 64).
    static_assert(static_cast<size_t>(op::kCount) <= 64,
                  "op::kCount exceeds uint64 bitmask capacity");
    mutable std::atomic<uint64_t> installed_last_ops_{0};
};

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_REGION_FUSION_H