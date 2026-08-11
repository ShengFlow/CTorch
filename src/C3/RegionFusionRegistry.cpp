/**
 * @file RegionFusionRegistry.cpp
 * @brief 区域融合注册表实现
 * @date 2026/08/05
 */
#include "C3/RegionFusion.h"

namespace ct {
namespace c3 {

/// 成本模型阈值（默认 20% 收益比）
double FusionCostModel::kMinGainRatio = FusionCostModel::kDefaultMinGainRatio;

RegionFusionRegistry& RegionFusionRegistry::getInstance() {
    static RegionFusionRegistry instance;
    // 确保 RollingHash 的 pow_base 表已初始化
    RollingHash::precompute(256);
    return instance;
}

void RegionFusionRegistry::install(uint64_t hash,
                                    const std::vector<op>& op_seq,
                                    std::shared_ptr<CompiledKernel> kernel,
                                    const std::vector<size_t>& input_shapes) {
    std::lock_guard<std::mutex> lock(mutex_);
    RegionEntry entry;
    entry.hash = hash;
    entry.op_seq = op_seq;
    entry.kernel = std::move(kernel);
    entry.input_shapes = input_shapes;
    entry.num_external_inputs = input_shapes.empty() ? 0 : 1;
    entry.len = op_seq.size();
    entry.active = true;
    entries_[hash] = std::move(entry);
    // [Dev 2026-08-11 tryRegionDispatch 位掩码] 同步末尾 op 位掩码 (atomic 置位, 无锁)
    if (!op_seq.empty()) {
        installed_last_ops_.fetch_or(
            uint64_t(1) << static_cast<size_t>(op_seq.back()),
            std::memory_order_release);
    }
    installed_count_.fetch_add(1, std::memory_order_release);
}

void RegionFusionRegistry::installFromCompiledKernel(
    const std::vector<op>& op_seq,
    std::shared_ptr<CompiledKernel> kernel) {
    // 计算 RollingHash
    RollingHash::precompute(256);
    auto prefix = RollingHash::computePrefixHashes(op_seq);
    uint64_t hash = RollingHash::getSubHash(prefix, 0, op_seq.size() - 1);

    // 扁平化输入形状（从 kernel 的 shapes 中提取）
    std::vector<size_t> input_shapes;

    std::lock_guard<std::mutex> lock(mutex_);
    RegionEntry entry;
    entry.hash = hash;
    entry.op_seq = op_seq;
    entry.kernel = std::move(kernel);
    entry.fused_func_ptr = nullptr;  // 函数指针由外部设置
    entry.input_shapes = input_shapes;
    entry.num_external_inputs = 0;
    entry.len = op_seq.size();
    entry.active = true;
    entries_[hash] = std::move(entry);
    // [Dev 2026-08-11 tryRegionDispatch 位掩码] 同步末尾 op 位掩码 (atomic 置位, 无锁)
    if (!op_seq.empty()) {
        installed_last_ops_.fetch_or(
            uint64_t(1) << static_cast<size_t>(op_seq.back()),
            std::memory_order_release);
    }
    installed_count_.fetch_add(1, std::memory_order_release);
}

void RegionFusionRegistry::installWithCost(
    const std::vector<op>& op_seq,
    std::shared_ptr<CompiledKernel> kernel,
    const std::vector<size_t>& out_numels,
    const std::vector<std::vector<size_t>>& first_input_shapes) {
    if (op_seq.empty() || !kernel) return;

    // 用成本模型评估融合收益；形状缺失或收益不足则不激活
    FusionCost cost = FusionCostModel::estimate(op_seq, out_numels);

    RollingHash::precompute(256);
    auto prefix = RollingHash::computePrefixHashes(op_seq);
    uint64_t op_hash = RollingHash::getSubHash(prefix, 0, op_seq.size() - 1);

    // DEBT-NEW-7 关键修复:把 first_input_shapes[0] 的 shape 信息混入 hash,
    // 避免不同 shape 的同 op_seq region 互相覆盖(MNIST 三层 W1/W2/W3 同 op_seq)
    uint64_t shape_hash = 0;
    if (!first_input_shapes.empty()) {
        for (auto s : first_input_shapes.front()) {
            shape_hash = shape_hash * 31 + s + 1;
        }
    }
    uint64_t hash = op_hash ^ (shape_hash << 32);

    std::lock_guard<std::mutex> lock(mutex_);
    RegionEntry entry;
    entry.hash = hash;
    entry.op_seq = op_seq;
    entry.kernel = std::move(kernel);
    entry.fused_func_ptr = nullptr;
    entry.num_external_inputs = 0;
    entry.len = op_seq.size();
    entry.cost = cost;
    entry.first_input_shapes = first_input_shapes;
    // DEBT-NEW-7 P1 修复:恢复成本模型 gating
    // 仅当:
    //   1. out_numels 跟 op_seq 长度一致(基本完整性检查)
    //   2. 成本模型判定值得融合(节省的内存/launch > 阈值)
    // 才激活 region entry。临时"全部激活"代码已删除,避免低收益 region 拖累 perf。
    bool basic_valid = (out_numels.size() == op_seq.size());
    bool worth_it = cost.worthwhile;  // FusionCost.worthwhile 是字段,不是函数
    entry.active = basic_valid && worth_it;
    entries_[hash] = std::move(entry);
    // [Dev 2026-08-11 tryRegionDispatch 位掩码] 同步末尾 op 位掩码
    //   installWithCost 路径: 即使 cost 判定不值, 仍置位 last op 位掩码
    //   (实际 match 时 entry.active=false 会被跳过, 掩码仅用于快速 O(1) 过滤)
    if (!op_seq.empty()) {
        installed_last_ops_.fetch_or(
            uint64_t(1) << static_cast<size_t>(op_seq.back()),
            std::memory_order_release);
    }
    installed_count_.fetch_add(1, std::memory_order_release);
}

RegionEntry* RegionFusionRegistry::find(uint64_t hash) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(hash);
    if (it != entries_.end() && it->second.active) {
        return &it->second;
    }
    return nullptr;
}

RegionEntry* RegionFusionRegistry::matchFromPosition(
    const std::vector<uint64_t>& prefix_hashes,
    size_t current_pos,
    const std::vector<size_t>& /*input_shapes*/) {
    if (current_pos >= prefix_hashes.size() - 1) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // 从最长到最短尝试匹配
    size_t max_len = prefix_hashes.size() - 1 - current_pos;
    // 限制最大尝试长度，避免过度遍历
    if (max_len > 32) max_len = 32;

    for (size_t len = max_len; len >= 2; --len) {
        size_t end = current_pos + len - 1;
        uint64_t sub_hash = RollingHash::getSubHash(prefix_hashes, current_pos, end);

        auto it = entries_.find(sub_hash);
        if (it != entries_.end() && it->second.active && it->second.len == len) {
            return &it->second;
        }
    }

    return nullptr;
}

size_t RegionFusionRegistry::entryCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

bool RegionFusionRegistry::mayMatchAsLastOp(op last_op) const {
    // [Dev 2026-08-11 tryRegionDispatch 位掩码] O(1) 无锁位测试
    //   atomic load + 位测试, 消除每次 dispatch 的 last_op_mutex_ 锁
    return (installed_last_ops_.load(std::memory_order_acquire) &
            (uint64_t(1) << static_cast<size_t>(last_op))) != 0;
}

void RegionFusionRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    installed_last_ops_.store(0, std::memory_order_release);
    installed_count_.store(0, std::memory_order_release);
}

void RegionFusionRegistry::uninstallAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [_, entry] : entries_) {
        entry.active = false;
    }
    entries_.clear();
    installed_last_ops_.store(0, std::memory_order_release);
    installed_count_.store(0, std::memory_order_release);
}

} // namespace c3
} // namespace ct