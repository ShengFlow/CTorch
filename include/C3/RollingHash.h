/**
 * @file RollingHash.h
 * @brief 可变进制滚动哈希工具
 * @details 为 op 序列提供 O(1) 子序列哈希提取能力，用于区域融合匹配。
 *          每个 op 类型分配一个唯一质数编码，使用 BASE=131, MOD=2^64 自然溢出。
 * @date 2026/08/05
 */
#ifndef CTORCH_C3_ROLLING_HASH_H
#define CTORCH_C3_ROLLING_HASH_H

#include <vector>
#include <cstdint>
#include <cstddef>
#include "../../include/Ctools.h"

namespace ct {
namespace c3 {

class RollingHash {
public:
    static constexpr uint64_t kBase = 131;

    /// 预计算 pow_base 表，支持最大长度为 max_len 的序列
    static void precompute(size_t max_len);

    /// 从 op 序列计算前缀哈希数组
    /// prefix[i] = hash of [ops[0], ..., ops[i-1]] (prefix[0] = 0)
    static std::vector<uint64_t> computePrefixHashes(const std::vector<op>& ops);

    /// 提取子序列 [l, r] (闭区间, 0-based) 的哈希
    /// 需要 prefix 数组长度 > r+1
    static uint64_t getSubHash(const std::vector<uint64_t>& prefix, size_t l, size_t r);

    /// 获取 op 的质数编码
    static uint64_t getOpCode(op op_type);

    /// 获取当前预计算的最大长度
    static size_t maxLen() { return max_len_; }

private:
    static std::vector<uint64_t> pow_base_;
    static size_t max_len_;
};

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_ROLLING_HASH_H