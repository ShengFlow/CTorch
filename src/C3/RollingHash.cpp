/**
 * @file RollingHash.cpp
 * @brief 可变进制滚动哈希工具实现
 * @date 2026/08/05
 */
#include "C3/RollingHash.h"

namespace ct {
namespace c3 {

std::vector<uint64_t> RollingHash::pow_base_;
size_t RollingHash::max_len_ = 0;

void RollingHash::precompute(size_t max_len) {
    if (max_len <= max_len_) return;
    pow_base_.resize(max_len + 1);
    pow_base_[0] = 1;
    for (size_t i = 1; i <= max_len; ++i) {
        pow_base_[i] = pow_base_[i - 1] * kBase;
    }
    max_len_ = max_len;
}

std::vector<uint64_t> RollingHash::computePrefixHashes(const std::vector<op>& ops) {
    size_t n = ops.size();
    if (n > max_len_) {
        precompute(n);
    }
    std::vector<uint64_t> prefix(n + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        prefix[i + 1] = prefix[i] * kBase + getOpCode(ops[i]);
    }
    return prefix;
}

uint64_t RollingHash::getSubHash(const std::vector<uint64_t>& prefix, size_t l, size_t r) {
    // hash = prefix[r+1] - prefix[l] * pow_base[r-l+1]
    size_t len = r - l + 1;
    return prefix[r + 1] - prefix[l] * pow_base_[len];
}

uint64_t RollingHash::getOpCode(op op_type) {
    // 每个 op 分配一个唯一质数编码
    switch (op_type) {
        case op::Add:     return 2;
        case op::Sub:     return 3;
        case op::Neg:     return 5;
        case op::Mul:     return 7;
        case op::Div:     return 11;
        case op::MatMul:  return 13;
        case op::Dot:     return 17;
        case op::Cos:     return 19;
        case op::Sin:     return 23;
        case op::ReLU:    return 29;
        case op::Tanh:    return 31;
        case op::Sigmoid: return 37;
        case op::GELU:    return 41;
        case op::LReLU:   return 43;
        case op::Log:     return 47;
        case op::Exp:     return 53;
        case op::Abs:     return 59;
        case op::Softmax: return 61;
        case op::Min:     return 67;
        case op::Max:     return 71;
        case op::MSE:     return 73;
        case op::CE:      return 79;
        case op::MAE:     return 83;
        case op::Conv:    return 89;
        case op::Pool:    return 97;
        default:          return 101; // kCount or unknown
    }
}

} // namespace c3
} // namespace ct