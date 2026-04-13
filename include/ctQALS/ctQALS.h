//
// Created by GhostFace on 2026/4/12.
//

#ifndef CTORCH_CTQALS_H
#define CTORCH_CTQALS_H
#pragma once
#include "Tensor.h"

#include <cstdint>
namespace ctQALS{
/**
 * 截断策略
 */
enum class TruncateMode {
    FIXED_RANK,      // 使用固定最大秩
    ENERGY_RATIO     // 保留指定比例的奇异值能量
};

/**
 * TT 分解配置
 */
struct TTDecomposeConfig {
    TruncateMode mode = TruncateMode::FIXED_RANK;
    std::int64_t max_rank = 16;          // 仅当 mode == FIXED_RANK 时有效
    double energy_ratio = 0.999;    // 仅当 mode == ENERGY_RATIO 时有效
};

/**
 * 将稠密矩阵分解为 Tensor Train 格式。
 *
 * @param W          输入矩阵，形状 [out_dim, in_dim]
 * @param in_shape   输入特征各维度的因子，乘积 = in_dim
 * @param out_shape  输出特征各维度的因子，乘积 = out_dim
 * @param config     分解配置
 * @return           TT 核心列表，每个核心为四维张量 [r_{k-1}, in_d_k, out_d_k, r_k]
 */
std::vector<Tensor> tt_decompose(const Tensor& W,
                                 const std::vector<int64_t>& in_shape,
                                 const std::vector<int64_t>& out_shape,
                                 const TTDecomposeConfig& config = {});

/**
 * 使用 TT 核心执行批量矩阵-向量乘法。
 *
 * @param cores       tt_decompose 返回的核心列表
 * @param batch_input 批量输入向量，形状 [batch_size, total_in_dim]
 * @return            批量输出向量，形状 [batch_size, total_out_dim]
 */
Tensor tt_matmul(const std::vector<Tensor>& cores, const Tensor& batch_input);

/**
 * 计算压缩统计信息
 */
struct TTCompressionStats {
    size_t original_elements;   // 原始矩阵元素数
    size_t compressed_elements; // 核心总元素数
    double compression_ratio;   // 压缩比 = original / compressed
    double mse;                 // 重建均方误差（基于随机测试输入）
};

TTCompressionStats tt_analyze(const Tensor& W,
                              const std::vector<int64_t>& in_shape,
                              const std::vector<int64_t>& out_shape,
                              const TTDecomposeConfig& config,
                              int num_test_samples = 100);

}
#endif //CTORCH_CTQALS_H
