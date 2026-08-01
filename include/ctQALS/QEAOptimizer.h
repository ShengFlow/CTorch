/**
 * @file QEAOptimizer.h
 * @brief Quantum-inspired Evolutionary Algorithm (QEA) optimizer
 * @details Implements QEA based on Han & Kim (2002). Uses Q-bit probability
 *          encoding with rotation gate updates. Suitable for discrete
 *          combinatorial optimization with expensive evaluation functions.
 *
 *          Key features:
 *          - Q-bit superposition: maintains exploration/exploitation balance
 *          - Small population (5-10) due to probabilistic representation
 *          - Rotation gate: converges toward best solution while preserving diversity
 *          - Evaluation cache: avoids re-evaluating duplicate configurations
 * @date 2026/8/1
 */

#ifndef CTORCH_QEA_OPTIMIZER_H
#define CTORCH_QEA_OPTIMIZER_H

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <unordered_map>
#include <vector>

#include "Random.h"

namespace ctQALS {
namespace optimize {

// ============================================================
// QEA 配置
// ============================================================

struct QEAConfig {
    size_t population_size = 10;       ///< 种群大小
    size_t max_generations = 20;       ///< 最大迭代代数
    double rotation_angle = 0.01 * 3.141592653589793;  ///< 旋转角 Δθ (弧度)
    size_t migration_interval = 0;     ///< 迁移间隔 (0 = 不迁移)
    uint64_t seed = 0;                 ///< 随机种子 (0 = 使用时间种子)
    bool verbose = false;              ///< 是否输出调试信息
};

// ============================================================
// QEA 优化结果
// ============================================================

struct QEAResult {
    std::vector<size_t> best_params;   ///< 最优参数索引
    double best_fitness;               ///< 最优适应度 (越小越好)
    size_t evaluations;                ///< 总评估次数
    size_t cache_hits;                 ///< 缓存命中次数
    std::vector<double> fitness_history; ///< 每代最优适应度历史
};

// ============================================================
// QEA 优化器
// ============================================================

/**
 * @class QEAOptimizer
 * @brief 量子启发进化算法优化器
 *
 * @tparam T 参数值类型 (通常为 int 或 size_t)
 *
 * 使用方式:
 * @code
 *   // 定义搜索空间: 3 个参数，每个有 4 个可选值
 *   std::vector<std::vector<int>> space = {
 *       {16, 32, 64, 128},
 *       {16, 32, 64, 128},
 *       {16, 32, 64, 128}
 *   };
 *   QEAOptimizer<int> qea(space);
 *   auto result = qea.optimize([](const std::vector<int>& params) {
 *       return benchmark(params[0], params[1], params[2]);
 *   });
 * @endcode
 */
template <typename T>
class QEAOptimizer {
public:
    /**
     * @param param_space 每个参数的可选值列表
     * @param config  QEA 配置
     */
    QEAOptimizer(std::vector<std::vector<T>> param_space,
                 QEAConfig config = {})
        : param_space_(std::move(param_space))
        , config_(config)
        , n_params_(param_space_.size())
    {
        assert(n_params_ > 0 && "QEA: param_space must not be empty");
        for (size_t i = 0; i < n_params_; ++i) {
            assert(!param_space_[i].empty() && "QEA: each param must have at least 1 value");
            // 计算每个参数需要的 Q-bit 数量
            size_t n_vals = param_space_[i].size();
            size_t n_bits = 1;
            while ((1u << n_bits) < n_vals) ++n_bits;
            n_qbits_per_param_.push_back(n_bits);
        }
    }

    /**
     * @brief 执行优化
     * @param fitness_fn 适应度函数: params → fitness (越小越好)
     * @return 优化结果
     */
    QEAResult optimize(std::function<double(const std::vector<T>&)> fitness_fn);

    /** @brief 获取搜索空间 */
    [[nodiscard]] const std::vector<std::vector<T>>& paramSpace() const { return param_space_; }

private:
    // ---- Q-bit 操作 ----
    struct QBit {
        double alpha;  ///< |0⟩ 振幅
        double beta;   ///< |1⟩ 振幅, α² + β² = 1
    };

    /// 初始化所有 Q-bit 为等概率叠加态 (1/√2, 1/√2)
    void initQBitPopulation(std::vector<std::vector<QBit>>& population);

    /// 观测 Q-bit 群体，坍缩为二进制解
    void observe(const std::vector<std::vector<QBit>>& q_pop,
                 std::vector<std::vector<size_t>>& bin_pop);

    /// 将二进制编码解码为参数索引
    std::vector<size_t> decodeToIndices(const std::vector<size_t>& bits) const;

    /// 将参数索引转换为实际参数值
    T indexToValue(size_t param_idx, size_t value_idx) const {
        return param_space_[param_idx][value_idx];
    }

    /// 旋转门更新: 将当前解向最优解方向旋转
    void rotateQBit(QBit& qb, size_t bit_i, size_t best_bit,
                    double fitness, double best_fitness);

    /// 迁移: 将局部最优分享给其他子种群
    void migrate(std::vector<std::vector<QBit>>& population,
                 const std::vector<size_t>& best_bits,
                 const std::vector<size_t>& best_indices);

    // ---- 成员 ----
    std::vector<std::vector<T>> param_space_;
    QEAConfig config_;
    size_t n_params_;
    std::vector<size_t> n_qbits_per_param_;
    ctQALS::rng::Xoshiro256PlusPlus rng_;
};

// ============================================================
// 实现 (模板类，需在头文件中)
// ============================================================

template <typename T>
QEAResult QEAOptimizer<T>::optimize(
    std::function<double(const std::vector<T>&)> fitness_fn)
{
    QEAResult result;
    result.best_fitness = std::numeric_limits<double>::max();
    result.evaluations = 0;
    result.cache_hits = 0;

    // 评估缓存: 避免重复评估相同配置
    std::unordered_map<size_t, double> eval_cache;
    auto hashConfig = [](const std::vector<size_t>& indices) -> size_t {
        size_t h = 0;
        for (auto v : indices) {
            h ^= std::hash<size_t>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    };

    // 缓存的评估函数
    auto cachedEval = [&](const std::vector<size_t>& indices) -> double {
        size_t h = hashConfig(indices);
        auto it = eval_cache.find(h);
        if (it != eval_cache.end()) {
            result.cache_hits++;
            return it->second;
        }
        std::vector<T> values;
        values.reserve(indices.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            values.push_back(indexToValue(i, indices[i]));
        }
        double f = fitness_fn(values);
        eval_cache[h] = f;
        result.evaluations++;
        return f;
    };

    // ---- 初始化 ----
    size_t pop_size = config_.population_size;
    size_t total_qbits = 0;
    for (auto n : n_qbits_per_param_) total_qbits += n;

    std::vector<std::vector<QBit>> q_pop(pop_size);
    initQBitPopulation(q_pop);

    std::vector<size_t> best_bits(total_qbits, 0);
    std::vector<size_t> best_indices(n_params_, 0);
    double best_fitness = std::numeric_limits<double>::max();

    // 首次观测与评估
    {
        std::vector<std::vector<size_t>> bin_pop(pop_size);
        observe(q_pop, bin_pop);

        for (size_t p = 0; p < pop_size; ++p) {
            auto indices = decodeToIndices(bin_pop[p]);
            double fitness = cachedEval(indices);
            if (fitness < best_fitness) {
                best_fitness = fitness;
                best_bits = bin_pop[p];
                best_indices = indices;
            }
        }
    }

    result.best_fitness = best_fitness;
    result.best_params = best_indices;
    result.fitness_history.push_back(best_fitness);

    if (config_.verbose) {
        printf("[QEA] Gen 0: best_fitness=%.6f, evals=%zu\n",
               best_fitness, result.evaluations);
    }

    // ---- 主循环 ----
    for (size_t gen = 1; gen <= config_.max_generations; ++gen) {
        std::vector<std::vector<size_t>> bin_pop(pop_size);
        observe(q_pop, bin_pop);

        // 评估并更新最优
        for (size_t p = 0; p < pop_size; ++p) {
            auto indices = decodeToIndices(bin_pop[p]);
            double fitness = cachedEval(indices);
            if (fitness < best_fitness) {
                best_fitness = fitness;
                best_bits = bin_pop[p];
                best_indices = indices;
            }
        }

        // 旋转门更新
        for (size_t p = 0; p < pop_size; ++p) {
            auto indices = decodeToIndices(bin_pop[p]);
            double fitness = cachedEval(indices);
            for (size_t b = 0; b < total_qbits; ++b) {
                rotateQBit(q_pop[p][b], bin_pop[p][b], best_bits[b],
                           fitness, best_fitness);
            }
        }

        // 迁移 (可选)
        if (config_.migration_interval > 0 &&
            (gen % config_.migration_interval) == 0) {
            migrate(q_pop, best_bits, best_indices);
        }

        result.best_fitness = best_fitness;
        result.best_params = best_indices;
        result.fitness_history.push_back(best_fitness);

        if (config_.verbose) {
            printf("[QEA] Gen %zu: best_fitness=%.6f, evals=%zu, cache_hits=%zu\n",
                   gen, best_fitness, result.evaluations, result.cache_hits);
        }
    }

    return result;
}

template <typename T>
void QEAOptimizer<T>::initQBitPopulation(
    std::vector<std::vector<QBit>>& population)
{
    size_t total_qbits = 0;
    for (auto n : n_qbits_per_param_) total_qbits += n;
    const double sqrt2_inv = 1.0 / std::sqrt(2.0);

    for (auto& individual : population) {
        individual.resize(total_qbits);
        for (auto& qb : individual) {
            qb.alpha = sqrt2_inv;
            qb.beta = sqrt2_inv;
        }
    }
}

template <typename T>
void QEAOptimizer<T>::observe(
    const std::vector<std::vector<QBit>>& q_pop,
    std::vector<std::vector<size_t>>& bin_pop)
{
    size_t total_qbits = q_pop[0].size();
    bin_pop.resize(q_pop.size());

    for (size_t p = 0; p < q_pop.size(); ++p) {
        bin_pop[p].resize(total_qbits);
        for (size_t b = 0; b < total_qbits; ++b) {
            double r = rng_.uniform_f64();
            // 坍缩: |β|² 概率观测到 1
            bin_pop[p][b] = (r < q_pop[p][b].beta * q_pop[p][b].beta) ? 1 : 0;
        }
    }
}

template <typename T>
std::vector<size_t> QEAOptimizer<T>::decodeToIndices(
    const std::vector<size_t>& bits) const
{
    std::vector<size_t> indices(n_params_, 0);
    size_t bit_offset = 0;

    for (size_t i = 0; i < n_params_; ++i) {
        size_t n_bits = n_qbits_per_param_[i];
        size_t n_vals = param_space_[i].size();

        // 将二进制位解码为索引
        size_t idx = 0;
        for (size_t b = 0; b < n_bits; ++b) {
            if (bits[bit_offset + b]) {
                idx |= (1u << b);
            }
        }
        // 截断到有效范围 (处理非 2 的幂的情况)
        if (idx >= n_vals) idx = n_vals - 1;
        indices[i] = idx;

        bit_offset += n_bits;
    }

    return indices;
}

template <typename T>
void QEAOptimizer<T>::rotateQBit(QBit& qb, size_t bit_i, size_t best_bit,
                                  double fitness, double best_fitness)
{
    // 标准 QEA 旋转门查找表 (Han & Kim, 2002)
    // 当当前解劣于最优解时，向最优解方向旋转
    double delta_theta = 0.0;

    if (fitness > best_fitness) {  // 当前解更差
        if (bit_i == 0 && best_bit == 1) {
            delta_theta = config_.rotation_angle;  // 向 |1⟩ 旋转
        } else if (bit_i == 1 && best_bit == 0) {
            delta_theta = -config_.rotation_angle; // 向 |0⟩ 旋转
        }
    }
    // 否则不旋转 (保持当前方向)

    if (delta_theta != 0.0) {
        double cos_dt = std::cos(delta_theta);
        double sin_dt = std::sin(delta_theta);
        double new_alpha = qb.alpha * cos_dt - qb.beta * sin_dt;
        double new_beta  = qb.alpha * sin_dt + qb.beta * cos_dt;
        qb.alpha = new_alpha;
        qb.beta  = new_beta;
    }
}

template <typename T>
void QEAOptimizer<T>::migrate(
    std::vector<std::vector<QBit>>& population,
    const std::vector<size_t>& best_bits,
    const std::vector<size_t>& /*best_indices*/)
{
    // 简单迁移: 将全局最优 Q-bit 状态复制到所有个体
    // 同时保留一定随机性以维持多样性
    for (auto& individual : population) {
        for (size_t b = 0; b < best_bits.size(); ++b) {
            double r = rng_.uniform_f64();
            if (r < 0.3) {  // 30% 概率接受迁移
                if (best_bits[b] == 0) {
                    // 向 |0⟩ 方向: α = 0.9, β = sqrt(1-0.81)
                    individual[b].alpha = 0.9;
                    individual[b].beta = std::sqrt(0.19);
                } else {
                    // 向 |1⟩ 方向: α = sqrt(0.19), β = 0.9
                    individual[b].alpha = std::sqrt(0.19);
                    individual[b].beta = 0.9;
                }
            }
        }
    }
}

} // namespace optimize
} // namespace ctQALS

#endif // CTORCH_QEA_OPTIMIZER_H