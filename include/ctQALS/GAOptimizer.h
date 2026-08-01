/**
 * @file GAOptimizer.h
 * @brief Standard Genetic Algorithm (GA) optimizer for comparison baseline
 * @details Binary-encoded GA with tournament selection, single-point crossover,
 *          bit-flip mutation, and elitism. Used as a classical baseline to
 *          compare against QEA.
 * @date 2026/8/1
 */

#ifndef CTORCH_GA_OPTIMIZER_H
#define CTORCH_GA_OPTIMIZER_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "Random.h"

namespace ctQALS {
namespace optimize {

// ============================================================
// GA 配置
// ============================================================

struct GAConfig {
    size_t population_size = 20;       ///< 种群大小
    size_t max_generations = 20;       ///< 最大迭代代数
    double crossover_rate = 0.8;       ///< 交叉概率
    double mutation_rate = 0.05;       ///< 变异概率 (per bit)
    size_t tournament_size = 3;        ///< 锦标赛选择大小
    size_t elitism_count = 2;          ///< 精英保留数量
    uint64_t seed = 0;                 ///< 随机种子
    bool verbose = false;              ///< 是否输出调试信息
};

// ============================================================
// GA 优化结果
// ============================================================

struct GAResult {
    std::vector<size_t> best_params;   ///< 最优参数索引
    double best_fitness;               ///< 最优适应度
    size_t evaluations;                ///< 总评估次数
    size_t cache_hits;                 ///< 缓存命中次数
    std::vector<double> fitness_history; ///< 每代最优适应度历史
};

// ============================================================
// GA 优化器
// ============================================================

template <typename T>
class GAOptimizer {
public:
    GAOptimizer(std::vector<std::vector<T>> param_space,
                GAConfig config = {})
        : param_space_(std::move(param_space))
        , config_(config)
        , n_params_(param_space_.size())
    {
        assert(n_params_ > 0 && "GA: param_space must not be empty");
        for (size_t i = 0; i < n_params_; ++i) {
            assert(!param_space_[i].empty() && "GA: each param must have at least 1 value");
            size_t n_vals = param_space_[i].size();
            size_t n_bits = 1;
            while ((1u << n_bits) < n_vals) ++n_bits;
            n_bits_per_param_.push_back(n_bits);
        }
    }

    GAResult optimize(std::function<double(const std::vector<T>&)> fitness_fn);

    [[nodiscard]] const std::vector<std::vector<T>>& paramSpace() const { return param_space_; }

private:
    using Individual = std::vector<size_t>;  ///< 二进制编码个体

    /// 随机初始化个体
    Individual randomIndividual();

    /// 解码为参数索引
    std::vector<size_t> decodeToIndices(const Individual& bits) const;

    /// 锦标赛选择
    size_t tournamentSelect(const std::vector<double>& fitness,
                            const std::vector<size_t>& sorted_indices);

    /// 单点交叉
    void crossover(const Individual& p1, const Individual& p2,
                   Individual& c1, Individual& c2);

    /// 位翻转变异
    void mutate(Individual& ind);

    std::vector<std::vector<T>> param_space_;
    GAConfig config_;
    size_t n_params_;
    std::vector<size_t> n_bits_per_param_;
    ctQALS::rng::Xoshiro256PlusPlus rng_;
};

// ============================================================
// 实现
// ============================================================

template <typename T>
GAResult GAOptimizer<T>::optimize(
    std::function<double(const std::vector<T>&)> fitness_fn)
{
    GAResult result;
    result.best_fitness = std::numeric_limits<double>::max();
    result.evaluations = 0;
    result.cache_hits = 0;

    size_t total_bits = 0;
    for (auto n : n_bits_per_param_) total_bits += n;

    // 评估缓存
    std::unordered_map<size_t, double> eval_cache;
    auto hashConfig = [](const std::vector<size_t>& indices) -> size_t {
        size_t h = 0;
        for (auto v : indices) {
            h ^= std::hash<size_t>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    };

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
            values.push_back(param_space_[i][indices[i]]);
        }
        double f = fitness_fn(values);
        eval_cache[h] = f;
        result.evaluations++;
        return f;
    };

    // 初始化种群
    std::vector<Individual> population;
    std::vector<double> pop_fitness;
    for (size_t i = 0; i < config_.population_size; ++i) {
        population.push_back(randomIndividual());
        auto indices = decodeToIndices(population.back());
        double f = cachedEval(indices);
        pop_fitness.push_back(f);
        if (f < result.best_fitness) {
            result.best_fitness = f;
            result.best_params = indices;
        }
    }
    result.fitness_history.push_back(result.best_fitness);

    if (config_.verbose) {
        printf("[GA] Gen 0: best_fitness=%.6f, evals=%zu\n",
               result.best_fitness, result.evaluations);
    }

    // 主循环
    for (size_t gen = 1; gen <= config_.max_generations; ++gen) {
        // 按适应度排序索引
        std::vector<size_t> sorted_idx(config_.population_size);
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::sort(sorted_idx.begin(), sorted_idx.end(),
                  [&](size_t a, size_t b) { return pop_fitness[a] < pop_fitness[b]; });

        std::vector<Individual> new_pop;
        std::vector<double> new_fitness;

        // 精英保留
        for (size_t e = 0; e < config_.elitism_count && e < config_.population_size; ++e) {
            new_pop.push_back(population[sorted_idx[e]]);
            new_fitness.push_back(pop_fitness[sorted_idx[e]]);
        }

        // 生成新个体
        while (new_pop.size() < config_.population_size) {
            // 选择
            size_t p1 = tournamentSelect(pop_fitness, sorted_idx);
            size_t p2 = tournamentSelect(pop_fitness, sorted_idx);

            Individual c1 = population[p1];
            Individual c2 = population[p2];

            // 交叉
            if (rng_.uniform_f64() < config_.crossover_rate) {
                crossover(population[p1], population[p2], c1, c2);
            }

            // 变异
            mutate(c1);
            mutate(c2);

            // 评估
            auto indices1 = decodeToIndices(c1);
            double f1 = cachedEval(indices1);
            new_pop.push_back(std::move(c1));
            new_fitness.push_back(f1);
            if (f1 < result.best_fitness) {
                result.best_fitness = f1;
                result.best_params = indices1;
            }

            if (new_pop.size() < config_.population_size) {
                auto indices2 = decodeToIndices(c2);
                double f2 = cachedEval(indices2);
                new_pop.push_back(std::move(c2));
                new_fitness.push_back(f2);
                if (f2 < result.best_fitness) {
                    result.best_fitness = f2;
                    result.best_params = indices2;
                }
            }
        }

        // 保持种群大小
        population = std::move(new_pop);
        pop_fitness = std::move(new_fitness);
        if (population.size() > config_.population_size) {
            population.resize(config_.population_size);
            pop_fitness.resize(config_.population_size);
        }

        result.fitness_history.push_back(result.best_fitness);

        if (config_.verbose) {
            printf("[GA] Gen %zu: best_fitness=%.6f, evals=%zu, cache_hits=%zu\n",
                   gen, result.best_fitness, result.evaluations, result.cache_hits);
        }
    }

    return result;
}

template <typename T>
std::vector<size_t> GAOptimizer<T>::randomIndividual() {
    size_t total_bits = 0;
    for (auto n : n_bits_per_param_) total_bits += n;
    Individual ind(total_bits, 0);
    for (size_t b = 0; b < total_bits; ++b) {
        ind[b] = (rng_.uniform_f64() < 0.5) ? 1 : 0;
    }
    return ind;
}

template <typename T>
std::vector<size_t> GAOptimizer<T>::decodeToIndices(const Individual& bits) const {
    std::vector<size_t> indices(n_params_, 0);
    size_t bit_offset = 0;

    for (size_t i = 0; i < n_params_; ++i) {
        size_t n_bits = n_bits_per_param_[i];
        size_t n_vals = param_space_[i].size();

        size_t idx = 0;
        for (size_t b = 0; b < n_bits; ++b) {
            if (bits[bit_offset + b]) {
                idx |= (1u << b);
            }
        }
        if (idx >= n_vals) idx = n_vals - 1;
        indices[i] = idx;

        bit_offset += n_bits;
    }

    return indices;
}

template <typename T>
size_t GAOptimizer<T>::tournamentSelect(
    const std::vector<double>& fitness,
    const std::vector<size_t>& sorted_indices)
{
    // 从排序列表的前部 (更好的个体) 中随机选择
    size_t best = 0;
    double best_f = std::numeric_limits<double>::max();
    for (size_t t = 0; t < config_.tournament_size; ++t) {
        // 偏向从较优个体中选择
        size_t idx = sorted_indices[rng_.next_u64() % std::min(config_.population_size, config_.tournament_size * 3)];
        if (fitness[idx] < best_f) {
            best_f = fitness[idx];
            best = idx;
        }
    }
    return best;
}

template <typename T>
void GAOptimizer<T>::crossover(const Individual& p1, const Individual& p2,
                                Individual& c1, Individual& c2)
{
    size_t n = p1.size();
    size_t point = rng_.next_u64() % (n - 1) + 1;

    for (size_t i = 0; i < point; ++i) {
        c1[i] = p1[i];
        c2[i] = p2[i];
    }
    for (size_t i = point; i < n; ++i) {
        c1[i] = p2[i];
        c2[i] = p1[i];
    }
}

template <typename T>
void GAOptimizer<T>::mutate(Individual& ind) {
    for (size_t b = 0; b < ind.size(); ++b) {
        if (rng_.uniform_f64() < config_.mutation_rate) {
            ind[b] = 1 - ind[b];  // 位翻转
        }
    }
}

} // namespace optimize
} // namespace ctQALS

#endif // CTORCH_GA_OPTIMIZER_H