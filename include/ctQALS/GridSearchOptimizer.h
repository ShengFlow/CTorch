/**
 * @file GridSearchOptimizer.h
 * @brief Exhaustive grid search optimizer for comparison baseline
 * @details Enumerates all combinations in the parameter space. Guarantees
 *          finding the global optimum but scales exponentially with the
 *          number of parameters. Used as a ground-truth baseline.
 * @date 2026/8/1
 */

#ifndef CTORCH_GRID_SEARCH_OPTIMIZER_H
#define CTORCH_GRID_SEARCH_OPTIMIZER_H

#include <cassert>
#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

namespace ctQALS {
namespace optimize {

// ============================================================
// GridSearch 优化结果
// ============================================================

struct GridSearchResult {
    std::vector<size_t> best_params;   ///< 最优参数索引
    double best_fitness;               ///< 最优适应度
    size_t evaluations;                ///< 总评估次数
    size_t total_combinations;         ///< 总组合数
};

// ============================================================
// GridSearch 优化器
// ============================================================

template <typename T>
class GridSearchOptimizer {
public:
    GridSearchOptimizer(std::vector<std::vector<T>> param_space)
        : param_space_(std::move(param_space))
        , n_params_(param_space_.size())
    {
        assert(n_params_ > 0 && "GridSearch: param_space must not be empty");
        total_combinations_ = 1;
        for (size_t i = 0; i < n_params_; ++i) {
            assert(!param_space_[i].empty() && "GridSearch: each param must have at least 1 value");
            total_combinations_ *= param_space_[i].size();
        }
    }

    GridSearchResult optimize(std::function<double(const std::vector<T>&)> fitness_fn);

    [[nodiscard]] size_t totalCombinations() const { return total_combinations_; }

private:
    std::vector<std::vector<T>> param_space_;
    size_t n_params_;
    size_t total_combinations_;
};

// ============================================================
// 实现
// ============================================================

template <typename T>
GridSearchResult GridSearchOptimizer<T>::optimize(
    std::function<double(const std::vector<T>&)> fitness_fn)
{
    GridSearchResult result;
    result.best_fitness = std::numeric_limits<double>::max();
    result.evaluations = 0;
    result.total_combinations = total_combinations_;

    // 使用嵌套循环穷举所有组合
    std::vector<size_t> indices(n_params_, 0);
    std::vector<size_t> best_indices(n_params_, 0);

    // 计算每个维度的步长
    std::vector<size_t> strides(n_params_, 1);
    for (size_t i = 1; i < n_params_; ++i) {
        strides[i] = strides[i - 1] * param_space_[i - 1].size();
    }

    for (size_t combo = 0; combo < total_combinations_; ++combo) {
        // 计算当前组合的索引
        size_t remaining = combo;
        for (size_t i = 0; i < n_params_; ++i) {
            indices[i] = remaining % param_space_[i].size();
            remaining /= param_space_[i].size();
        }

        // 评估
        std::vector<T> values;
        values.reserve(n_params_);
        for (size_t i = 0; i < n_params_; ++i) {
            values.push_back(param_space_[i][indices[i]]);
        }

        double f = fitness_fn(values);
        result.evaluations++;

        if (f < result.best_fitness) {
            result.best_fitness = f;
            best_indices = indices;
        }
    }

    result.best_params = best_indices;
    return result;
}

} // namespace optimize
} // namespace ctQALS

#endif // CTORCH_GRID_SEARCH_OPTIMIZER_H