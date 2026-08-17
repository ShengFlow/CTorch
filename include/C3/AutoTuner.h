/**
 * @file AutoTuner.h
 * @generation JIT-2.0 手写后端调优工具
 * @brief C3 kernel auto-tuner using QEA, GA, and GridSearch optimizers
 * @details Defines the tuning search space for MatMul tile parameters and
 *          provides comparison utilities for evaluating different optimization
 *          strategies. Uses ctQALS optimizers as the backend.
 *
 *          Tuning parameters:
 *          - TILE_M: {16, 32, 64, 96, 128}
 *          - TILE_N: {16, 32, 64, 96, 128}
 *          - TILE_K: {16, 32, 64, 96, 128}
 *          - unroll:  {1, 2, 4, 8}
 *
 *          搜索空间大小: 5×5×5×4 = 500 种组合
 * @date 2026/8/1
 */

#ifndef CTORCH_C3_AUTO_TUNER_H
#define CTORCH_C3_AUTO_TUNER_H

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "ctQALS/QEAOptimizer.h"
#include "ctQALS/GAOptimizer.h"
#include "ctQALS/GridSearchOptimizer.h"

namespace ct {
namespace c3 {

// ============================================================
// AutoTuner 配置
// ============================================================

struct AutoTunerConfig {
    // 搜索空间: MatMul 分块大小
    std::vector<int> tile_m_candidates = {16, 32, 64, 96, 128};
    std::vector<int> tile_n_candidates = {16, 32, 64, 96, 128};
    std::vector<int> tile_k_candidates = {16, 32, 64, 96, 128};
    std::vector<int> unroll_candidates   = {1, 2, 4, 8};

    // QEA 参数
    size_t qea_population = 10;
    size_t qea_generations = 20;

    // GA 参数
    size_t ga_population = 20;
    size_t ga_generations = 20;

    // Benchmark 参数
    size_t benchmark_runs = 50;  ///< 每次评估的 benchmark 运行次数
    bool verbose = false;
};

// ============================================================
// AutoTuner 结果
// ============================================================

struct TuningResult {
    std::vector<size_t> best_indices;  ///< 最优参数索引
    int tile_m, tile_n, tile_k, unroll; ///< 最优参数值
    double best_fitness;                ///< 最优适应度 (us)
    size_t evaluations;                 ///< 评估次数
    size_t cache_hits;                  ///< 缓存命中次数
    double elapsed_ms;                  ///< 调优耗时 (ms)
    std::vector<double> fitness_history; ///< 收敛历史
    std::string method;                 ///< 优化方法名称
};

// ============================================================
// AutoTuner
// ============================================================

class AutoTuner {
public:
    explicit AutoTuner(AutoTunerConfig config = {})
        : config_(std::move(config))
    {
        buildSearchSpace();
    }

    /**
     * @brief 使用 QEA 优化器调优
     * @param fitness_fn 适应度函数: (tile_m, tile_n, tile_k, unroll) → 耗时 (us)
     */
    TuningResult tuneWithQEA(
        std::function<double(int, int, int, int)> fitness_fn);

    /**
     * @brief 使用 GA 优化器调优
     */
    TuningResult tuneWithGA(
        std::function<double(int, int, int, int)> fitness_fn);

    /**
     * @brief 使用 GridSearch 穷举调优 (ground truth)
     */
    TuningResult tuneWithGridSearch(
        std::function<double(int, int, int, int)> fitness_fn);

    /**
     * @brief 运行完整对比 Benchmark
     * @param fitness_fn 适应度函数
     * @param known_optimal 已知最优参数 (可选，用于验证)
     */
    void runComparison(
        std::function<double(int, int, int, int)> fitness_fn,
        const std::vector<int>& known_optimal = {});

    /** @brief 获取搜索空间大小 */
    [[nodiscard]] size_t searchSpaceSize() const { return search_space_size_; }

    /** @brief 获取搜索空间 */
    [[nodiscard]] const std::vector<std::vector<int>>& searchSpace() const {
        return search_space_;
    }

private:
    void buildSearchSpace();

    /// 将 QEA/GA/GridSearch 结果转换为 TuningResult
    TuningResult toTuningResult(
        const ctQALS::optimize::QEAResult& r,
        const std::string& method, double elapsed_ms);

    TuningResult toTuningResult(
        const ctQALS::optimize::GAResult& r,
        const std::string& method, double elapsed_ms);

    TuningResult toTuningResult(
        const ctQALS::optimize::GridSearchResult& r,
        const std::string& method, double elapsed_ms);

    AutoTunerConfig config_;
    std::vector<std::vector<int>> search_space_;
    size_t search_space_size_;
};

// ============================================================
// 实现
// ============================================================

inline void AutoTuner::buildSearchSpace() {
    search_space_ = {
        config_.tile_m_candidates,
        config_.tile_n_candidates,
        config_.tile_k_candidates,
        config_.unroll_candidates
    };
    search_space_size_ = 1;
    for (auto& dim : search_space_) {
        search_space_size_ *= dim.size();
    }
}

inline TuningResult AutoTuner::tuneWithQEA(
    std::function<double(int, int, int, int)> fitness_fn)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    ctQALS::optimize::QEAConfig qea_cfg;
    qea_cfg.population_size = config_.qea_population;
    qea_cfg.max_generations = config_.qea_generations;
    qea_cfg.verbose = config_.verbose;

    ctQALS::optimize::QEAOptimizer<int> qea(search_space_, qea_cfg);
    auto result = qea.optimize([&](const std::vector<int>& params) {
        return fitness_fn(params[0], params[1], params[2], params[3]);
    });

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return toTuningResult(result, "QEA", elapsed);
}

inline TuningResult AutoTuner::tuneWithGA(
    std::function<double(int, int, int, int)> fitness_fn)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    ctQALS::optimize::GAConfig ga_cfg;
    ga_cfg.population_size = config_.ga_population;
    ga_cfg.max_generations = config_.ga_generations;
    ga_cfg.verbose = config_.verbose;

    ctQALS::optimize::GAOptimizer<int> ga(search_space_, ga_cfg);
    auto result = ga.optimize([&](const std::vector<int>& params) {
        return fitness_fn(params[0], params[1], params[2], params[3]);
    });

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return toTuningResult(result, "GA", elapsed);
}

inline TuningResult AutoTuner::tuneWithGridSearch(
    std::function<double(int, int, int, int)> fitness_fn)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    ctQALS::optimize::GridSearchOptimizer<int> gs(search_space_);
    auto result = gs.optimize([&](const std::vector<int>& params) {
        return fitness_fn(params[0], params[1], params[2], params[3]);
    });

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return toTuningResult(result, "GridSearch", elapsed);
}

inline void AutoTuner::runComparison(
    std::function<double(int, int, int, int)> fitness_fn,
    const std::vector<int>& known_optimal)
{
    std::cout << "\n========== AutoTuner Comparison ==========" << std::endl;
    std::cout << "Search space: "
              << config_.tile_m_candidates.size() << "x"
              << config_.tile_n_candidates.size() << "x"
              << config_.tile_k_candidates.size() << "x"
              << config_.unroll_candidates.size()
              << " = " << searchSpaceSize() << " combinations" << std::endl;

    if (!known_optimal.empty()) {
        std::cout << "Known optimal: ("
                  << known_optimal[0] << ", " << known_optimal[1] << ", "
                  << known_optimal[2] << ", " << known_optimal[3] << ")" << std::endl;
    }

    auto qea_result = tuneWithQEA(fitness_fn);
    auto ga_result  = tuneWithGA(fitness_fn);
    auto gs_result  = tuneWithGridSearch(fitness_fn);

    // 打印结果
    std::cout << "\n" << std::setw(16) << std::left << "Method"
              << " | " << std::setw(22) << "Best Params"
              << " | " << std::setw(12) << std::right << "Fitness"
              << " | " << std::setw(10) << "Evals"
              << " | " << std::setw(10) << "CacheHit"
              << " | " << std::setw(10) << "Time(ms)" << std::endl;
    std::cout << std::string(95, '-') << std::endl;

    auto printResult = [&](const TuningResult& r) {
        char buf[32];
        snprintf(buf, sizeof(buf), "(%d,%d,%d,%d)",
                 r.tile_m, r.tile_n, r.tile_k, r.unroll);
        std::cout << std::setw(16) << std::left << r.method
                  << " | " << std::setw(22) << buf
                  << " | " << std::setw(12) << std::right << std::fixed << std::setprecision(1)
                  << r.best_fitness
                  << " | " << std::setw(10) << r.evaluations
                  << " | " << std::setw(10) << r.cache_hits
                  << " | " << std::setw(10) << std::fixed << std::setprecision(1)
                  << r.elapsed_ms << std::endl;
    };

    printResult(qea_result);
    printResult(ga_result);
    printResult(gs_result);

    std::cout << std::string(95, '-') << std::endl;

    // 效率对比
    double qea_efficiency = (gs_result.evaluations > 0)
        ? (double)searchSpaceSize() / qea_result.evaluations : 0.0;
    double ga_efficiency = (gs_result.evaluations > 0)
        ? (double)searchSpaceSize() / ga_result.evaluations : 0.0;

    std::cout << "\nEfficiency (search_space / evals):" << std::endl;
    std::cout << "  QEA: " << std::fixed << std::setprecision(1) << qea_efficiency
              << "x (" << qea_result.evaluations << " evals / " << searchSpaceSize() << " total)" << std::endl;
    std::cout << "  GA:  " << std::fixed << std::setprecision(1) << ga_efficiency
              << "x (" << ga_result.evaluations << " evals / " << searchSpaceSize() << " total)" << std::endl;
    std::cout << "  GS:  1.0x (" << gs_result.evaluations << " evals / " << searchSpaceSize() << " total)" << std::endl;

    // 精度验证
    if (!known_optimal.empty()) {
        bool qea_ok = (qea_result.tile_m == known_optimal[0] &&
                       qea_result.tile_n == known_optimal[1] &&
                       qea_result.tile_k == known_optimal[2] &&
                       qea_result.unroll == known_optimal[3]);
        bool ga_ok  = (ga_result.tile_m == known_optimal[0] &&
                       ga_result.tile_n == known_optimal[1] &&
                       ga_result.tile_k == known_optimal[2] &&
                       ga_result.unroll == known_optimal[3]);

        std::cout << "\nAccuracy (found optimal):" << std::endl;
        std::cout << "  QEA: " << (qea_ok ? "YES" : "NO") << std::endl;
        std::cout << "  GA:  " << (ga_ok ? "YES" : "NO") << std::endl;
        std::cout << "  GS:  YES (guaranteed)" << std::endl;
    }
}

inline TuningResult AutoTuner::toTuningResult(
    const ctQALS::optimize::QEAResult& r,
    const std::string& method, double elapsed_ms)
{
    TuningResult tr;
    tr.method = method;
    tr.best_indices = r.best_params;
    tr.tile_m = search_space_[0][r.best_params[0]];
    tr.tile_n = search_space_[1][r.best_params[1]];
    tr.tile_k = search_space_[2][r.best_params[2]];
    tr.unroll = search_space_[3][r.best_params[3]];
    tr.best_fitness = r.best_fitness;
    tr.evaluations = r.evaluations;
    tr.cache_hits = r.cache_hits;
    tr.elapsed_ms = elapsed_ms;
    tr.fitness_history = r.fitness_history;
    return tr;
}

inline TuningResult AutoTuner::toTuningResult(
    const ctQALS::optimize::GAResult& r,
    const std::string& method, double elapsed_ms)
{
    TuningResult tr;
    tr.method = method;
    tr.best_indices = r.best_params;
    tr.tile_m = search_space_[0][r.best_params[0]];
    tr.tile_n = search_space_[1][r.best_params[1]];
    tr.tile_k = search_space_[2][r.best_params[2]];
    tr.unroll = search_space_[3][r.best_params[3]];
    tr.best_fitness = r.best_fitness;
    tr.evaluations = r.evaluations;
    tr.cache_hits = r.cache_hits;
    tr.elapsed_ms = elapsed_ms;
    tr.fitness_history = r.fitness_history;
    return tr;
}

inline TuningResult AutoTuner::toTuningResult(
    const ctQALS::optimize::GridSearchResult& r,
    const std::string& method, double elapsed_ms)
{
    TuningResult tr;
    tr.method = method;
    tr.best_indices = r.best_params;
    tr.tile_m = search_space_[0][r.best_params[0]];
    tr.tile_n = search_space_[1][r.best_params[1]];
    tr.tile_k = search_space_[2][r.best_params[2]];
    tr.unroll = search_space_[3][r.best_params[3]];
    tr.best_fitness = r.best_fitness;
    tr.evaluations = r.evaluations;
    tr.cache_hits = 0;
    tr.elapsed_ms = elapsed_ms;
    return tr;
}

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_AUTO_TUNER_H