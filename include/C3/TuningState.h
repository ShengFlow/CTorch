/**
 * @file TuningState.h
 * @brief C3 自动调优状态管理
 * @details 存储由 AutoTuner 产出的最优分块参数，供 kernel 生成器使用。
 *          线程安全，单次调优后全局复用。
 * @date 2026/8/1
 */

#ifndef CTORCH_C3_TUNING_STATE_H
#define CTORCH_C3_TUNING_STATE_H

#include <cstddef>
#include <mutex>

namespace ct {
namespace c3 {

struct TuningParams {
    int tile_m = 64;   ///< MatMul M 分块大小
    int tile_n = 64;   ///< MatMul N 分块大小
    int tile_k = 64;   ///< MatMul K 分块大小
    int unroll = 4;    ///< 内层循环展开因子
    bool tuned = false; ///< 是否已完成调优
};

class TuningState {
public:
    static TuningState& instance() {
        static TuningState s;
        return s;
    }

    TuningParams get() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return params_;
    }

    void set(const TuningParams& p) {
        std::lock_guard<std::mutex> lk(mutex_);
        params_ = p;
    }

    bool isTuned() const {
        std::lock_guard<std::mutex> lk(mutex_);
        return params_.tuned;
    }

private:
    TuningState() = default;
    mutable std::mutex mutex_;
    TuningParams params_;
};

} // namespace c3
} // namespace ct

#endif // CTORCH_C3_TUNING_STATE_H