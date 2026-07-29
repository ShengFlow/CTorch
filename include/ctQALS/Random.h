//
// Created by GhostFace on 2026/4/19.
//

#ifndef CTORCH_RANDOM_H
#define CTORCH_RANDOM_H
#include <ctime>
#pragma once

#include <cstdint>
#include <cstddef>

namespace ctQALS {
namespace rng {

// ============================================================
// 引擎：Xoshiro256++
// ============================================================
class Xoshiro256PlusPlus {
public:
    explicit Xoshiro256PlusPlus(uint64_t seed = 0);

    // 生成下一个 64 位均匀随机数
    uint64_t next_u64();

    // 生成 [0,1) 均匀双精度浮点数
    double uniform_f64();

    // 生成 [0,1) 均匀单精度浮点数
    float uniform_f32();

private:
    uint64_t s_[4];
    static inline uint64_t rotl(uint64_t x, int k);
    void seed(uint64_t seed);
};

// ============================================================
// 分布：Ziggurat 正态分布
// ============================================================
class ZigguratNormal {
public:
    explicit ZigguratNormal(Xoshiro256PlusPlus& engine);

    // 生成一个标准正态分布 N(0,1) 单精度浮点数
    float operator()();

    // 批量生成
    void fill(float* out, size_t n);

private:
    Xoshiro256PlusPlus& engine_;
    float tail_sample();
};

// ============================================================
// 线程本地实例（便捷访问）
// ============================================================
Xoshiro256PlusPlus& local_engine();
ZigguratNormal& local_normal();

} // namespace rng
} // namespace ctQALS
#endif //CTORCH_RANDOM_H
