#include "../include/ctQALS/Random.h"
#include <limits>
#include <random>

namespace ctQALS {
namespace rng {

// SplitMix64 播种器
static uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

void Xoshiro256PlusPlus::seed(uint64_t seed) {
    uint64_t sm_state = seed;
    s_[0] = splitmix64(sm_state);
    s_[1] = splitmix64(sm_state);
    s_[2] = splitmix64(sm_state);
    s_[3] = splitmix64(sm_state);
}

Xoshiro256PlusPlus::Xoshiro256PlusPlus(uint64_t seed) {
    this->seed(seed);
}

inline uint64_t Xoshiro256PlusPlus::rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

uint64_t Xoshiro256PlusPlus::next_u64() {
    const uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
    const uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
}

double Xoshiro256PlusPlus::uniform_f64() {
    // 取高 53 位，生成 [0,1) 双精度
    return (next_u64() >> 11) * 0x1.0p-53;
}

float Xoshiro256PlusPlus::uniform_f32() {
    // 取高 24 位，生成 [0,1) 单精度
    return (next_u64() >> 40) * 0x1.0p-24f;
}

// 线程本地引擎实例
thread_local Xoshiro256PlusPlus g_local_engine(std::random_device{}());

Xoshiro256PlusPlus& local_engine() {
    return g_local_engine;
}

} // namespace rng
} // namespace ctQALS