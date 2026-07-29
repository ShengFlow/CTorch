#include "../include/ctQALS/Random.h"
#include <cmath>
#include <algorithm>

namespace ctQALS {
namespace rng {

// ============================================================
// 硬编码 Ziggurat 表（256 层，标准正态分布）
// ============================================================
struct ZigguratTable {
    static constexpr int N = 256;
    float x[N + 1];
    float y[N + 1];
    float r;  // 尾部起始点
};

static const ZigguratTable g_zig_table = []() {
    ZigguratTable tbl;
    constexpr double R = 3.6541528853610088;  // 尾部起始点
    constexpr double A = 0.00492867323399;    // 面积参数

    tbl.r = static_cast<float>(R);
    tbl.x[0] = static_cast<float>(R);
    tbl.y[0] = std::exp(-0.5 * R * R);

    double x = R;
    for (int i = 1; i <= 256; ++i) {
        double y = std::exp(-0.5 * x * x);
        tbl.x[i] = static_cast<float>(x);
        tbl.y[i] = static_cast<float>(y);
        // 牛顿迭代求下一层边界
        x = std::sqrt(-2.0 * std::log(A / x + y));
    }
    // 确保最后一层边界为 0
    tbl.x[256] = 0.0f;
    tbl.y[256] = 1.0f;
    return tbl;
}();

// ============================================================
// ZigguratNormal 实现
// ============================================================
ZigguratNormal::ZigguratNormal(Xoshiro256PlusPlus& engine)
    : engine_(engine) {}

float ZigguratNormal::tail_sample() {
    float x, y;
    do {
        x = -std::log(engine_.uniform_f32()) / g_zig_table.r;
        y = -std::log(engine_.uniform_f32());
    } while (y + y < x * x);
    return (engine_.uniform_f32() < 0.5f) ? (g_zig_table.r + x) : -(g_zig_table.r + x);
}

float ZigguratNormal::operator()() {
    for (;;) {
        // 1. 随机选择层 0..255
        uint64_t u64 = engine_.next_u64();
        int layer = u64 & 0xFF;

        // 2. 生成 x ∈ [-layer_width, layer_width]
        float x = static_cast<float>((u64 >> 11) * 0x1.0p-53);  // [0,1)
        x = x * 2.0f - 1.0f;
        x *= g_zig_table.x[layer];

        // 3. 快速接受：落在矩形内
        if (std::abs(x) < g_zig_table.x[layer + 1]) {
            return x;
        }

        // 4. 处理尾部
        if (layer == 0) {
            return tail_sample();
        }

        // 5. 处理角落：拒绝采样
        float y = engine_.uniform_f32();
        float y_bound = (g_zig_table.y[layer] - g_zig_table.y[layer + 1]);
        float pdf_x = std::exp(-0.5f * x * x);
        if (y * y_bound < pdf_x - g_zig_table.y[layer + 1]) {
            return x;
        }
    }
}

void ZigguratNormal::fill(float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = (*this)();
    }
}

// 线程本地分布实例
thread_local ZigguratNormal g_local_normal(local_engine());

ZigguratNormal& local_normal() {
    return g_local_normal;
}

} // namespace rng
} // namespace ctQALS