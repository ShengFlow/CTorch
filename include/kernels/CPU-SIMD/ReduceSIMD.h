/**
 * @file ReduceSIMD.h
 * @brief reduce/broadcast 的 SIMD 辅助 (header-only)
 * @details sum(dim)/mean(dim) 沿最内维归约与反向梯度广播填充。aarch64 走 NEON 4-wide,
 *          其它平台回退标量(可后续补 AVX2/AVX-512)。
 * @date 2026/09/06
 */
#ifndef CTORCH_REDUCE_SIMD_H
#define CTORCH_REDUCE_SIMD_H

#include <cstddef>
#include <cstring>

#if defined(__aarch64__)
#include <arm_neon.h>
#define CTORCH_REDUCE_NEON 1
#endif

namespace ctorch {
namespace kernels {
namespace simd {

/// 把连续 n 个 float 求和(4-wide 累加器 + 尾部)
inline float reduce_row_sum_f32(const float* p, size_t n) {
    float r = 0.0f;
#if CTORCH_REDUCE_NEON
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        acc = vaddq_f32(acc, vld1q_f32(p + i));
    }
    r = vaddvq_f32(acc);
    for (; i < n; ++i) r += p[i];
#else
    for (size_t i = 0; i < n; ++i) r += p[i];
#endif
    return r;
}

/// 每行连续 dim_size 个元素求和: in[nrows*dim_size] -> out[nrows]
inline void reduce_rows_sum_f32(const float* in, size_t nrows, size_t dim_size, float* out) {
    for (size_t r = 0; r < nrows; ++r) {
        out[r] = reduce_row_sum_f32(in + r * dim_size, dim_size);
    }
}

/// 用标量 v 填充 n 个 float (4-wide 存储 + 尾部)
inline void fill_f32(float* out, size_t n, float v) {
#if CTORCH_REDUCE_NEON
    float32x4_t vec = vdupq_n_f32(v);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(out + i, vec);
    }
    for (; i < n; ++i) out[i] = v;
#else
    for (size_t i = 0; i < n; ++i) out[i] = v;
#endif
}

/// 原地缩放: out[i] *= s (4-wide 乘 + 尾部)
inline void scale_f32(float* out, size_t n, float s) {
#if CTORCH_REDUCE_NEON
    float32x4_t vec = vdupq_n_f32(s);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(out + i, vmulq_f32(vld1q_f32(out + i), vec));
    }
    for (; i < n; ++i) out[i] *= s;
#else
    for (size_t i = 0; i < n; ++i) out[i] *= s;
#endif
}

}  // namespace simd
}  // namespace kernels
}  // namespace ctorch

#endif  // CTORCH_REDUCE_SIMD_H
