/**
 * @file SIMDWrapper.cpp
 * @brief C-ABI 包装层实现
 * @details 将 SIMD 超越函数库暴露为 C ABI 符号，供 MLIR JIT 后端调用。
 *          每个函数只做一层转发，性能开销可忽略（函数调用被内联）。
 *
 * @date 2026/08/03
 * @see SIMDWrapper.h
 */

#include "kernels/SIMDWrapper.h"
#include "kernels/SIMDMath.h"

// ======================= extern "C" 包装 =======================
// 使用 __attribute__((used)) 防止 LTO 移除符号（MLIR JIT 需通过 dlsym 查找）
// 使用 __attribute__((visibility("default"))) 确保符号导出

extern "C" __attribute__((used, visibility("default"))) void ct_simd_vexp(const float* in, float* out, size_t n) {
    ct::kernels::simd::vexp(in, out, n);
}

extern "C" __attribute__((used, visibility("default"))) void ct_simd_vlog(const float* in, float* out, size_t n) {
    ct::kernels::simd::vlog(in, out, n);
}

extern "C" __attribute__((used, visibility("default"))) void ct_simd_vtanh(const float* in, float* out, size_t n) {
    ct::kernels::simd::vtanh(in, out, n);
}

extern "C" __attribute__((used, visibility("default"))) void ct_simd_vsigmoid(const float* in, float* out, size_t n) {
    ct::kernels::simd::vsigmoid(in, out, n);
}

extern "C" __attribute__((used, visibility("default"))) void ct_simd_vgelu(const float* in, float* out, size_t n) {
    ct::kernels::simd::vgelu(in, out, n);
}

// ======================= 批量算术运算 =======================

extern "C" __attribute__((used, visibility("default"))) void ct_simd_vadd(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
#ifdef __aarch64__
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vaddq_f32(va, vb));
    }
#elif defined(__x86_64__)
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    }
#endif
    for (; i < n; ++i) out[i] = a[i] + b[i];
}

extern "C" __attribute__((used, visibility("default"))) void ct_simd_vmul(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
#ifdef __aarch64__
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vmulq_f32(va, vb));
    }
#elif defined(__x86_64__)
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
    }
#endif
    for (; i < n; ++i) out[i] = a[i] * b[i];
}

extern "C" __attribute__((used, visibility("default"))) void ct_simd_vsub(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
#ifdef __aarch64__
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vsubq_f32(va, vb));
    }
#elif defined(__x86_64__)
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_sub_ps(va, vb));
    }
#endif
    for (; i < n; ++i) out[i] = a[i] - b[i];
}

extern "C" __attribute__((used, visibility("default"))) void ct_simd_vdiv(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0;
#ifdef __aarch64__
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vdivq_f32(va, vb));
    }
#elif defined(__x86_64__)
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_div_ps(va, vb));
    }
#endif
    for (; i < n; ++i) out[i] = a[i] / b[i];
}

// ======================= 批量一元运算 =======================

extern "C" __attribute__((used, visibility("default"))) void ct_simd_vneg(const float* in, float* out, size_t n) {
    size_t i = 0;
#ifdef __aarch64__
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(in + i);
        vst1q_f32(out + i, vnegq_f32(v));
    }
#elif defined(__x86_64__)
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(in + i);
        _mm256_storeu_ps(out + i, _mm256_sub_ps(_mm256_setzero_ps(), v));
    }
#endif
    for (; i < n; ++i) out[i] = -in[i];
}

extern "C" __attribute__((used, visibility("default"))) void ct_simd_vrelu(const float* in, float* out, size_t n) {
    size_t i = 0;
#ifdef __aarch64__
    float32x4_t zero = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(in + i);
        vst1q_f32(out + i, vmaxq_f32(v, zero));
    }
#elif defined(__x86_64__)
    __m256 zero = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(in + i);
        _mm256_storeu_ps(out + i, _mm256_max_ps(v, zero));
    }
#endif
    for (; i < n; ++i) out[i] = (in[i] > 0.0f) ? in[i] : 0.0f;
}