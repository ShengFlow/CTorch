/**
 * @file SwiGLU.cpp
 * @brief SwiGLU 算子 Eager CPU 实现 (BASIC + SIMD AVX2)
 * @details swiglu(x, gate) = silu(x) * gate, 双输入 elementwise
 * @author Mavis (PEL25 §6 SwiGLU 算子开发协议 Stage 1)
 * @date 2026-09-05
 */

#include "ops/SwiGLU.h"
#include "CtorchError.h"

#include <cmath>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace ct::ops {

namespace {
inline std::pair<float, float> swiglu_backward_scalar_impl(float grad_y, float x, float gate) {
    float s = 1.0f / (1.0f + std::exp(-x));
    float silu_d = s + x * s * (1.0f - s);
    float grad_x = grad_y * gate * silu_d;
    float grad_gate = grad_y * (x * s);
    return {grad_x, grad_gate};
}
}  // namespace

#if defined(__AVX2__)

namespace {
inline void swiglu_avx2(const float* x_ptr, const float* g_ptr, float* y_ptr, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x_ptr + i);
        __m256 vg = _mm256_loadu_ps(g_ptr + i);
        __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), vx);
        const __m256 clamp_hi = _mm256_set1_ps(15.0f);
        const __m256 clamp_lo = _mm256_set1_ps(-15.0f);
        neg_x = _mm256_min_ps(_mm256_max_ps(neg_x, clamp_lo), clamp_hi);

        alignas(32) float xs[8];
        _mm256_store_ps(xs, neg_x);
        for (int j = 0; j < 8; ++j) xs[j] = std::exp(xs[j]);
        __m256 exp_neg_x = _mm256_load_ps(xs);
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 vsigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_x));

        // silu = x * sigmoid
        __m256 vsilu = _mm256_mul_ps(vx, vsigmoid);
        // y = silu * gate
        __m256 vy = _mm256_mul_ps(vsilu, vg);
        _mm256_storeu_ps(y_ptr + i, vy);
    }
    for (; i < n; ++i) {
        float s = 1.0f / (1.0f + std::exp(-x_ptr[i]));
        y_ptr[i] = x_ptr[i] * s * g_ptr[i];
    }
}

inline void swiglu_backward_avx2(const float* go_ptr, const float* x_ptr, const float* g_ptr,
                                  float* gx_ptr, float* gg_ptr, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vgo = _mm256_loadu_ps(go_ptr + i);
        __m256 vx = _mm256_loadu_ps(x_ptr + i);
        __m256 vg = _mm256_loadu_ps(g_ptr + i);

        __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), vx);
        const __m256 clamp_hi = _mm256_set1_ps(15.0f);
        const __m256 clamp_lo = _mm256_set1_ps(-15.0f);
        neg_x = _mm256_min_ps(_mm256_max_ps(neg_x, clamp_lo), clamp_hi);
        alignas(32) float xs[8];
        _mm256_store_ps(xs, neg_x);
        for (int j = 0; j < 8; ++j) xs[j] = std::exp(xs[j]);
        __m256 exp_neg_x = _mm256_load_ps(xs);
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 vsigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_x));

        // silu_derivative = sigmoid + x * sigmoid * (1 - sigmoid)
        __m256 vx_sigmoid = _mm256_mul_ps(vx, vsigmoid);
        __m256 v1_minus_s = _mm256_sub_ps(one, vsigmoid);
        __m256 vx_sigmoid_1ms = _mm256_mul_ps(vx_sigmoid, v1_minus_s);
        __m256 vsilu_d = _mm256_add_ps(vsigmoid, vx_sigmoid_1ms);

        // grad_x = grad_y * gate * silu_d
        __m256 vgx = _mm256_mul_ps(_mm256_mul_ps(vgo, vg), vsilu_d);
        _mm256_storeu_ps(gx_ptr + i, vgx);

        // grad_gate = grad_y * silu(x) = grad_y * x * sigmoid
        __m256 vgg = _mm256_mul_ps(vgo, vx_sigmoid);
        _mm256_storeu_ps(gg_ptr + i, vgg);
    }
    for (; i < n; ++i) {
        auto [gx, gg] = swiglu_backward_scalar_impl(go_ptr[i], x_ptr[i], g_ptr[i]);
        gx_ptr[i] = gx;
        gg_ptr[i] = gg;
    }
}
}  // namespace

#endif  // __AVX2__

Tensor swiglu_forward(const Tensor& x, const Tensor& gate) {
    if (x.dtype() != DType::kFloat || gate.dtype() != DType::kFloat) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "SwiGLU forward: only float32 supported (Stage 1)");
    }
    if (x.device() != DeviceType::kCPU || gate.device() != DeviceType::kCPU) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT, "SwiGLU forward: only CPU device supported (Stage 1)");
    }
    if (x.shape() != gate.shape()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION, "SwiGLU forward: x and gate shape mismatch (Stage 1 不做 broadcasting)");
    }

    const auto& shape = x.shape();
    size_t n = 1;
    for (auto dim : shape) n *= dim;

    Tensor output(ShapeTag{}, shape, DType::kFloat, DeviceType::kCPU);

    const float* x_ptr = x.data_read<float>();
    const float* g_ptr = gate.data_read<float>();
    float* y_ptr = output.data_write<float>();

#if defined(__AVX2__)
    swiglu_avx2(x_ptr, g_ptr, y_ptr, n);
#else
    for (size_t i = 0; i < n; ++i) {
        float s = 1.0f / (1.0f + std::exp(-x_ptr[i]));
        y_ptr[i] = x_ptr[i] * s * g_ptr[i];
    }
#endif

    return output;
}

std::pair<Tensor, Tensor> swiglu_backward(const Tensor& grad_output, const Tensor& x, const Tensor& gate) {
    if (grad_output.dtype() != DType::kFloat || x.dtype() != DType::kFloat || gate.dtype() != DType::kFloat) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "SwiGLU backward: only float32 supported (Stage 1)");
    }
    if (grad_output.shape() != x.shape() || x.shape() != gate.shape()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION, "SwiGLU backward: shape mismatch");
    }

    const auto& shape = x.shape();
    size_t n = 1;
    for (auto dim : shape) n *= dim;

    Tensor grad_x(ShapeTag{}, shape, DType::kFloat, DeviceType::kCPU);
    Tensor grad_gate(ShapeTag{}, shape, DType::kFloat, DeviceType::kCPU);

    const float* go_ptr = grad_output.data_read<float>();
    const float* x_ptr = x.data_read<float>();
    const float* g_ptr = gate.data_read<float>();
    float* gx_ptr = grad_x.data_write<float>();
    float* gg_ptr = grad_gate.data_write<float>();

#if defined(__AVX2__)
    swiglu_backward_avx2(go_ptr, x_ptr, g_ptr, gx_ptr, gg_ptr, n);
#else
    for (size_t i = 0; i < n; ++i) {
        auto [gx, gg] = swiglu_backward_scalar_impl(go_ptr[i], x_ptr[i], g_ptr[i]);
        gx_ptr[i] = gx;
        gg_ptr[i] = gg;
    }
#endif

    return {grad_x, grad_gate};
}

}  // namespace ct::ops
