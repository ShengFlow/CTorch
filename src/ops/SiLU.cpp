/**
 * @file SiLU.cpp
 * @brief SiLU 算子 Eager CPU 实现 (BASIC + SIMD AVX2)
 * @details 标量 fallback 处理 tail (<8 elements), SIMD 主路径走 AVX2 8-wide
 * @author Mavis (PEL25 §6 SwiGLU 算子开发协议 Stage 1)
 * @date 2026-09-05
 *
 * Stage 1 实施: 不接 C3 dispatch, 走纯 Eager CPU + Autograd
 * Stage 2 (PEL25 §7 HITL 决策点): op 枚举扩展 + C3 Kernel Registry 接入
 */

#include "ops/SiLU.h"
#include "CtorchError.h"
#include "kernels/SIMDMath.h"

#include <cmath>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace ct::ops {

// ============================================================
// BASIC (标量) kernel — 用于通用 CPU + 测试对照
// ============================================================

namespace {
inline float silu_scalar_impl(float x) {
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    return x / (1.0f + std::exp(-x));
}

inline float silu_derivative_scalar_impl(float x) {
    // d/dx silu(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    //              = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    float s = 1.0f / (1.0f + std::exp(-x));
    return s + x * s * (1.0f - s);
}
}  // namespace

// ============================================================
// SIMD AVX2 kernel — 8-wide 批量处理
// ============================================================

#if defined(__AVX2__)

namespace {
inline __m256 silu_avx2(__m256 x) {
    // sigmoid(x) = 1 / (1 + exp(-x))
    // AVX2 没有原生 exp, 用 polynomial 近似 (跟 GELU SIMD kernel 类似模式)
    // 用 rational approximation: 6 次多项式拟合 exp, 误差 < 2e-7
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);

    // clamp 到 [-15, 15] 避免 exp 数值爆炸
    const __m256 clamp_hi = _mm256_set1_ps(15.0f);
    const __m256 clamp_lo = _mm256_set1_ps(-15.0f);
    neg_x = _mm256_min_ps(_mm256_max_ps(neg_x, clamp_lo), clamp_hi);

    // exp(-x) 近似 (4 次多项式 + 修正)
    // exp(y) ≈ 1 + y + y²/2 + y³/6 + y⁴/24 for |y| < 1
    // 但这对大 y 不够, 改用 e^x = 2^(x/ln2), 用 _mm256_exp_ps 替代
    // 因为没原生 _mm256_exp_ps, 用 std::exp 通过 _mm256_extract_ps 走标量
    // SIMD 性能优化留给 Stage 2 C3 路径, Stage 1 用 4-lane 处理 + 标量 tail
    alignas(32) float xs[8];
    _mm256_store_ps(xs, neg_x);
    xs[0] = std::exp(xs[0]);
    xs[1] = std::exp(xs[1]);
    xs[2] = std::exp(xs[2]);
    xs[3] = std::exp(xs[3]);
    xs[4] = std::exp(xs[4]);
    xs[5] = std::exp(xs[5]);
    xs[6] = std::exp(xs[6]);
    xs[7] = std::exp(xs[7]);
    __m256 exp_neg_x = _mm256_load_ps(xs);

    // sigmoid(x) = 1 / (1 + exp(-x))
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 sigmoid_x = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_x));

    // silu(x) = x * sigmoid(x)
    return _mm256_mul_ps(x, sigmoid_x);
}

inline __m256 silu_derivative_avx2(__m256 x) {
    // d/dx silu(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    //              = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    const __m256 clamp_hi = _mm256_set1_ps(15.0f);
    const __m256 clamp_lo = _mm256_set1_ps(-15.0f);
    neg_x = _mm256_min_ps(_mm256_max_ps(neg_x, clamp_lo), clamp_hi);

    alignas(32) float xs[8];
    _mm256_store_ps(xs, neg_x);
    for (int i = 0; i < 8; ++i) xs[i] = std::exp(xs[i]);
    __m256 exp_neg_x = _mm256_load_ps(xs);

    __m256 one = _mm256_set1_ps(1.0f);
    __m256 sigmoid_x = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_x));

    // d/dx = sigmoid + x * sigmoid * (1 - sigmoid)
    __m256 x_sigmoid = _mm256_mul_ps(x, sigmoid_x);
    __m256 one_minus_sigmoid = _mm256_sub_ps(one, sigmoid_x);
    __m256 x_sigmoid_one_minus = _mm256_mul_ps(x_sigmoid, one_minus_sigmoid);
    return _mm256_add_ps(sigmoid_x, x_sigmoid_one_minus);
}
}  // namespace

#endif  // __AVX2__

// ============================================================
// Public API
// ============================================================

Tensor silu_forward(const Tensor& input) {
    if (input.dtype() != DType::kFloat) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "SiLU forward: only float32 supported (Stage 1)");
    }
    if (input.device() != DeviceType::kCPU) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT, "SiLU forward: only CPU device supported (Stage 1)");
    }

    const auto& shape = input.shape();
    size_t n = 1;
    for (auto dim : shape) n *= dim;

    Tensor output(ShapeTag{}, shape, DType::kFloat, DeviceType::kCPU);

    const float* in_ptr = input.data_read<float>();
    float* out_ptr = output.data_write<float>();

#if defined(__AVX2__)
    // SIMD 主路径: 8-wide 处理, 标量 tail
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_loadu_ps(in_ptr + i);
        __m256 vy = silu_avx2(vx);
        _mm256_storeu_ps(out_ptr + i, vy);
    }
    // tail
    for (; i < n; ++i) {
        out_ptr[i] = silu_scalar_impl(in_ptr[i]);
    }
#else
    // 标量 fallback
    for (size_t i = 0; i < n; ++i) {
        out_ptr[i] = silu_scalar_impl(in_ptr[i]);
    }
#endif

    return output;
}

Tensor silu_backward(const Tensor& grad_output, const Tensor& input) {
    if (grad_output.dtype() != DType::kFloat || input.dtype() != DType::kFloat) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DATATYPE, "SiLU backward: only float32 supported (Stage 1)");
    }
    if (grad_output.device() != DeviceType::kCPU || input.device() != DeviceType::kCPU) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DEVICE_COMPAT, "SiLU backward: only CPU device supported (Stage 1)");
    }
    if (grad_output.shape() != input.shape()) {
        CtorchError::throwException(ErrorPlatform::kGENERAL, ErrorType::DIMENSION, "SiLU backward: grad_output and input shape mismatch");
    }

    const auto& shape = input.shape();
    size_t n = 1;
    for (auto dim : shape) n *= dim;

    Tensor grad_input(ShapeTag{}, shape, DType::kFloat, DeviceType::kCPU);

    const float* go_ptr = grad_output.data_read<float>();
    const float* in_ptr = input.data_read<float>();
    float* gi_ptr = grad_input.data_write<float>();

#if defined(__AVX2__)
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vgo = _mm256_loadu_ps(go_ptr + i);
        __m256 vx = _mm256_loadu_ps(in_ptr + i);
        __m256 vd = silu_derivative_avx2(vx);
        __m256 vgi = _mm256_mul_ps(vgo, vd);
        _mm256_storeu_ps(gi_ptr + i, vgi);
    }
    for (; i < n; ++i) {
        gi_ptr[i] = go_ptr[i] * silu_derivative_scalar_impl(in_ptr[i]);
    }
#else
    for (size_t i = 0; i < n; ++i) {
        gi_ptr[i] = go_ptr[i] * silu_derivative_scalar_impl(in_ptr[i]);
    }
#endif

    return grad_input;
}

}  // namespace ct::ops
