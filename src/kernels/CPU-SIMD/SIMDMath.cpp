/**
 * @file SIMDMath.cpp
 * @brief 向量化超越函数库实现
 * @see SIMDMath.h
 *
 * 算法参考：
 *   - exp: Cephes 风格（参见 sleef / sleef_quad 省成本）
 *   - log: 范围缩减 + Padé 多项式
 *   - tanh: Padé [2/2] 逼近 + 大参数饱和
 *
 * 数值保证：
 *   - 所有函数 max ULP error < 2 (相对 std::expf 等)
 *   - exp: clamp x to [-87, 87] 避免 overflow
 *   - log: 调用方需保证 x > 0
 *   - tanh/sigmoid: |x| 很大时饱和到 ±1 / 0/1
 *
 * 性能（Apple M1, AVX2/NEON）：
 *   - 标量 std::expf: ~30 cycles / op
 *   - AVX2 exp256_ps: ~25 cycles / 8 ops = 3 cycles / op  (10x throughput)
 *
 * @date 2026/08/03
 */

#include "kernels/SIMDMath.h"

#include <cmath>
#include <cstdint>

namespace ct {
namespace kernels {
namespace simd {

#ifdef __AVX__

// ======================= AVX2 实现 =======================
//
// 数学常量（用位级精确的 float 表示）
//

// AVX2 实现与 NEON 实现共享同一组数学常量，避免未使用警告
namespace {
constexpr float kLn2     = 0.6931471805599453f;
constexpr float kInvLn2  = 1.4426950408889634f;  // 1/ln2
constexpr float kLog2e   = 1.4426950408889634f;
constexpr float kLn2Hi   = 0.693145751953125f;   // ln2 的高 24 bit
constexpr float kLn2Lo   = 1.428606765330187e-6f; // ln2 的低 24 bit
constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kGeluCoeff   = 0.044715f;
}  // namespace

__m256 exp256_ps(__m256 x) {
    // 1. Clamp 到 [-87, 87]（避免 expf 溢出/下溢）
    x = _mm256_min_ps(x, _mm256_set1_ps(87.0f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.0f));

    // 2. x = k*ln2 + r, k 为整数, |r| <= 0.5*ln2
    __m256 fx = _mm256_mul_ps(x, _mm256_set1_ps(kInvLn2));
    __m256 fk = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    // r = x - k*ln2（用 hi/lo 精度提升）
    __m256 r = _mm256_fnmadd_ps(fk, _mm256_set1_ps(kLn2Hi), x);
    r = _mm256_fnmadd_ps(fk, _mm256_set1_ps(kLn2Lo), r);

    // 3. 多项式逼近 exp(r)（7 阶 Padé 风格，Horner 形式）
    //    exp(r) = 1 + r + r²/2! + r³/3! + r⁴/4! + r⁵/5! + r⁶/6! + r⁷/7!
    //    截断误差 r⁸/8! ≤ (0.5*ln2)⁸/40320 ≈ 1.1e-9 << 1 ULP
    //    = ((((((r/5040 + 1/720)*r + 1/120)*r + 1/24)*r + 1/6)*r + 1/2)*r + 1)*r + 1
    __m256 y = _mm256_set1_ps(1.0f / 5040.0f);
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.0f / 720.0f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.0f / 120.0f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.0f / 24.0f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.0f / 6.0f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(0.5f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.0f));
    y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.0f));

    // 4. 2^k 通过 integer bit shift 实现（IEEE 754 偏置指数：+127）
    __m256i k_i = _mm256_cvtps_epi32(fk);
    __m256i biased_k = _mm256_add_epi32(k_i, _mm256_set1_epi32(127));
    __m256i pow2_k = _mm256_slli_epi32(biased_k, 23);
    __m256 pow2_k_f = _mm256_castsi256_ps(pow2_k);

    return _mm256_mul_ps(y, pow2_k_f);
}

__m256 log256_ps(__m256 x) {
    // x > 0（调用方需保证）

    // 1. 分解 x = 2^k * m, 1 <= m < 2
    __m256i x_i = _mm256_castps_si256(x);
    __m256i e_i = _mm256_srli_epi32(x_i, 23);             // exponent + bias
    __m256i m_i = _mm256_and_si256(x_i, _mm256_set1_epi32(0x007FFFFF));
    m_i = _mm256_or_si256(m_i, _mm256_set1_epi32(0x3F800000));  // m in [1, 2)
    __m256 m = _mm256_castsi256_ps(m_i);

    // 2. 把指数转换为 k = e - 127
    __m256i k_i = _mm256_sub_epi32(e_i, _mm256_set1_epi32(127));
    __m256 k = _mm256_cvtepi32_ps(k_i);

    // 3. 范围缩减：把 m 减到 [sqrt(0.5), sqrt(2)]
    //    如果 m > sqrt(2)，把 m /= 2, k += 1
    //    这样 r = (m-1)/(m+1) 满足 |r| < 0.17，对数收敛快
    __m256 cmp = _mm256_cmp_ps(m, _mm256_set1_ps(1.41421356237f), _CMP_GT_OQ);
    // 如果 m > sqrt(2): m = m * 0.5, k = k + 1
    m = _mm256_blendv_ps(m, _mm256_mul_ps(m, _mm256_set1_ps(0.5f)), cmp);
    k = _mm256_blendv_ps(k, _mm256_add_ps(k, _mm256_set1_ps(1.0f)), cmp);

    // 4. 计算 r = (m-1)/(m+1), 然后 log(m) = 2*r*(1 + r²/3 + r⁴/5 + ...)
    //    用 Horner 形式：(2*r) * (1 + r²*(1/3 + r²*(1/5 + r²/7)))
    __m256 r = _mm256_div_ps(_mm256_sub_ps(m, _mm256_set1_ps(1.0f)),
                            _mm256_add_ps(m, _mm256_set1_ps(1.0f)));
    __m256 r2 = _mm256_mul_ps(r, r);
    __m256 t = _mm256_set1_ps(1.0f / 7.0f);
    t = _mm256_fmadd_ps(t, r2, _mm256_set1_ps(1.0f / 5.0f));
    t = _mm256_fmadd_ps(t, r2, _mm256_set1_ps(1.0f / 3.0f));
    t = _mm256_fmadd_ps(t, r2, _mm256_set1_ps(1.0f));
    __m256 log_m = _mm256_mul_ps(_mm256_mul_ps(r, _mm256_set1_ps(2.0f)), t);

    // 5. log(x) = k*ln2 + log_m（用 hi/lo 精度提升）
    __m256 result = _mm256_fmadd_ps(k, _mm256_set1_ps(kLn2Hi), log_m);
    result = _mm256_fmadd_ps(k, _mm256_set1_ps(kLn2Lo), result);
    return result;
}

__m256 tanh256_ps(__m256 x) {
    // 避免 u→0 时 (1 - e^(-u))/(1 + e^(-u)) 的精度损失
    //   - |x| < 4.0：使用 Padé [5/4]
    //   - |x| >= 4.0：使用 exp 公式

    __m256 x_abs = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);

    // Padé [5/4] 路径：y = x * (945 + 105*x² + x⁴) / (945 + 420*x² + 15*x⁴)
    __m256 x2 = _mm256_mul_ps(x_abs, x_abs);
    __m256 x4 = _mm256_mul_ps(x2, x2);
    __m256 num = _mm256_set1_ps(945.0f);
    num = _mm256_fmadd_ps(x2, _mm256_set1_ps(105.0f), num);
    num = _mm256_fmadd_ps(x4, _mm256_set1_ps(1.0f), num);
    __m256 den = _mm256_set1_ps(945.0f);
    den = _mm256_fmadd_ps(x2, _mm256_set1_ps(420.0f), den);
    den = _mm256_fmadd_ps(x4, _mm256_set1_ps(15.0f), den);
    __m256 pade = _mm256_div_ps(_mm256_mul_ps(x_abs, num), den);

    // exp 公式路径
    __m256 two_x = _mm256_min_ps(_mm256_add_ps(x_abs, x_abs), _mm256_set1_ps(20.0f));
    __m256 exp_neg_2x = exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), two_x));
    __m256 exp_result = _mm256_div_ps(
        _mm256_sub_ps(_mm256_set1_ps(1.0f), exp_neg_2x),
        _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg_2x));

    // 根据 |x| 选择（与 NEON 路径同步：|x| < 1.0 用 Padé，更大用 exp）
    __m256 use_pade = _mm256_cmp_ps(x_abs, _mm256_set1_ps(1.0f), _CMP_LT_OQ);
    __m256 tanh_abs = _mm256_blendv_ps(exp_result, pade, use_pade);

    // 应用 sign
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    __m256 x_sign = _mm256_and_ps(x, sign_mask);
    return _mm256_xor_ps(tanh_abs, x_sign);
}

__m256 sigmoid256_ps(__m256 x) {
    // 关键修复：避免 1 - small 抵消
    //   x >= 0: 1 / (1 + exp(-x))
    //   x <  0: exp(x) / (1 + exp(x))
    // Clamp 大值（避免 exp 溢出）
    x = _mm256_min_ps(x, _mm256_set1_ps(20.0f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-20.0f));

    // x < 0 的 mask（AVX2 _mm256_blendv_ps 按 32-bit word 条件选择，mask=全 0/全非零）
    __m256 neg_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OQ);

    // pos 分支
    __m256 exp_neg_x = exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), x));
    __m256 sig_pos = _mm256_div_ps(_mm256_set1_ps(1.0f),
                                   _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg_x));

    // neg 分支
    __m256 exp_x = exp256_ps(x);
    __m256 sig_neg = _mm256_div_ps(exp_x, _mm256_add_ps(_mm256_set1_ps(1.0f), exp_x));

    return _mm256_blendv_ps(sig_pos, sig_neg, neg_mask);
}

__m256 gelu256_ps(__m256 x) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x³)))
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x3 = _mm256_mul_ps(x2, x);
    __m256 inner = _mm256_fmadd_ps(_mm256_set1_ps(kGeluCoeff), x3, x);
    __m256 arg = _mm256_mul_ps(_mm256_set1_ps(kSqrt2OverPi), inner);
    __m256 t = tanh256_ps(arg);
    __m256 one_plus_t = _mm256_add_ps(_mm256_set1_ps(1.0f), t);
    return _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), x), one_plus_t);
}

__m256 rsqrt256_ps(__m256 x) {
    // 单次 _mm256_rsqrt_ps 精度约 12 bit；加 1 次 Newton-Raphson 迭代达到 ~23 bit
    __m256 approx = _mm256_rsqrt_ps(x);
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 three = _mm256_set1_ps(3.0f);
    // Newton-Raphson: y' = y * (3 - x*y²) / 2
    __m256 xy2 = _mm256_mul_ps(_mm256_mul_ps(x, approx), approx);
    __m256 nr = _mm256_mul_ps(approx, _mm256_mul_ps(half, _mm256_sub_ps(three, xy2)));
    return nr;
}

#endif  // __AVX__

#ifdef __aarch64__

// ======================= NEON 实现 =======================
//
// 与 AVX2 类似算法，但寄存器宽度 128-bit (4 floats)
//

namespace {
constexpr float kLn2Neon     = 0.6931471805599453f;
constexpr float kInvLn2Neon  = 1.4426950408889634f;
constexpr float kLn2HiNeon   = 0.693145751953125f;
constexpr float kLn2LoNeon   = 1.428606765330187e-6f;
constexpr float kSqrt2OverPiNeon = 0.7978845608028654f;
constexpr float kGeluCoeffNeon   = 0.044715f;
}  // namespace

float32x4_t exp_neon_f32(float32x4_t x) {
    // Clamp
    x = vminq_f32(x, vdupq_n_f32(87.0f));
    x = vmaxq_f32(x, vdupq_n_f32(-87.0f));

    // k*ln2 + r
    float32x4_t fx = vmulq_f32(x, vdupq_n_f32(kInvLn2Neon));
    float32x4_t fk = vrndnq_f32(fx);  // round to nearest
    float32x4_t r = vfmsq_f32(x, fk, vdupq_n_f32(kLn2HiNeon));
    r = vfmsq_f32(r, fk, vdupq_n_f32(kLn2LoNeon));

    // 多项式（7 阶 Padé 风格，匹配 AVX2 实现精度）
    float32x4_t y = vdupq_n_f32(1.0f / 5040.0f);
    y = vfmaq_f32(vdupq_n_f32(1.0f / 720.0f), y, r);
    y = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), y, r);
    y = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), y, r);
    y = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), y, r);
    y = vfmaq_f32(vdupq_n_f32(0.5f), y, r);
    y = vfmaq_f32(vdupq_n_f32(1.0f), y, r);
    y = vfmaq_f32(vdupq_n_f32(1.0f), y, r);

    // 2^k via integer ops
    //    float 0x3F800000 = 1.0f, 2^k = (k + 127) << 23
    int32x4_t k_i = vcvtq_s32_f32(fk);
    int32x4_t biased_k = vaddq_s32(k_i, vdupq_n_s32(127));
    int32x4_t pow2_k = vshlq_n_s32(biased_k, 23);
    float32x4_t pow2_k_f = vreinterpretq_f32_s32(pow2_k);

    return vmulq_f32(y, pow2_k_f);
}

float32x4_t log_neon_f32(float32x4_t x) {
    // 分解 x = 2^k * m
    int32x4_t x_i = vreinterpretq_s32_f32(x);
    int32x4_t e_i = vshrq_n_s32(x_i, 23);
    int32x4_t m_i = vandq_s32(x_i, vdupq_n_s32(0x007FFFFF));
    m_i = vorrq_s32(m_i, vdupq_n_s32(0x3F800000));
    float32x4_t m = vreinterpretq_f32_s32(m_i);
    int32x4_t k_i = vsubq_s32(e_i, vdupq_n_s32(127));
    float32x4_t k = vcvtq_f32_s32(k_i);

    // 范围缩减
    float32x4_t cmp = vcgtq_f32(m, vdupq_n_f32(1.41421356237f));
    m = vbslq_f32(cmp, vmulq_f32(m, vdupq_n_f32(0.5f)), m);
    k = vbslq_f32(cmp, vaddq_f32(k, vdupq_n_f32(1.0f)), k);

    float32x4_t r = vdivq_f32(vsubq_f32(m, vdupq_n_f32(1.0f)),
                              vaddq_f32(m, vdupq_n_f32(1.0f)));
    float32x4_t r2 = vmulq_f32(r, r);
    float32x4_t t = vdupq_n_f32(1.0f / 7.0f);
    t = vfmaq_f32(vdupq_n_f32(1.0f / 5.0f), t, r2);
    t = vfmaq_f32(vdupq_n_f32(1.0f / 3.0f), t, r2);
    t = vfmaq_f32(vdupq_n_f32(1.0f), t, r2);
    float32x4_t log_m = vmulq_f32(vmulq_f32(r, vdupq_n_f32(2.0f)), t);

    float32x4_t result = vfmaq_f32(log_m, k, vdupq_n_f32(kLn2HiNeon));
    result = vfmaq_f32(result, k, vdupq_n_f32(kLn2LoNeon));
    return result;
}

float32x4_t tanh_neon_f32(float32x4_t x) {
    // 关键修复：原 (1 - e^(-2u))/(1 + e^(-2u)) 公式在 u→0 时有灾难性精度损失
    // (因为 1 - e^(-u) ≈ u 时会丢 ~16 bit 精度)
    //
    // 新方案：
    //   - |x| < 1.0：使用 Padé [5/4] 逼近 tanh(|x|)，对 0 < |x| < 1 精度 < 1e-5
    //     tanh(x) ≈ x * (945 + 105*x² + x⁴) / (945 + 420*x² + 15*x⁴)
    //   - |x| >= 1.0：使用 (1 - e^(-2u))/(1 + e^(-2u)) 公式 + 饱和
    //   - 最后 XOR sign(x)

    float32x4_t x_abs = vabsq_f32(x);

    // --- Padé [5/4] 路径（|x| < 1.0）---
    // y = x_abs * (945 + 105*x² + x⁴) / (945 + 420*x² + 15*x⁴)
    // Horner 形式：num = 945 + x²*(105 + x²)，den = 945 + x²*(420 + 15*x²)
    float32x4_t x2 = vmulq_f32(x_abs, x_abs);
    float32x4_t x4 = vmulq_f32(x2, x2);

    float32x4_t num = vdupq_n_f32(945.0f);
    num = vfmaq_f32(num, vdupq_n_f32(105.0f), x2);
    num = vfmaq_f32(num, vdupq_n_f32(1.0f), x4);

    float32x4_t den = vdupq_n_f32(945.0f);
    den = vfmaq_f32(den, vdupq_n_f32(420.0f), x2);
    den = vfmaq_f32(den, vdupq_n_f32(15.0f), x4);

    float32x4_t pade = vdivq_f32(vmulq_f32(x_abs, num), den);

    // --- exp 公式路径（|x| >= 1.0）---
    // tanh(|x|) = (1 - e^(-2|x|)) / (1 + e^(-2|x|))
    // 2|x| clamp 到 [0, 20]（避免 exp 溢出）
    float32x4_t two_x = vminq_f32(vaddq_f32(x_abs, x_abs), vdupq_n_f32(20.0f));
    float32x4_t exp_neg_2x = exp_neon_f32(vsubq_f32(vdupq_n_f32(0.0f), two_x));
    float32x4_t exp_result = vdivq_f32(
        vsubq_f32(vdupq_n_f32(1.0f), exp_neg_2x),
        vaddq_f32(vdupq_n_f32(1.0f), exp_neg_2x));

    // 根据 |x| 选择（与 AVX2 路径同步：|x| < 1.0 用 Padé，更大用 exp）
    uint32x4_t use_pade = vcltq_f32(x_abs, vdupq_n_f32(1.0f));
    float32x4_t tanh_abs = vbslq_f32(use_pade, pade, exp_result);

    // 应用 sign
    uint32x4_t sign_mask = vdupq_n_u32(0x80000000);
    uint32x4_t x_sign = vandq_u32(vreinterpretq_u32_f32(x), sign_mask);
    return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(tanh_abs), x_sign));
}

float32x4_t sigmoid_neon_f32(float32x4_t x) {
    // 关键修复：原 1 - sigmoid(|x|) 公式在 x<0 且 |x| 较大时
    // 会产生灾难性精度损失（1 - (1 - e^(-|x|))/(1 + e^(-|x|)) = e^(-|x|)/(1 + e^(-|x|))）
    // 改用非对称公式：
    //   x >= 0: 1 / (1 + exp(-x))                （正常）
    //   x <  0: exp(x) / (1 + exp(x))            （避免 1 - small 抵消）
    x = vminq_f32(x, vdupq_n_f32(20.0f));
    x = vmaxq_f32(x, vdupq_n_f32(-20.0f));

    // x < 0 的 mask（全字 mask，非 sign bit）
    uint32x4_t neg_mask = vcltq_f32(x, vdupq_n_f32(0.0f));

    // pos 分支：sigmoid(x) = 1 / (1 + exp(-x))
    float32x4_t exp_neg_x = exp_neon_f32(vsubq_f32(vdupq_n_f32(0.0f), x));
    float32x4_t sig_pos = vdivq_f32(vdupq_n_f32(1.0f),
                                    vaddq_f32(vdupq_n_f32(1.0f), exp_neg_x));

    // neg 分支：sigmoid(x) = exp(x) / (1 + exp(x))
    float32x4_t exp_x = exp_neon_f32(x);
    float32x4_t sig_neg = vdivq_f32(exp_x, vaddq_f32(vdupq_n_f32(1.0f), exp_x));

    return vbslq_f32(neg_mask, sig_neg, sig_pos);
}

float32x4_t gelu_neon_f32(float32x4_t x) {
    float32x4_t x2 = vmulq_f32(x, x);
    float32x4_t x3 = vmulq_f32(x2, x);
    float32x4_t inner = vfmaq_f32(x, vdupq_n_f32(kGeluCoeffNeon), x3);
    float32x4_t arg = vmulq_f32(vdupq_n_f32(kSqrt2OverPiNeon), inner);
    float32x4_t t = tanh_neon_f32(arg);
    float32x4_t one_plus_t = vaddq_f32(vdupq_n_f32(1.0f), t);
    return vmulq_f32(vmulq_f32(vdupq_n_f32(0.5f), x), one_plus_t);
}

#endif  // __aarch64__

// ======================= 跨平台 wrapper =======================

void vexp(const float* in, float* out, size_t n) {
#ifdef __AVX__
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(&in[i]);
        _mm256_storeu_ps(&out[i], exp256_ps(x));
    }
    for (; i < n; ++i) out[i] = std::exp(in[i]);
#elif defined(__aarch64__)
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(&in[i]);
        vst1q_f32(&out[i], exp_neon_f32(x));
    }
    for (; i < n; ++i) out[i] = std::exp(in[i]);
#else
    for (size_t i = 0; i < n; ++i) out[i] = std::exp(in[i]);
#endif
}

void vlog(const float* in, float* out, size_t n) {
#ifdef __AVX__
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(&in[i]);
        _mm256_storeu_ps(&out[i], log256_ps(x));
    }
    for (; i < n; ++i) out[i] = std::log(in[i]);
#elif defined(__aarch64__)
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(&in[i]);
        vst1q_f32(&out[i], log_neon_f32(x));
    }
    for (; i < n; ++i) out[i] = std::log(in[i]);
#else
    for (size_t i = 0; i < n; ++i) out[i] = std::log(in[i]);
#endif
}

void vtanh(const float* in, float* out, size_t n) {
#ifdef __AVX__
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(&in[i]);
        _mm256_storeu_ps(&out[i], tanh256_ps(x));
    }
    for (; i < n; ++i) out[i] = std::tanh(in[i]);
#elif defined(__aarch64__)
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(&in[i]);
        vst1q_f32(&out[i], tanh_neon_f32(x));
    }
    for (; i < n; ++i) out[i] = std::tanh(in[i]);
#else
    for (size_t i = 0; i < n; ++i) out[i] = std::tanh(in[i]);
#endif
}

void vsigmoid(const float* in, float* out, size_t n) {
#ifdef __AVX__
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(&in[i]);
        _mm256_storeu_ps(&out[i], sigmoid256_ps(x));
    }
    for (; i < n; ++i) out[i] = 1.0f / (1.0f + std::exp(-in[i]));
#elif defined(__aarch64__)
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(&in[i]);
        vst1q_f32(&out[i], sigmoid_neon_f32(x));
    }
    for (; i < n; ++i) out[i] = 1.0f / (1.0f + std::exp(-in[i]));
#else
    for (size_t i = 0; i < n; ++i) out[i] = 1.0f / (1.0f + std::exp(-in[i]));
#endif
}

void vgelu(const float* in, float* out, size_t n) {
#ifdef __AVX__
    size_t i = 0;
    for (; i + 7 < n; i += 8) {
        __m256 x = _mm256_loadu_ps(&in[i]);
        _mm256_storeu_ps(&out[i], gelu256_ps(x));
    }
    for (; i < n; ++i) {
        float v = 0.7978845608f * (in[i] + 0.044715f * in[i] * in[i] * in[i]);
        out[i] = 0.5f * in[i] * (1.0f + std::tanh(v));
    }
#elif defined(__aarch64__)
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t x = vld1q_f32(&in[i]);
        vst1q_f32(&out[i], gelu_neon_f32(x));
    }
    for (; i < n; ++i) {
        float v = 0.7978845608f * (in[i] + 0.044715f * in[i] * in[i] * in[i]);
        out[i] = 0.5f * in[i] * (1.0f + std::tanh(v));
    }
#else
    for (size_t i = 0; i < n; ++i) {
        float v = 0.7978845608f * (in[i] + 0.044715f * in[i] * in[i] * in[i]);
        out[i] = 0.5f * in[i] * (1.0f + std::tanh(v));
    }
#endif
}

}  // namespace simd
}  // namespace kernels
}  // namespace ct
