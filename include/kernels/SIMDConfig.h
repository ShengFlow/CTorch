/**
 * @file SIMDConfig.h
 * @brief 编译期 SIMD 架构检测与向量宽度统一抽象（NEON / AVX2 / AVX-512）
 * @details 本头文件在编译期（由 -march=native 触发的架构宏）确定宿主机当前
 *          float32 向量宽度与类型，作为手写 SIMD 内核、C-ABI wrapper 与
 *          MLIR 代码生成的统一宽度来源，替代散落的硬编码 \`VL = 8\`。
 *
 *          架构优先顺序（x86 上 -march=native 会同时定义 __AVX2__ 与
 *          __AVX512F__，故必须先判 512 再判 256）：
 *            1. __AVX512F__   -> 16 x f32 (512-bit)
 *            2. __AVX2__      ->  8 x f32 (256-bit)
 *            3. __aarch64__   ->  4 x f32 (128-bit NEON)
 *            4. 其他          ->  1 (标量回退)
 *
 * @date 2026/09/05
 */
#ifndef CTORCH_KERNELS_SIMD_CONFIG_H
#define CTORCH_KERNELS_SIMD_CONFIG_H

#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace ct {
namespace kernels {
namespace simd {

// ======================= 编译期架构枚举 =======================
enum class SimdArch : std::uint8_t {
    Scalar = 0,
    Neon   = 1,   ///< aarch64 NEON, 128-bit
    Avx2   = 2,   ///< x86-64 AVX2+FMA, 256-bit
    Avx512 = 3,   ///< x86-64 AVX-512F+DQ, 512-bit
};

// ======================= 编译期检测当前架构 =======================
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    constexpr SimdArch kSimdArch = SimdArch::Avx512;
    using F32Vec = __m512;
    constexpr std::size_t kSimdFloatLanes = 16;  ///< float32 lanes per vector
    constexpr int kVecWidthBits = 512;
    constexpr std::size_t kSimdLoadStep = 16;
#elif defined(__AVX2__)
    constexpr SimdArch kSimdArch = SimdArch::Avx2;
    using F32Vec = __m256;
    constexpr std::size_t kSimdFloatLanes = 8;
    constexpr int kVecWidthBits = 256;
    constexpr std::size_t kSimdLoadStep = 8;
#elif defined(__aarch64__)
    constexpr SimdArch kSimdArch = SimdArch::Neon;
    using F32Vec = float32x4_t;
    constexpr std::size_t kSimdFloatLanes = 4;
    constexpr int kVecWidthBits = 128;
    constexpr std::size_t kSimdLoadStep = 4;
#else
    constexpr SimdArch kSimdArch = SimdArch::Scalar;
    constexpr std::size_t kSimdFloatLanes = 1;
    constexpr int kVecWidthBits = 32;
    constexpr std::size_t kSimdLoadStep = 1;
#endif

// ======================= 便捷别名 =======================
/// 当前架构下一条 float32 向量的元素数（用于步进与 lane 展开）
inline constexpr std::size_t simdLanes() { return kSimdFloatLanes; }
/// 当前架构向量位宽
inline constexpr int simdBits() { return kVecWidthBits; }

}  // namespace simd
}  // namespace kernels
}  // namespace ct

#endif  // CTORCH_KERNELS_SIMD_CONFIG_H
