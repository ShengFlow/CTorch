//
// Created by renyz on 2026/3/13.
//

#ifndef CTORCH_COREDEFS_H
#define CTORCH_COREDEFS_H

#include <csignal>
#include <cstdint>  // for standard int defs

#include "Features.h"

/**
 * Debug & release flags, currently controlled by NDEBUG (TODO raw handling, nerf it)
 */
#if defined(NDEBUG)
#define CT_RELEASE 1
#else
#define CT_DEBUG 1
#endif

/**
 * Definition for CT_NOINLINE, CT_FORCEINLINE, and CT_ALWAYS_FORCEINLINE
 * Inline controller.
 * CT_NOINLINE: function never inline.
 * CT_FORCEINLINE: function always inline, except when compiling in debug mode (for easy debugging).
 * CT_ALWAYS_FORCEINLINE: function always inline, used for primitives.
 */
#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
  #define CT_NOINLINE __attribute__((noinline))
  #define CT_ALWAYS_FORCEINLINE __attribute__((always_inline)) inline
#elif defined(COMPILER_MSVC)
  #define CT_NOINLINE __declspec(noinline)
  #define CT_ALWAYS_FORCEINLINE __forceinline
#else
  #define CT_NOINLINE
  #define CT_ALWAYS_FORCEINLINE inline
#endif
#if CT_RELEASE
  #define CT_FORCEINLINE CT_ALWAYS_FORCEINLINE
#else
  #define CT_FORCEINLINE inline
#endif

/**
 * Pretty function name
 */
#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
  #define CT_FUNC_NAME __PRETTY_FUNCTION__
#elif defined(COMPILER_MSVC)
  #define CT_FUNC_NAME __FUNCSIG__
#else
  #define CT_FUNC_NAME __func__
#endif

/**
 * Explicit breakpoint
 */
#if defined(COMPILER_GCC)
  #define CT_BREAKPOINT std::raise(SIGTRAP)
#elif defined(COMPILER_CLANG)
  #define CT_BREAKPOINT __builtin_debugtrap()
#elif defined(COMPILER_MSVC)
  #define CT_BREAKPOINT __debugbreak()
#else
  #define CT_BREAKPOINT std::raise(SIGTRAP)
#endif

/**
 * Pure marker - const functions (no side effects, no global state access)
 */
#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
#define CT_PURE __attribute__((const))
#else
#define CT_PURE
#endif

/**
 * Pure Read marker - pure functions (no side effects, but may read global state)
 */
#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
#define CT_PURE_READ __attribute__((pure))
#else
#define CT_PURE_READ
#endif

/**
 * Restrict marker - pointer aliasing hint
 */
#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
#define CT_RESTRICT __restrict__
#elif defined(COMPILER_MSVC)
#define CT_RESTRICT __restrict
#else
#define CT_RESTRICT
#endif

/**
 * Hot marker - frequently executed code (function attribute)
 */
#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
#define CT_HOT __attribute__((hot))
#else
#define CT_HOT
#endif

/**
 * Cold marker - rarely executed code (function attribute)
 */
#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
#define CT_COLD __attribute__((cold))
#else
#define CT_COLD
#endif

/**
 * Malloc marker - function returns newly allocated memory with no aliasing
 */
#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
#define CT_MALLOC __attribute__((malloc))
#else
#define CT_MALLOC
#endif

/**
 * Unroll pragma for loop
 */
#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
#define CT_UNROLL _Pragma("GCC unroll 16")
#elif defined(COMPILER_MSVC)
#define CT_UNROLL __pragma(loop(unroll))
#endif

/**
 * Unreachable hint
 */
#if defined(__cplusplus) && __cplusplus >= 202302L
  #define CT_UNREACHABLE() std::unreachable()
#elif defined(COMPILER_GCC) || defined(COMPILER_CLANG)
  #define CT_UNREACHABLE() __builtin_unreachable()
#elif defined(COMPILER_MSVC)
  #define CT_UNREACHABLE() __assume(false)
#else
  #define CT_UNREACHABLE() ((void)0)
#endif

/**
 * Check for constant expression
 */
#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
  #define CT_IS_CONSTANT_EXPR(x) (__builtin_constant_p(x))
#else
  #define CT_IS_CONSTANT_EXPR(x) (0)
#endif


namespace ct {

/**
 * Float types
 *
 * @par 为什么 bfloat16/float16 需要条件定义
 *
 * `__bf16` 与 `_Float16` 不是可移植类型：
 *   - aarch64（Apple Silicon / ARMv8）：GCC 与 Clang 均原生提供；
 *   - x86-64：Clang 原生提供，**GCC 需 13+**（GCC 12 在 x86 上完全没有 `__bf16`）。
 *
 * 因此「macOS 能编、Linux x86 GCC 12 挂掉」不是环境配置问题，而是这两个别名
 * 原先无条件依赖了非可移植类型。本处按（架构, 编译器, 版本）判定可用性，
 * 不可用时回退到语义等价的位容器 —— 见下方 CT_HAS_NATIVE_BF16。
 *
 * @par 回退类型为何安全
 *
 * bf16 在本库中**只作 2 字节位容器**：ScalarConvert 的 Bf16ToBits/BitsToBf16/
 * Bf16ToFloat/FloatToBf16 全部通过 uint16_t 位模式实现（Bf16ToFloat 即左移 16
 * 再 BitCast），所有算术都在 float 域完成。故用「内部存 uint16_t + 显式转换」的
 * 最小包装替换，行为与原生类型逐位一致；**不可回退为裸 uint16_t** —— 那会让
 * `static_cast<bfloat16_t>(bits)` 静默失去浮点语义。
 */
#if defined(__aarch64__) || defined(__ARM_ARCH) || defined(_M_ARM64) ||        \
    defined(COMPILER_CLANG) || (defined(COMPILER_GCC) && __GNUC__ >= 13)
  #define CT_HAS_NATIVE_BF16 1
#else
  #define CT_HAS_NATIVE_BF16 0
#endif

#if CT_HAS_NATIVE_BF16
  using bfloat16_t = __bf16;
#else
  /// x86 + GCC < 13 的回退：bf16 位模式的等价容器（sizeof == 2）
  struct bfloat16_t {
      uint16_t bits = 0;
      constexpr bfloat16_t() = default;
      constexpr explicit bfloat16_t(uint16_t b) : bits(b) {}
      constexpr explicit bfloat16_t(float f)
          : bits(static_cast<uint16_t>(__builtin_bit_cast(uint32_t, f) >> 16)) {}
      constexpr operator uint16_t() const { return bits; }
      constexpr operator float() const {
          return __builtin_bit_cast(float, static_cast<uint32_t>(bits) << 16);
      }
  };
  static_assert(sizeof(bfloat16_t) == 2, "bfloat16_t must be 2 bytes");
#endif

#if defined(__aarch64__) || defined(__ARM_ARCH) || defined(_M_ARM64) ||        \
    defined(COMPILER_CLANG) || (defined(COMPILER_GCC) && __GNUC__ >= 12)
  #define CT_HAS_NATIVE_F16 1
#else
  #define CT_HAS_NATIVE_F16 0
#endif

#if CT_HAS_NATIVE_F16
  using float16_t = _Float16;
#else
  /// x86 + GCC < 12 的回退：float16 位模式的等价容器（sizeof == 2）
  struct float16_t {
      uint16_t bits = 0;
      constexpr float16_t() = default;
      constexpr explicit float16_t(uint16_t b) : bits(b) {}
      constexpr operator uint16_t() const { return bits; }
  };
  static_assert(sizeof(float16_t) == 2, "float16_t must be 2 bytes");
#endif
using float32_t = float;
using float64_t = double;
/**
 * Signed native int, having the same width as machine word.
 */
using nint_t = ptrdiff_t;
/**
 * Unsigned native int, having the same width as machine word.
 */
using nuint_t = size_t;

template <typename T>
struct TypeTraits {
  static constexpr bool is_integer = std::is_integral_v<T>;
  static constexpr bool is_signed = std::is_signed_v<T>;
  static constexpr bool is_float = std::is_floating_point_v<T>;
  static constexpr size_t bits = sizeof(T) * 8;
  static constexpr bool is_bfloat16 = false;
  static constexpr bool is_float16 = false;
};

template <>
struct TypeTraits<bfloat16_t> {
  static constexpr bool is_integer = false;
  static constexpr bool is_signed = true;
  static constexpr bool is_float = true;
  static constexpr size_t bits = 16;
  static constexpr bool is_bfloat16 = true;
  static constexpr bool is_float16 = false;
};

template <>
struct TypeTraits<float16_t> {
  static constexpr bool is_integer = false;
  static constexpr bool is_signed = true;
  static constexpr bool is_float = true;
  static constexpr size_t bits = 16;
  static constexpr bool is_bfloat16 = false;
  static constexpr bool is_float16 = true;
};

template <typename T> constexpr bool IsIntV = TypeTraits<T>::is_integer;
template <typename T> constexpr bool IsFloatV = TypeTraits<T>::is_float;
template <typename T> constexpr bool IsSignedV = TypeTraits<T>::is_signed;
template <typename T> constexpr bool IsBfloat16V = TypeTraits<T>::is_bfloat16;
template <typename T> constexpr bool IsFloat16V = TypeTraits<T>::is_float16;
template <typename T> constexpr bool IsStandardFloatV = IsFloatV<T> && !IsBfloat16V<T> && !IsFloat16V<T>;
template <typename T> constexpr size_t TypeBitsV = TypeTraits<T>::bits;


} // namespace ct

#endif //CTORCH_COREDEFS_H
