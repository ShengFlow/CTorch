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
 */
using bfloat16_t = __bf16;
using float16_t = _Float16;
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
