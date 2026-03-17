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
 * Pure marker
 */
#if defined(COMPILER_GCC) || defined(COMPILER_CLANG)
#define CT_PURE __attribute__((const))
#else
#define CT_PURE
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

/**
 * Float types
 */
namespace ct {
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
} // namespace ct

#endif //CTORCH_COREDEFS_H
