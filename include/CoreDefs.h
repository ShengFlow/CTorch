//
// Created by renyz on 2026/3/13.
//

#ifndef CTORCH_COREDEFS_H
#define CTORCH_COREDEFS_H

#include <csignal>

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
#if defined(__GNUC__) || defined(__clang__)
  #define CT_NOINLINE __attribute__((noinline))
  #define CT_ALWAYS_FORCEINLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
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
#if defined(__GNUC__) || defined(__clang__)
  #define CT_FUNC_NAME __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
  #define CT_FUNC_NAME __FUNCSIG__
#else
  #define CT_FUNC_NAME __func__
#endif

/**
 * Explicit breakpoint
 */
#if defined(__GNUC__)
  #define CT_BREAKPOINT std::raise(SIGTRAP)
#elif defined(__clang__)
  #define CT_BREAKPOINT __builtin_debugtrap()
#elif defined(_MSC_VER)
  #define CT_BREAKPOINT __debugbreak()
#else
  #define CT_BREAKPOINT std::raise(SIGTRAP)
#endif

#endif //CTORCH_COREDEFS_H
