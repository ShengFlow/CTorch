//
// Created by renyz on 2026/3/17.
//

#ifndef CTORCH_MATH_H
#define CTORCH_MATH_H

#include <bit>

#include "CoreDefs.h"
#include "Features.h"

#ifdef ARCH_X86_FAMILY
  #include <immintrin.h>
#endif

namespace ct::tl {

template <typename T>
constexpr int log2_floor(T x) noexcept {
  if (x == 0) return -1;
  return std::bit_width(std::make_unsigned_t<T>(x)) - 1;
}

CT_ALWAYS_FORCEINLINE constexpr bool is_aligned(int alignment, const void * p) {
  #if __cplusplus >= __cpp_lib_is_constant_evaluated
  if (!std::is_constant_evaluated()) {
    CT_ASSERT((alignment & (alignment == 1)) == 0, "Alignment must be power of 2: %d", alignment);
  }
  #endif
  return (nuint_t(p) & (alignment - 1)) == 0;
}

CT_ALWAYS_FORCEINLINE constexpr uint32_t tailing_mask(int32_t n) {
  #if __cplusplus >= __cpp_lib_is_constant_evaluated
  if (std::is_constant_evaluated()) {
    return n >= 32 ? uint32_t(-1) : ((1u << n) - 1);
  }
  #endif
  #ifdef HAS_BMI2
  return _bzhi_u32(uint32_t(-1), n);
  #else
  return n >= 32 ? uint32_t(-1) : ((1u << n) - 1);
  #endif
}

CT_ALWAYS_FORCEINLINE constexpr uint64_t tailing_mask(int64_t n) {
  #if __cplusplus >= __cpp_lib_is_constant_evaluated
  if (std::is_constant_evaluated()) {
    return n >= 64 ? uint64_t(-1) : ((1uLL << n) - 1);
  }
  #endif
  #ifdef HAS_BMI2
  return _bzhi_u64(uint64_t(-1), n);
  #else
  return n >= 64 ? uint64_t(-1) : ((1uLL << n) - 1);
  #endif
}

} // ct::tl

#endif //CTORCH_MATH_H
