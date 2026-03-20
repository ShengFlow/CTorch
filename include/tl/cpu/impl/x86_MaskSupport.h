//
// Created by renyz on 2026/3/19.
//

#ifndef CTORCH_X86_MASKSUPPORT_H
#define CTORCH_X86_MASKSUPPORT_H

#include "CoreDefs.h"
#include "tl/cpu/VecBase.h"

#ifndef ARCH_X86_FAMILY
#error "Not x86 family"
#endif
#include <immintrin.h>
#include <x86intrin.h>

namespace ct::tl::vec::x86 {
CT_ALWAYS_FORCEINLINE __m128i mask8_to_epi16_simd(uint8_t mask) {
  static const __m128i bits = _mm_set_epi16(0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);
  __m128i v = _mm_set1_epi16(mask);
  return _mm_cmpeq_epi16(_mm_and_si128(v, bits), bits);
}

CT_ALWAYS_FORCEINLINE uint8_t epi16_to_mask8_simd(__m128i vec) {
  static const __m128i bits = _mm_set_epi16(0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);
  __m128i shifted = _mm_srai_epi16(vec, 15); // apply from MSB of 16-bit int
  __m128i masked = _mm_and_si128(shifted, bits);
  masked = _mm_or_si128(masked, _mm_srli_si128(masked, 8));
  masked = _mm_or_si128(masked, _mm_srli_si128(masked, 4));
  masked = _mm_or_si128(masked, _mm_srli_si128(masked, 2));
  return (uint8_t)_mm_extract_epi16(masked, 0);
}

CT_ALWAYS_FORCEINLINE __m128i mask16_to_epi8_simd(uint16_t mask) {
  static const __m128i bits = _mm_set_epi16(0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);
  __m128i vh = _mm_set1_epi16((uint16_t)(mask >> 8));
  __m128i resulth = _mm_cmpeq_epi16(_mm_and_si128(vh, bits), bits);
  resulth = _mm_packus_epi16(resulth, resulth); // [x, ..., x, M15, ..., M8], where Mx = 0xff or 0
  __m128i vl = _mm_set1_epi16((uint16_t)(mask & 0xFF));
  __m128i resultl = _mm_cmpeq_epi16(_mm_and_si128(vl, bits), bits);
  resultl = _mm_packus_epi16(resultl, resultl); // [x, ..., x, M7, ..., M0]
  return _mm_unpacklo_epi64(resultl, resulth); // [M15, ..., M0]
}

CT_ALWAYS_FORCEINLINE __m128i mask4_to_epi32_simd(uint8_t mask) {
  static const __m128i bits = _mm_set_epi32(0x08, 0x04, 0x02, 0x01);
  __m128i v = _mm_set1_epi32(mask);
  return _mm_cmpeq_epi32(_mm_and_si128(v, bits), bits);
}

#ifdef HAS_BMI2

CT_ALWAYS_FORCEINLINE __m128i mask8_to_epi16_bmi2(uint8_t mask) {
  uint64_t low  = _pdep_u64(mask & 0xF, 0x80008000'80008000ULL);
  uint64_t high = _pdep_u64(mask >> 4, 0x80008000'80008000ULL);
  __m128i v = _mm_set_epi64x(high, low);
  return _mm_srai_epi16(v, 15);
}

CT_ALWAYS_FORCEINLINE uint8_t epi16_to_mask8_bmi2(__m128i vec) {
  uint32_t bits = (uint32_t)_mm_movemask_epi8(vec);
  uint32_t extracted = _pext_u32(bits, 0xAAAA); // 0b1010 ...
  return (uint8_t)extracted;
}

CT_ALWAYS_FORCEINLINE __m128i mask16_to_epi8_bmi2(uint16_t mask) {
  uint64_t low  = _pdep_u64(mask >> 8, 0x01010101'01010101ULL);
  uint64_t high = _pdep_u64(mask & 0xFF, 0x01010101'01010101ULL);
  __m128i v = _mm_set_epi64x(high, low);
  return _mm_sub_epi8(_mm_setzero_si128(), v);
}

CT_ALWAYS_FORCEINLINE __m128i mask4_to_epi32_bmi2(uint8_t mask) {
  uint32_t expanded = _pdep_u32(mask & 0xF, 0x80808080);
  __m128i v = _mm_cvtsi32_si128(expanded);
  v = _mm_unpacklo_epi8(v, v); // [x, ..., x, b3, b3, b2, ..., b0, b0]
  v = _mm_unpacklo_epi16(v, v); // [b3, b3, b3, b3, ..., b0, b0, b0, b0]
  return _mm_srai_epi32(v, 31);
}

#endif // HAS_BMI2

// 8位mask → 8个16位元素 (0x0 or 0xffff)
CT_ALWAYS_FORCEINLINE __m128i mask8_to_epi16(uint8_t mask) {
  return mask8_to_epi16_simd(mask);
}

// 8个16位元素 (MSB) → 8位mask
CT_ALWAYS_FORCEINLINE uint8_t epi16_to_mask8(__m128i vec) {
  #ifdef HAS_BMI2
  return epi16_to_mask8_bmi2(vec);
  #else
  return epi16_to_mask8_simd(vec);
  #endif
}

// 16位mask → 16个8位元素
CT_ALWAYS_FORCEINLINE __m128i mask16_to_epi8(uint16_t mask) {
  #ifdef HAS_BMI2
  return mask16_to_epi8_bmi2(mask);
  #else
  return mask16_to_epi8_simd(mask);
  #endif
}

// 16个8位元素 → 16位mask
CT_ALWAYS_FORCEINLINE uint16_t epi8_to_mask16(__m128i vec) {
  return (uint16_t)_mm_movemask_epi8(vec);
}

// 4位mask → 4个32位元素
CT_ALWAYS_FORCEINLINE __m128i mask4_to_epi32(uint8_t mask) {
  return mask4_to_epi32_simd(mask);
}

// 4个32位元素 → 4位mask
CT_ALWAYS_FORCEINLINE uint8_t epi32_to_mask4(__m128i vec) {
  return (uint8_t)_mm_movemask_ps(_mm_castsi128_ps(vec));
}

} // namespace ct::tl::vec::x86

#endif //CTORCH_X86_MASKSUPPORT_H
