//
// Created by renyz on 2026/3/28.
//

#ifndef CTORCH_X86_BIT_H
#define CTORCH_X86_BIT_H

#include "tl/cpu/Capabilities.h"
#include "tl/cpu/impl/x86_Basic.h"

//@formatter:off
namespace ct::tl::vec::CPU_CAPABILITY {
namespace word {

/* ************************************************************************** */
//                             Bitwise Operations                             //
/* ************************************************************************** */
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_and(V a, V b) {
  return _mm_and_si128(a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_or(V a, V b) {
  return _mm_or_si128(a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_xor(V a, V b) {
  return _mm_xor_si128(a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_andnot(V a, V b) {
  return _mm_andnot_si128(a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_not(V v) {
  #ifdef HAS_AVX512F
  return _mm_ternarylogic_epi32(v.v, v.v, v.v, ~_MM_TERNLOG_A);
  #else
  return word::bit_xor(v, V{_mm_set1_epi32(0xFFFFFFFF)});
  #endif
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_blend(V a, V m, V b) {
  #ifdef HAS_AVX512F
  return _mm_ternarylogic_epi32(a.v, m.v, b.v, (_MM_TERNLOG_B & _MM_TERNLOG_C) | (~_MM_TERNLOG_B & _MM_TERNLOG_A));
  #else
  return word::bit_or(word::bit_and(m, b), word::bit_and(word::bit_not(m), a));
  #endif
}

#if VEC_WIDTH >= 256
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_and(V a, V b) {
  return _mm256_and_si256(a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_or(V a, V b) {
  return _mm256_or_si256(a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_xor(V a, V b) {
  return _mm256_xor_si256(a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_andnot(V a, V b) {
  return _mm256_andnot_si256(a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_not(V v) {
  #ifdef HAS_AVX512F
  return _mm256_ternarylogic_epi32(v.v, v.v, v.v, ~_MM_TERNLOG_A);
  #else
  return word::bit_xor(v, V{_mm256_set1_epi32(0xFFFFFFFF)});
  #endif
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_blend(V a, V m, V b) {
  #ifdef HAS_AVX512F
  return _mm256_ternarylogic_epi32(a.v, m.v, b.v, (_MM_TERNLOG_B & _MM_TERNLOG_C) | (~_MM_TERNLOG_B & _MM_TERNLOG_A));
  #else
  return word::bit_or(word::bit_and(m, b), word::bit_and(word::bit_not(m), a));
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_and(V a, V b) {
  return _mm512_and_si512(a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_or(V a, V b) {
  return _mm512_or_si512(a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_xor(V a, V b) {
  return _mm512_xor_si512(a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_andnot(V a, V b) {
  return _mm512_andnot_si512(a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_not(V v) {
  return _mm512_ternarylogic_epi32(v.v, v.v, v.v, ~_MM_TERNLOG_B);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_blend(V a, V m, V b) {
  return _mm512_ternarylogic_epi32(a.v, m.v, b.v, (_MM_TERNLOG_B & _MM_TERNLOG_C) | (~_MM_TERNLOG_B & _MM_TERNLOG_A));
}
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_and(V a, V b, Mask<T> m) {
  return _mm_mask_and_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_and(V a, V b, Mask<T> m) {
  return _mm_mask_and_epi64(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>> && is_none<TypeOf<T>, int32_t, uint32_t, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_and(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::bit_and(a, b));
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_or(V a, V b, Mask<T> m) {
  return _mm_mask_or_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_or(V a, V b, Mask<T> m) {
  return _mm_mask_or_epi64(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>> && is_none<TypeOf<T>, int32_t, uint32_t, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_or(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::bit_or(a, b));
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_xor(V a, V b, Mask<T> m) {
  return _mm_mask_xor_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_xor(V a, V b, Mask<T> m) {
  return _mm_mask_xor_epi64(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>> && is_none<TypeOf<T>, int32_t, uint32_t, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_xor(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::bit_xor(a, b));
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_andnot(V a, V b, Mask<T> m) {
  return _mm_mask_andnot_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_andnot(V a, V b, Mask<T> m) {
  return _mm_mask_andnot_epi64(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>> && is_none<TypeOf<T>, int32_t, uint32_t, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_andnot(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::bit_andnot(a, b));
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_not(V v, Mask<T> m, V default_v) {
  return _mm_mask_ternarylogic_epi32(default_v.v, m.v, v.v, v.v, ~_MM_TERNLOG_B);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_not(V v, Mask<T> m, V default_v) {
  return _mm_mask_ternarylogic_epi64(default_v.v, m.v, v.v, v.v, ~_MM_TERNLOG_B);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>> && is_none<TypeOf<T>, int32_t, uint32_t, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_not(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, word::bit_not(v));
}


template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_and(V a, V b, Mask<T> m) {
  return _mm256_mask_and_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_and(V a, V b, Mask<T> m) {
  return _mm256_mask_and_epi64(a.v, m.v, a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_or(V a, V b, Mask<T> m) {
  return _mm256_mask_or_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_or(V a, V b, Mask<T> m) {
  return _mm256_mask_or_epi64(a.v, m.v, a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_xor(V a, V b, Mask<T> m) {
  return _mm256_mask_xor_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_xor(V a, V b, Mask<T> m) {
  return _mm256_mask_xor_epi64(a.v, m.v, a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_andnot(V a, V b, Mask<T> m) {
  return _mm256_mask_andnot_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_andnot(V a, V b, Mask<T> m) {
  return _mm256_mask_andnot_epi64(a.v, m.v, a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_not(V v, Mask<T> m, V default_v) {
  return _mm256_mask_ternarylogic_epi32(default_v.v, m.v, v.v, v.v, ~_MM_TERNLOG_B);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_not(V v, Mask<T> m, V default_v) {
  return _mm256_mask_ternarylogic_epi64(default_v.v, m.v, v.v, v.v, ~_MM_TERNLOG_B);
}


template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_and(V a, V b, Mask<T> m) {
  return _mm512_mask_and_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_and(V a, V b, Mask<T> m) {
  return _mm512_mask_and_epi64(a.v, m.v, a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_or(V a, V b, Mask<T> m) {
  return _mm512_mask_or_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_or(V a, V b, Mask<T> m) {
  return _mm512_mask_or_epi64(a.v, m.v, a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_xor(V a, V b, Mask<T> m) {
  return _mm512_mask_xor_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_xor(V a, V b, Mask<T> m) {
  return _mm512_mask_xor_epi64(a.v, m.v, a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_andnot(V a, V b, Mask<T> m) {
  return _mm512_mask_andnot_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_andnot(V a, V b, Mask<T> m) {
  return _mm512_mask_andnot_epi64(a.v, m.v, a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_not(V v, Mask<T> m, V default_v) {
  return _mm512_mask_ternarylogic_epi32(default_v.v, m.v, v.v, v.v, ~_MM_TERNLOG_B);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_not(V v, Mask<T> m, V default_v) {
  return _mm512_mask_ternarylogic_epi64(default_v.v, m.v, v.v, v.v, ~_MM_TERNLOG_B);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_and(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::bit_and(a, b));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_or(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::bit_or(a, b));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_xor(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::bit_xor(a, b));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_andnot(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::bit_andnot(a, b));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_not(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, word::bit_not(v));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                              Bit shift left                                //
/* ************************************************************************** */
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  static const __m128i mask = _mm_set1_epi8(~tailing_mask(int32_t(Shift)));
  auto u = _mm_slli_epi16(v.v, Shift);
  return _mm_and_si128(u, mask);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm_slli_epi16(v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm_slli_epi32(v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm_slli_epi64(v.v, Shift);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  auto mask = _mm_set1_epi8(~tailing_mask(int32_t(shift)));
  auto u = _mm_sll_epi16(v.v, _mm_cvtsi32_si128(shift));
  return _mm_and_si128(u, mask);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm_sll_epi16(v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm_sll_epi32(v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm_sll_epi64(v.v, _mm_cvtsi32_si128(shift));
}


#if VEC_WIDTH >= 256
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  static const __m256i mask = _mm256_set1_epi8(~tailing_mask(int32_t(Shift)));
  auto u = _mm256_slli_epi16(v.v, Shift);
  return _mm256_and_si256(u, mask);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm256_slli_epi16(v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm256_slli_epi32(v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm256_slli_epi64(v.v, Shift);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  auto mask = _mm256_set1_epi8(~tailing_mask(int32_t(shift)));
  auto u = _mm256_sll_epi16(v.v, _mm_cvtsi32_si128(shift));
  return _mm256_and_si256(u, mask);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm256_sll_epi16(v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm256_sll_epi32(v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm256_sll_epi64(v.v, _mm_cvtsi32_si128(shift));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  static const __m512i mask = _mm512_set1_epi8(~tailing_mask(int32_t(Shift)));
  auto u = _mm512_slli_epi16(v.v, Shift);
  return _mm512_and_si512(u, mask);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm512_slli_epi16(v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm512_slli_epi32(v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm512_slli_epi64(v.v, Shift);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  auto mask = _mm512_set1_epi8(~tailing_mask(int32_t(shift)));
  auto u = _mm512_sll_epi16(v.v, _mm_cvtsi32_si128(shift));
  return _mm512_and_si512(u, mask);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm512_sll_epi16(v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm512_sll_epi32(v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm512_sll_epi64(v.v, _mm_cvtsi32_si128(shift));
}
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, int8_t, uint8_t, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, Mask<T> m) {
  return word::blend(v, m, word::bit_shl<Shift>(v));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, int8_t, uint8_t, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, int shift, Mask<T> m) {
  return word::blend(v, m, word::bit_shl(v, shift));
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, Mask<T> m) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm_mask_slli_epi32(v.v, m.v, v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, Mask<T> m) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm_mask_slli_epi64(v.v, m.v, v.v, Shift);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, int shift, Mask<T> m) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm_mask_sll_epi32(v.v, m.v, v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, int shift, Mask<T> m) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm_mask_sll_epi64(v.v, m.v, v.v, _mm_cvtsi32_si128(shift));
}

template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, Mask<T> m) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm256_mask_slli_epi32(v.v, m.v, v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, Mask<T> m) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm256_mask_slli_epi64(v.v, m.v, v.v, Shift);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, int shift, Mask<T> m) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm256_mask_sll_epi32(v.v, m.v, v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, int shift, Mask<T> m) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm256_mask_sll_epi64(v.v, m.v, v.v, _mm_cvtsi32_si128(shift));
}

template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, Mask<T> m) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm512_mask_slli_epi32(v.v, m.v, v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, Mask<T> m) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm512_mask_slli_epi64(v.v, m.v, v.v, Shift);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, int shift, Mask<T> m) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm512_mask_sll_epi32(v.v, m.v, v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, int shift, Mask<T> m) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm512_mask_sll_epi64(v.v, m.v, v.v, _mm_cvtsi32_si128(shift));
}
#else // HAS_AVX512DQ
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, Mask<T> m) {
  return word::blend(v, m, word::bit_shl<Shift>(v));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_shl(V v, int shift, Mask<T> m) {
  return word::blend(v, m, word::bit_shl(v, shift));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                              Bit shift right                               //
/* ************************************************************************** */
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  static const __m128i mask = _mm_set1_epi8(tailing_mask(int32_t(8 - Shift)));
  auto u = _mm_srli_epi16(v.v, Shift);
  return _mm_and_si128(u, mask);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  static const __m128i mask = _mm_set1_epi8(tailing_mask(int32_t(8 - Shift)));
  auto signmask = _mm_cmpgt_epi8(_mm_setzero_si128(), v.v);
  auto u = _mm_srli_epi16(v.v, Shift);
  #ifdef HAS_AVX512DQ
  // (mask & u) | (~mask & signmask), bitwise MUX
  u = _mm_ternarylogic_epi32(mask, u, signmask, (_MM_TERNLOG_A & _MM_TERNLOG_B) | (~_MM_TERNLOG_A & _MM_TERNLOG_C));
  #else
  static const __m128i inv_mask = _mm_set1_epi8(~tailing_mask(int32_t(8 - Shift)));
  signmask = _mm_and_si128(signmask, inv_mask);
  u = _mm_and_si128(u, mask);
  u = _mm_or_si128(u, signmask);
  #endif
  return u;
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm_srli_epi16(v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm_srai_epi16(v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm_srli_epi32(v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm_srai_epi32(v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm_srli_epi64(v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  #ifdef HAS_AVX512DQ
  return _mm_srai_epi64(v.v, Shift);
  #else
  ViewAs<int32_t, T> t32; ViewAs<uint64_t, T> t64;
  auto signs = word::bit_shr<31>(word::bitcast(t32, v));
  auto sign_hi = word::local_shuf<3, 3, 1, 1>(signs);
  static const __m128i mask = _mm_set1_epi64x(~tailing_mask(int64_t(64 - Shift)));  // 0b1...10...0
  auto u = word::bit_shr<Shift>(word::bitcast(t64, v)); // unsigned shift
  auto masked_sign = _mm_and_si128(sign_hi.v, mask);
  return _mm_or_si128(masked_sign, u.v);
  #endif
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  auto mask = _mm_set1_epi8(tailing_mask(int32_t(8 - shift)));
  auto u = _mm_srl_epi16(v.v, _mm_cvtsi32_si128(shift));
  return _mm_and_si128(u, mask);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  auto mask = _mm_set1_epi8(tailing_mask(int32_t(8 - shift)));
  auto signmask = _mm_cmpgt_epi8(_mm_setzero_si128(), v.v);
  auto u = _mm_srl_epi16(v.v, _mm_cvtsi32_si128(shift));
  #ifdef HAS_AVX512DQ
  // (mask & u) | (~mask & signmask), bitwise MUX
  u = _mm_ternarylogic_epi32(mask, u, signmask, (_MM_TERNLOG_A & _MM_TERNLOG_B) | (~_MM_TERNLOG_A & _MM_TERNLOG_C));
  #else
  auto inv_mask = _mm_set1_epi8(~tailing_mask(int32_t(8 - shift)));
  signmask = _mm_and_si128(signmask, inv_mask);
  u = _mm_and_si128(u, mask);
  u = _mm_or_si128(u, signmask);
  #endif
  return u;
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm_srl_epi16(v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm_sra_epi16(v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm_srl_epi32(v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm_sra_epi32(v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm_srl_epi64(v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  #ifdef HAS_AVX512DQ
  return _mm_sra_epi64(v.v, _mm_cvtsi32_si128(shift));
  #else
  ViewAs<int32_t, T> t32; ViewAs<uint64_t, T> t64;
  auto signs = word::bit_shr<31>(word::bitcast(t32, v));
  auto sign_hi = word::local_shuf<3, 3, 1, 1>(signs);
  auto mask = word::fill(t64, ~tailing_mask(int64_t(64 - shift)));  // 0b1...10...0
  auto u = word::bit_shr(word::bitcast(t64, v), shift); // unsigned shift
  auto masked_sign = _mm_and_si128(sign_hi.v, mask.v);
  return _mm_or_si128(masked_sign, u.v);
  #endif
}

#if VEC_WIDTH >= 256
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  static const __m256i mask = _mm256_set1_epi8(tailing_mask(int32_t(8 - Shift)));
  auto u = _mm256_srli_epi16(v.v, Shift);
  return _mm256_and_si256(u, mask);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  static const __m256i mask = _mm256_set1_epi8(tailing_mask(int32_t(8 - Shift)));
  auto signmask = _mm256_cmpgt_epi8(_mm256_setzero_si256(), v.v);
  auto u = _mm256_srli_epi16(v.v, Shift);
  #ifdef HAS_AVX512DQ
  // (mask & u) | (~mask & signmask), bitwise MUX
  u = _mm256_ternarylogic_epi32(mask, u, signmask, (_MM_TERNLOG_A & _MM_TERNLOG_B) | (~_MM_TERNLOG_A & _MM_TERNLOG_C));
  #else
  static const __m256i inv_mask = _mm256_set1_epi8(~tailing_mask(int32_t(8 - Shift)));
  signmask = _mm256_and_si256(signmask, inv_mask);
  u = _mm256_and_si256(u, mask);
  u = _mm256_or_si256(u, signmask);
  #endif
  return u;
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm256_srli_epi16(v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm256_srai_epi16(v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm256_srli_epi32(v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm256_srai_epi32(v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm256_srli_epi64(v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  #ifdef HAS_AVX512DQ
  return _mm256_srai_epi64(v.v, Shift);
  #else
  ViewAs<int32_t, T> t32; ViewAs<uint64_t, T> t64;
  auto signs = word::bit_shr<31>(word::bitcast(t32, v));
  auto sign_hi = word::local_shuf<3, 3, 1, 1>(signs);
  static const __m256i mask = _mm256_set1_epi64x(~tailing_mask(int64_t(64 - Shift)));  // 0b1...10...0
  auto u = word::bit_shr<Shift>(word::bitcast(t64, v)); // unsigned shift
  auto masked_sign = _mm256_and_si256(sign_hi.v, mask);
  return _mm256_or_si256(masked_sign, u.v);
  #endif
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  auto mask = _mm256_set1_epi8(tailing_mask(int32_t(8 - shift)));
  auto u = _mm256_srl_epi16(v.v, _mm_cvtsi32_si128(shift));
  return _mm256_and_si256(u, mask);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  auto mask = _mm256_set1_epi8(tailing_mask(int32_t(8 - shift)));
  auto signmask = _mm256_cmpgt_epi8(_mm256_setzero_si256(), v.v);
  auto u = _mm256_srl_epi16(v.v, _mm_cvtsi32_si128(shift));
  #ifdef HAS_AVX512DQ
  // (mask & u) | (~mask & signmask), bitwise MUX
  u = _mm256_ternarylogic_epi32(mask, u, signmask, (_MM_TERNLOG_A & _MM_TERNLOG_B) | (~_MM_TERNLOG_A & _MM_TERNLOG_C));
  #else
  auto inv_mask = _mm256_set1_epi8(~tailing_mask(int32_t(8 - shift)));
  signmask = _mm256_and_si256(signmask, inv_mask);
  u = _mm256_and_si256(u, mask);
  u = _mm256_or_si256(u, signmask);
  #endif
  return u;
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm256_srl_epi16(v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm256_sra_epi16(v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm256_srl_epi32(v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm256_sra_epi32(v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm256_srl_epi64(v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  #ifdef HAS_AVX512DQ
  return _mm256_sra_epi64(v.v, _mm_cvtsi32_si128(shift));
  #else
  ViewAs<int32_t, T> t32; ViewAs<uint64_t, T> t64;
  auto signs = word::bit_shr<31>(word::bitcast(t32, v));
  static_assert(sizeof(word::bitcast(t32, v)) == 32);
  auto sign_hi = word::local_shuf<3, 3, 1, 1>(signs);
  auto mask = word::fill(t64, ~tailing_mask(int64_t(64 - shift)));  // 0b1...10...0
  auto u = word::bit_shr(word::bitcast(t64, v), shift); // unsigned shift
  auto masked_sign = _mm256_and_si256(sign_hi.v, mask.v);
  return _mm256_or_si256(masked_sign, u.v);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  static const __m512i mask = _mm512_set1_epi8(tailing_mask(int32_t(8 - Shift)));
  auto u = _mm512_srli_epi16(v.v, Shift);
  return _mm512_and_si512(u, mask);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  static const __m512i mask = _mm512_set1_epi8(tailing_mask(int32_t(8 - Shift)));
  auto signmask = _mm512_cmpgt_epi8(_mm512_setzero_si512(), v.v);
  auto u = _mm512_srli_epi16(v.v, Shift);
  // (mask & u) | (~mask & signmask), bitwise MUX
  u = _mm512_ternarylogic_epi32(mask, u, signmask, (_MM_TERNLOG_A & _MM_TERNLOG_B) | (~_MM_TERNLOG_A & _MM_TERNLOG_C));
  return u;
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm512_srli_epi16(v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm512_srai_epi16(v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm512_srli_epi32(v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm512_srai_epi32(v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm512_srli_epi64(v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm512_srai_epi64(v.v, Shift);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  auto mask = _mm512_set1_epi8(tailing_mask(int32_t(8 - shift)));
  auto u = _mm512_srl_epi16(v.v, _mm_cvtsi32_si128(shift));
  return _mm512_and_si512(u, mask);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  auto mask = _mm512_set1_epi8(tailing_mask(int32_t(8 - shift)));
  auto signmask = _mm512_cmpgt_epi8(_mm512_setzero_si512(), v.v);
  auto u = _mm512_srl_epi16(v.v, _mm_cvtsi32_si128(shift));
  // (mask & u) | (~mask & signmask), bitwise MUX
  u = _mm512_ternarylogic_epi32(mask, u, signmask, (_MM_TERNLOG_A & _MM_TERNLOG_B) | (~_MM_TERNLOG_A & _MM_TERNLOG_C));
  return u;
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm512_srl_epi16(v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm512_sra_epi16(v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm512_srl_epi32(v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm512_sra_epi32(v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm512_srl_epi64(v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm512_sra_epi64(v.v, _mm_cvtsi32_si128(shift));
}
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, int8_t, uint8_t, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, Mask<T> m) {
  return word::blend(v, m, word::bit_shr<Shift>(v));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, int8_t, uint8_t, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift, Mask<T> m) {
  return word::blend(v, m, word::bit_shr(v, shift));
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, Mask<T> m) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm_mask_srai_epi32(v.v, m.v, v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, Mask<T> m) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm_mask_srli_epi32(v.v, m.v, v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, Mask<T> m) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm_mask_srai_epi64(v.v, m.v, v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, Mask<T> m) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm_mask_srli_epi64(v.v, m.v, v.v, Shift);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift, Mask<T> m) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm_mask_sra_epi32(v.v, m.v, v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift, Mask<T> m) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm_mask_srl_epi32(v.v, m.v, v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift, Mask<T> m) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm_mask_sra_epi64(v.v, m.v, v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift, Mask<T> m) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm_mask_srl_epi64(v.v, m.v, v.v, _mm_cvtsi32_si128(shift));
}

template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, Mask<T> m) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm256_mask_srai_epi32(v.v, m.v, v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, Mask<T> m) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm256_mask_srli_epi32(v.v, m.v, v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, Mask<T> m) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm256_mask_srai_epi64(v.v, m.v, v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, Mask<T> m) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm256_mask_srli_epi64(v.v, m.v, v.v, Shift);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift, Mask<T> m) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm256_mask_sra_epi32(v.v, m.v, v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift, Mask<T> m) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm256_mask_srl_epi32(v.v, m.v, v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift, Mask<T> m) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm256_mask_sra_epi64(v.v, m.v, v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift, Mask<T> m) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm256_mask_srl_epi64(v.v, m.v, v.v, _mm_cvtsi32_si128(shift));
}

template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, Mask<T> m) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm512_mask_srai_epi32(v.v, m.v, v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, Mask<T> m) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm512_mask_srli_epi32(v.v, m.v, v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, Mask<T> m) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm512_mask_srai_epi64(v.v, m.v, v.v, Shift);
}
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, Mask<T> m) {
  static_assert(0 <= Shift && Shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range");
  return _mm512_mask_srli_epi64(v.v, m.v, v.v, Shift);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift, Mask<T> m) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm512_mask_sra_epi32(v.v, m.v, v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift, Mask<T> m) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm512_mask_srl_epi32(v.v, m.v, v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift, Mask<T> m) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm512_mask_sra_epi64(v.v, m.v, v.v, _mm_cvtsi32_si128(shift));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift, Mask<T> m) {
  CT_ASSERT(0 <= shift && shift <= sizeof(TypeOf<T>) * CHAR_BIT, "Shift out of range: %d", shift);
  return _mm512_mask_srl_epi64(v.v, m.v, v.v, _mm_cvtsi32_si128(shift));
}
#else // HAS_AVX512DQ
template <int Shift, typename V, typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, Mask<T> m) {
  return word::blend(v, m, word::bit_shr<Shift>(v));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V bit_shr(V v, int shift, Mask<T> m) {
  return word::blend(v, m, word::bit_shr(v, shift));
}
#endif // HAS_AVX512DQ

} // namespace word
} // namespace ct::tl::vec::CPU_CAPABILITY
//@formatter:on

#endif //CTORCH_X86_BIT_H
