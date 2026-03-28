//
// Created by renyz on 2026/3/28.
//

#ifndef CTORCH_X86_ARITHMETIC_H
#define CTORCH_X86_ARITHMETIC_H

#include <cmath>

#include "tl/cpu/impl/x86_Basic.h"
#include "tl/cpu/impl/x86_Bit.h"

//@formatter:off
namespace ct::tl::vec::CPU_CAPABILITY {
namespace word {

/* ************************************************************************** */
//                                   Add                                      //
/* ************************************************************************** */
// TODO arithmetics for float16_t and bfloat16_t
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b) {
  return _mm_add_ps(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b) {
  return _mm_add_pd(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b) {
  return _mm_add_epi8(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b) {
  return _mm_add_epi16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b) {
  return _mm_add_epi32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b) {
  return _mm_add_epi64(a.v, b.v);
}

#if VEC_WIDTH >= 256
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b) {
  return _mm256_add_ps(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b) {
  return _mm256_add_pd(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b) {
  return _mm256_add_epi8(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b) {
  return _mm256_add_epi16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b) {
  return _mm256_add_epi32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b) {
  return _mm256_add_epi64(a.v, b.v);
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b) {
  return _mm512_add_ps(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b) {
  return _mm512_add_pd(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b) {
  return _mm512_add_epi8(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b) {
  return _mm512_add_epi16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b) {
  return _mm512_add_epi32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b) {
  return _mm512_add_epi64(a.v, b.v);
}
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b, Mask<T> m) {
  return _mm_mask_add_ps(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b, Mask<T> m) {
  return _mm_mask_add_pd(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b, Mask<T> m) {
  return _mm_mask_add_epi8(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b, Mask<T> m) {
  return _mm_mask_add_epi16(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b, Mask<T> m) {
  return _mm_mask_add_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b, Mask<T> m) {
  return _mm_mask_add_epi64(a.v, m.v, a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b, Mask<T> m) {
  return _mm256_mask_add_ps(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b, Mask<T> m) {
  return _mm256_mask_add_pd(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b, Mask<T> m) {
  return _mm256_mask_add_epi8(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b, Mask<T> m) {
  return _mm256_mask_add_epi16(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b, Mask<T> m) {
  return _mm256_mask_add_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b, Mask<T> m) {
  return _mm256_mask_add_epi64(a.v, m.v, a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b, Mask<T> m) {
  return _mm512_mask_add_ps(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b, Mask<T> m) {
  return _mm512_mask_add_pd(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b, Mask<T> m) {
  return _mm512_mask_add_epi8(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b, Mask<T> m) {
  return _mm512_mask_add_epi16(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b, Mask<T> m) {
  return _mm512_mask_add_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V add(V a, V b, Mask<T> m) {
  return _mm512_mask_add_epi64(a.v, m.v, a.v, b.v);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>>
CT_ALWAYS_FORCEINLINE V add(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::add(a, b));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                                   Sub                                      //
/* ************************************************************************** */
// TODO arithmetics for float16_t and bfloat16_t
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b) {
  return _mm_sub_ps(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b) {
  return _mm_sub_pd(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b) {
  return _mm_sub_epi8(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b) {
  return _mm_sub_epi16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b) {
  return _mm_sub_epi32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b) {
  return _mm_sub_epi64(a.v, b.v);
}

#if VEC_WIDTH >= 256
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b) {
  return _mm256_sub_ps(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b) {
  return _mm256_sub_pd(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b) {
  return _mm256_sub_epi8(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b) {
  return _mm256_sub_epi16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b) {
  return _mm256_sub_epi32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b) {
  return _mm256_sub_epi64(a.v, b.v);
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b) {
  return _mm512_sub_ps(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b) {
  return _mm512_sub_pd(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b) {
  return _mm512_sub_epi8(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b) {
  return _mm512_sub_epi16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b) {
  return _mm512_sub_epi32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b) {
  return _mm512_sub_epi64(a.v, b.v);
}
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b, Mask<T> m) {
  return _mm_mask_sub_ps(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b, Mask<T> m) {
  return _mm_mask_sub_pd(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b, Mask<T> m) {
  return _mm_mask_sub_epi8(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b, Mask<T> m) {
  return _mm_mask_sub_epi16(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b, Mask<T> m) {
  return _mm_mask_sub_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b, Mask<T> m) {
  return _mm_mask_sub_epi64(a.v, m.v, a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b, Mask<T> m) {
  return _mm256_mask_sub_ps(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b, Mask<T> m) {
  return _mm256_mask_sub_pd(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b, Mask<T> m) {
  return _mm256_mask_sub_epi8(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b, Mask<T> m) {
  return _mm256_mask_sub_epi16(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b, Mask<T> m) {
  return _mm256_mask_sub_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b, Mask<T> m) {
  return _mm256_mask_sub_epi64(a.v, m.v, a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b, Mask<T> m) {
  return _mm512_mask_sub_ps(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b, Mask<T> m) {
  return _mm512_mask_sub_pd(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b, Mask<T> m) {
  return _mm512_mask_sub_epi8(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b, Mask<T> m) {
  return _mm512_mask_sub_epi16(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b, Mask<T> m) {
  return _mm512_mask_sub_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V sub(V a, V b, Mask<T> m) {
  return _mm512_mask_sub_epi64(a.v, m.v, a.v, b.v);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>>
CT_ALWAYS_FORCEINLINE V sub(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::sub(a, b));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                                   Mul                                      //
/* ************************************************************************** */
// TODO arithmetics for float16_t and bfloat16_t
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b) {
  return _mm_mul_ps(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b) {
  return _mm_mul_pd(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b) {
  auto even = _mm_mullo_epi16(a.v, b.v);
  auto odd = _mm_mullo_epi16(_mm_srli_epi16(a.v, 8), _mm_srli_epi16(b.v, 8));
  return _mm_or_si128(_mm_slli_epi16(odd, 8), _mm_and_si128(even, _mm_set1_epi16(0xFF)));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b) {
  return _mm_mullo_epi16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b) {
  return _mm_mullo_epi32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b) {
  #ifdef HAS_AVX512DQ
  return _mm_mullo_epi64(a.v, b.v);
  #else
  ViewAs<int32_t, T> t32;
  auto lo_lo = _mm_mul_epu32(a.v, b.v);
  auto a_hi = word::local_shuf<3, 3, 1, 1>(word::bitcast(t32, a));
  auto b_hi = word::local_shuf<3, 3, 1, 1>(word::bitcast(t32, b));
  auto hi_lo = _mm_mul_epu32(a_hi.v, b.v); // a_hi × b_lo
  auto lo_hi = _mm_mul_epu32(a.v, b_hi.v); // a_lo × b_hi
  auto cross = _mm_add_epi64(hi_lo, lo_hi);
  cross = _mm_slli_epi64(cross, 32);
  return _mm_add_epi64(lo_lo, cross);
  #endif
}

#if VEC_WIDTH >= 256
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b) {
  return _mm256_mul_ps(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b) {
  return _mm256_mul_pd(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b) {
  auto even = _mm256_mullo_epi16(a.v, b.v);
  auto odd = _mm256_mullo_epi16(_mm256_srli_epi16(a.v, 8), _mm256_srli_epi16(b.v, 8));
  return _mm256_or_si256(_mm256_slli_epi16(odd, 8), _mm256_and_si256(even, _mm256_set1_epi16(0xFF)));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b) {
  return _mm256_mullo_epi16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b) {
  return _mm256_mullo_epi32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b) {
  #ifdef HAS_AVX512DQ
  return _mm256_mullo_epi64(a.v, b.v);
  #else
  auto lo_lo = _mm256_mul_epu32(a.v, b.v);
  auto a_hi = word::local_shuf<3, 3, 1, 1>(a);
  auto b_hi = word::local_shuf<3, 3, 1, 1>(b);
  auto hi_lo = _mm256_mul_epu32(a_hi.v, b.v); // a_hi × b_lo
  auto lo_hi = _mm256_mul_epu32(a.v, b_hi.v); // a_lo × b_hi
  auto cross = _mm256_add_epi64(hi_lo, lo_hi);
  cross = _mm256_slli_epi64(cross, 32);
  return _mm256_add_epi64(lo_lo, cross);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b) {
  return _mm512_mul_ps(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b) {
  return _mm512_mul_pd(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b) {
  auto even = _mm512_mullo_epi16(a.v, b.v);
  auto odd = _mm512_mullo_epi16(_mm512_srli_epi16(a.v, 8), _mm512_srli_epi16(b.v, 8));
  return _mm512_or_si128(_mm512_slli_epi16(odd, 8), _mm512_and_si512(even, _mm512_set1_epi16(0xFF)));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b) {
  return _mm512_mullo_epi16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b) {
  return _mm512_mullo_epi32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b) {
  return _mm512_mullo_epi64(a.v, b.v);
}
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b, Mask<T> m) {
  return _mm_mask_mul_ps(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b, Mask<T> m) {
  return _mm_mask_mul_pd(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::mul(a, b));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b, Mask<T> m) {
  return _mm_mask_mullo_epi16(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b, Mask<T> m) {
  return _mm_mask_mullo_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b, Mask<T> m) {
  return _mm_mask_mullo_epi64(a.v, m.v, a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b, Mask<T> m) {
  return _mm256_mask_mul_ps(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b, Mask<T> m) {
  return _mm256_mask_mul_pd(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b, Mask<T> m) {
  return _mm256_mask_mullo_epi16(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b, Mask<T> m) {
  return _mm256_mask_mullo_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b, Mask<T> m) {
  return _mm256_mask_mullo_epi64(a.v, m.v, a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b, Mask<T> m) {
  return _mm512_mask_mul_ps(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b, Mask<T> m) {
  return _mm512_mask_mul_pd(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b, Mask<T> m) {
  return _mm512_mask_mullo_epi16(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b, Mask<T> m) {
  return _mm512_mask_mullo_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE V mul(V a, V b, Mask<T> m) {
  return _mm512_mask_mullo_epi64(a.v, m.v, a.v, b.v);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>>
CT_ALWAYS_FORCEINLINE V mul(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::mul(a, b));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                                   Div                                      //
/* ************************************************************************** */
// TODO arithmetics for float16_t and bfloat16_t
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V div(V a, V b) {
  return _mm_div_ps(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V div(V a, V b) {
  return _mm_div_pd(a.v, b.v);
}

#if VEC_WIDTH >= 256
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V div(V a, V b) {
  return _mm256_div_ps(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V div(V a, V b) {
  return _mm256_div_pd(a.v, b.v);
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V div(V a, V b) {
  return _mm512_div_ps(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V div(V a, V b) {
  return _mm512_div_pd(a.v, b.v);
}
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V div(V a, V b, Mask<T> m) {
  return _mm_mask_div_ps(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V div(V a, V b, Mask<T> m) {
  return _mm_mask_div_pd(a.v, m.v, a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V div(V a, V b, Mask<T> m) {
  return _mm256_mask_div_ps(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V div(V a, V b, Mask<T> m) {
  return _mm256_mask_div_pd(a.v, m.v, a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V div(V a, V b, Mask<T> m) {
  return _mm512_mask_div_ps(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V div(V a, V b, Mask<T> m) {
  return _mm512_mask_div_pd(a.v, m.v, a.v, b.v);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>>
CT_ALWAYS_FORCEINLINE V div(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::div(a, b));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                                   Min                                      //
/* ************************************************************************** */
// TODO arithmetics for float16_t and bfloat16_t
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm_min_ps(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm_min_pd(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm_min_epi8(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm_min_epu8(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm_min_epi16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm_min_epu16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm_min_epi32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm_min_epu32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  #ifdef HAS_AVX512F
  return _mm_min_epi64(a.v, b.v);
  #else
  return _mm_blendv_epi8(b.v, a.v, _mm_cmpgt_epi64(b.v, a.v));
  #endif
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  #ifdef HAS_AVX512F
  return _mm_min_epu64(a.v, b.v);
  #else
  static const __m128i flip = _mm_set1_epi64x((int64_t)0x8000000000000000LL);
  auto a_flip = _mm_xor_si128(a.v, flip);
  auto b_flip = _mm_xor_si128(b.v, flip);
  return _mm_blendv_epi8(b.v, a.v, _mm_cmpgt_epi64(b_flip, a_flip));
  #endif
}

#if VEC_WIDTH >= 256
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm256_min_ps(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm256_min_pd(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm256_min_epi8(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm256_min_epu8(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm256_min_epi16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm256_min_epu16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm256_min_epi32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm256_min_epu32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  #ifdef HAS_AVX512F
  return _mm256_min_epi64(a.v, b.v);
  #else
  return _mm256_blendv_epi8(b.v, a.v, _mm256_cmpgt_epi64(b.v, a.v));
  #endif
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  #ifdef HAS_AVX512F
  return _mm256_min_epu64(a.v, b.v);
  #else
  static const __m256i flip = _mm256_set1_epi64x((int64_t)0x8000000000000000LL);
  auto a_flip = _mm_xor_si256(a.v, flip);
  auto b_flip = _mm_xor_si256(b.v, flip);
  return _mm256_blendv_epi8(b.v, a.v, _mm256_cmpgt_epi64(b_flip, a_flip));
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm512_min_ps(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm512_min_pd(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm512_min_epi8(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm512_min_epu8(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm512_min_epi16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm512_min_epu16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm512_min_epi32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm512_min_epu32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm512_min_epi64(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b) {
  return _mm512_min_epu64(a.v, b.v);
}
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm_mask_min_ps(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm_mask_min_pd(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm_mask_min_epi8(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm_mask_min_epu8(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm_mask_min_epi16(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm_mask_min_epu16(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm_mask_min_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm_mask_min_epu32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm_mask_min_epi64(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm_mask_min_epu64(a.v, m.v, a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm256_mask_min_ps(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm256_mask_min_pd(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm256_mask_min_epi8(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm256_mask_min_epu8(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm256_mask_min_epi16(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm256_mask_min_epu16(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm256_mask_min_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm256_mask_min_epu32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm256_mask_min_epi64(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm256_mask_min_epu64(a.v, m.v, a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm512_mask_min_ps(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm512_mask_min_pd(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm512_mask_min_epi8(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm512_mask_min_epu8(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm512_mask_min_epi16(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm512_mask_min_epu16(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm512_mask_min_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm512_mask_min_epu32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm512_mask_min_epi64(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return _mm512_mask_min_epu64(a.v, m.v, a.v, b.v);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>>
CT_ALWAYS_FORCEINLINE V min(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::min(a, b));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                                   Max                                      //
/* ************************************************************************** */
// TODO arithmetics for float16_t and bfloat16_t
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm_max_ps(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm_max_pd(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm_max_epi8(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm_max_epu8(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm_max_epi16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm_max_epu16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm_max_epi32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm_max_epu32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  #ifdef HAS_AVX512F
  return _mm_max_epi64(a.v, b.v);
  #else
  return _mm_blendv_epi8(b.v, a.v, _mm_cmpgt_epi64(a.v, b.v));
  #endif
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  #ifdef HAS_AVX512F
  return _mm_max_epu64(a.v, b.v);
  #else
  static const __m128i sign_bit = _mm_set1_epi64x((int64_t)0x8000000000000000LL);
  auto a_flip = _mm_xor_si128(a.v, sign_bit);
  auto b_flip = _mm_xor_si128(b.v, sign_bit);
  return _mm_blendv_epi8(b.v, a.v, _mm_cmpgt_epi64(a_flip, b_flip));
  #endif
}

#if VEC_WIDTH >= 256
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm256_max_ps(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm256_max_pd(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm256_max_epi8(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm256_max_epu8(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm256_max_epi16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm256_max_epu16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm256_max_epi32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm256_max_epu32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  #ifdef HAS_AVX512F
  return _mm256_max_epi64(a.v, b.v);
  #else
  return _mm256_blendv_epi8(b.v, a.v, _mm256_cmpgt_epi64(a.v, b.v));
  #endif
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  #ifdef HAS_AVX512F
  return _mm256_max_epu64(a.v, b.v);
  #else
  static const __m256i flip = _mm256_set1_epi64x((int64_t)0x8000000000000000LL);
  auto a_flip = _mm_xor_si256(a.v, flip);
  auto b_flip = _mm_xor_si256(b.v, flip);
  return _mm256_blendv_epi8(b.v, a.v, _mm256_cmpgt_epi64(a_flip, b_flip));
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm512_max_ps(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm512_max_pd(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm512_max_epi8(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm512_max_epu8(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm512_max_epi16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm512_max_epu16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm512_max_epi32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm512_max_epu32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm512_max_epi64(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b) {
  return _mm512_max_epu64(a.v, b.v);
}
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm_mask_max_ps(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm_mask_max_pd(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm_mask_max_epi8(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm_mask_max_epu8(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm_mask_max_epi16(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm_mask_max_epu16(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm_mask_max_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm_mask_max_epu32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm_mask_max_epi64(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm_mask_max_epu64(a.v, m.v, a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm256_mask_max_ps(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm256_mask_max_pd(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm256_mask_max_epi8(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm256_mask_max_epu8(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm256_mask_max_epi16(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm256_mask_max_epu16(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm256_mask_max_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm256_mask_max_epu32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm256_mask_max_epi64(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm256_mask_max_epu64(a.v, m.v, a.v, b.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm512_mask_max_ps(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm512_mask_max_pd(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm512_mask_max_epi8(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm512_mask_max_epu8(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm512_mask_max_epi16(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm512_mask_max_epu16(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm512_mask_max_epi32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm512_mask_max_epu32(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm512_mask_max_epi64(a.v, m.v, a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return _mm512_mask_max_epu64(a.v, m.v, a.v, b.v);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>>
CT_ALWAYS_FORCEINLINE V max(V a, V b, Mask<T> m) {
  return word::blend(a, m, word::max(a, b));
}
#endif // HAS_AVX512DQ

/* ************************************************************************** */
//                             Rcp / Reciprocal                               //
/* ************************************************************************** */
// TODO arithmetics for float16_t and bfloat16_t
// Note: AVX512 uses rcp14 which gives higher accuracy than rcp used not in AVX512
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V rcp(V v) {
  #ifdef HAS_AVX512F
  return _mm_rcp14_ps(v.v);
  #else
  return _mm_rcp_ps(v.v);
  #endif
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V rcp(V v) {
  #ifdef HAS_AVX512F
  return _mm_rcp14_pd(v.v);
  #else
  return word::div(fill(T(), 1), v);
  #endif
}

#if VEC_WIDTH >= 256
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V rcp(V v) {
  #ifdef HAS_AVX512F
  return _mm256_rcp14_ps(v.v);
  #else
  return _mm256_rcp_ps(v.v);
  #endif
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V rcp(V v) {
  #ifdef HAS_AVX512F
  return _mm256_rcp14_pd(v.v);
  #else
  return word::div(fill(T(), 1), v);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V rcp(V v) {
  return _mm512_rcp14_ps(v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V rcp(V v) {
  return _mm512_rcp14_pd(v.v);
}
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V rcp(V v, Mask<T> m, V default_v) {
  return _mm_mask_rcp14_ps(default_v.v, m.v, v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V rcp(V v, Mask<T> m, V default_v) {
  return _mm_mask_rcp14_pd(default_v.v, m.v, v.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V rcp(V v, Mask<T> m, V default_v) {
  return _mm256_mask_rcp14_ps(default_v.v, m.v, v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V rcp(V v, Mask<T> m, V default_v) {
  return _mm256_mask_rcp14_pd(default_v.v, m.v, v.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V rcp(V v, Mask<T> m, V default_v) {
  return _mm512_mask_rcp14_ps(default_v.v, m.v, v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V rcp(V v, Mask<T> m, V default_v) {
  return _mm512_mask_rcp14_pd(default_v.v, m.v, v.v);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>>
CT_ALWAYS_FORCEINLINE V rcp(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, word::rcp(v));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                                   Sqrt                                     //
/* ************************************************************************** */
// TODO arithmetics for float16_t and bfloat16_t
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V sqrt(V v) {
  return _mm_sqrt_ps(v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V sqrt(V v) {
  return _mm_sqrt_pd(v.v);
}

#if VEC_WIDTH >= 256
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V sqrt(V v) {
  return _mm256_sqrt_ps(v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V sqrt(V v) {
  return _mm256_sqrt_pd(v.v);
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V sqrt(V v) {
  return _mm512_sqrt_ps(v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V sqrt(V v) {
  return _mm512_sqrt_pd(v.v);
}
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V sqrt(V v, Mask<T> m, V default_v) {
  return _mm_mask_sqrt_ps(default_v.v, m.v, v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V sqrt(V v, Mask<T> m, V default_v) {
  return _mm_mask_sqrt_pd(default_v.v, m.v, v.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V sqrt(V v, Mask<T> m, V default_v) {
  return _mm256_mask_sqrt_ps(default_v.v, m.v, v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V sqrt(V v, Mask<T> m, V default_v) {
  return _mm256_mask_sqrt_pd(default_v.v, m.v, v.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V sqrt(V v, Mask<T> m, V default_v) {
  return _mm512_mask_sqrt_ps(default_v.v, m.v, v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V sqrt(V v, Mask<T> m, V default_v) {
  return _mm512_mask_sqrt_pd(default_v.v, m.v, v.v);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>>
CT_ALWAYS_FORCEINLINE V sqrt(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, word::sqrt(v));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                        Rsqrt / Reciprocal of Sqrt                          //
/* ************************************************************************** */
// TODO arithmetics for float16_t and bfloat16_t
// Note: AVX512 uses rsqrt14 which gives higher accuracy than rsqrt used not in AVX512
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V rsqrt(V v) {
  #ifdef HAS_AVX512F
  return _mm_rsqrt14_ps(v.v);
  #else
  return _mm_rsqrt_ps(v.v);
  #endif
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V rsqrt(V v) {
  #ifdef HAS_AVX512F
  return _mm_rsqrt14_pd(v.v);
  #else
  return word::div(fill(T(), 1), word::sqrt(v));
  #endif
}

#if VEC_WIDTH >= 256
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V rsqrt(V v) {
  #ifdef HAS_AVX512F
  return _mm256_rsqrt14_ps(v.v);
  #else
  return _mm256_rsqrt_ps(v.v);
  #endif
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V rsqrt(V v) {
  #ifdef HAS_AVX512F
  return _mm256_rsqrt14_pd(v.v);
  #else
  return word::div(fill(T(), 1), word::sqrt(v));
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V rsqrt(V v) {
  return _mm512_rsqrt14_ps(v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V rsqrt(V v) {
  return _mm512_rsqrt14_pd(v.v);
}
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V rsqrt(V v, Mask<T> m, V default_v) {
  return _mm_mask_rsqrt14_ps(default_v.v, m.v, v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V rsqrt(V v, Mask<T> m, V default_v) {
  return _mm_mask_rsqrt14_pd(default_v.v, m.v, v.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V rsqrt(V v, Mask<T> m, V default_v) {
  return _mm256_mask_rsqrt14_ps(default_v.v, m.v, v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V rsqrt(V v, Mask<T> m, V default_v) {
  return _mm256_mask_rsqrt14_pd(default_v.v, m.v, v.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V rsqrt(V v, Mask<T> m, V default_v) {
  return _mm512_mask_rsqrt14_ps(default_v.v, m.v, v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V rsqrt(V v, Mask<T> m, V default_v) {
  return _mm512_mask_rsqrt14_pd(default_v.v, m.v, v.v);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>>
CT_ALWAYS_FORCEINLINE V rsqrt(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, word::rsqrt(v));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                                  Negate                                    //
/* ************************************************************************** */
// Also defined for unsigned ints
template <typename V, typename T = Vec2Tag<V>>
CT_ALWAYS_FORCEINLINE V neg(V v) {
  return word::sub(word::zeros(T()), v);
}
template <typename V, typename T = Vec2Tag<V>>
CT_ALWAYS_FORCEINLINE V neg(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, word::sub(word::zeros(T()), v));
}


/* ************************************************************************** */
//                              Absolute Value                                //
/* ************************************************************************** */
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v) {
  return _mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)), v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v) {
  return _mm_and_pd(_mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)), v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>> && std::is_unsigned_v<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V abs(V v) {
  return v;
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v) {
  return _mm_abs_epi8(v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v) {
  return _mm_abs_epi16(v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v) {
  return _mm_abs_epi32(v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v) {
  #ifdef HAS_AVX512DQ
  return _mm_abs_epi64(v.v);
  #else
  ViewAs<int32_t, T> t32; T t64;
  auto high32 = word::local_shuf<3, 3, 1, 1>(word::bitcast(t32, v));
  auto sign_mask = word::bit_shr<31>(high32);
  auto xored = word::bit_xor(word::bitcast(t32, v), sign_mask);
  return word::sub(word::bitcast(t64, xored), word::bitcast(t64, sign_mask));
  #endif
}

#if VEC_WIDTH >= 256
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v) {
  return _mm256_and_ps(_mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)), v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v) {
  return _mm256_and_pd(_mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)), v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v) {
  return _mm256_abs_epi8(v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v) {
  return _mm256_abs_epi16(v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v) {
  return _mm256_abs_epi32(v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v) {
  #ifdef HAS_AVX512DQ
  return _mm256_abs_epi64(v.v);
  #else
  ViewAs<int32_t, T> t32; T t64;
  auto high32 = word::shuf<3, 3, 1, 1>(word::bitcast(t32, v));
  auto sign_mask = word::bit_shr<31>(high32);
  auto xored = word::bit_xor(v, sign_mask);
  return word::sub(word::bitcast(t64, xored), word::bitcast(t64, sign_mask));
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v) {
  return _mm512_and_ps(_mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)), v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v) {
  return _mm512_and_pd(_mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFFLL)), v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v) {
  return _mm512_abs_epi8(v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v) {
  return _mm512_abs_epi16(v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v) {
  return _mm512_abs_epi32(v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v) {
  return _mm512_abs_epi64(v.v);
}
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>> && std::is_unsigned_v<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE V abs(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v, Mask<T> m, V default_v) {
  return _mm_mask_and_ps(default_v.v, m.v, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)), v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v, Mask<T> m, V default_v) {
  return _mm_mask_and_pd(default_v.v, m.v, _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)), v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v, Mask<T> m, V default_v) {
  return _mm_mask_abs_epi8(default_v.v, m.v, v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v, Mask<T> m, V default_v) {
  return _mm_mask_abs_epi16(default_v.v, m.v, v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v, Mask<T> m, V default_v) {
  return _mm_mask_abs_epi32(default_v.v, m.v, v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v, Mask<T> m, V default_v) {
  return _mm_mask_abs_epi64(default_v.v, m.v, v.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v, Mask<T> m, V default_v) {
  return _mm256_mask_and_ps(default_v.v, m.v, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)), v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v, Mask<T> m, V default_v) {
  return _mm256_mask_and_pd(default_v.v, m.v, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)), v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v, Mask<T> m, V default_v) {
  return _mm256_mask_abs_epi8(default_v.v, m.v, v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v, Mask<T> m, V default_v) {
  return _mm256_mask_abs_epi16(default_v.v, m.v, v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v, Mask<T> m, V default_v) {
  return _mm256_mask_abs_epi32(default_v.v, m.v, v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v, Mask<T> m, V default_v) {
  return _mm256_mask_abs_epi64(default_v.v, m.v, v.v);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v, Mask<T> m, V default_v) {
  return _mm512_mask_and_ps(default_v.v, m.v, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)), v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v, Mask<T> m, V default_v) {
  return _mm512_mask_and_pd(default_v.v, m.v, _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFFLL)), v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v, Mask<T> m, V default_v) {
  return _mm512_mask_abs_epi8(default_v.v, m.v, v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v, Mask<T> m, V default_v) {
  return _mm512_mask_abs_epi16(default_v.v, m.v, v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v, Mask<T> m, V default_v) {
  return _mm512_mask_abs_epi32(default_v.v, m.v, v.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE V abs(V v, Mask<T> m, V default_v) {
  return _mm512_mask_abs_epi64(default_v.v, m.v, v.v);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>>
CT_ALWAYS_FORCEINLINE V abs(V v, Mask<T> m, V default_v) {
  return word::blend(default_v, m, word::abs(v));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                              Compare Equals                                //
/* ************************************************************************** */
#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm_cmp_ps_mask(a.v, b.v, _CMP_EQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm_cmp_pd_mask(a.v, b.v, _CMP_EQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_EQ);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm256_cmp_ps_mask(a.v, b.v, _CMP_EQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm256_cmp_pd_mask(a.v, b.v, _CMP_EQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm256_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm256_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm256_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm256_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm256_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_EQ);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm512_cmp_ps_mask(a.v, b.v, _CMP_EQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm512_cmp_pd_mask(a.v, b.v, _CMP_EQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm512_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm512_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm512_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm512_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_EQ);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm_castps_si128(_mm_cmpeq_ps(a.v, b.v));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm_castpd_si128(_mm_cmpeq_pd(a.v, b.v));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm_cmpeq_epi8(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm_cmpeq_epi16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm_cmpeq_epi32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm_cmpeq_epi64(a.v, b.v);
}

#if VEC_WIDTH >= 256
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_EQ_OQ));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_EQ_OQ));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm256_cmpeq_epi8(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm256_cmpeq_epi16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm256_cmpeq_epi32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b) {
  return _mm256_cmpeq_epi64(a.v, b.v);
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
  #error "Unreachable"
#endif // VEC_WIDTH >= 512
#endif // HAS_AVX512DQ

#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_EQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_EQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_EQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_EQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_EQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_EQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_EQ);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>>
CT_ALWAYS_FORCEINLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return word::bit_and(m, word::cmpeq(a, b));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                            Compare Not Equals                              //
/* ************************************************************************** */
#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm_cmp_ps_mask(a.v, b.v, _CMP_NEQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm_cmp_pd_mask(a.v, b.v, _CMP_NEQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NE);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm256_cmp_ps_mask(a.v, b.v, _CMP_NEQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm256_cmp_pd_mask(a.v, b.v, _CMP_NEQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm256_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm256_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm256_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm256_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm256_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NE);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm512_cmp_ps_mask(a.v, b.v, _CMP_NEQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm512_cmp_pd_mask(a.v, b.v, _CMP_NEQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm512_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm512_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm512_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm512_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NE);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm_castps_si128(_mm_cmpneq_ps(a.v, b.v));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm_castpd_si128(_mm_cmpneq_pd(a.v, b.v));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 32), TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return word::bit_not(V{word::cmpeq(a, b).v}).v; // TODO dabian
}

#if VEC_WIDTH >= 256
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_NEQ_OQ));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b) {
  return _mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_NEQ_OQ));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
  #error "Unreachable"
#endif // VEC_WIDTH >= 512
#endif // HAS_AVX512DQ

#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_NEQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_NEQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_NEQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_NEQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_NEQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_NEQ_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NE);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>>
CT_ALWAYS_FORCEINLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  using Vi = Vec<ViewAs<Index<TypeOf<T>>, T>>;
  return word::bit_and(Vi{m.v}, Vi{word::cmpne(a, b).v}).v;
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                            Compare Less Than                               //
/* ************************************************************************** */
#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm_cmp_ps_mask(a.v, b.v, _CMP_LT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm_cmp_pd_mask(a.v, b.v, _CMP_LT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_LT);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm256_cmp_ps_mask(a.v, b.v, _CMP_LT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm256_cmp_pd_mask(a.v, b.v, _CMP_LT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm256_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm256_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm256_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm256_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm256_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_LT);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm512_cmp_ps_mask(a.v, b.v, _CMP_LT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm512_cmp_pd_mask(a.v, b.v, _CMP_LT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm512_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm512_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm512_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm512_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_LT);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm_castps_si128(_mm_cmplt_ps(a.v, b.v));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm_castpd_si128(_mm_cmplt_pd(a.v, b.v));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm_cmpgt_epi8(b.v, a.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm_cmpgt_epi16(b.v, a.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm_cmpgt_epi32(b.v, a.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm_cmpgt_epi64(b.v, a.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>> && std::is_unsigned_v<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  using Es = std::make_signed_t<TypeOf<T>>;
  constexpr Es val = 1uLL << (sizeof(Es) * CHAR_BIT - 1);
  ViewAs<Es, T> ts;
  const auto bits = word::fill(ts, val);
  return word::cmplt(
      word::bit_xor(word::bitcast(ts, a), bits),
      word::bit_xor(word::bitcast(ts, b), bits)
  );
}

#if VEC_WIDTH >= 256
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_LT_OQ));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_LT_OQ));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm256_cmpgt_epi8(b.v, a.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm256_cmpgt_epi16(b.v, a.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm256_cmpgt_epi32(b.v, a.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b) {
  return _mm256_cmpgt_epi64(b.v, a.v);
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
  #error "Unreachable"
#endif // VEC_WIDTH >= 512
#endif // HAS_AVX512DQ

#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_LT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_LT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_LT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_LT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_LT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_LT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_LT);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>>
CT_ALWAYS_FORCEINLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return word::bit_and(m, word::cmplt(a, b));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                          Compare Greater Than                              //
/* ************************************************************************** */
#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm_cmp_ps_mask(a.v, b.v, _CMP_GT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm_cmp_pd_mask(a.v, b.v, _CMP_GT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NLE);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm256_cmp_ps_mask(a.v, b.v, _CMP_GT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm256_cmp_pd_mask(a.v, b.v, _CMP_GT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm256_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm256_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm256_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm256_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm256_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NLE);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm512_cmp_ps_mask(a.v, b.v, _CMP_GT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm512_cmp_pd_mask(a.v, b.v, _CMP_GT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm512_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm512_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm512_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm512_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NLE);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm_castps_si128(_mm_cmpgt_ps(a.v, b.v));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm_castpd_si128(_mm_cmpgt_pd(a.v, b.v));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm_cmpgt_epi8(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm_cmpgt_epi16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm_cmpgt_epi32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm_cmpgt_epi64(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>> && std::is_unsigned_v<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  using Es = std::make_signed_t<TypeOf<T>>;
  constexpr Es val = 1uLL << (sizeof(Es) * CHAR_BIT - 1);
  ViewAs<Es, T> ts;
  const auto bits = word::fill(ts, val);
  return word::cmpgt(
      word::bit_xor(word::bitcast(ts, a), bits),
      word::bit_xor(word::bitcast(ts, b), bits)
  );
}

#if VEC_WIDTH >= 256
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_GT_OQ));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_GT_OQ));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm256_cmpgt_epi8(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm256_cmpgt_epi16(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm256_cmpgt_epi32(a.v, b.v);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b) {
  return _mm256_cmpgt_epi64(a.v, b.v);
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
  #error "Unreachable"
#endif // VEC_WIDTH >= 512
#endif // HAS_AVX512DQ

#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_GT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_GT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_GT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_GT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_GT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_GT_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NLE);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>>
CT_ALWAYS_FORCEINLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return word::bit_and(m, word::cmpgt(a, b));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                       Compare Less Than or Equals                          //
/* ************************************************************************** */
#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm_cmp_ps_mask(a.v, b.v, _CMP_LE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm_cmp_pd_mask(a.v, b.v, _CMP_LE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_LE);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm256_cmp_ps_mask(a.v, b.v, _CMP_LE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm256_cmp_pd_mask(a.v, b.v, _CMP_LE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm256_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm256_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm256_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm256_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm256_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_LE);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm512_cmp_ps_mask(a.v, b.v, _CMP_LE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm512_cmp_pd_mask(a.v, b.v, _CMP_LE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm512_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm512_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm512_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm512_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_LE);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm_castps_si128(_mm_cmple_ps(a.v, b.v));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm_castpd_si128(_mm_cmple_pd(a.v, b.v));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return word::bit_not(V{word::cmpgt(a, b).v}).v; // TODO dabian
}

#if VEC_WIDTH >= 256
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_LE_OQ));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b) {
  return _mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_LE_OQ));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
  #error "Unreachable"
#endif // VEC_WIDTH >= 512
#endif // HAS_AVX512DQ

#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_LE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_LE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_LE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_LE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_LE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_LE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_LE);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>>
CT_ALWAYS_FORCEINLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return word::bit_and(m, word::cmple(a, b));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                      Compare Greater Than or Equals                        //
/* ************************************************************************** */
#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm_cmp_ps_mask(a.v, b.v, _CMP_GE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm_cmp_pd_mask(a.v, b.v, _CMP_GE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NLT);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm256_cmp_ps_mask(a.v, b.v, _CMP_GE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm256_cmp_pd_mask(a.v, b.v, _CMP_GE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm256_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm256_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm256_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm256_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm256_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NLT);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm512_cmp_ps_mask(a.v, b.v, _CMP_GE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm512_cmp_pd_mask(a.v, b.v, _CMP_GE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm512_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm512_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm512_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm512_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NLT);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm_castps_si128(_mm_cmpge_ps(a.v, b.v));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm_castpd_si128(_mm_cmpge_pd(a.v, b.v));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return word::bit_not(V{word::cmplt(a, b).v}).v; // TODO dabian
}

#if VEC_WIDTH >= 256
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_GE_OQ));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b) {
  return _mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_GE_OQ));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
  #error "Unreachable"
#endif // VEC_WIDTH >= 512
#endif // HAS_AVX512DQ

#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_GE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_GE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_GE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_GE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm256_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_GE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_GE_OQ);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return _mm512_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NLT);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>>
CT_ALWAYS_FORCEINLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return word::bit_and(m, word::cmpge(a, b));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                                  Is NaN                                    //
/* ************************************************************************** */
#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> isnan(V v) {
  return _mm_cmp_ps_mask(v.v, v.v, _CMP_UNORD_Q);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> isnan(V v) {
  return _mm_cmp_pd_mask(v.v, v.v, _CMP_UNORD_Q);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> isnan(V v) {
  return _mm256_cmp_ps_mask(v.v, v.v, _CMP_UNORD_Q);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> isnan(V v) {
  return _mm256_cmp_pd_mask(v.v, v.v, _CMP_UNORD_Q);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> isnan(V v) {
  return _mm512_cmp_ps_mask(v.v, v.v, _CMP_UNORD_Q);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> isnan(V v) {
  return _mm512_cmp_pd_mask(v.v, v.v, _CMP_UNORD_Q);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> isnan(V v) {
  return _mm_castps_si128(_mm_cmpunord_ps(v.v, v.v));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> isnan(V v) {
  return _mm_castpd_si128(_mm_cmpunord_pd(v.v, v.v));
}

#if VEC_WIDTH >= 256
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> isnan(V v) {
  return _mm256_castps_si256(_mm256_cmp_ps(v.v, v.v, _CMP_UNORD_Q));
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> isnan(V v) {
  return _mm256_castpd_si256(_mm256_cmp_pd(v.v, v.v, _CMP_UNORD_Q));
}
#endif // VEC_WIDTH >= 256
#endif // HAS_AVX512DQ

#ifdef HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> isnan(V v, Mask<T> m) {
  return _mm_mask_cmp_ps_mask(m.v, v.v, v.v, _CMP_UNORD_Q);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> isnan(V v, Mask<T> m) {
  return _mm_mask_cmp_pd_mask(m.v, v.v, v.v, _CMP_UNORD_Q);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> isnan(V v, Mask<T> m) {
  return _mm256_mask_cmp_ps_mask(m.v, v.v, v.v, _CMP_UNORD_Q);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> isnan(V v, Mask<T> m) {
  return _mm256_mask_cmp_pd_mask(m.v, v.v, v.v, _CMP_UNORD_Q);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> isnan(V v, Mask<T> m) {
  return _mm512_mask_cmp_ps_mask(m.v, v.v, v.v, _CMP_UNORD_Q);
}
template <typename V, typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Mask<T> isnan(V v, Mask<T> m) {
  return _mm512_mask_cmp_pd_mask(m.v, v.v, v.v, _CMP_UNORD_Q);
}
#else // HAS_AVX512DQ
template <typename V, typename T = Vec2Tag<V>>
CT_ALWAYS_FORCEINLINE Mask<T> isnan(V v, Mask<T> m) {
  return word::bit_and(m, word::isnan(v));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                                Is Infinity                                 //
/* ************************************************************************** */
template <typename V, typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE Mask<T> isposinf(V v) {
  return word::cmpeq(v, word::fill(T(), TypeOf<T>(INFINITY)));
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE Mask<T> isposinf(V v, Mask<T> m) {
  return word::cmpeq(v, word::fill(T(), TypeOf<T>(INFINITY)), m);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE Mask<T> isneginf(V v) {
  return word::cmpeq(v, word::fill(T(), TypeOf<T>(-INFINITY)));
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE Mask<T> isneginf(V v, Mask<T> m) {
  return word::cmpeq(v, word::fill(T(), TypeOf<T>(-INFINITY)), m);
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE Mask<T> isinf(V v) {
  return word::isposinf(word::abs(v));
}

template <typename V, typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>> || is_small_float<TypeOf<T>>)>
CT_ALWAYS_FORCEINLINE Mask<T> isinf(V v, Mask<T> m) {
  return word::isposinf(word::abs(v), m);
}

} // namespace word
} // namespace ct::tl::vec::CPU_CAPABILITY
//@formatter:on

#endif //CTORCH_X86_ARITHMETIC_H
