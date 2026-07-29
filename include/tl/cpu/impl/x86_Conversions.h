//
// Created by renyz on 2026/3/23.
//

#ifndef CTORCH_X86_CONVERSIONS_H
#define CTORCH_X86_CONVERSIONS_H

#include "tl/cpu/impl/x86_Basic.h"

//@formatter:off
namespace ct::tl::vec::CPU_CAPABILITY {
namespace word {

template <TLV_DECL_TAG(To), TLV_DECL_VEC(Vi), typename Ti = Vec2Tag<Vi>>
TLV_INLINE Vec<To> reshape(To t_out, Vi v_in) {
  static_assert(std::is_same_v<TypeOf<To>, TypeOf<Ti>>, "Not same type");
  static_assert(std::is_same_v<Vec<To>, Vi>, "What");
  return v_in;
}

/* ************************************************************************** */
//                           Generic Conversions                              //
/* ************************************************************************** */
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> convert(T t, Vec<T> v) { return v; }

template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> promote(T t, Vec<T> v) { return v; }

template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> demote(T t, Vec<T> v) { return v; }


/* ************************************************************************** */
//                          int64_t <=> float64_t                             //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<float64_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm_cvttpd_epi64(v.v);
  #else
  alignas(16) float64_t data[2];
  alignas(16) int64_t conv[2];
  _mm_store_pd(data, v.v);
  for (int i = 0; i < 2; ++i) conv[i] = int64_t(data[i]);
  return _mm_load_si128((const __m128i*)conv);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<int64_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm_cvtepi64_pd(v.v);
  #else
  alignas(16) int64_t data[2];
  alignas(16) float64_t conv[2];
  _mm_store_si128((__m128i*)data, v.v);
  for (int i = 0; i < 2; ++i) conv[i] = float64_t(data[i]);
  return _mm_load_pd(conv);
  #endif
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<float64_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm256_cvttpd_epi64(v.v);
  #elif VEC_WIDTH >= 256
  alignas(32) float64_t data[4];
  alignas(32) int64_t conv[4];
  _mm256_store_pd(data, v.v);
  for (int i = 0; i < 4; ++i) conv[i] = int64_t(data[i]);
  return _mm256_load_si256((const __m256i*)conv);
  #else
  Tag<int64_t, 2> t1;
  Rebind<float64_t, T> t2;
  auto lo = word::convert(t1, word::lower(t2, v));
  auto hi = word::convert(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<int64_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm256_cvtepi64_pd(v.v);
  #elif VEC_WIDTH >= 256
  alignas(32) int64_t data[4];
  alignas(32) float64_t conv[4];
  _mm256_store_si256((__m256i*)data, v.v);
  for (int i = 0; i < 4; ++i) conv[i] = float64_t(data[i]);
  return _mm256_load_pd(conv);
  #else
  Tag<float64_t, 2> t1;
  Rebind<int64_t, T> t2;
  auto lo = word::convert(t1, word::lower(t2, v));
  auto hi = word::convert(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<float64_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvttpd_epi64(v.v);
  #else
  Tag<int64_t, 4> t1;
  Rebind<float64_t, T> t2;
  auto lo = word::convert(t1, word::lower(t2, v));
  auto hi = word::convert(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<int64_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvtepi64_pd(v.v);
  #else
  Tag<float64_t, 4> t1;
  Rebind<int64_t, T> t2;
  auto lo = word::convert(t1, lower(t2, v));
  auto hi = word::convert(t1, upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 512

/* ************************************************************************** */
//                          uint64_t <=> float64_t                            //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, uint64_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<float64_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm_cvttpd_epu64(v.v);
  #else
  alignas(16) float64_t data[2];
  alignas(16) uint64_t conv[2];
  _mm_store_pd(data, v.v);
  for (int i = 0; i < 2; ++i) conv[i] = uint64_t(data[i]);
  return _mm_load_si128((const __m128i*)conv);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<uint64_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm_cvtepu64_pd(v.v);
  #else
  alignas(16) uint64_t data[2];
  alignas(16) float64_t conv[2];
  _mm_store_si128((__m128i*)data, v.v);
  for (int i = 0; i < 2; ++i) conv[i] = float64_t(data[i]);
  return _mm_load_pd(conv);
  #endif
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, uint64_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<float64_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm256_cvttpd_epu64(v.v);
  #elif VEC_WIDTH >= 256
  alignas(32) float64_t data[4];
  alignas(32) uint64_t conv[4];
  _mm256_store_pd(data, v.v);
  for (int i = 0; i < 4; ++i) conv[i] = uint64_t(data[i]);
  return _mm256_load_si256((const __m256i*)conv);
  #else
  Tag<uint64_t, 2> t1;
  Rebind<float64_t, T> t2;
  auto lo = word::convert(t1, word::lower(t2, v));
  auto hi = word::convert(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<uint64_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm256_cvtepu64_pd(v.v);
  #elif VEC_WIDTH >= 256
  alignas(32) uint64_t data[4];
  alignas(32) float64_t conv[4];
  _mm256_store_si256((__m256i*)data, v.v);
  for (int i = 0; i < 4; ++i) conv[i] = float64_t(data[i]);
  return _mm256_load_pd(conv);
  #else
  Tag<float64_t, 2> t1;
  Rebind<uint64_t, T> t2;
  auto lo = word::convert(t1, word::lower(t2, v));
  auto hi = word::convert(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, uint64_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<float64_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvttpd_epu64(v.v);
  #else
  Tag<uint64_t, 4> t1;
  Rebind<float64_t, T> t2;
  auto lo = word::convert(t1, word::lower(t2, v));
  auto hi = word::convert(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<uint64_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvtepu64_pd(v.v);
  #else
  Tag<float64_t, 4> t1;
  Rebind<uint64_t, T> t2;
  auto lo = word::convert(t1, lower(t2, v));
  auto hi = word::convert(t1, upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                           int64_t <=> uint64_t                             //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<uint64_t, T>> v) {
  return v.v;
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint64_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<int64_t, T>> v) {
  return v.v;
}


/* ************************************************************************** */
//                         float32_t <=> float64_t                            //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  return _mm_cvtpd_ps(v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
  return _mm_cvtps_pd(v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  #if VEC_WIDTH >= 256
  return _mm256_cvtpd_ps(v.v);
  #else
  Tag<float32_t, 2> t1;
  Rebind<float64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
  #if VEC_WIDTH >= 256
  return _mm256_cvtps_pd(v.v);
  #else
  Tag<float64_t, 2> t1;
  Rebind<float32_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvtpd_ps(v.v);
  #else
  Tag<float32_t, 4> t1;
  Rebind<float64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvtps_pd(v.v);
  #else
  Tag<float64_t, 4> t1;
  Rebind<float32_t, T> t2;
  auto lo = word::promote(t1, lower(t2, v));
  auto hi = word::promote(t1, upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  Tag<float32_t, 8> t1;
  Rebind<float64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
  Tag<float64_t, 8> t1;
  Rebind<float32_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                          float32_t <=> int64_t                             //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm_cvtepi64_ps(v.v);
  #else
  alignas(16) int64_t data[2];
  alignas(16) float32_t conv[4];
  _mm_store_si128((__m128i*)data, v.v);
  for (int i = 0; i < 2; ++i) conv[i] = float32_t(data[i]);
  return _mm_load_ps(conv);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm_cvttps_epi64(v.v);
  #else
  alignas(16) float32_t data[4];
  alignas(16) int64_t conv[2];
  _mm_store_ps(data, v.v);
  for (int i = 0; i < 2; ++i) conv[i] = int64_t(data[i]);
  return _mm_load_si128((const __m128i*)conv);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm256_cvtepi64_ps(v.v);
  #elif VEC_WIDTH >= 256
  alignas(32) int64_t data[4];
  alignas(32) float32_t conv[4];
  _mm256_store_si256((__m256i*)data, v.v);
  for (int i = 0; i < 4; ++i) conv[i] = float32_t(data[i]);
  return _mm_load_ps(conv);
  #else
  Tag<float32_t, 2> t1;
  Rebind<int64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm256_cvttps_epi64(v.v);
  #elif VEC_WIDTH >= 256
  alignas(32) float32_t data[4];
  alignas(32) int64_t conv[4];
  _mm_store_ps(data, v.v);
  for (int i = 0; i < 4; ++i) conv[i] = int64_t(data[i]);
  return _mm256_load_si256((const __m256i*)conv);
  #else
  Tag<int64_t, 2> t1;
  Rebind<float32_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  #if HAS_AVX512DQ
  return _mm512_cvtepi64_ps(v.v);
  #else
  Tag<float32_t, 4> t1;
  Rebind<int64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvttps_epi64(v.v);
  #else
  Tag<int64_t, 4> t1;
  Rebind<float32_t, T> t2;
  auto lo = word::promote(t1, lower(t2, v));
  auto hi = word::promote(t1, upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  Tag<float32_t, 8> t1;
  Rebind<int64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
  Tag<int64_t, 8> t1;
  Rebind<float32_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                          float32_t <=> uint64_t                            //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm_cvtepu64_ps(v.v);
  #else
  alignas(16) uint64_t data[2];
  alignas(16) float32_t conv[4];
  _mm_store_si128((__m128i*)data, v.v);
  for (int i = 0; i < 2; ++i) conv[i] = float32_t(data[i]);
  return _mm_load_ps(conv);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, uint64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm_cvttps_epu64(v.v);
  #else
  alignas(16) float32_t data[4];
  alignas(16) uint64_t conv[2];
  _mm_store_ps(data, v.v);
  for (int i = 0; i < 2; ++i) conv[i] = uint64_t(data[i]);
  return _mm_load_si128((const __m128i*)conv);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm256_cvtepu64_ps(v.v);
  #elif VEC_WIDTH >= 256
  alignas(32) uint64_t data[4];
  alignas(32) float32_t conv[4];
  _mm256_store_si256((__m256i*)data, v.v);
  for (int i = 0; i < 4; ++i) conv[i] = float32_t(data[i]);
  return _mm_load_ps(conv);
  #else
  Tag<float32_t, 2> t1;
  Rebind<uint64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, uint64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm256_cvttps_epu64(v.v);
  #elif VEC_WIDTH >= 256
  alignas(32) float32_t data[4];
  alignas(32) uint64_t conv[4];
  _mm_store_ps(data, v.v);
  for (int i = 0; i < 4; ++i) conv[i] = uint64_t(data[i]);
  return _mm256_load_si256((const __m256i*)conv);
  #else
  Tag<uint64_t, 2> t1;
  Rebind<float32_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
  #if HAS_AVX512DQ
  return _mm512_cvtepu64_ps(v.v);
  #else
  Tag<float32_t, 4> t1;
  Rebind<uint64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, uint64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvttps_epu64(v.v);
  #else
  Tag<uint64_t, 4> t1;
  Rebind<float32_t, T> t2;
  auto lo = word::promote(t1, lower(t2, v));
  auto hi = word::promote(t1, upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
  Tag<float32_t, 8> t1;
  Rebind<uint64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
  Tag<uint64_t, 8> t1;
  Rebind<float32_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                         int32_t <=> float64_t                            //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  return _mm_cvttpd_epi32(v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
  return _mm_cvtepi32_pd(v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  #if VEC_WIDTH >= 256
  return _mm256_cvttpd_epi32(v.v);
  #else
  Tag<int32_t, 2> t1;
  Rebind<float64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
  #if VEC_WIDTH >= 256
  return _mm256_cvtepi32_pd(v.v);
  #else
  Tag<float64_t, 2> t1;
  Rebind<int32_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvttpd_epi32(v.v);
  #else
  Tag<int32_t, 4> t1;
  Rebind<float64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvtepi32_pd(v.v);
  #else
  Tag<float64_t, 4> t1;
  Rebind<int32_t, T> t2;
  auto lo = word::promote(t1, lower(t2, v));
  auto hi = word::promote(t1, upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  Tag<int32_t, 8> t1;
  Rebind<float64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
  Tag<float64_t, 8> t1;
  Rebind<int32_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                          int32_t <=> int64_t                             //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  static constexpr int64_t min_val = INT32_MIN;
  static constexpr int64_t max_val = INT32_MAX;
  #ifdef HAS_AVX512DQ
  auto u = _mm_min_epi64(_mm_max_epi64(v.v, _mm_set1_epi64x(min_val)), _mm_set1_epi64x(max_val));
  return _mm_cvtepi64_epi32(u);
  #else
  alignas(16) int64_t data[2];
  alignas(16) int32_t conv[4];
  _mm_store_si128((__m128i*)data, v.v);
  for (int i = 0; i < 2; ++i) conv[i] = int32_t(std::clamp(data[i], min_val, max_val));
  return _mm_load_si128((const __m128i*)conv);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
  return _mm_cvtepi32_epi64(v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  static constexpr int64_t min_val = INT32_MIN;
  static constexpr int64_t max_val = INT32_MAX;
  #ifdef HAS_AVX512DQ
  auto u = _mm256_min_epi64(_mm256_max_epi64(v.v, _mm256_set1_epi64x(min_val)), _mm256_set1_epi64x(max_val));
  return _mm256_cvtepi64_epi32(u);
  #elif VEC_WIDTH >= 256
  alignas(32) int64_t data[4];
  alignas(32) int32_t conv[4];
  _mm256_store_si256((__m256i*)data, v.v);
  for (int i = 0; i < 4; ++i) conv[i] = int32_t(std::clamp(data[i], min_val, max_val));
  return _mm_load_si128((const __m128i*)conv);
  #else
  Tag<int32_t, 2> t1;
  Rebind<int64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
  return _mm256_cvtepi32_epi64(v.v);
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  static constexpr int64_t min_val = INT32_MIN;
  static constexpr int64_t max_val = INT32_MAX;
  #if HAS_AVX512DQ
  auto u = _mm512_min_epi64(_mm512_max_epi64(v.v, _mm512_set1_epi64(min_val)), _mm512_set1_epi64(max_val));
  return _mm512_cvtepi64_epi32(u);
  #else
  Tag<int32_t, 4> t1;
  Rebind<int64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvtepi32_epi64(v.v);
  #else
  Tag<int64_t, 4> t1;
  Rebind<int32_t, T> t2;
  auto lo = word::promote(t1, lower(t2, v));
  auto hi = word::promote(t1, upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  Tag<int32_t, 8> t1;
  Rebind<int64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
  Tag<int64_t, 8> t1;
  Rebind<int32_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                          int32_t <=> uint64_t                            //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
  static constexpr uint64_t max_val = INT32_MAX;
  #ifdef HAS_AVX512DQ
  auto u = _mm_min_epu64(v.v, _mm_set1_epi64x(max_val));
  return _mm_cvtepi64_epi32(u);
  #else
  alignas(16) uint64_t data[2];
  alignas(16) int32_t conv[4];
  _mm_store_si128((__m128i*)data, v.v);
  for (int i = 0; i < 2; ++i) conv[i] = int32_t(std::min(data[i], max_val));
  return _mm_load_si128((const __m128i*)conv);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
  Rebind<int64_t, T> t1;
  return word::bitcast(t, word::promote(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
  static constexpr uint64_t max_val = INT32_MAX;
  #ifdef HAS_AVX512DQ
  auto u = _mm256_min_epu64(v.v, _mm256_set1_epi64x(max_val));
  return _mm256_cvtepi64_epi32(u);
  #elif VEC_WIDTH >= 256
  alignas(32) uint64_t data[4];
  alignas(32) int32_t conv[4];
  _mm256_store_si256((__m256i*)data, v.v);
  for (int i = 0; i < 4; ++i) conv[i] = int32_t(std::min(data[i], max_val));
  return _mm_load_si128((const __m128i*)conv);
  #else
  Tag<int32_t, 2> t1;
  Rebind<int64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
  static constexpr uint64_t max_val = INT32_MAX;
  #if HAS_AVX512DQ
  auto u = _mm512_min_epu64(v.v, _mm512_set1_epi64(max_val));
  return _mm512_cvtepi64_epi32(u);
  #else
  Tag<int32_t, 4> t1;
  Rebind<int64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
  Tag<int32_t, 8> t1;
  Rebind<uint64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                         uint32_t <=> float64_t                            //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, uint32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm_cvttpd_epu32(v.v);
  #else
  alignas(16) float64_t data[2];
  alignas(16) uint32_t conv[4];
  _mm_store_pd(data, v.v);
  for (int i = 0; i < 2; ++i) conv[i] = uint32_t(data[i]);
  return _mm_load_si128((const __m128i*)conv);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm_cvtepu32_pd(v.v);
  #else
  alignas(16) uint32_t data[4];
  alignas(16) float64_t conv[2];
  _mm_store_si128((__m128i*)data, v.v);
  for (int i = 0; i < 2; ++i) conv[i] = float64_t(data[i]);
  return _mm_load_pd(conv);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, uint32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm256_cvttpd_epu32(v.v);
  #elif VEC_WIDTH >= 256
  alignas(32) float64_t data[4];
  alignas(32) uint32_t conv[4];
  _mm256_store_pd(data, v.v);
  for (int i = 0; i < 4; ++i) conv[i] = uint32_t(data[i]);
  return _mm_load_si128((const __m128i*)conv);
  #else
  Tag<uint32_t, 2> t1;
  Rebind<float64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm256_cvtepi32_pd(v.v);
  #elif VEC_WIDTH >= 256
  alignas(32) uint32_t data[4];
  alignas(32) float64_t conv[4];
  _mm_store_si128((__m128i*)data, v.v);
  for (int i = 0; i < 4; ++i) conv[i] = float64_t(data[i]);
  return _mm256_load_pd(conv);
  #else
  Tag<float64_t, 2> t1;
  Rebind<uint32_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, uint32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvttpd_epu32(v.v);
  #else
  Tag<uint32_t, 4> t1;
  Rebind<float64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvtepu32_pd(v.v);
  #else
  Tag<float64_t, 4> t1;
  Rebind<uint32_t, T> t2;
  auto lo = word::promote(t1, lower(t2, v));
  auto hi = word::promote(t1, upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  Tag<uint32_t, 8> t1;
  Rebind<float64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
  Tag<float64_t, 8> t1;
  Rebind<uint32_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                          uint32_t <=> int64_t                             //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, uint32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  static constexpr int64_t min_val = 0;
  static constexpr int64_t max_val = UINT32_MAX;
  #ifdef HAS_AVX512DQ
  auto u = _mm_min_epi64(_mm_max_epi64(v.v, _mm_set1_epi64x(min_val)), _mm_set1_epi64x(max_val));
  return _mm_cvtepi64_epi32(u);
  #else
  alignas(16) int64_t data[2];
  alignas(16) uint32_t conv[4];
  _mm_store_si128((__m128i*)data, v.v);
  for (int i = 0; i < 2; ++i) conv[i] = uint32_t(std::clamp(data[i], min_val, max_val));
  return _mm_load_si128((const __m128i*)conv);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
  return _mm_cvtepu32_epi64(v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, uint32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  static constexpr int64_t min_val = 0;
  static constexpr int64_t max_val = UINT32_MAX;
  #ifdef HAS_AVX512DQ
  auto u = _mm256_min_epi64(_mm256_max_epi64(v.v, _mm256_set1_epi64x(min_val)), _mm256_set1_epi64x(max_val));
  return _mm256_cvtepi64_epi32(u);
  #elif VEC_WIDTH >= 256
  alignas(32) int64_t data[4];
  alignas(32) uint32_t conv[4];
  _mm256_store_si256((__m256i*)data, v.v);
  for (int i = 0; i < 4; ++i) conv[i] = uint32_t(std::clamp(data[i], min_val, max_val));
  return _mm_load_si128((const __m128i*)conv);
  #else
  Tag<uint32_t, 2> t1;
  Rebind<int64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
  return _mm256_cvtepu32_epi64(v.v);
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, uint32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  static constexpr int64_t min_val = 0;
  static constexpr int64_t max_val = UINT32_MAX;
  #if HAS_AVX512DQ
  auto u = _mm512_min_epi64(_mm512_max_epi64(v.v, _mm512_set1_epi64(min_val)), _mm512_set1_epi64(max_val));
  return _mm512_cvtepi64_epi32(u);
  #else
  Tag<uint32_t, 4> t1;
  Rebind<int64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvtepu32_epi64(v.v);
  #else
  Tag<int64_t, 4> t1;
  Rebind<uint32_t, T> t2;
  auto lo = word::promote(t1, lower(t2, v));
  auto hi = word::promote(t1, upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  Tag<uint32_t, 8> t1;
  Rebind<int64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
  Tag<int64_t, 8> t1;
  Rebind<uint32_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                          uint32_t <=> uint64_t                            //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, uint32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
  static constexpr uint64_t max_val = UINT32_MAX;
  #ifdef HAS_AVX512DQ
  auto u = _mm_min_epu64(v.v, _mm_set1_epi64x(max_val));
  return _mm_cvtepi64_epi32(u);
  #else
  alignas(16) uint64_t data[2];
  alignas(16) uint32_t conv[4];
  _mm_store_si128((__m128i*)data, v.v);
  for (int i = 0; i < 2; ++i) conv[i] = uint32_t(std::min(data[i], max_val));
  return _mm_load_si128((const __m128i*)conv);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, uint64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
  return _mm_cvtepu32_epi64(v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, uint32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
  static constexpr uint64_t max_val = UINT32_MAX;
  #ifdef HAS_AVX512DQ
  auto u = _mm256_min_epu64(v.v, _mm256_set1_epi64x(max_val));
  return _mm256_cvtepi64_epi32(u);
  #elif VEC_WIDTH >= 256
  alignas(32) uint64_t data[4];
  alignas(32) uint32_t conv[4];
  _mm256_store_si256((__m256i*)data, v.v);
  for (int i = 0; i < 4; ++i) conv[i] = uint32_t(std::min(data[i], max_val));
  return _mm_load_si128((const __m128i*)conv);
  #else
  Tag<uint32_t, 2> t1;
  Rebind<int64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, uint64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
  return _mm256_cvtepu32_epi64(v.v);
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, uint32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
  static constexpr uint64_t max_val = UINT32_MAX;
  #if HAS_AVX512DQ
  auto u = _mm512_min_epu64(v.v, _mm512_set1_epi64(max_val));
  return _mm512_cvtepi64_epi32(u);
  #else
  Tag<uint32_t, 4> t1;
  Rebind<int64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, uint64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvtepu32_epi64(v.v);
  #else
  Tag<uint64_t, 4> t1;
  Rebind<uint32_t, T> t2;
  auto lo = word::promote(t1, lower(t2, v));
  auto hi = word::promote(t1, upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
  Tag<uint32_t, 8> t1;
  Rebind<uint64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
  Tag<uint64_t, 8> t1;
  Rebind<uint32_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                          float32_t <=> int32_t                             //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<int32_t, T>> v) {
  return _mm_cvtepi32_ps(v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<float32_t, T>> v) {
  return _mm_cvttps_epi32(v.v);
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<int32_t, T>> v) {
  return _mm256_cvtepi32_ps(v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<float32_t, T>> v) {
  return _mm256_cvttps_epi32(v.v);
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<int32_t, T>> v) {
  return _mm512_cvtepi32_ps(v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<float32_t, T>> v) {
  return _mm512_cvttps_epi32(v.v);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                          float32_t <=> uint32_t                             //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<uint32_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm_cvtepu32_ps(v.v);
  #else
  static const __m128 two31 = _mm_set1_ps(2147483648.0f);  // 2^31 as float
  static const __m128i mask_31 = _mm_set1_epi32(0x7FFFFFFF);
  auto is_big = _mm_castsi128_ps(_mm_cmplt_epi32(v.v, _mm_setzero_si128()));
  auto cleared = _mm_and_si128(v.v, mask_31);
  auto result = _mm_cvtepi32_ps(cleared);
  result = _mm_add_ps(result, _mm_and_ps(is_big, two31));
  return result;
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, uint32_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<float32_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm_cvttps_epu32(v.v);
  #else
  const static __m128 two_31 = _mm_set1_ps(2147483648.0f);
  auto mask = _mm_cmpge_ps(v.v, two_31);
  auto a_adjusted = _mm_sub_ps(v.v, _mm_and_ps(mask, two_31));
  auto result = _mm_cvttps_epi32(a_adjusted);
  auto offset = _mm_and_si128(_mm_castps_si128(mask), _mm_set1_epi32(0x80000000));
  return _mm_add_epi32(result, offset);
  #endif
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<uint32_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm256_cvtepu32_ps(v.v);
  #else
  static const __m256 two31 = _mm256_set1_ps(2147483648.0f);  // 2^31 as float
  static const __m256i mask_31 = _mm256_set1_epi32(0x7FFFFFFF);
  auto is_big = _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_setzero_si256(), v.v));
  auto cleared = _mm256_and_si256(v.v, mask_31);
  auto result = _mm256_cvtepi32_ps(cleared);
  result = _mm256_add_ps(result, _mm256_and_ps(is_big, two31));
  return result;
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, uint32_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<float32_t, T>> v) {
  #ifdef HAS_AVX512DQ
  return _mm256_cvttps_epu32(v.v);
  #else
  const static __m256 two_31 = _mm256_set1_ps(2147483648.0f);
  auto mask = _mm256_cmp_ps(v.v, two_31, _CMP_GE_OS);
  auto a_adjusted = _mm256_sub_ps(v.v, _mm256_and_ps(mask, two_31));
  auto result = _mm256_cvttps_epi32(a_adjusted);
  auto offset = _mm256_and_si256(_mm256_castps_si256(mask), _mm256_set1_epi32(0x80000000));
  return _mm256_add_epi32(result, offset);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<uint32_t, T>> v) {
  return _mm512_cvtepu32_ps(v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<float32_t, T>> v) {
  return _mm512_cvttps_epu32(v.v);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                          int32_t <=> uint32_t                             //
/* ************************************************************************** */

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<uint32_t, T>> v) {
  return v.v;
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint32_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<int32_t, T>> v) {
  return v.v;
}


/* ************************************************************************** */
//                           int16_t <=> int32_t                              //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, int16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int32_t, T>> v) {
  return _mm_packs_epi32(v.v, v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  return _mm_cvtepi16_epi32(v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int32_t, T>> v) {
  Rebind<int32_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  return _mm_packs_epi32(lo.v, hi.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  #if VEC_WIDTH >= 256
  return _mm256_cvtepi16_epi32(v.v);
  #else
  Tag<int32_t, 2> t1;
  Rebind<int16_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int32_t, T>> v) {
  Rebind<int32_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  auto u = _mm256_packs_epi32(lo.v, hi.v);
  return _mm256_permute4x64_epi64(u, _MM_SHUFFLE(3, 1, 2, 0));
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvtepi16_epi32(v.v);
  #else
  Tag<int32_t, 4> t1;
  Rebind<int16_t, T> t2;
  auto lo = word::promote(t1, lower(t2, v));
  auto hi = word::promote(t1, upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int32_t, T>> v) {
  Rebind<int32_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  auto u = _mm512_packs_epi32(lo.v, hi.v);
  static const __m512i idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_epi64(u, idx);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  Tag<int32_t, 8> t1;
  Rebind<int16_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                           int16_t <=> uint32_t                             //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, int16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint32_t, T>> v) {
  static constexpr uint32_t max_val = INT32_MAX;
  auto u = _mm_min_epu32(v.v, _mm_set1_epi32(max_val));
  return _mm_packs_epi32(u, u);
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::bitcast(t, word::promote(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint32_t, T>> v) {
  Rebind<uint32_t, T> t1;
  static constexpr uint32_t max_val = INT32_MAX;
  Vec<decltype(t1)> u = _mm256_min_epu32(v.v, _mm256_set1_epi32(max_val));
  auto lo = word::lower(t1, u);
  auto hi = word::upper(t1, u);
  return _mm_packs_epi32(lo.v, hi.v);
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint32_t, T>> v) {
  static constexpr uint32_t max_val = INT32_MAX;
  #if VEC_WIDTH >= 512
  Rebind<uint32_t, T> t1;
  Vec<decltype(t1)> u = _mm512_min_epu32(v.v, _mm512_set1_epi32(max_val));
  auto lo = word::lower(t1, u).v;
  auto hi = word::upper(t1, u).v;
  #else
  Rebind<int16_t, T> t1;
  auto lo = _mm256_min_epu32(lower(t1, v).v, _mm256_set1_epi32(max_val));
  auto hi = _mm256_min_epu32(upper(t1, v).v, _mm256_set1_epi32(max_val));
  #endif
  auto w = _mm256_packs_epi32(lo, hi);
  return _mm256_permute4x64_epi64(w, _MM_SHUFFLE(3, 1, 2, 0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint32_t, T>> v) {
  Rebind<uint32_t, T> t1;
  static constexpr uint32_t max_val = INT32_MAX;
  auto lo = _mm512_min_epu32(word::lower(t1, v).v, _mm512_set1_epi32(max_val));
  auto hi = _mm512_min_epu32(word::upper(t1, v).v, _mm512_set1_epi32(max_val));
  auto u = _mm512_packs_epi32(lo.v, hi.v);
  static const __m512i idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_epi64(u, idx);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                          int16_t <=> float32_t                             //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<float32_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                           uint16_t <=> int32_t                             //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, uint16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int32_t, T>> v) {
  return _mm_packus_epi32(v.v, v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  return _mm_cvtepu16_epi32(v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, uint16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int32_t, T>> v) {
  Rebind<int32_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  return _mm_packus_epi32(lo.v, hi.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  #if VEC_WIDTH >= 256
  return _mm256_cvtepu16_epi32(v.v);
  #else
  Tag<int32_t, 2> t1;
  Rebind<uint16_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int32_t, T>> v) {
  Rebind<int32_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  auto u = _mm256_packus_epi32(lo.v, hi.v);
  return _mm256_permute4x64_epi64(u, _MM_SHUFFLE(3, 1, 2, 0));
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvtepu16_epi32(v.v);
  #else
  Tag<int32_t, 4> t1;
  Rebind<uint16_t, T> t2;
  auto lo = word::promote(t1, lower(t2, v));
  auto hi = word::promote(t1, upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int32_t, T>> v) {
  Rebind<int32_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  auto u = _mm512_packus_epi32(lo.v, hi.v);
  static const __m512i idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_epi64(u, idx);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  Tag<int32_t, 8> t1;
  Rebind<uint16_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                          uint16_t <=> uint32_t                             //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, uint16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint32_t, T>> v) {
  static constexpr uint32_t max_val = INT32_MAX;
  auto u = _mm_min_epu32(v.v, _mm_set1_epi32(max_val));
  return _mm_packus_epi32(u, u);
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::bitcast(t, word::promote(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, uint16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint32_t, T>> v) {
  Rebind<uint32_t, T> t1;
  static constexpr uint32_t max_val = INT32_MAX;
  Vec<decltype(t1)> u = _mm256_min_epu32(v.v, _mm256_set1_epi32(max_val));
  auto lo = word::lower(t1, u);
  auto hi = word::upper(t1, u);
  return _mm_packus_epi32(lo.v, hi.v);
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint32_t, T>> v) {
  static constexpr uint32_t max_val = INT32_MAX;
  #if VEC_WIDTH >= 512
  Rebind<uint32_t, T> t1;
  Vec<decltype(t1)> u = _mm512_min_epu32(v.v, _mm512_set1_epi32(max_val));
  auto lo = word::lower(t1, u).v;
  auto hi = word::upper(t1, u).v;
  #else
  Rebind<uint16_t, T> t1;
  auto lo = _mm256_min_epu32(lower(t1, v).v, _mm256_set1_epi32(max_val));
  auto hi = _mm256_min_epu32(upper(t1, v).v, _mm256_set1_epi32(max_val));
  #endif
  auto w = _mm256_packus_epi32(lo, hi);
  return _mm256_permute4x64_epi64(w, _MM_SHUFFLE(3, 1, 2, 0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint32_t, T>> v) {
  Rebind<uint32_t, T> t1;
  static constexpr uint32_t max_val = INT32_MAX;
  auto lo = _mm512_min_epu32(word::lower(t1, v).v, _mm512_set1_epi32(max_val));
  auto hi = _mm512_min_epu32(word::upper(t1, v).v, _mm512_set1_epi32(max_val));
  auto u = _mm512_packus_epi32(lo.v, hi.v);
  static const __m512i idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_epi64(u, idx);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                          uint16_t <=> float32_t                            //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<float32_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                           int16_t <=> float64_t                            //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                           int16_t <=> int64_t                              //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  return _mm_cvtepi16_epi64(v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  #if VEC_WIDTH >= 256
  return _mm256_cvtepi16_epi64(v.v);
  #else
  Tag<int64_t, 2> t1;
  Rebind<int16_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvtepi16_epi64(v.v);
  #else
  Tag<int64_t, 4> t1;
  Rebind<int16_t, T> t2;
  auto lo = word::promote(t1, lower(t2, v));
  auto hi = word::promote(t1, upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  Tag<int64_t, 8> t1;
  Rebind<int16_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                           int16_t <=> uint64_t                             //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int64_t, T> t1;
  return word::bitcast(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                          uint16_t <=> float64_t                            //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return demote(t, convert(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return convert(t, promote(t1, v));
}


/* ************************************************************************** */
//                           uint16_t <=> int64_t                             //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  return _mm_cvtepu16_epi64(v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  #if VEC_WIDTH >= 256
  return _mm256_cvtepu16_epi64(v.v);
  #else
  Tag<int64_t, 2> t1;
  Rebind<uint16_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvtepu16_epi64(v.v);
  #else
  Tag<int64_t, 4> t1;
  Rebind<uint16_t, T> t2;
  auto lo = word::promote(t1, lower(t2, v));
  auto hi = word::promote(t1, upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  Tag<int64_t, 8> t1;
  Rebind<uint16_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                          uint16_t <=> uint64_t                             //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint16_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<int64_t, T> t1;
  return word::bitcast(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                            int8_t <=> int16_t                              //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::N <= 8), TL_IF(is_any<TypeOf<T>, int8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int16_t, T>> v) {
  return _mm_packs_epi16(v.v, v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 8), TL_IF(is_any<TypeOf<T>, int16_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  return _mm_cvtepi8_epi16(v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int16_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  return _mm_packs_epi16(lo.v, hi.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  #if VEC_WIDTH >= 256
  return _mm256_cvtepi8_epi16(v.v);
  #else
  Tag<int16_t, 8> t1;
  Rebind<int8_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int16_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  auto u = _mm256_packs_epi16(lo.v, hi.v);
  return _mm256_permute4x64_epi64(u, _MM_SHUFFLE(3, 1, 2, 0));
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvtepi8_epi16(v.v);
  #else
  Tag<int16_t, 16> t1;
  Rebind<int8_t, T> t2;
  auto lo = word::promote(t1, lower(t2, v));
  auto hi = word::promote(t1, upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int16_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  auto u = _mm512_packs_epi16(lo.v, hi.v);
  static const __m512i idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_epi64(u, idx);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Tag<int16_t, 32> t1;
  Rebind<int8_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                          int16_t <=> uint16_t                             //
/* ************************************************************************** */

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int16_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<uint16_t, T>> v) {
  return v.v;
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint16_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<int16_t, T>> v) {
  return v.v;
}

/* ************************************************************************** */
//                           int8_t <=> uint16_t                              //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::N <= 8), TL_IF(is_any<TypeOf<T>, int8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint16_t, T>> v) {
  static constexpr uint16_t max_val = INT16_MAX;
  auto u = _mm_min_epu16(v.v, _mm_set1_epi16(max_val));
  return _mm_packs_epi16(u, u);
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint16_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int16_t, T> t1;
  return word::bitcast(t, word::promote(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<uint16_t, T> t1;
  static constexpr uint16_t max_val = INT16_MAX;
  Vec<decltype(t1)> u = _mm256_min_epu16(v.v, _mm256_set1_epi16(max_val));
  auto lo = word::lower(t1, u);
  auto hi = word::upper(t1, u);
  return _mm_packs_epi16(lo.v, hi.v);
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint16_t, T>> v) {
  static constexpr uint16_t max_val = INT16_MAX;
  #if VEC_WIDTH >= 512
  Rebind<uint16_t, T> t1;
  Vec<decltype(t1)> u = _mm512_min_epu16(v.v, _mm512_set1_epi16(max_val));
  auto lo = word::lower(t1, u).v;
  auto hi = word::upper(t1, u).v;
  #else
  Rebind<int8_t, T> t1;
  auto lo = _mm256_min_epu16(lower(t1, v).v, _mm256_set1_epi16(max_val));
  auto hi = _mm256_min_epu16(upper(t1, v).v, _mm256_set1_epi16(max_val));
  #endif
  auto w = _mm256_packs_epi16(lo, hi);
  return _mm256_permute4x64_epi64(w, _MM_SHUFFLE(3, 1, 2, 0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<uint16_t, T> t1;
  static constexpr uint16_t max_val = INT16_MAX;
  auto lo = _mm512_min_epu16(word::lower(t1, v).v, _mm512_set1_epi16(max_val));
  auto hi = _mm512_min_epu16(word::upper(t1, v).v, _mm512_set1_epi16(max_val));
  auto u = _mm512_packs_epi16(lo.v, hi.v);
  static const __m512i idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_epi64(u, idx);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                           uint8_t <=> int16_t                              //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::N <= 8), TL_IF(is_any<TypeOf<T>, uint8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int16_t, T>> v) {
  return _mm_packus_epi16(v.v, v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 8), TL_IF(is_any<TypeOf<T>, int16_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  return _mm_cvtepu8_epi16(v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int16_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  return _mm_packus_epi16(lo.v, hi.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  #if VEC_WIDTH >= 256
  return _mm256_cvtepu8_epi16(v.v);
  #else
  Tag<int16_t, 8> t1;
  Rebind<uint8_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int16_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  auto u = _mm256_packus_epi16(lo.v, hi.v);
  return _mm256_permute4x64_epi64(u, _MM_SHUFFLE(3, 1, 2, 0));
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvtepu8_epi16(v.v);
  #else
  Tag<int16_t, 16> t1;
  Rebind<uint8_t, T> t2;
  auto lo = word::promote(t1, lower(t2, v));
  auto hi = word::promote(t1, upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int16_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  auto u = _mm512_packus_epi16(lo.v, hi.v);
  static const __m512i idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_epi64(u, idx);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Tag<int16_t, 32> t1;
  Rebind<uint8_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                           uint8_t <=> uint16_t                             //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::N <= 8), TL_IF(is_any<TypeOf<T>, uint8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint16_t, T>> v) {
  static constexpr uint16_t max_val = INT16_MAX;
  auto u = _mm_min_epu16(v.v, _mm_set1_epi16(max_val));
  return _mm_packus_epi16(u, u);
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint16_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<int16_t, T> t1;
  return word::bitcast(t, word::promote(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<uint16_t, T> t1;
  static constexpr uint16_t max_val = INT16_MAX;
  Vec<decltype(t1)> u = _mm256_min_epu16(v.v, _mm256_set1_epi16(max_val));
  auto lo = word::lower(t1, u);
  auto hi = word::upper(t1, u);
  return _mm_packus_epi16(lo.v, hi.v);
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint16_t, T>> v) {
  static constexpr uint16_t max_val = INT16_MAX;
  #if VEC_WIDTH >= 512
  Rebind<uint16_t, T> t1;
  Vec<decltype(t1)> u = _mm512_min_epu16(v.v, _mm512_set1_epi16(max_val));
  auto lo = word::lower(t1, u).v;
  auto hi = word::upper(t1, u).v;
  #else
  Rebind<uint8_t, T> t1;
  auto lo = _mm256_min_epu16(lower(t1, v).v, _mm256_set1_epi16(max_val));
  auto hi = _mm256_min_epu16(upper(t1, v).v, _mm256_set1_epi16(max_val));
  #endif
  auto w = _mm256_packus_epi16(lo, hi);
  return _mm256_permute4x64_epi64(w, _MM_SHUFFLE(3, 1, 2, 0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<uint16_t, T> t1;
  static constexpr uint16_t max_val = INT16_MAX;
  auto lo = _mm512_min_epu16(word::lower(t1, v).v, _mm512_set1_epi16(max_val));
  auto hi = _mm512_min_epu16(word::upper(t1, v).v, _mm512_set1_epi16(max_val));
  auto u = _mm512_packus_epi16(lo.v, hi.v);
  static const __m512i idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_epi64(u, idx);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                           int8_t <=> float32_t                             //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<float32_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                            int8_t <=> int32_t                              //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int32_t, T>> v) {
  Rebind<int16_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  return _mm_cvtepi8_epi32(v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  #if VEC_WIDTH >= 256
  return _mm256_cvtepi8_epi32(v.v);
  #else
  Tag<int32_t, 4> t1;
  Rebind<int8_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvtepi8_epi32(v.v);
  #else
  Tag<int32_t, 8> t1;
  Rebind<int8_t, T> t2;
  auto lo = word::promote(t1, lower(t2, v));
  auto hi = word::promote(t1, upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Tag<int32_t, 16> t1;
  Rebind<int8_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                           int8_t <=> uint32_t                              //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint32_t, T>> v) {
  Rebind<int16_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::bitcast(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                           uint8_t <=> float32_t                            //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<float32_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                           uint8_t <=> int32_t                              //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int32_t, T>> v) {
  Rebind<int16_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  return _mm_cvtepu8_epi32(v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  #if VEC_WIDTH >= 256
  return _mm256_cvtepu8_epi32(v.v);
  #else
  Tag<int32_t, 4> t1;
  Rebind<uint8_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvtepu8_epi32(v.v);
  #else
  Tag<int32_t, 8> t1;
  Rebind<uint8_t, T> t2;
  auto lo = word::promote(t1, lower(t2, v));
  auto hi = word::promote(t1, upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Tag<int32_t, 16> t1;
  Rebind<uint8_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                           uint8_t <=> uint32_t                             //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint32_t, T>> v) {
  Rebind<uint16_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint32_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::bitcast(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                           int8_t <=> float64_t                             //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                            int8_t <=> int64_t                              //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  return _mm_cvtepi8_epi64(v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  #if VEC_WIDTH >= 256
  return _mm256_cvtepi8_epi64(v.v);
  #else
  Tag<int64_t, 2> t1;
  Rebind<int8_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvtepi8_epi64(v.v);
  #else
  Tag<int64_t, 4> t1;
  Rebind<int8_t, T> t2;
  auto lo = word::promote(t1, lower(t2, v));
  auto hi = word::promote(t1, upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Tag<int64_t, 8> t1;
  Rebind<int8_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                           int8_t <=> uint64_t                              //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int64_t, T> t1;
  return word::bitcast(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                           uint8_t <=> float64_t                            //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                           uint8_t <=> int64_t                              //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  return _mm_cvtepu8_epi64(v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  #if VEC_WIDTH >= 256
  return _mm256_cvtepu8_epi64(v.v);
  #else
  Tag<int64_t, 2> t1;
  Rebind<uint8_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  #if VEC_WIDTH >= 512
  return _mm512_cvtepu8_epi64(v.v);
  #else
  Tag<int64_t, 4> t1;
  Rebind<uint8_t, T> t2;
  auto lo = word::promote(t1, lower(t2, v));
  auto hi = word::promote(t1, upper(t2, v));
  return word::concat(t, lo, hi);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Tag<int64_t, 8> t1;
  Rebind<uint8_t, T> t2;
  auto lo = word::promote(t1, word::lower(t2, v));
  auto hi = word::promote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                           uint8_t <=> uint64_t                             //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint8_t>)>
TLV_INLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint64_t>)>
TLV_INLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<int64_t, T> t1;
  return word::bitcast(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                            int8_t <=> uint8_t                              //
/* ************************************************************************** */

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, int8_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<uint8_t, T>> v) {
  return v.v;
}

template <TLV_DECL_TAG(T), TL_IF(is_any<TypeOf<T>, uint8_t>)>
TLV_INLINE Vec<T> convert(T t, Vec<Rebind<int8_t, T>> v) {
  return v.v;
}

} // namespace word
} // namespace ct::tl::vec::CPU_CAPABILITY
//@formatter:on

#endif //CTORCH_X86_CONVERSIONS_H
