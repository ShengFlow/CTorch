//
// Created by renyz on 2026/3/28.
//

#ifndef CTORCH_X86_BASIC_H
#define CTORCH_X86_BASIC_H

#include "tl/cpu/VecBase.h"
#include "tl/cpu/Capabilities.h"
#include "tl/util/TypeTraits.h"

#include "tl/cpu/impl/x86_Types.h"

//@formatter:off
namespace ct::tl::vec::CPU_CAPABILITY {
namespace word {

/* ************************************************************************** */
//                             Mask constructors                              //
/* ************************************************************************** */
#ifdef HAS_AVX512DQ
template <TLV_DECL_TAG(T), TL_IF(T::N <= 8)>
TLV_INLINE Mask<T> mfill(T t, bool value) {
  /* we do not guarantee that padded elements are zero */
  uint32_t x = value ? 0xffffffffu : 0x00;
  return _cvtu32_mask8(x);
}
template <TLV_DECL_TAG(T), TL_IF(T::N == 16)>
TLV_INLINE Mask<T> mfill(T t, bool value) {
  uint32_t x = value ? 0xffffffffu : 0x00;
  return _cvtu32_mask16(x);
}
template <TLV_DECL_TAG(T), TL_IF(T::N == 32)>
TLV_INLINE Mask<T> mfill(T t, bool value) {
  uint32_t x = value ? 0xffffffffu : 0x00;
  return _cvtu32_mask32(x);
}
template <TLV_DECL_TAG(T), TL_IF(T::N == 64)>
TLV_INLINE Mask<T> mfill(T t, bool value) {
  uint64_t x = value ? 0xffffffffffffffffLLu : 0x00;
  return _cvtu64_mask64(x);
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 8)>
TLV_INLINE Mask<T> mwhilelt(T t, nint_t a, nint_t b) {
  nint_t end = std::clamp<nint_t>(b - a, 0, 8);
  return _cvtu32_mask8(tailing_mask(int32_t(end)));
}
template <TLV_DECL_TAG(T), TL_IF(T::N == 16)>
TLV_INLINE Mask<T> mwhilelt(T t, nint_t a, nint_t b) {
  nint_t end = std::clamp<nint_t>(b - a, 0, 16);
  return _cvtu32_mask16(tailing_mask(int32_t(end)));
}
template <TLV_DECL_TAG(T), TL_IF(T::N == 32)>
TLV_INLINE Mask<T> mwhilelt(T t, nint_t a, nint_t b) {
  nint_t end = std::clamp<nint_t>(b - a, 0, 32);
  return _cvtu32_mask32(tailing_mask(int32_t(end)));
}
template <TLV_DECL_TAG(T), TL_IF(T::N == 64)>
TLV_INLINE Mask<T> mwhilelt(T t, nint_t a, nint_t b) {
  nint_t end = std::clamp<nint_t>(b - a, 0, 64);
  return _cvtu64_mask64(tailing_mask(int64_t(end)));
}

template <TLV_DECL_TAG(T), TL_IF(T::N <= 8)>
TLV_INLINE Mask<T> mwhilege(T t, nint_t a, nint_t b) {
  nint_t end = std::clamp<nint_t>(b - a, 0, 8);
  return _cvtu32_mask8(~tailing_mask(int32_t(end)));
}
template <TLV_DECL_TAG(T), TL_IF(T::N == 16)>
TLV_INLINE Mask<T> mwhilege(T t, nint_t a, nint_t b) {
  nint_t end = std::clamp<nint_t>(b - a, 0, 16);
  return _cvtu32_mask16(~tailing_mask(int32_t(end)));
}
template <TLV_DECL_TAG(T), TL_IF(T::N == 32)>
TLV_INLINE Mask<T> mwhilege(T t, nint_t a, nint_t b) {
  nint_t end = std::clamp<nint_t>(b - a, 0, 32);
  return _cvtu32_mask32(~tailing_mask(int32_t(end)));
}
template <TLV_DECL_TAG(T), TL_IF(T::N == 64)>
TLV_INLINE Mask<T> mwhilege(T t, nint_t a, nint_t b) {
  nint_t end = std::clamp<nint_t>(b - a, 0, 64);
  return _cvtu64_mask64(~tailing_mask(int64_t(end)));
}
#else // HAS_AVX512DQ
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16)>
TLV_INLINE Mask<T> mfill(T t, bool value) {
  /* we do not guarantee that padded elements are zero */
  int64_t x = value ? int64_t(-1) : 0;
  return _mm_set1_epi64x(x);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(sizeof(TypeOf<T>) == 1)>
TLV_INLINE Mask<T> mwhilelt(T t, nint_t a, nint_t b) {
  static const __m128i idx = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  int8_t diff = (int8_t)std::clamp<nint_t>(b - a, INT8_MIN, INT8_MAX);
  auto end = _mm_set1_epi8(diff);
  return _mm_cmplt_epi8(idx, end);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(sizeof(TypeOf<T>) == 2)>
TLV_INLINE Mask<T> mwhilelt(T t, nint_t a, nint_t b) {
  static const __m128i idx = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
  int16_t diff = (int16_t)std::clamp<nint_t>(b - a, INT16_MIN, INT16_MAX);
  auto end = _mm_set1_epi16(diff);
  return _mm_cmplt_epi16(idx, end);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(sizeof(TypeOf<T>) == 4)>
TLV_INLINE Mask<T> mwhilelt(T t, nint_t a, nint_t b) {
  static const __m128i idx = _mm_setr_epi32(0, 1, 2, 3);
  int32_t diff = (int32_t)std::clamp<nint_t>(b - a, INT32_MIN, INT32_MAX);
  auto end = _mm_set1_epi32(diff);
  return _mm_cmplt_epi32(idx, end);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(sizeof(TypeOf<T>) == 8)>
TLV_INLINE Mask<T> mwhilelt(T t, nint_t a, nint_t b) {
  static const __m128i idx = _mm_setr_epi32(0, 0, 1, 1);
  int32_t diff = (int32_t)std::clamp<nint_t>(b - a, INT32_MIN, INT32_MAX);
  auto end = _mm_set1_epi32(diff);
  return _mm_cmplt_epi32(idx, end);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(sizeof(TypeOf<T>) == 1)>
TLV_INLINE Mask<T> mwhilege(T t, nint_t a, nint_t b) {
  static const __m128i idx = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
  int8_t diff = (int8_t)std::clamp<nint_t>(b - a, INT8_MIN, INT8_MAX);
  auto start = _mm_set1_epi8(diff);
  return _mm_cmplt_epi8(start, idx);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(sizeof(TypeOf<T>) == 2)>
TLV_INLINE Mask<T> mwhilege(T t, nint_t a, nint_t b) {
  static const __m128i idx = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
  int16_t diff = (int16_t)std::clamp<nint_t>(b - a, INT16_MIN, INT16_MAX);
  auto start = _mm_set1_epi16(diff);
  return _mm_cmplt_epi16(start, idx);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(sizeof(TypeOf<T>) == 4)>
TLV_INLINE Mask<T> mwhilege(T t, nint_t a, nint_t b) {
  static const __m128i idx = _mm_setr_epi32(1, 2, 3, 4);
  int32_t diff = (int32_t)std::clamp<nint_t>(b - a, INT32_MIN, INT32_MAX);
  auto start = _mm_set1_epi32(diff);
  return _mm_cmplt_epi32(start, idx);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(sizeof(TypeOf<T>) == 8)>
TLV_INLINE Mask<T> mwhilege(T t, nint_t a, nint_t b) {
  static const __m128i idx = _mm_setr_epi32(1, 1, 2, 2);
  int32_t diff = (int32_t)std::clamp<nint_t>(b - a, INT32_MIN, INT32_MAX);
  auto start = _mm_set1_epi32(diff);
  return _mm_cmplt_epi32(start, idx);
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32)>
TLV_INLINE Mask<T> mfill(T t, bool value) {
  /* we do not guarantee that padded elements are zero */
  int64_t x = value ? int64_t(-1) : 0;
  return _mm256_set1_epi64x(x);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(sizeof(TypeOf<T>) == 1)>
TLV_INLINE Mask<T> mwhilelt(T t, nint_t a, nint_t b) {
  static const __m256i idx = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                           16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
  int8_t diff = (int8_t)std::clamp<nint_t>(b - a, INT8_MIN, INT8_MAX);
  auto end = _mm256_set1_epi8(diff);
  return _mm256_cmpgt_epi8(end, idx);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(sizeof(TypeOf<T>) == 2)>
TLV_INLINE Mask<T> mwhilelt(T t, nint_t a, nint_t b) {
  static const __m256i idx = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  int16_t diff = (int16_t)std::clamp<nint_t>(b - a, INT16_MIN, INT16_MAX);
  auto end = _mm256_set1_epi16(diff);
  return _mm256_cmpgt_epi16(end, idx);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(sizeof(TypeOf<T>) == 4)>
TLV_INLINE Mask<T> mwhilelt(T t, nint_t a, nint_t b) {
  static const __m256i idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  int32_t diff = (int32_t)std::clamp<nint_t>(b - a, INT32_MIN, INT32_MAX);
  auto end = _mm256_set1_epi32(diff);
  return _mm256_cmpgt_epi32(end, idx);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(sizeof(TypeOf<T>) == 8)>
TLV_INLINE Mask<T> mwhilelt(T t, nint_t a, nint_t b) {
  static const __m256i idx = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
  int32_t diff = (int32_t)std::clamp<nint_t>(b - a, INT32_MIN, INT32_MAX);
  auto end = _mm256_set1_epi32(diff);
  return _mm256_cmpgt_epi32(end, idx);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(sizeof(TypeOf<T>) == 1)>
TLV_INLINE Mask<T> mwhilege(T t, nint_t a, nint_t b) {
  static const __m256i idx = _mm256_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                              17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);
  int8_t diff = (int8_t)std::clamp<nint_t>(b - a, INT8_MIN, INT8_MAX);
  auto start = _mm256_set1_epi8(diff);
  return _mm256_cmpgt_epi8(idx, start);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(sizeof(TypeOf<T>) == 2)>
TLV_INLINE Mask<T> mwhilege(T t, nint_t a, nint_t b) {
  static const __m256i idx = _mm256_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
  int16_t diff = (int16_t)std::clamp<nint_t>(b - a, INT16_MIN, INT16_MAX);
  auto start = _mm256_set1_epi16(diff);
  return _mm256_cmpgt_epi16(idx, start);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(sizeof(TypeOf<T>) == 4)>
TLV_INLINE Mask<T> mwhilege(T t, nint_t a, nint_t b) {
  static const __m256i idx = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
  int32_t diff = (int32_t)std::clamp<nint_t>(b - a, INT32_MIN, INT32_MAX);
  auto start = _mm256_set1_epi32(diff);
  return _mm256_cmpgt_epi32(idx, start);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(sizeof(TypeOf<T>) == 8)>
TLV_INLINE Mask<T> mwhilege(T t, nint_t a, nint_t b) {
  static const __m256i idx = _mm256_setr_epi32(1, 1, 2, 2, 3, 3, 4, 4);
  int32_t diff = (int32_t)std::clamp<nint_t>(b - a, INT32_MIN, INT32_MAX);
  auto start = _mm256_set1_epi32(diff);
  return _mm256_cmpgt_epi32(idx, start);
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
  #error "Unreachable"
#endif // VEC_WIDTH >= 512
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                            Mask Bit Operation                              //
/* ************************************************************************** */
#ifdef HAS_AVX512DQ
template <TLV_DECL_MASK(M), TL_IF(M::N <= 8)>
TLV_INLINE M bit_and(M a, M b) {
  return _kand_mask8(a.v, b.v);
}
template <TLV_DECL_MASK(M), TL_IF(M::N == 16)>
TLV_INLINE M bit_and(M a, M b) {
  return _kand_mask16(a.v, b.v);
}
template <TLV_DECL_MASK(M), TL_IF(M::N == 32)>
TLV_INLINE M bit_and(M a, M b) {
  return _kand_mask32(a.v, b.v);
}
template <TLV_DECL_MASK(M), TL_IF(M::N == 64)>
TLV_INLINE M bit_and(M a, M b) {
  return _kand_mask64(a.v, b.v);
}

template <TLV_DECL_MASK(M), TL_IF(M::N <= 8)>
TLV_INLINE M bit_or(M a, M b) {
  return _kor_mask8(a.v, b.v);
}
template <TLV_DECL_MASK(M), TL_IF(M::N == 16)>
TLV_INLINE M bit_or(M a, M b) {
  return _kor_mask16(a.v, b.v);
}
template <TLV_DECL_MASK(M), TL_IF(M::N == 32)>
TLV_INLINE M bit_or(M a, M b) {
  return _kor_mask32(a.v, b.v);
}
template <TLV_DECL_MASK(M), TL_IF(M::N == 64)>
TLV_INLINE M bit_or(M a, M b) {
  return _kor_mask64(a.v, b.v);
}

template <TLV_DECL_MASK(M), TL_IF(M::N <= 8)>
TLV_INLINE M bit_xor(M a, M b) {
  return _kxor_mask8(a.v, b.v);
}
template <TLV_DECL_MASK(M), TL_IF(M::N == 16)>
TLV_INLINE M bit_xor(M a, M b) {
  return _kxor_mask16(a.v, b.v);
}
template <TLV_DECL_MASK(M), TL_IF(M::N == 32)>
TLV_INLINE M bit_xor(M a, M b) {
  return _kxor_mask32(a.v, b.v);
}
template <TLV_DECL_MASK(M), TL_IF(M::N == 64)>
TLV_INLINE M bit_xor(M a, M b) {
  return _kxor_mask64(a.v, b.v);
}

template <TLV_DECL_MASK(M), TL_IF(M::N <= 8)>
TLV_INLINE M bit_andnot(M a, M b) {
  return _kandn_mask8(a.v, b.v);
}
template <TLV_DECL_MASK(M), TL_IF(M::N == 16)>
TLV_INLINE M bit_andnot(M a, M b) {
  return _kandn_mask16(a.v, b.v);
}
template <TLV_DECL_MASK(M), TL_IF(M::N == 32)>
TLV_INLINE M bit_andnot(M a, M b) {
  return _kandn_mask32(a.v, b.v);
}
template <TLV_DECL_MASK(M), TL_IF(M::N == 64)>
TLV_INLINE M bit_andnot(M a, M b) {
  return _kandn_mask64(a.v, b.v);
}

template <TLV_DECL_MASK(M), TL_IF(M::N <= 8)>
TLV_INLINE M bit_not(M a, M b) {
  return _knot_mask8(a.v, b.v);
}
template <TLV_DECL_MASK(M), TL_IF(M::N == 16)>
TLV_INLINE M bit_not(M a, M b) {
  return _knot_mask16(a.v, b.v);
}
template <TLV_DECL_MASK(M), TL_IF(M::N == 32)>
TLV_INLINE M bit_not(M a, M b) {
  return _knot_mask32(a.v, b.v);
}
template <TLV_DECL_MASK(M), TL_IF(M::N == 64)>
TLV_INLINE M bit_not(M a, M b) {
  return _knot_mask64(a.v, b.v);
}
#else // HAS_AVX512DQ
template <TLV_DECL_MASK(M), TL_IF(M::Bytes <= 16)>
TLV_INLINE M bit_and(M a, M b) {
  return _mm_and_si128(a.v, b.v);
}
template <TLV_DECL_MASK(M), TL_IF(M::Bytes <= 16)>
TLV_INLINE M bit_or(M a, M b) {
  return _mm_or_si128(a.v, b.v);
}
template <TLV_DECL_MASK(M), TL_IF(M::Bytes <= 16)>
TLV_INLINE M bit_xor(M a, M b) {
  return _mm_xor_si128(a.v, b.v);
}
template <TLV_DECL_MASK(M), TL_IF(M::Bytes <= 16)>
TLV_INLINE M bit_andnot(M a, M b) {
  return _mm_andnot_si128(a.v, b.v);
}
template <TLV_DECL_MASK(M), TL_IF(M::Bytes <= 16)>
TLV_INLINE M bit_not(M m) {
  return _mm_xor_si128(m.v, _mm_set1_epi32(-1));
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                               Mask selection                               //
/* ************************************************************************** */
#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  return _mm_mask_blend_ps(m.v, v0.v, v1.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  return _mm_mask_blend_pd(m.v, v0.v, v1.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  return _mm_mask_blend_epi8(m.v, v0.v, v1.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, bfloat16_t, float16_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  return _mm_mask_blend_epi16(m.v, v0.v, v1.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  return _mm_mask_blend_epi32(m.v, v0.v, v1.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  return _mm_mask_blend_epi64(m.v, v0.v, v1.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  return _mm256_mask_blend_ps(m.v, v0.v, v1.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  return _mm256_mask_blend_pd(m.v, v0.v, v1.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  return _mm256_mask_blend_epi8(m.v, v0.v, v1.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, bfloat16_t, float16_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  return _mm256_mask_blend_epi16(m.v, v0.v, v1.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  return _mm256_mask_blend_epi32(m.v, v0.v, v1.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  return _mm256_mask_blend_epi64(m.v, v0.v, v1.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  return _mm512_mask_blend_ps(m.v, v0.v, v1.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  return _mm512_mask_blend_pd(m.v, v0.v, v1.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  return _mm512_mask_blend_epi8(m.v, v0.v, v1.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, bfloat16_t, float16_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  return _mm512_mask_blend_epi16(m.v, v0.v, v1.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  return _mm512_mask_blend_epi32(m.v, v0.v, v1.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  return _mm512_mask_blend_epi64(m.v, v0.v, v1.v);
}
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  return _mm_blendv_ps(v0.v, v1.v, _mm_castsi128_ps(m.v));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  return _mm_blendv_pd(v0.v, v1.v, _mm_castsi128_pd(m.v));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_none<TypeOf<T>, float32_t, float64_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  // assuming that mask is sanitized (all bits corresponding to active element are 1)
  return _mm_blendv_epi8(v0.v, v1.v, m.v);
}

#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  return _mm256_blendv_ps(v0.v, v1.v, _mm256_castsi256_ps(m.v));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  return _mm256_blendv_pd(v0.v, v1.v, _mm256_castsi256_pd(m.v));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_none<TypeOf<T>, float32_t, float64_t>)>
TLV_INLINE V blend(V v0, Mask<T> m, V v1) {
  // assuming that mask is sanitized (all bits corresponding to active element are 1)
  return _mm256_blendv_epi8(v0.v, v1.v, m.v);
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
  #error "Unreachable"
#endif // VEC_WIDTH >= 512
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                                Bit Casts                                   //
/* ************************************************************************** */
namespace details {
template <typename T> struct BitCastFromInt128 {
  TLV_INLINE __m128i operator()(__m128i v) { return v; }
};
template <> struct BitCastFromInt128<float32_t> {
  TLV_INLINE __m128 operator()(__m128i v) { return _mm_castsi128_ps(v); }
};
template <> struct BitCastFromInt128<float64_t> {
  TLV_INLINE __m128d operator()(__m128i v) { return _mm_castsi128_pd(v); }
};
template <typename T> struct BitCastToInt128 {
  TLV_INLINE __m128i operator()(__m128i v) { return v; }
};
template <> struct BitCastToInt128<float32_t> {
  TLV_INLINE __m128i operator()(__m128 v) { return _mm_castps_si128(v); }
};
template <> struct BitCastToInt128<float64_t> {
  TLV_INLINE __m128i operator()(__m128d v) { return _mm_castpd_si128(v); }
};

template <typename T> TLV_INLINE
x86::RegType<uint8_t, 16> bitcast_to_int(x86::RegType<T, 16 / sizeof(T)> v) { return BitCastToInt128<T>()(v.v); }
template <typename T> TLV_INLINE
x86::RegType<T, 16 / sizeof(T)> bitcast_from_int(x86::RegType<uint8_t, 16> v) { return BitCastFromInt128<T>()(v.v); }

#if VEC_WIDTH >= 256
template <typename T> struct BitCastFromInt256 {
  TLV_INLINE __m256i operator()(__m256i v) { return v; }
};
template <> struct BitCastFromInt256<float32_t> {
  TLV_INLINE __m256 operator()(__m256i v) { return _mm256_castsi256_ps(v); }
};
template <> struct BitCastFromInt256<float64_t> {
  TLV_INLINE __m256d operator()(__m256i v) { return _mm256_castsi256_pd(v); }
};
template <typename T> struct BitCastToInt256 {
  TLV_INLINE __m256i operator()(__m256i v) { return v; }
};
template <> struct BitCastToInt256<float32_t> {
  TLV_INLINE __m256i operator()(__m256 v) { return _mm256_castps_si256(v); }
};
template <> struct BitCastToInt256<float64_t> {
  TLV_INLINE __m256i operator()(__m256d v) { return _mm256_castpd_si256(v); }
};

template <typename T> TLV_INLINE
x86::RegType<uint8_t, 32> bitcast_to_int(x86::RegType<T, 32 / sizeof(T)> v) { return BitCastToInt256<T>()(v.v); }
template <typename T> TLV_INLINE
x86::RegType<T, 32 / sizeof(T)> bitcast_from_int(x86::RegType<uint8_t, 32> v) { return BitCastFromInt256<T>()(v.v); }
#endif // VEC_WIDTH >= 256


#if VEC_WIDTH >= 512
template <typename T> struct BitCastFromInt512 {
  TLV_INLINE __m512i operator()(__m512i v) { return v; }
};
template <> struct BitCastFromInt512<float32_t> {
  TLV_INLINE __m512 operator()(__m512i v) { return _mm512_castsi512_ps(v); }
};
template <> struct BitCastFromInt512<float64_t> {
  TLV_INLINE __m512d operator()(__m512i v) { return _mm512_castsi512_pd(v); }
};
template <typename T> struct BitCastToInt512 {
  TLV_INLINE __m512i operator()(__m512i v) { return v; }
};
template <> struct BitCastToInt512<float32_t> {
  TLV_INLINE __m512i operator()(__m512 v) { return _mm512_castps_si512(v); }
};
template <> struct BitCastToInt512<float64_t> {
  TLV_INLINE __m512i operator()(__m512d v) { return _mm512_castpd_si512(v); }
};

template <typename T> TLV_INLINE
x86::RegType<uint8_t, 64> bitcast_to_int(x86::RegType<T, 64 / sizeof(T)> v) { return BitCastToInt512<T>()(v.v); }
template <typename T> TLV_INLINE
x86::RegType<T, 64 / sizeof(T)> bitcast_from_int(x86::RegType<uint8_t, 64> v) { return BitCastFromInt512<T>()(v.v); }
#endif // VEC_WIDTH >= 512

} // namespace details

template <TLV_DECL_TAG(T), TLV_DECL_VEC(V), TL_IF(sizeof(V) == sizeof(Vec<T>)), TL_IF(sizeof(V) <= VEC_WIDTH / 8)>
TLV_INLINE Vec<T> bitcast(T t, V v) {
  return details::bitcast_from_int<TypeOf<T>>(details::bitcast_to_int(v));
}

template <TLV_DECL_TAG(T), TLV_DECL_VEC(V), TL_IF(sizeof(V) == sizeof(Vec<T>)), TL_IF(sizeof(V) == 2 * VEC_WIDTH / 8)>
TLV_INLINE Vec<T> bitcast(T t, V v) {
  Tag<TypeOf<T>, T::N / 2> t1;
  return Vec<T>{ word::bitcast(t1, v[0]), word::bitcast(t1, v[1]) };
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TLV_DECL_VEC(V), TL_IF(sizeof(V) == 16 && sizeof(Vec<T>) == 32)>
TLV_INLINE Vec<T> bitcast(T t, V v) {
  return details::bitcast_from_int<TypeOf<T>>(
      x86::RegType<uint8_t, 32>{_mm256_castsi128_si256(details::bitcast_to_int(v).v)});
}
template <TLV_DECL_TAG(T), TLV_DECL_VEC(V), TL_IF(sizeof(V) == 32 && sizeof(Vec<T>) == 16)>
TLV_INLINE Vec<T> bitcast(T t, V v) {
  return details::bitcast_from_int<TypeOf<T>>(
      x86::RegType<uint8_t, 16>{_mm256_castsi256_si128(details::bitcast_to_int(v).v)});
}
#endif // VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TLV_DECL_VEC(V), TL_IF(sizeof(V) == 16 && sizeof(Vec<T>) == 64)>
TLV_INLINE Vec<T> bitcast(T t, V v) {
  return details::bitcast_from_int<TypeOf<T>>(
      x86::RegType<uint8_t, 64>{_mm512_castsi128_si512(details::bitcast_to_int(v).v)});
}
template <TLV_DECL_TAG(T), TLV_DECL_VEC(V), TL_IF(sizeof(V) == 64 && sizeof(Vec<T>) == 16)>
TLV_INLINE Vec<T> bitcast(T t, V v) {
  return details::bitcast_from_int<TypeOf<T>>(
      x86::RegType<uint8_t, 16>{_mm512_castsi512_si128(details::bitcast_to_int(v).v)});
}
template <TLV_DECL_TAG(T), TLV_DECL_VEC(V), TL_IF(sizeof(V) == 32 && sizeof(Vec<T>) == 64)>
TLV_INLINE Vec<T> bitcast(T t, V v) {
  return details::bitcast_from_int<TypeOf<T>>(
      x86::RegType<uint8_t, 64>{_mm512_castsi256_si512(details::bitcast_to_int(v).v)});
}
template <TLV_DECL_TAG(T), TLV_DECL_VEC(V), TL_IF(sizeof(V) == 64 && sizeof(Vec<T>) == 32)>
TLV_INLINE Vec<T> bitcast(T t, V v) {
  return details::bitcast_from_int<TypeOf<T>>(
      x86::RegType<uint8_t, 32>{_mm512_castsi512_si256(details::bitcast_to_int(v).v)});
}


/* ************************************************************************** */
//                           Intra-block Shuffle                              //
/* ************************************************************************** */
namespace details {
template <nint_t N, int... Is>
struct StaticIndexChecker;

template <nint_t N>
struct StaticIndexChecker<N> {
  TLV_INLINE void operator()() {}
};

template <nint_t N, int I0, int... Is>
struct StaticIndexChecker<N, I0, Is...> {
  TLV_INLINE void operator()() {
    static_assert(0 <= I0 && I0 < N, "Index out of range");
    StaticIndexChecker<N, Is...>()();
  }
};

template <nint_t N, nint_t J, typename... Is>
struct RuntimeIndexChecker;

template <nint_t N, nint_t J>
struct RuntimeIndexChecker<N, J> {
  TLV_INLINE void operator()() {}
};

template <nint_t N, nint_t J, typename I0, typename... Is>
struct RuntimeIndexChecker<N, J, I0, Is...> {
  TLV_INLINE void operator()(I0 i0, Is... is) {
    CT_ASSERT(0 <= i0 && i0 < N, "Index #%zd out of range: %d !in 0:%zd", N - J - 1, int(i0), N);
    RuntimeIndexChecker<N, J + 1, Is...>()(is...);
  }
};

template <int... Is>
TLV_INLINE void assert_index() { StaticIndexChecker<sizeof...(Is), Is...>()(); }

template <typename... Is>
TLV_INLINE void assert_index(Is... is) { RuntimeIndexChecker<sizeof...(is), 0, Is...>()(is...); }
} // namespace details

/* ******************************** float32_t ******************************* */

template <int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<float32_t, 4>> local_shuf(Vec<Tag<float32_t, 4>> v) {
  details::assert_index<I3, I2, I1, I0>();
  return _mm_shuffle_ps(v.v, v.v, _MM_SHUFFLE(I3, I2, I1, I0));
}

TLV_INLINE Vec<Tag<float32_t, 4>> local_shuf(Vec<Tag<float32_t, 4>> v, Vec<Tag<int32_t, 4>> i) {
  return _mm_permutevar_ps(v.v, i.v);
}

TLV_INLINE Vec<Tag<float32_t, 4>> local_shuf(Vec<Tag<float32_t, 4>> v, int i3, int i2, int i1, int i0) {
  details::assert_index(i3, i2, i1, i0);
  return word::local_shuf(v, _mm_set_epi32(i3, i2, i1, i0));
}

#if VEC_WIDTH >= 256
template <int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<float32_t, 8>> local_shuf(Vec<Tag<float32_t, 8>> v) {
  details::assert_index<I3, I2, I1, I0>();
  return _mm256_shuffle_ps(v.v, v.v, _MM_SHUFFLE(I3, I2, I1, I0));
}

TLV_INLINE Vec<Tag<float32_t, 8>> local_shuf(Vec<Tag<float32_t, 8>> v, Vec<Tag<int32_t, 8>> i) {
  return _mm256_permutevar_ps(v.v, i.v);
}

TLV_INLINE Vec<Tag<float32_t, 8>> local_shuf(Vec<Tag<float32_t, 8>> v, int i3, int i2, int i1, int i0) {
  details::assert_index(i3, i2, i1, i0);
  return word::local_shuf(v, _mm256_set_epi32(i3, i2, i1, i0, i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<float32_t, 16>> local_shuf(Vec<Tag<float32_t, 16>> v) {
  details::assert_index<I3, I2, I1, I0>();
  return _mm512_shuffle_ps(v.v, v.v, _MM_SHUFFLE(I3, I2, I1, I0));
}

TLV_INLINE Vec<Tag<float32_t, 16>> local_shuf(Vec<Tag<float32_t, 16>> v, Vec<Tag<int32_t, 16>> i) {
  return _mm512_permutevar_ps(v.v, i.v);
}

TLV_INLINE Vec<Tag<float32_t, 16>> local_shuf(Vec<Tag<float32_t, 16>> v, int i3, int i2, int i1, int i0) {
  details::assert_index(i3, i2, i1, i0);
  return word::local_shuf(v, _mm512_set_epi32(i3, i2, i1, i0, i3, i2, i1, i0, i3, i2, i1, i0, i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 512

/* ********************************* int32_t ******************************** */

template <int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<int32_t, 4>> local_shuf(Vec<Tag<int32_t, 4>> v) {
  details::assert_index<I3, I2, I1, I0>();
  return _mm_shuffle_epi32(v.v, _MM_SHUFFLE(I3, I2, I1, I0));
}

TLV_INLINE Vec<Tag<int32_t, 4>> local_shuf(Vec<Tag<int32_t, 4>> v, Vec<Tag<int32_t, 4>> i) {
  Tag<float32_t, 4> t1; Tag<int32_t, 4> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

TLV_INLINE Vec<Tag<int32_t, 4>> local_shuf(Vec<Tag<int32_t, 4>> v, int i3, int i2, int i1, int i0) {
  Tag<float32_t, 4> t1; Tag<int32_t, 4> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i3, i2, i1, i0));
}

#if VEC_WIDTH >= 256
template <int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<int32_t, 8>> local_shuf(Vec<Tag<int32_t, 8>> v) {
  details::assert_index<I3, I2, I1, I0>();
  return _mm256_shuffle_epi32(v.v, _MM_SHUFFLE(I3, I2, I1, I0));
}

TLV_INLINE Vec<Tag<int32_t, 8>> local_shuf(Vec<Tag<int32_t, 8>> v, Vec<Tag<int32_t, 8>> i) {
  Tag<float32_t, 8> t1; Tag<int32_t, 8> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

TLV_INLINE Vec<Tag<int32_t, 8>> local_shuf(Vec<Tag<int32_t, 8>> v, int i3, int i2, int i1, int i0) {
  details::assert_index(i3, i2, i1, i0);
  return word::local_shuf(v, _mm256_set_epi32(i3, i2, i1, i0, i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<int32_t, 16>> local_shuf(Vec<Tag<int32_t, 16>> v) {
  details::assert_index<I3, I2, I1, I0>();
  return _mm512_shuffle_epi32(v.v, _MM_SHUFFLE(I3, I2, I1, I0));
}

TLV_INLINE Vec<Tag<int32_t, 16>> local_shuf(Vec<Tag<int32_t, 16>> v, Vec<Tag<int32_t, 16>> i) {
  Tag<float32_t, 16> t1; Tag<int32_t, 16> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

TLV_INLINE Vec<Tag<int32_t, 16>> local_shuf(Vec<Tag<int32_t, 16>> v, int i3, int i2, int i1, int i0) {
  details::assert_index(i3, i2, i1, i0);
  return word::local_shuf(v, _mm512_set_epi32(i3, i2, i1, i0, i3, i2, i1, i0, i3, i2, i1, i0, i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 512

/* ********************************* uint32_t ******************************* */

template <int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<uint32_t, 4>> local_shuf(Vec<Tag<uint32_t, 4>> v) {
  Tag<int32_t, 4> t1; Tag<uint32_t, 4> t2;
  return word::bitcast(t2, word::local_shuf<I3, I2, I1, I0>(word::bitcast(t1, v)));
}

TLV_INLINE Vec<Tag<uint32_t, 4>> local_shuf(Vec<Tag<uint32_t, 4>> v, Vec<Tag<int32_t, 4>> i) {
  Tag<int32_t, 4> t1; Tag<uint32_t, 4> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

TLV_INLINE Vec<Tag<uint32_t, 4>> local_shuf(Vec<Tag<uint32_t, 4>> v, int i3, int i2, int i1, int i0) {
  Tag<int32_t, 4> t1; Tag<uint32_t, 4> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i3, i2, i1, i0));
}

#if VEC_WIDTH >= 256
template <int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<uint32_t, 8>> local_shuf(Vec<Tag<uint32_t, 8>> v) {
  Tag<int32_t, 8> t1; Tag<uint32_t, 8> t2;
  return word::bitcast(t2, word::local_shuf<I3, I2, I1, I0>(bitcast(t1, v)));
}

TLV_INLINE Vec<Tag<uint32_t, 8>> local_shuf(Vec<Tag<uint32_t, 8>> v, Vec<Tag<int32_t, 8>> i) {
  Tag<int32_t, 8> t1; Tag<uint32_t, 8> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

TLV_INLINE Vec<Tag<uint32_t, 8>> local_shuf(Vec<Tag<uint32_t, 8>> v, int i3, int i2, int i1, int i0) {
  Tag<int32_t, 8> t1; Tag<uint32_t, 8> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<uint32_t, 16>> local_shuf(Vec<Tag<uint32_t, 16>> v) {
  Tag<int32_t, 16> t1; Tag<uint32_t, 16> t2;
  return word::bitcast(t2, word::local_shuf<I3, I2, I1, I0>(bitcast(t1, v)));
}

TLV_INLINE Vec<Tag<uint32_t, 16>> local_shuf(Vec<Tag<uint32_t, 16>> v, Vec<Tag<int32_t, 16>> i) {
  Tag<int32_t, 16> t1; Tag<uint32_t, 16> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

TLV_INLINE Vec<Tag<uint32_t, 16>> local_shuf(Vec<Tag<uint32_t, 16>> v, int i3, int i2, int i1, int i0) {
  Tag<int32_t, 16> t1; Tag<uint32_t, 16> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 512

/* ********************************* float64_t ****************************** */

template <int I1, int I0>
TLV_INLINE Vec<Tag<float64_t, 2>> local_shuf(Vec<Tag<float64_t, 2>> v) {
  details::assert_index<I1, I0>();
  return _mm_shuffle_pd(v.v, v.v, _MM_SHUFFLE2(I1, I0));
}

TLV_INLINE Vec<Tag<float64_t, 2>> local_shuf(Vec<Tag<float64_t, 2>> v, Vec<Tag<int64_t, 2>> i) {
  return _mm_permutevar_pd(v.v, _mm_slli_epi64(i.v, 1));
}

TLV_INLINE Vec<Tag<float64_t, 2>> local_shuf(Vec<Tag<float64_t, 2>> v, int i1, int i0) {
  details::assert_index(i1, i0);
  return local_shuf(v, _mm_set_epi64x(i1, i0));
}

#if VEC_WIDTH >= 256
template <int I1, int I0>
TLV_INLINE Vec<Tag<float64_t, 4>> local_shuf(Vec<Tag<float64_t, 4>> v) {
  details::assert_index<I1, I0>();
  return _mm256_shuffle_pd(v.v, v.v, ((I1 << 3) | (I0 << 2) | (I1 << 1) | (I0 << 0)));
}

TLV_INLINE Vec<Tag<float64_t, 4>> local_shuf(Vec<Tag<float64_t, 4>> v, Vec<Tag<int64_t, 4>> i) {
  return _mm256_permutevar_pd(v.v, _mm256_slli_epi64(i.v, 1));
}

TLV_INLINE Vec<Tag<float64_t, 4>> local_shuf(Vec<Tag<float64_t, 4>> v, int i1, int i0) {
  details::assert_index(i1, i0);
  return local_shuf(v, _mm256_set_epi64x(i1, i0, i1, i0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <int I1, int I0>
TLV_INLINE Vec<Tag<float64_t, 8>> local_shuf(Vec<Tag<float64_t, 8>> v) {
  details::assert_index<I1, I0>();
  return _mm512_shuffle_pd(v.v, v.v, ((I1 << 7) | (I0 << 6) | (I1 << 5) | (I0 << 4) | (I1 << 3) | (I0 << 2) | (I1 << 1) | (I0 << 0)));
}

TLV_INLINE Vec<Tag<float64_t, 8>> local_shuf(Vec<Tag<float64_t, 8>> v, Vec<Tag<int64_t, 8>> i) {
  return _mm512_permutevar_pd(v.v, _mm512_slli_epi64(i.v, 1));
}

TLV_INLINE Vec<Tag<float64_t, 8>> local_shuf(Vec<Tag<float64_t, 8>> v, int i1, int i0) {
  details::assert_index(i1, i0);
  return local_shuf(v, _mm512_set_epi64(i1, i0, i1, i0, i1, i0, i1, i0));
}
#endif // VEC_WIDTH >= 512

/* ********************************** int64_t ******************************* */

template <int I1, int I0>
TLV_INLINE Vec<Tag<int64_t, 2>> local_shuf(Vec<Tag<int64_t, 2>> v) {
  Tag<float64_t, 2> t1; Tag<int64_t, 2> t2;
  return word::bitcast(t2, word::local_shuf<I1, I0>(bitcast(t1, v)));
}

TLV_INLINE Vec<Tag<int64_t, 2>> local_shuf(Vec<Tag<int64_t, 2>> v, Vec<Tag<int64_t, 2>> i) {
  Tag<float64_t, 2> t1; Tag<int64_t, 2> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

TLV_INLINE Vec<Tag<int64_t, 2>> local_shuf(Vec<Tag<int64_t, 2>> v, int i1, int i0) {
  Tag<float64_t, 2> t1; Tag<int64_t, 2> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i1, i0));
}

#if VEC_WIDTH >= 256
template <int I1, int I0>
TLV_INLINE Vec<Tag<int64_t, 4>> local_shuf(Vec<Tag<int64_t, 4>> v) {
  Tag<float64_t, 4> t1; Tag<int64_t, 4> t2;
  return word::bitcast(t2, word::local_shuf<I1, I0>(bitcast(t1, v)));
}

TLV_INLINE Vec<Tag<int64_t, 4>> local_shuf(Vec<Tag<int64_t, 4>> v, Vec<Tag<int64_t, 4>> i) {
  Tag<float64_t, 4> t1; Tag<int64_t, 4> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

TLV_INLINE Vec<Tag<int64_t, 4>> local_shuf(Vec<Tag<int64_t, 4>> v, int i1, int i0) {
  Tag<float64_t, 4> t1; Tag<int64_t, 4> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i1, i0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <int I1, int I0>
TLV_INLINE Vec<Tag<int64_t, 8>> local_shuf(Vec<Tag<int64_t, 8>> v) {
  Tag<float64_t, 8> t1; Tag<int64_t, 8> t2;
  return word::bitcast(t2, word::local_shuf<I1, I0>(bitcast(t1, v)));
}

TLV_INLINE Vec<Tag<int64_t, 8>> local_shuf(Vec<Tag<int64_t, 8>> v, Vec<Tag<int64_t, 8>> i) {
  Tag<float64_t, 8> t1; Tag<int64_t, 8> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

TLV_INLINE Vec<Tag<int64_t, 8>> local_shuf(Vec<Tag<int64_t, 8>> v, int i1, int i0) {
  Tag<float64_t, 8> t1; Tag<int64_t, 8> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i1, i0));
}
#endif // VEC_WIDTH >= 512

/* ********************************* uint64_t ******************************* */

template <int I1, int I0>
TLV_INLINE Vec<Tag<uint64_t, 2>> local_shuf(Vec<Tag<uint64_t, 2>> v) {
  Tag<int64_t, 2> t1; Tag<uint64_t, 2> t2;
  return word::bitcast(t2, word::local_shuf<I1, I0>(bitcast(t1, v)));
}

TLV_INLINE Vec<Tag<uint64_t, 2>> local_shuf(Vec<Tag<uint64_t, 2>> v, Vec<Tag<int64_t, 2>> i) {
  Tag<int64_t, 2> t1; Tag<uint64_t, 2> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

TLV_INLINE Vec<Tag<uint64_t, 2>> local_shuf(Vec<Tag<uint64_t, 2>> v, int i1, int i0) {
  Tag<int64_t, 2> t1; Tag<uint64_t, 2> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i1, i0));
}


#if VEC_WIDTH >= 256
template <int I1, int I0>
TLV_INLINE Vec<Tag<uint64_t, 4>> local_shuf(Vec<Tag<uint64_t, 4>> v) {
  Tag<int64_t, 4> t1; Tag<uint64_t, 4> t2;
  return word::bitcast(t2, word::local_shuf<I1, I0>(bitcast(t1, v)));
}

TLV_INLINE Vec<Tag<uint64_t, 4>> local_shuf(Vec<Tag<uint64_t, 4>> v, Vec<Tag<int64_t, 4>> i) {
  Tag<int64_t, 4> t1; Tag<uint64_t, 4> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

TLV_INLINE Vec<Tag<uint64_t, 4>> local_shuf(Vec<Tag<uint64_t, 4>> v, int i1, int i0) {
  Tag<int64_t, 4> t1; Tag<uint64_t, 4> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i1, i0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <int I1, int I0>
TLV_INLINE Vec<Tag<uint64_t, 8>> local_shuf(Vec<Tag<uint64_t, 8>> v) {
  Tag<int64_t, 8> t1; Tag<uint64_t, 8> t2;
  return word::bitcast(t2, word::local_shuf<I1, I0>(bitcast(t1, v)));
}

TLV_INLINE Vec<Tag<uint64_t, 8>> local_shuf(Vec<Tag<uint64_t, 8>> v, Vec<Tag<int64_t, 8>> i) {
  Tag<int64_t, 8> t1; Tag<uint64_t, 8> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

TLV_INLINE Vec<Tag<uint64_t, 8>> local_shuf(Vec<Tag<uint64_t, 8>> v, int i1, int i0) {
  Tag<int64_t, 8> t1; Tag<uint64_t, 8> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i1, i0));
}
#endif // VEC_WIDTH >= 512

/* ********************************** int8_t ******************************** */

TLV_INLINE Vec<Tag<int8_t, 16>> local_shuf(Vec<Tag<int8_t, 16>> v, Vec<Tag<int8_t, 16>> i) {
  return _mm_shuffle_epi8(v.v, i.v);
}

TLV_INLINE Vec<Tag<int8_t, 16>> local_shuf(Vec<Tag<int8_t, 16>> v, int i15, int i14, int i13, int i12, int i11, int i10, int i9, int i8, int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  details::assert_index(i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0);
  return local_shuf(v, _mm_set_epi8(i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0));
}

template <int I15, int I14, int I13, int I12, int I11, int I10, int I9, int I8, int I7, int I6, int I5, int I4, int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<int8_t, 16>> local_shuf(Vec<Tag<int8_t, 16>> v) {
  details::assert_index<I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0>();
  return local_shuf(v, _mm_set_epi8(I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0));
}


#if VEC_WIDTH >= 256
TLV_INLINE Vec<Tag<int8_t, 32>> local_shuf(Vec<Tag<int8_t, 32>> v, Vec<Tag<int8_t, 32>> i) {
  return _mm256_shuffle_epi8(v.v, i.v);
}

TLV_INLINE Vec<Tag<int8_t, 32>> local_shuf(Vec<Tag<int8_t, 32>> v, int i15, int i14, int i13, int i12, int i11, int i10, int i9, int i8, int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  details::assert_index(i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0);
  return local_shuf(v, _mm256_set_epi8(i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0, i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0));
}

template <int I15, int I14, int I13, int I12, int I11, int I10, int I9, int I8, int I7, int I6, int I5, int I4, int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<int8_t, 32>> local_shuf(Vec<Tag<int8_t, 32>> v) {
  details::assert_index<I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0>();
  return local_shuf(v, _mm256_set_epi8(I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0, I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
TLV_INLINE Vec<Tag<int8_t, 64>> local_shuf(Vec<Tag<int8_t, 64>> v, Vec<Tag<int8_t, 64>> i) {
  return _mm512_shuffle_epi8(v.v, i.v);
}

TLV_INLINE Vec<Tag<int8_t, 64>> local_shuf(Vec<Tag<int8_t, 64>> v, int i15, int i14, int i13, int i12, int i11, int i10, int i9, int i8, int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  details::assert_index(i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0);
  return local_shuf(v, _mm512_set_epi8(i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0, i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0, i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0, i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0));
}

template <int I15, int I14, int I13, int I12, int I11, int I10, int I9, int I8, int I7, int I6, int I5, int I4, int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<int8_t, 64>> local_shuf(Vec<Tag<int8_t, 64>> v) {
  details::assert_index<I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0>();
  return local_shuf(v, _mm512_set_epi8(I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0, I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0, I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0, I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0));
}

#endif // VEC_WIDTH >= 512


/* ********************************* uint8_t ******************************** */

template <int I15, int I14, int I13, int I12, int I11, int I10, int I9, int I8, int I7, int I6, int I5, int I4, int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<uint8_t, 16>> local_shuf(Vec<Tag<uint8_t, 16>> v) {
  Tag<int8_t, 16> t1; Tag<uint8_t, 16> t2;
  return word::bitcast(t2, word::local_shuf<I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0>(bitcast(t1, v)));
}

TLV_INLINE Vec<Tag<uint8_t, 16>> local_shuf(Vec<Tag<uint8_t, 16>> v, Vec<Tag<int8_t, 16>> i) {
  Tag<int8_t, 16> t1; Tag<uint8_t, 16> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

TLV_INLINE Vec<Tag<uint8_t, 16>> local_shuf(Vec<Tag<uint8_t, 16>> v, int i15, int i14, int i13, int i12, int i11, int i10, int i9, int i8, int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  Tag<int8_t, 16> t1; Tag<uint8_t, 16> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0));
}


#if VEC_WIDTH >= 256
template <int I15, int I14, int I13, int I12, int I11, int I10, int I9, int I8, int I7, int I6, int I5, int I4, int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<uint8_t, 32>> local_shuf(Vec<Tag<uint8_t, 32>> v) {
  Tag<int8_t, 32> t1; Tag<uint8_t, 32> t2;
  return word::bitcast(t2, word::local_shuf<I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0>(bitcast(t1, v)));
}

TLV_INLINE Vec<Tag<uint8_t, 32>> local_shuf(Vec<Tag<uint8_t, 32>> v, Vec<Tag<int8_t, 32>> i) {
  Tag<int8_t, 32> t1; Tag<uint8_t, 32> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

TLV_INLINE Vec<Tag<uint8_t, 32>> local_shuf(Vec<Tag<uint8_t, 32>> v, int i15, int i14, int i13, int i12, int i11, int i10, int i9, int i8, int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  Tag<int8_t, 32> t1; Tag<uint8_t, 32> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <int I15, int I14, int I13, int I12, int I11, int I10, int I9, int I8, int I7, int I6, int I5, int I4, int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<uint8_t, 64>> local_shuf(Vec<Tag<uint8_t, 64>> v) {
  Tag<int8_t, 64> t1; Tag<uint8_t, 64> t2;
  return word::bitcast(t2, word::local_shuf<I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0>(bitcast(t1, v)));
}

TLV_INLINE Vec<Tag<uint8_t, 64>> local_shuf(Vec<Tag<uint8_t, 64>> v, Vec<Tag<int8_t, 64>> i) {
  Tag<int8_t, 64> t1; Tag<uint8_t, 64> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

TLV_INLINE Vec<Tag<uint8_t, 64>> local_shuf(Vec<Tag<uint8_t, 64>> v, int i15, int i14, int i13, int i12, int i11, int i10, int i9, int i8, int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  Tag<int8_t, 64> t1; Tag<uint8_t, 64> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 512

/* ********************************* int16_t ******************************** */

template <int I7, int I6, int I5, int I4, int I3, int I2, int I1, int I0, TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 16), TL_IF(sizeof(TypeOf<T>) == 2)>
TLV_INLINE V local_shuf(V v) {
  if constexpr (I3 < 4 && I2 < 4 && I1 < 4 && I0 < 4 &&
                I7 >= 4 && I6 >= 4 && I5 >= 4 && I4 >= 4) {
    // Fast path: low 4 elements all from [0,3], high 4 all from [4,7]
    auto u = _mm_shufflelo_epi16(v.v, _MM_SHUFFLE(I3, I2, I1, I0));
    return _mm_shufflehi_epi16(u, _MM_SHUFFLE(I7 - 4, I6 - 4, I5 - 4, I4 - 4));
  } else {
    Tag<int8_t, 16> t1;
    return word::bitcast(T(), word::local_shuf<
        2 * I7 + 1, 2 * I7, 2 * I6 + 1, 2 * I6, 2 * I5 + 1, 2 * I5, 2 * I4 + 1, 2 * I4,
        2 * I3 + 1, 2 * I3, 2 * I2 + 1, 2 * I2, 2 * I1 + 1, 2 * I1, 2 * I0 + 1, 2 * I0
    >(word::bitcast(t1, v)));
  }
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 16), TL_IF(sizeof(TypeOf<T>) == 2)>
TLV_INLINE V local_shuf(V v, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  Tag<int8_t, 16> t1;
  // we assume that high byte of each index in i are zero
  auto i2 = _mm_slli_epi32(i.v, 1); // * 2
  auto i3 = _mm_slli_epi32(i.v, 9); // move to hi byte
  // idx = i2 | i3 | 0x0100
#ifdef HAS_AVX512F
  auto idx = _mm_ternarylogic_epi32(i2, i3, _mm_set1_epi16(0x0100), _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C);
#else
  auto idx = _mm_or_si128(_mm_or_si128(i2, i3), _mm_set1_epi16(0x0100));
#endif
  return word::bitcast(T(), word::local_shuf(word::bitcast(t1, v), Vec<decltype(t1)>{idx}));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 16), TL_IF(sizeof(TypeOf<T>) == 2)>
TLV_INLINE V local_shuf(V v, int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  return word::local_shuf(v, _mm_set_epi16(i7, i6, i5, i4, i3, i2, i1, i0));
}

#if VEC_WIDTH >= 256
template <int I7, int I6, int I5, int I4, int I3, int I2, int I1, int I0, TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(sizeof(TypeOf<T>) == 2)>
TLV_INLINE V local_shuf(V v) {
  if constexpr (I3 < 4 && I2 < 4 && I1 < 4 && I0 < 4 &&
                I7 >= 4 && I6 >= 4 && I5 >= 4 && I4 >= 4) {
    // Fast path: low 4 elements all from [0,3], high 4 all from [4,7]
    auto u = _mm256_shufflelo_epi16(v.v, _MM_SHUFFLE(I3, I2, I1, I0));
    return _mm256_shufflehi_epi16(u, _MM_SHUFFLE(I7 - 4, I6 - 4, I5 - 4, I4 - 4));
  } else {
    Tag<int8_t, 32> t1;
    return word::bitcast(T(), word::local_shuf<
        2 * I7 + 1, 2 * I7, 2 * I6 + 1, 2 * I6, 2 * I5 + 1, 2 * I5, 2 * I4 + 1, 2 * I4,
        2 * I3 + 1, 2 * I3, 2 * I2 + 1, 2 * I2, 2 * I1 + 1, 2 * I1, 2 * I0 + 1, 2 * I0
    >(word::bitcast(t1, v)));
  }
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(sizeof(TypeOf<T>) == 2)>
TLV_INLINE V local_shuf(V v, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  Tag<int8_t, 32> t1;
  // we assume that high byte of each index in i are zero
  auto i2 = _mm256_slli_epi32(i.v, 1); // * 2
  auto i3 = _mm256_slli_epi32(i.v, 9); // move to hi byte
  // idx = i2 | i3 | 0x0100
  #ifdef HAS_AVX512F
  auto idx = _mm256_ternarylogic_epi32(i2, i3, _mm256_set1_epi16(0x0100), _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C);
  #else
  auto idx = _mm256_or_si256(_mm256_or_si256(i2, i3), _mm256_set1_epi16(0x0100));
  #endif
  return word::bitcast(T(), word::local_shuf(word::bitcast(t1, v), Vec<decltype(t1)>{idx}));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(sizeof(TypeOf<T>) == 2)>
TLV_INLINE V local_shuf(V v, int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  return word::local_shuf(v, _mm256_set_epi16(i7, i6, i5, i4, i3, i2, i1, i0, i7, i6, i5, i4, i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <int I7, int I6, int I5, int I4, int I3, int I2, int I1, int I0, TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(sizeof(TypeOf<T>) == 2)>
TLV_INLINE V local_shuf(V v) {
  if constexpr (I3 < 4 && I2 < 4 && I1 < 4 && I0 < 4 &&
                I7 >= 4 && I6 >= 4 && I5 >= 4 && I4 >= 4) {
    // Fast path: low 4 elements all from [0,3], high 4 all from [4,7]
    auto u = _mm512_shufflelo_epi16(v.v, _MM_SHUFFLE(I3, I2, I1, I0));
    return _mm512_shufflehi_epi16(u, _MM_SHUFFLE(I7 - 4, I6 - 4, I5 - 4, I4 - 4));
  } else {
    Tag<int8_t, 64> t1;
    return word::bitcast(T(), word::local_shuf<
        2 * I7 + 1, 2 * I7, 2 * I6 + 1, 2 * I6, 2 * I5 + 1, 2 * I5, 2 * I4 + 1, 2 * I4,
        2 * I3 + 1, 2 * I3, 2 * I2 + 1, 2 * I2, 2 * I1 + 1, 2 * I1, 2 * I0 + 1, 2 * I0
    >( word::bitcast(t1, v)));
  }
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(sizeof(TypeOf<T>) == 2)>
TLV_INLINE V local_shuf(V v, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  Tag<int8_t, 64> t1;
  // we assume that high byte of each index in i are zero
  auto i2 = _mm512_slli_epi32(i.v, 1); // * 2
  auto i3 = _mm512_slli_epi32(i.v, 9); // move to hi byte
  // idx = i2 | i3 | 0x0100
  auto idx = _mm512_ternarylogic_epi32(i2, i3, _mm512_set1_epi16(0x0100), _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C);
  return word::bitcast(T(), word::local_shuf(word::bitcast(t1, v), Vec<decltype(t1)>{idx}));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(sizeof(TypeOf<T>) == 2)>
TLV_INLINE V local_shuf(V v, int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  return word::local_shuf(v, _mm512_set_epi16(i7, i6, i5, i4, i3, i2, i1, i0, i7, i6, i5, i4, i3, i2, i1, i0, i7, i6, i5, i4, i3, i2, i1, i0, i7, i6, i5, i4, i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                             Block-level Shuffle                            //
/* ************************************************************************** */
// xmm only have 1 block/lane
template <int I0, TLV_DECL_VEC(V), TL_IF(sizeof(V) == 16)>
TLV_INLINE V block_shuf(V v) {
  details::assert_index<I0>();
  return v;
}

template <TLV_DECL_VEC(V), TL_IF(sizeof(V) == 16)>
TLV_INLINE V block_shuf(V v, int i0) {
  details::assert_index(i0);
  return v;
}

#if VEC_WIDTH >= 256
template <int I1, int I0>
TLV_INLINE Vec<Tag<float32_t, 8>> block_shuf(Vec<Tag<float32_t, 8>> v) {
  details::assert_index<I1, I0>();
  return _mm256_permute2f128_ps(v.v, v.v, ((I1 << 4) | (I0)));
}

template <int I1, int I0, TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(sizeof(V) == 32), TL_IF(is_none<TypeOf<T>, float32_t>)>
TLV_INLINE V block_shuf(V v) {
  Tag<float32_t, 8> t1;
  return word::bitcast(T(), word::block_shuf<I1, I0>(word::bitcast(t1, v)));
}

TLV_INLINE Vec<Tag<float32_t, 8>> block_shuf(Vec<Tag<float32_t, 8>> v, int i1, int i0) {
  details::assert_index(i1, i0);
  return _mm256_permutevar8x32_ps(v.v, _mm256_set_epi32(i1 + 3, i1 + 2, i1 + 1, i1 + 0, i0 + 3, i0 + 2, i0 + 1, i0 + 0));
}

TLV_INLINE Vec<Tag<float64_t, 4>> block_shuf(Vec<Tag<float64_t, 4>> v, int i1, int i0) {
  Tag<float32_t, 8> t1; Tag<float64_t, 4> t2;
  return word::bitcast(t2, word::block_shuf(word::bitcast(t1, v), i1, i0));
}

TLV_INLINE Vec<Tag<int32_t, 8>> block_shuf(Vec<Tag<int32_t, 8>> v, int i1, int i0) {
  details::assert_index(i1, i0);
  return _mm256_permutevar8x32_epi32(v.v, _mm256_set_epi32(i1 + 3, i1 + 2, i1 + 1, i1 + 0, i0 + 3, i0 + 2, i0 + 1, i0 + 0));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(sizeof(V) == 32), TL_IF(is_none<TypeOf<T>, float32_t, float64_t, int32_t>)>
TLV_INLINE V block_shuf(V v, int i1, int i0) {
  Tag<int32_t, 8> t1;
  return word::bitcast(T(), word::block_shuf(word::bitcast(t1, v), i1, i0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<float32_t, 16>> block_shuf(Vec<Tag<float32_t, 16>> v) {
  details::assert_index<I3, I2, I1, I0>();
  return _mm512_shuffle_f32x4(v.v, v.v, _MM_SHUFFLE(I3, I2, I1, I0));
}

template <int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<float64_t, 8>> block_shuf(Vec<Tag<float64_t, 8>> v) {
  details::assert_index<I3, I2, I1, I0>();
  return _mm512_shuffle_f64x2(v.v, v.v, _MM_SHUFFLE(I3, I2, I1, I0));
}

template <int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<int32_t, 16>> block_shuf(Vec<Tag<int32_t, 16>> v) {
  details::assert_index<I3, I2, I1, I0>();
  return _mm512_shuffle_i32x4(v.v, v.v, _MM_SHUFFLE(I3, I2, I1, I0));
}

template <int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<uint32_t, 16>> block_shuf(Vec<Tag<uint32_t, 16>> v) {
  Tag<int32_t, 16> t1; Tag<uint32_t, 16> t2;
  return word::bitcast(t2, word::block_shuf<I3, I2, I1, I0>(word::bitcast(t1, v)));
}

template <int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<int64_t, 8>> block_shuf(Vec<Tag<int64_t, 8>> v) {
  details::assert_index<I3, I2, I1, I0>();
  return _mm512_shuffle_i64x2(v.v, v.v, _MM_SHUFFLE(I3, I2, I1, I0));
}

template <int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<uint64_t, 8>> block_shuf(Vec<Tag<uint64_t, 8>> v) {
  Tag<int64_t, 8> t1; Tag<uint64_t, 8> t2;
  return word::bitcast(t2, word::block_shuf<I3, I2, I1, I0>(word::bitcast(t1, v)));
}

template <int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<int16_t, 32>> block_shuf(Vec<Tag<int16_t, 32>> v) {
  Tag<int32_t, 16> t1; Tag<int16_t, 32> t2;
  return word::bitcast(t2, word::block_shuf<I3, I2, I1, I0>(word::bitcast(t1, v)));
}

template <int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<uint16_t, 32>> block_shuf(Vec<Tag<uint16_t, 32>> v) {
  Tag<int16_t, 32> t1; Tag<uint16_t, 32> t2;
  return word::bitcast(t2, word::block_shuf<I3, I2, I1, I0>(word::bitcast(t1, v)));
}

template <int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<int8_t, 64>> block_shuf(Vec<Tag<int8_t, 64>> v) {
  Tag<int32_t, 16> t1; Tag<int8_t, 64> t2;
  return word::bitcast(t2, word::block_shuf<I3, I2, I1, I0>(word::bitcast(t1, v)));
}

template <int I3, int I2, int I1, int I0>
TLV_INLINE Vec<Tag<uint8_t, 64>> block_shuf(Vec<Tag<uint8_t, 64>> v) {
  Tag<int8_t, 64> t1; Tag<uint8_t, 64> t2;
  return word::bitcast(t2, word::block_shuf<I3, I2, I1, I0>(word::bitcast(t1, v)));
}

TLV_INLINE Vec<Tag<float64_t, 8>> block_shuf(Vec<Tag<float64_t, 8>> v, int i3, int i2, int i1, int i0) {
  details::assert_index(i3, i2, i1, i0);
  return _mm512_permutexvar_pd(_mm512_set_epi64(i3 + 1, i3, i2 + 1, i2, i1 + 1, i1, i0 + 1, i0), v.v);
}

TLV_INLINE Vec<Tag<float32_t, 16>> block_shuf(Vec<Tag<float32_t, 16>> v, int i3, int i2, int i1, int i0) {
  Tag<float64_t, 8> t1; Tag<float32_t, 16> t2;
  return word::bitcast(t2, word::block_shuf(word::bitcast(t1, v), i3, i2, i1, i0));
}

TLV_INLINE Vec<Tag<int64_t, 8>> block_shuf(Vec<Tag<int64_t, 8>> v, int i3, int i2, int i1, int i0) {
  details::assert_index(i3, i2, i1, i0);
  return _mm512_permutexvar_epi64(_mm512_set_epi64(i3 + 1, i3, i2 + 1, i2, i1 + 1, i1, i0 + 1, i0), v.v);
}

template <TLV_DECL_VEC(V), TL_IF(sizeof(V) == 64), TL_IF(is_none<TypeOf<Vec2Tag<V>>, float32_t, float64_t, int64_t>)>
TLV_INLINE V block_shuf(V v, int i3, int i2, int i1, int i0) {
  Tag<int64_t, 8> t1; Vec2Tag<V> t2;
  return word::bitcast(t2, word::block_shuf(word::bitcast(t1, v), i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                       Global Shuffle / Table Lookup                        //
/* ************************************************************************** */
template <TLV_DECL_VEC(V), TL_IF(sizeof(TypeOf<Vec2Tag<V>>) == 1), TL_IF(sizeof(V) == 16)>
TLV_INLINE V shuf(V v, Vec<Tag<int8_t, 16>> i) {
  return word::local_shuf(v, i);
}

template <TLV_DECL_VEC(V), TL_IF(sizeof(TypeOf<Vec2Tag<V>>) == 2), TL_IF(sizeof(V) == 16)>
TLV_INLINE V shuf(V v, Vec<Tag<int16_t, 8>> i) {
  return word::local_shuf(v, i);
}

template <TLV_DECL_VEC(V), TL_IF(sizeof(TypeOf<Vec2Tag<V>>) == 4), TL_IF(sizeof(V) == 16)>
TLV_INLINE V shuf(V v, Vec<Tag<int32_t, 4>> i) {
  return word::local_shuf(v, i);
}

template <TLV_DECL_VEC(V), TL_IF(sizeof(TypeOf<Vec2Tag<V>>) == 8), TL_IF(sizeof(V) == 16)>
TLV_INLINE V shuf(V v, Vec<Tag<int64_t, 2>> i) {
  return word::local_shuf(v, i);
}

#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE V shuf(V v, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  return _mm256_permutevar8x32_ps(v.v, i.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE V shuf(V v, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  #ifdef HAS_AVX512F
  return _mm256_permutexvar_pd(i.v, v.v);
  #else
  auto i1 = _mm256_slli_epi64(i.v, 1); // i * 2
  auto i2 = _mm256_slli_epi64(i.v, 33); // move to hi 4 bytes
  auto idx = _mm256_or_si256(_mm256_or_si256(i1, i2), _mm256_set1_epi64x(0x00000001'00000000));
  Tag<float32_t, 8> t1; Tag<int32_t, 8> ti;
  return word::bitcast(T(), word::shuf(word::bitcast(t1, v), Vec<decltype(ti)>{idx}));
  #endif
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE V shuf(V v, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  #ifdef HAS_AVX512F
  return _mm256_permutexvar_epi64(i.v, v.v);
  #else
  Rebind<float64_t, T> t1;
  return word::bitcast(T(), word::shuf(word::bitcast(t1, v), i));
  #endif
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE V shuf(V v, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  return _mm256_permutevar8x32_epi32(v.v, i.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE V shuf(V v, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  #ifdef HAS_AVX512VBMI
  return _mm256_permutexvar_epi8(i.v, v.v);
  #else
  Vec<Tag<int8_t, 32>> vlane_0 = _mm256_permute2f128_si256(v.v, v.v, 0x00);
  Vec<Tag<int8_t, 32>> vlane_1 = _mm256_permute2f128_si256(v.v, v.v, 0x11);
  auto u0 = local_shuf(vlane_0, i);
  #ifdef HAS_AVX512BW
  // 4th bit becomes the 7th bit, for blendv
  auto lane_mask = _mm256_movepi8_mask(_mm256_slli_epi16(i.v, 3));
  auto u = _mm256_mask_shuffle_epi8(u0.v, lane_mask, vlane_1.v, i.v);
  #else
  auto lane_mask = _mm256_slli_epi16(i.v, 3);
  auto u1 = local_shuf(vlane_1, i);
  auto u = _mm256_blendv_epi8(u0.v, u1.v, lane_mask);
  #endif
  return u;
  #endif
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE V shuf(V v, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  #ifdef HAS_AVX512BW
  return _mm256_permutexvar_epi16(i.v, v.v);
  #else
  Tag<int8_t, 32> t1;
  auto i2 = _mm256_slli_epi32(i.v, 1); // * 2
  auto i3 = _mm256_slli_epi32(i.v, 9); // move to hi byte
  // idx = i2 | i3 | 0x0100
  #ifdef HAS_AVX512F
  auto idx = _mm256_ternarylogic_epi32(i2, i3, _mm256_set1_epi16(0x0100), _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C);
  #else
  auto idx = _mm256_or_si256(_mm256_or_si256(i2, i3), _mm256_set1_epi16(0x0100));
  #endif
  return word::bitcast(T(), word::shuf(word::bitcast(t1, v), idx));
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE V shuf(V v, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  return _mm512_permutexvar_ps(i.v, v.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE V shuf(V v, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  return _mm512_permutexvar_pd(i.v, v.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE V shuf(V v, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  return _mm512_permutexvar_epi64(i.v, v.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE V shuf(V v, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  return _mm512_permutexvar_epi32(i.v, v.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE V shuf(V v, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  #ifdef HAS_AVX512VBMI
  return _mm512_permutexvar_epi8(i.v, v.v);
  #else
  Vec<Tag<int8_t, 64>> vlane_0 = _mm512_shuffle_i32x4(v.v, v.v, _MM_SHUFFLE(0, 0, 0, 0));
  Vec<Tag<int8_t, 64>> vlane_1 = _mm512_shuffle_i32x4(v.v, v.v, _MM_SHUFFLE(1, 1, 1, 1));
  Vec<Tag<int8_t, 64>> vlane_2 = _mm512_shuffle_i32x4(v.v, v.v, _MM_SHUFFLE(2, 2, 2, 2));
  Vec<Tag<int8_t, 64>> vlane_3 = _mm512_shuffle_i32x4(v.v, v.v, _MM_SHUFFLE(3, 3, 3, 3));
  auto lane_mask_0 = _mm512_movepi8_mask(_mm512_slli_epi16(i.v, 3)); // low bit of lane no
  auto lane_mask_1 = _mm512_movepi8_mask(_mm512_slli_epi16(i.v, 2)); // high bit of lane no
  auto u0 = local_shuf(vlane_0, i);
  auto u1 = local_shuf(vlane_2, i);
  auto w0 = _mm512_mask_shuffle_epi8(u0.v, lane_mask_0, vlane_1.v, i.v);
  auto w1 = _mm512_mask_shuffle_epi8(u1.v, lane_mask_0, vlane_3.v, i.v);
  return _mm512_mask_blend_epi8(lane_mask_1, w0, w1);
  #endif
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE V shuf(V v, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  return _mm512_permutexvar_epi16(i.v, v.v);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                           Concat & Upper Lower                             //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 2), TL_IF(sizeof(TypeOf<T>) == 1)>
TLV_INLINE Vec<T> concat(T t, Vec<Half<T>> v1, Vec<Half<T>> v2) {
Tag<uint8_t, 16> t1;
  auto u1 = word::bitcast(t1, v1);
  auto u2 = word::bitcast(t1, v2);
  auto w1 = _mm_cvtsi128_si32(v1.v);
  auto w2 = _mm_cvtsi128_si32(v2.v);
  auto w3 = (w1 & 0xff) | ((w2 && 0xff) << 8);
  auto r = _mm_cvtsi32_si128(w3);
  return word::bitcast(t, Vec<decltype(t1)>{r});
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 4), TL_IF(sizeof(TypeOf<T>) <= 2)>
TLV_INLINE Vec<T> concat(T t, Vec<Half<T>> v1, Vec<Half<T>> v2) {
Tag<uint8_t, 16> t1;
  auto u1 = word::bitcast(t1, v1);
  auto u2 = word::bitcast(t1, v2);
  auto shuf = _mm_shufflelo_epi16(u2.v, _MM_SHUFFLE(0, 0, 0, 0));
  auto r = _mm_blend_epi16(u1.v, shuf, 0b10);
  return word::bitcast(t, Vec<decltype(t1)>{r});
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> concat(T t, Vec<Half<T>> v1, Vec<Half<T>> v2) {
  auto hi = word::local_shuf<0, 0, 0, 0>(v1);
  return _mm_move_ss(hi.v, v2.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8), TL_IF(is_none<TypeOf<T>, float32_t> && sizeof(TypeOf<T>) <= 4)>
TLV_INLINE Vec<T> concat(T t, Vec<Half<T>> v1, Vec<Half<T>> v2) {
  Tag<uint8_t, 16> t1;
  Tag<float32_t, 4> t2;
  auto hi = word::bitcast(t2, word::local_shuf<0, 0, 0, 0>(word::bitcast(t1, v1)));
  return word::bitcast(t, Vec<decltype(t2)>{_mm_move_ss(hi.v, v2.v)});
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8)>
TLV_INLINE Vec<T> concat(T t, Vec<Half<T>> v1, Vec<Half<T>> v2) {
  Tag<float32_t, 4> t1;
  auto u1 = word::bitcast(t1, v1);
  auto u2 = word::bitcast(t1, v2);
  return word::bitcast(t, Vec<decltype(t1)>{_mm_movelh_ps(u1.v, u2.v)});
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16)>
TLV_INLINE Vec<T> concat(T t, Vec<Half<T>> v1, Vec<Half<T>> v2) {
  Tag<float32_t, 4> t1;
  auto u1 = word::bitcast(t1, v1);
  auto u2 = word::bitcast(t1, v2);
  return word::bitcast(t, Vec<decltype(t1)>{_mm_movelh_ps(u1.v, u2.v)});
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  return _mm_unpacklo_ps(a.v, b.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  return _mm_shuffle_ps(a.v, b.v, _MM_SHUFFLE(2, 0, 2, 0));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  return _mm_shuffle_pd(a.v, b.v, _MM_SHUFFLE2(0, 0));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  Rebind<float32_t, T> tf;
  return word::bitcast(t, word::concat_even(tf, word::bitcast(tf, a), word::bitcast(tf, b)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  Rebind<float64_t, T> tf;
  return word::bitcast(t, word::concat_even(tf, word::bitcast(tf, a), word::bitcast(tf, b)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 4), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  return _mm_unpacklo_epi16(a.v, b.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  auto u_a = _mm_shufflelo_epi16(a.v, _MM_SHUFFLE(2, 0, 2, 0));
  auto u_b = _mm_shufflelo_epi16(b.v, _MM_SHUFFLE(2, 0, 2, 0));
  return _mm_blend_epi16(u_a, u_b, 0b11001100);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  auto u_a = _mm_shufflelo_epi16(a.v, _MM_SHUFFLE(2, 0, 2, 0));
  auto u_b = _mm_shufflelo_epi16(b.v, _MM_SHUFFLE(2, 0, 2, 0));
  u_a = _mm_shufflehi_epi16(u_a, _MM_SHUFFLE(2, 0, 2, 0));
  u_b = _mm_shufflehi_epi16(u_b, _MM_SHUFFLE(2, 0, 2, 0));
  return _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(u_a), _mm_castsi128_ps(u_b), _MM_SHUFFLE(2, 0, 2, 0)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 2), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  return _mm_unpacklo_epi8(a.v, b.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 4), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  static const auto idx = _mm_set_epi8(2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0);
  auto u_a = _mm_shuffle_epi8(a.v, idx);
  auto u_b = _mm_shuffle_epi8(b.v, idx);
  return _mm_blend_epi16(u_a, u_b, 0b10101010);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  static const auto idx = _mm_set_epi8(6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0, 6, 4, 2, 0);
  auto u_a = _mm_shuffle_epi8(a.v, idx);
  auto u_b = _mm_shuffle_epi8(b.v, idx);
  #ifdef HAS_AVX2
  return _mm_blend_epi32(u_a, u_b, 0b1010);
  #else
  return _mm_blend_epi16(u_a, u_b, 0b11001100);
  #endif
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  static const auto idx = _mm_set_epi8(14, 12, 10, 8, 6, 4, 2, 0, 14, 12, 10, 8, 6, 4, 2, 0);
  auto u_a = _mm_shuffle_epi8(a.v, idx);
  auto u_b = _mm_shuffle_epi8(b.v, idx);
  #ifdef HAS_AVX2
  return _mm_blend_epi32(u_a, u_b, 0b1100);
  #else
  return _mm_blend_epi16(u_a, u_b, 0b11110000);
  #endif
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  return _mm_unpacklo_ps(
      _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(a.v), 4)),
      _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(b.v), 4))
  );
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  return _mm_shuffle_ps(a.v, b.v, _MM_SHUFFLE(3, 1, 3, 1));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  return _mm_shuffle_pd(a.v, b.v, _MM_SHUFFLE2(1, 1));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  Rebind<float32_t, T> tf;
  return word::bitcast(t, word::concat_odd(tf, word::bitcast(tf, a), word::bitcast(tf, b)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  Rebind<float64_t, T> tf;
  return word::bitcast(t, word::concat_odd(tf, word::bitcast(tf, a), word::bitcast(tf, b)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 4), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  return _mm_unpacklo_epi16(_mm_srli_si128(a.v, 2), _mm_srli_si128(b.v, 2));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  auto u_a = _mm_shufflelo_epi16(a.v, _MM_SHUFFLE(3, 1, 3, 1));
  auto u_b = _mm_shufflelo_epi16(b.v, _MM_SHUFFLE(3, 1, 3, 1));
  return _mm_blend_epi16(u_a, u_b, 0b11001100);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  auto u_a = _mm_shufflelo_epi16(a.v, _MM_SHUFFLE(3, 1, 3, 1));
  auto u_b = _mm_shufflelo_epi16(b.v, _MM_SHUFFLE(3, 1, 3, 1));
  u_a = _mm_shufflehi_epi16(u_a, _MM_SHUFFLE(3, 1, 3, 1));
  u_b = _mm_shufflehi_epi16(u_b, _MM_SHUFFLE(3, 1, 3, 1));
  return _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(u_a), _mm_castsi128_ps(u_b), _MM_SHUFFLE(3, 1, 3, 1)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 2), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  return _mm_unpacklo_epi8(_mm_srli_si128(a.v, 1), _mm_srli_si128(b.v, 1));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 4), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  static const auto idx = _mm_set_epi8(3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1);
  auto u_a = _mm_shuffle_epi8(a.v, idx);
  auto u_b = _mm_shuffle_epi8(b.v, idx);
  return _mm_blend_epi16(u_a, u_b, 0b10101010);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  static const auto idx = _mm_set_epi8(7, 5, 3, 1, 7, 5, 3, 1, 7, 5, 3, 1, 7, 5, 3, 1);
  auto u_a = _mm_shuffle_epi8(a.v, idx);
  auto u_b = _mm_shuffle_epi8(b.v, idx);
  #ifdef HAS_AVX2
  return _mm_blend_epi32(u_a, u_b, 0b1010);
  #else
  return _mm_blend_epi16(u_a, u_b, 0b11001100);
  #endif
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  static const auto idx = _mm_set_epi8(15, 13, 11, 9, 7, 5, 3, 1, 15, 13, 11, 9, 7, 5, 3, 1);
  auto u_a = _mm_shuffle_epi8(a.v, idx);
  auto u_b = _mm_shuffle_epi8(b.v, idx);
  #ifdef HAS_AVX2
  return _mm_blend_epi32(u_a, u_b, 0b1100);
  #else
  return _mm_blend_epi16(u_a, u_b, 0b11110000);
  #endif
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32)>
TLV_INLINE Vec<T> concat(T t, Vec<Half<T>> v1, Vec<Half<T>> v2) {
  Tag<uint8_t, 32> t1;
  Tag<uint8_t, 16> t2;
  auto u1 = word::bitcast(t1, v1);
  auto u2 = word::bitcast(t2, v2);
  return word::bitcast(t, Vec<decltype(t1)>{_mm256_insertf128_si256(u1.v, u2.v, 1)});
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  auto u = _mm256_shuffle_ps(a.v, b.v, _MM_SHUFFLE(2, 0, 2, 0));
  return _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(u), _MM_SHUFFLE(3, 1, 2, 0)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  auto u = _mm256_shuffle_pd(a.v, b.v, 0b0000); // (b[2], a[2], b[0], a[0])
  return _mm256_permute4x64_pd(u, _MM_SHUFFLE(3, 1, 2, 0));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  Rebind<float32_t, T> tf;
  return word::bitcast(t, word::concat_even(tf, word::bitcast(tf, a), word::bitcast(tf, b)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  Rebind<float64_t, T> tf;
  return word::bitcast(t, word::concat_even(tf, word::bitcast(tf, a), word::bitcast(tf, b)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  auto u_a = _mm256_shufflelo_epi16(a.v, _MM_SHUFFLE(2, 0, 2, 0));
  auto u_b = _mm256_shufflelo_epi16(b.v, _MM_SHUFFLE(2, 0, 2, 0));
  u_a = _mm256_shufflehi_epi16(u_a, _MM_SHUFFLE(2, 0, 2, 0));
  u_b = _mm256_shufflehi_epi16(u_b, _MM_SHUFFLE(2, 0, 2, 0));
  auto u = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(u_a), _mm256_castsi256_ps(u_b), _MM_SHUFFLE(2, 0, 2, 0)));
  return _mm256_permute4x64_epi64(u, _MM_SHUFFLE(3, 1, 2, 0));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  static const auto idx = _mm256_set_epi8(14, 12, 10, 8, 6, 4, 2, 0, 14, 12, 10, 8, 6, 4, 2, 0, 14, 12, 10, 8, 6, 4, 2, 0, 14, 12, 10, 8, 6, 4, 2, 0);
  auto u_a = _mm256_shuffle_epi8(a.v, idx);
  auto u_b = _mm256_shuffle_epi8(b.v, idx);
  #ifdef HAS_AVX2
  auto u = _mm256_blend_epi32(u_a, u_b, 0b11001100);
  #else
  auto u = _mm256_blend_epi16(u_a, u_b, 0b11110000);
  #endif
  return _mm256_permute4x64_epi64(u, _MM_SHUFFLE(3, 1, 2, 0));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  auto u = _mm256_shuffle_ps(a.v, b.v, _MM_SHUFFLE(3, 1, 3, 1));
  return _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(u), _MM_SHUFFLE(3, 1, 2, 0)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  auto u = _mm256_shuffle_pd(a.v, b.v, 0b1111); // (b[3], a[3], b[1], a[1])
  return _mm256_permute4x64_pd(u, _MM_SHUFFLE(3, 1, 2, 0));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  Rebind<float32_t, T> tf;
  return word::bitcast(t, word::concat_odd(tf, word::bitcast(tf, a), word::bitcast(tf, b)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  Rebind<float64_t, T> tf;
  return word::bitcast(t, word::concat_odd(tf, word::bitcast(tf, a), word::bitcast(tf, b)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  auto u_a = _mm256_shufflelo_epi16(a.v, _MM_SHUFFLE(3, 1, 3, 1));
  auto u_b = _mm256_shufflelo_epi16(b.v, _MM_SHUFFLE(3, 1, 3, 1));
  u_a = _mm256_shufflehi_epi16(u_a, _MM_SHUFFLE(3, 1, 3, 1));
  u_b = _mm256_shufflehi_epi16(u_b, _MM_SHUFFLE(3, 1, 3, 1));
  auto u = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(u_a), _mm256_castsi256_ps(u_b), _MM_SHUFFLE(3, 1, 3, 1)));
  return _mm256_permute4x64_epi64(u, _MM_SHUFFLE(3, 1, 2, 0));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  static const auto idx = _mm256_set_epi8(15, 13, 11, 9, 7, 5, 3, 1, 15, 13, 11, 9, 7, 5, 3, 1, 15, 13, 11, 9, 7, 5, 3, 1, 15, 13, 11, 9, 7, 5, 3, 1);
  auto u_a = _mm256_shuffle_epi8(a.v, idx);
  auto u_b = _mm256_shuffle_epi8(b.v, idx);
  #ifdef HAS_AVX2
  auto u = _mm256_blend_epi32(u_a, u_b, 0b11001100);
  #else
  auto u = _mm256_blend_epi16(u_a, u_b, 0b11110000);
  #endif
  return _mm256_permute4x64_epi64(u, _MM_SHUFFLE(3, 1, 2, 0));
}
#else
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32)>
  TLV_INLINE Vec<T> concat(T t, Vec<Half<T>> v1, Vec<Half<T>> v2) {
  return Vec<T>{ v1, v2 };
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64)>
TLV_INLINE Vec<T> concat(T t, Vec<Half<T>> v1, Vec<Half<T>> v2) {
  Tag<uint8_t, 64> t1;
  Tag<uint8_t, 32> t2;
  auto u1 = word::bitcast(t1, v1);
  auto u2 = word::bitcast(t2, v2);
  return word::bitcast(t, Vec<decltype(t1)>{_mm512_inserti64x4(u1.v, u2.v, 1)});
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 128)>
TLV_INLINE Vec<T> concat(T t, Vec<Half<T>> v1, Vec<Half<T>> v2) {
  return Vec<T>{ v1, v2 };
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  auto u = _mm512_shuffle_ps(a.v, b.v, _MM_SHUFFLE(2, 0, 2, 0));
  static const auto idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_castpd_ps(_mm512_permutexvar_pd(idx, _mm512_castps_pd(u)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  auto u = _mm512_shuffle_pd(a.v, b.v, 0b00000000); // (b[6], a[6], b[4], a[4], b[2], a[2], b[0], a[0])
  static const auto idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_pd(idx, u);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  Rebind<float32_t, T> tf;
  return word::bitcast(t, word::concat_even(tf, word::bitcast(tf, a), word::bitcast(tf, b)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  Rebind<float64_t, T> tf;
  return word::bitcast(t, word::concat_even(tf, word::bitcast(tf, a), word::bitcast(tf, b)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  auto u_a = _mm512_shufflelo_epi16(a.v, _MM_SHUFFLE(2, 0, 2, 0));
  auto u_b = _mm512_shufflelo_epi16(b.v, _MM_SHUFFLE(2, 0, 2, 0));
  u_a = _mm512_shufflehi_epi16(u_a, _MM_SHUFFLE(2, 0, 2, 0));
  u_b = _mm512_shufflehi_epi16(u_b, _MM_SHUFFLE(2, 0, 2, 0));
  auto u = _mm512_shuffle_ps(_mm512_castsi512_ps(u_a), _mm512_castsi512_ps(u_b), _MM_SHUFFLE(2, 0, 2, 0));
  static const auto idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_epi64(idx, _mm512_castps_si512(u));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> a, Vec<T> b) {
  static const auto idx = _mm512_set_epi8(14, 12, 10, 8, 6, 4, 2, 0, 14, 12, 10, 8, 6, 4, 2, 0, 14, 12, 10, 8, 6, 4, 2, 0, 14, 12, 10, 8, 6, 4, 2, 0, 14, 12, 10, 8, 6, 4, 2, 0, 14, 12, 10, 8, 6, 4, 2, 0, 14, 12, 10, 8, 6, 4, 2, 0, 14, 12, 10, 8, 6, 4, 2, 0);
  auto u_a = _mm512_shuffle_epi8(a.v, idx);
  auto u_b = _mm512_shuffle_epi8(b.v, idx);
  auto u = _mm512_mask_blend_epi32(_cvtu32_mask16(0b1100110011001100), u_a, u_b);
  static const auto idx2 = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_epi64(idx2, u);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  auto u = _mm512_shuffle_ps(a.v, b.v, _MM_SHUFFLE(3, 1, 3, 1));
  static const auto idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_castpd_ps(_mm512_permutexvar_pd(idx, _mm512_castps_pd(u)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  auto u = _mm512_shuffle_pd(a.v, b.v, 0b11111111); // (b[7], a[7], b[5], a[5], b[3], a[3], b[1], a[1])
  static const auto idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_pd(idx, u);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  Rebind<float32_t, T> tf;
  return word::bitcast(t, word::concat_odd(tf, word::bitcast(tf, a), word::bitcast(tf, b)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  Rebind<float64_t, T> tf;
  return word::bitcast(t, word::concat_odd(tf, word::bitcast(tf, a), word::bitcast(tf, b)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  auto u_a = _mm512_shufflelo_epi16(a.v, _MM_SHUFFLE(3, 1, 3, 1));
  auto u_b = _mm512_shufflelo_epi16(b.v, _MM_SHUFFLE(3, 1, 3, 1));
  u_a = _mm512_shufflehi_epi16(u_a, _MM_SHUFFLE(3, 1, 3, 1));
  u_b = _mm512_shufflehi_epi16(u_b, _MM_SHUFFLE(3, 1, 3, 1));
  auto u = _mm512_shuffle_ps(_mm512_castsi512_ps(u_a), _mm512_castsi512_ps(u_b), _MM_SHUFFLE(3, 1, 3, 1));
  static const auto idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_epi64(idx, _mm512_castps_si512(u));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> a, Vec<T> b) {
  static const auto idx = _mm512_set_epi8(15, 13, 11, 9, 7, 5, 3, 1, 15, 13, 11, 9, 7, 5, 3, 1, 15, 13, 11, 9, 7, 5, 3, 1, 15, 13, 11, 9, 7, 5, 3, 1, 15, 13, 11, 9, 7, 5, 3, 1, 15, 13, 11, 9, 7, 5, 3, 1, 15, 13, 11, 9, 7, 5, 3, 1, 15, 13, 11, 9, 7, 5, 3, 1);
  auto u_a = _mm512_shuffle_epi8(a.v, idx);
  auto u_b = _mm512_shuffle_epi8(b.v, idx);
  auto u = _mm512_mask_blend_epi32(_cvtu32_mask16(0b1100110011001100), u_a, u_b);
  static const auto idx2 = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_epi64(idx2, u);
}
#elif VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64)>
TLV_INLINE Vec<T> concat(T t, Vec<Half<T>> v1, Vec<Half<T>> v2) {
  return Vec<T>{ v1, v2 };
}
#else
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64)>
  TLV_INLINE Vec<T> concat(T t, Vec<Half<T>> v1, Vec<Half<T>> v2) {
  return Vec<T>{ v1[0], v1[1], v2[0], v2[1] };
}
#endif // VEC_WIDTH >= 512

template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= VEC_WIDTH / 8)>
TLV_INLINE Vec<Half<T>> lower(T t, Vec<T> v) {
  Half<T> t1;
  return word::bitcast(t1, v);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 2 * VEC_WIDTH / 8)>
TLV_INLINE Vec<Half<T>> lower(T t, Vec<T> v) {
  return v[0];
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 2), TL_IF(sizeof(TypeOf<T>) == 1)>
TLV_INLINE Vec<Half<T>> upper(T t, Vec<T> v) {
  Tag<uint8_t, 16> t1;
  Half<T> tr;
  auto u = word::bitcast(t1, v);
  auto r = _mm_srli_epi16(u.v, 8);
  return word::bitcast(tr, Vec<decltype(t1)>{r});
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 4), TL_IF(sizeof(TypeOf<T>) <= 2)>
TLV_INLINE Vec<Half<T>> upper(T t, Vec<T> v) {
  Tag<uint8_t, 16> t1;
  Half<T> tr;
  auto u = word::bitcast(t1, v);
  auto r = _mm_srli_epi32(u.v, 16);
  return word::bitcast(tr, Vec<decltype(t1)>{r});
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<Half<T>> upper(T t, Vec<T> v) {
  return word::local_shuf<0, 0, 0, 1>(v);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8), TL_IF(is_none<TypeOf<T>, float32_t> && sizeof(TypeOf<T>) <= 4)>
TLV_INLINE Vec<Half<T>> upper(T t, Vec<T> v) {
  Tag<uint8_t, 16> t1;
  Half<T> tr;
  auto u = word::bitcast(t1, v);
  auto r = _mm_srli_epi64(u.v, 32);
  return word::bitcast(tr, Vec<decltype(t1)>{r});
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<Half<T>> upper(T t, Vec<T> v) {
  return word::local_shuf<0, 0, 3, 2>(v);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<Half<T>> upper(T t, Vec<T> v) {
  return word::local_shuf<0, 1>(v);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_none<TypeOf<T>, float32_t, float64_t>)>
TLV_INLINE Vec<Half<T>> upper(T t, Vec<T> v) {
  Tag<uint32_t, 4> t1;
  Half<T> tr;
  auto u = word::bitcast(t1, v);
  auto r = word::local_shuf<0, 0, 3, 2>(u);
  return word::bitcast(tr, r);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t, int32_t, uint32_t>)>
TLV_INLINE Vec<Half<T>> even(T t, Vec<T> v) {
  return word::local_shuf<2, 0, 2, 0>(v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t, int64_t, uint64_t>)>
TLV_INLINE Vec<Half<T>> even(T t, Vec<T> v) {
  return word::local_shuf<0, 0>(v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE Vec<Half<T>> even(T t, Vec<T> v) {
  return word::local_shuf<6, 4, 2, 0, 6, 4, 2, 0>(v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<Half<T>> even(T t, Vec<T> v) {
  return word::local_shuf<14, 12, 10, 8, 6, 4, 2, 0, 14, 12, 10, 8, 6, 4, 2, 0>(v);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t, int32_t, uint32_t>)>
TLV_INLINE Vec<Half<T>> odd(T t, Vec<T> v) {
  return word::local_shuf<3, 1, 3, 1>(v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t, int64_t, uint64_t>)>
TLV_INLINE Vec<Half<T>> odd(T t, Vec<T> v) {
  return word::local_shuf<1, 1>(v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE Vec<Half<T>> odd(T t, Vec<T> v) {
  return word::local_shuf<7, 5, 3, 1, 7, 5, 3, 1>(v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<Half<T>> odd(T t, Vec<T> v) {
  return word::local_shuf<15, 13, 11, 9, 7, 5, 3, 1, 15, 13, 11, 9, 7, 5, 3, 1>(v);
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t, float64_t>)>
TLV_INLINE Vec<Half<T>> upper(T t, Vec<T> v) {
  Tag<uint8_t, 32> t1;
  Tag<uint8_t, 16> t2;
  Half<T> tr;
  auto u = word::bitcast(t1, v);
  auto r = _mm256_extractf128_si256(u.v, 1);
  return word::bitcast(tr, Vec<decltype(t2)>{r});
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_none<TypeOf<T>, float32_t, float64_t>)>
TLV_INLINE Vec<Half<T>> upper(T t, Vec<T> v) {
  Tag<uint8_t, 32> t1;
  Tag<uint8_t, 16> t2;
  Half<T> tr;
  auto u = word::bitcast(t1, v);
  auto r = _mm256_extracti128_si256(u.v, 1);
  return word::bitcast(tr, Vec<decltype(t2)>{r});
}


template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<Half<T>> even(T t, Vec<T> v) {
  auto w = _mm256_permute4x64_pd(v.v, _MM_SHUFFLE(2, 0, 2, 0));
  return _mm256_castpd256_pd128(w);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<Half<T>> even(T t, Vec<T> v) {
  auto w = _mm256_permute4x64_epi64(v.v, _MM_SHUFFLE(2, 0, 2, 0));
  return _mm256_castsi256_si128(w);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<Half<T>> even(T t, Vec<T> v) {
  ViewAs<float64_t, T> tf;
  auto u = word::local_shuf<2, 0, 2, 0>(v);
  return word::bitcast(Half<T>{}, word::even(tf, word::bitcast(tf, u)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<Half<T>> even(T t, Vec<T> v) {
  ViewAs<int64_t, T> tf;
  auto u = word::local_shuf<2, 0, 2, 0>(v);
  return word::bitcast(Half<T>{}, word::even(tf, word::bitcast(tf, u)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE Vec<Half<T>> even(T t, Vec<T> v) {
  ViewAs<int64_t, T> tf;
  auto u = word::local_shuf<6, 4, 2, 0, 6, 4, 2, 0>(v);
  return word::bitcast(Half<T>{}, word::even(tf, word::bitcast(tf, u)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<Half<T>> even(T t, Vec<T> v) {
  ViewAs<int64_t, T> tf;
  auto u = word::local_shuf<14, 12, 10, 8, 6, 4, 2, 0, 14, 12, 10, 8, 6, 4, 2, 0>(v);
  return word::bitcast(Half<T>{}, word::even(tf, word::bitcast(tf, u)));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<Half<T>> odd(T t, Vec<T> v) {
  auto w = _mm256_permute4x64_pd(v.v, _MM_SHUFFLE(3, 1, 3, 1));
  return _mm256_castpd256_pd128(w);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<Half<T>> odd(T t, Vec<T> v) {
  auto w = _mm256_permute4x64_epi64(v.v, _MM_SHUFFLE(3, 1, 3, 1));
  return _mm256_castsi256_si128(w);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<Half<T>> odd(T t, Vec<T> v) {
  ViewAs<float64_t, T> tf;
  auto u = word::local_shuf<3, 1, 3, 1>(v);
  return word::bitcast(Half<T>{}, word::odd(tf, word::bitcast(tf, u)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<Half<T>> odd(T t, Vec<T> v) {
  ViewAs<int64_t, T> tf;
  auto u = word::local_shuf<3, 1, 3, 1>(v);
  return word::bitcast(Half<T>{}, word::odd(tf, word::bitcast(tf, u)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE Vec<Half<T>> odd(T t, Vec<T> v) {
  ViewAs<int64_t, T> tf;
  auto u = word::local_shuf<7, 5, 3, 1, 7, 5, 3, 1>(v);
  return word::bitcast(Half<T>{}, word::odd(tf, word::bitcast(tf, u)));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<Half<T>> odd(T t, Vec<T> v) {
  ViewAs<int64_t, T> tf;
  auto u = word::local_shuf<15, 13, 11, 9, 7, 5, 3, 1, 15, 13, 11, 9, 7, 5, 3, 1>(v);
  return word::bitcast(Half<T>{}, word::odd(tf, word::bitcast(tf, u)));
}
#else
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32)>
TLV_INLINE Vec<Half<T>> upper(T t, Vec<T> v) {
  return v[1];
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32)>
TLV_INLINE Vec<Half<T>> even(T t, Vec<T> v) {
  Half<T> th;
  auto u_lo = word::even(th, v[0]);
  auto u_hi = word::even(th, v[1]);
  return word::concat(th, u_lo, u_hi);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32)>
TLV_INLINE Vec<Half<T>> odd(T t, Vec<T> v) {
  Half<T> th;
  auto u_lo = word::odd(th, v[0]);
  auto u_hi = word::odd(th, v[1]);
  return word::concat(th, u_lo, u_hi);
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<Half<T>> upper(T t, Vec<T> v) {
  return _mm512_extractf32x8_ps(v.v, 1);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<Half<T>> upper(T t, Vec<T> v) {
  return _mm512_extractf64x4_pd(v.v, 1);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_none<TypeOf<T>, float32_t, float64_t>)>
TLV_INLINE Vec<Half<T>> upper(T t, Vec<T> v) {
  return _mm512_extracti32x8_epi32(v.v, 1);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 128)>
TLV_INLINE Vec<Half<T>> upper(T t, Vec<T> v) {
  return v[1];
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t, int64_t, uint64_t>)>
TLV_INLINE Vec<Half<T>> even(T t, Vec<T> v) {
  static const auto idx = _mm512_set_epi64(6, 4, 2, 0, 6, 4, 2, 0);
  using Ti = Rebind<Index<TypeOf<T>>, T>;
  return word::bitcast(Half<T>{}, word::shuf(v, Vec<Ti>{idx}));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t, int32_t, uint32_t>)>
TLV_INLINE Vec<Half<T>> even(T t, Vec<T> v) {
  static const auto idx = _mm512_set_epi32(14, 12, 10, 8, 6, 4, 2, 0, 14, 12, 10, 8, 6, 4, 2, 0);
  using Ti = Rebind<Index<TypeOf<T>>, T>;
  return word::bitcast(Half<T>{}, word::shuf(v, Vec<Ti>{idx}));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE Vec<Half<T>> even(T t, Vec<T> v) {
  static const auto idx = _mm512_set_epi16(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
  using Ti = Rebind<Index<TypeOf<T>>, T>;
  return word::bitcast(Half<T>{}, word::shuf(v, Vec<Ti>{idx}));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<Half<T>> even(T t, Vec<T> v) {
  static const auto idx = _mm512_set_epi8(62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
  using Ti = Rebind<Index<TypeOf<T>>, T>;
  return word::bitcast(Half<T>{}, word::shuf(v, Vec<Ti>{idx}));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t, int64_t, uint64_t>)>
TLV_INLINE Vec<Half<T>> odd(T t, Vec<T> v) {
  static const auto idx = _mm512_set_epi64(7, 5, 3, 1, 7, 5, 3, 1);
  using Ti = Rebind<Index<TypeOf<T>>, T>;
  return word::bitcast(Half<T>{}, word::shuf(v, Vec<Ti>{idx}));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t, int32_t, uint32_t>)>
TLV_INLINE Vec<Half<T>> odd(T t, Vec<T> v) {
  static const auto idx = _mm512_set_epi32(15, 13, 11, 9, 7, 5, 3, 1, 15, 13, 11, 9, 7, 5, 3, 1);
  using Ti = Rebind<Index<TypeOf<T>>, T>;
  return word::bitcast(Half<T>{}, word::shuf(v, Vec<Ti>{idx}));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE Vec<Half<T>> odd(T t, Vec<T> v) {
  static const auto idx = _mm512_set_epi16(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1, 31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);
  using Ti = Rebind<Index<TypeOf<T>>, T>;
  return word::bitcast(Half<T>{}, word::shuf(v, Vec<Ti>{idx}));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<Half<T>> odd(T t, Vec<T> v) {
  static const auto idx = _mm512_set_epi8(63, 61, 59, 57, 55, 53, 51, 49, 47, 45, 43, 41, 39, 37, 35, 33, 31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1, 63, 61, 59, 57, 55, 53, 51, 49, 47, 45, 43, 41, 39, 37, 35, 33, 31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);
  using Ti = Rebind<Index<TypeOf<T>>, T>;
  return word::bitcast(Half<T>{}, word::shuf(v, Vec<Ti>{idx}));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 128)>
TLV_INLINE Vec<Half<T>> even(T t, Vec<T> v) {
  Half<T> th;
  auto u_lo = word::even(th, v[0]);
  auto u_hi = word::even(th, v[1]);
  return word::concat(th, u_lo, u_hi);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 128)>
TLV_INLINE Vec<Half<T>> odd(T t, Vec<T> v) {
  Half<T> th;
  auto u_lo = word::odd(th, v[0]);
  auto u_hi = word::odd(th, v[1]);
  return word::concat(th, u_lo, u_hi);
}
#elif VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64)>
TLV_INLINE Vec<Half<T>> upper(T t, Vec<T> v) {
  return v[1];
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64)>
TLV_INLINE Vec<Half<T>> even(T t, Vec<T> v) {
  Half<T> th;
  auto u_lo = word::even(th, v[0]);
  auto u_hi = word::even(th, v[1]);
  return word::concat(th, u_lo, u_hi);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64)>
TLV_INLINE Vec<Half<T>> odd(T t, Vec<T> v) {
  Half<T> th;
  auto u_lo = word::odd(th, v[0]);
  auto u_hi = word::odd(th, v[1]);
  return word::concat(th, u_lo, u_hi);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                                Interleave                                  //
/* ************************************************************************** */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE V local_interleave_lower(V a, V b) {
  return _mm_unpacklo_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE V local_interleave_lower(V a, V b) {
  return _mm_unpacklo_pd(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE V local_interleave_lower(V a, V b) {
  return _mm_unpacklo_epi8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE V local_interleave_lower(V a, V b) {
  return _mm_unpacklo_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE V local_interleave_lower(V a, V b) {
  return _mm_unpacklo_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE V local_interleave_lower(V a, V b) {
  return _mm_unpacklo_epi64(a.v, b.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE V local_interleave_upper(V a, V b) {
  return _mm_unpackhi_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE V local_interleave_upper(V a, V b) {
  return _mm_unpackhi_pd(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE V local_interleave_upper(V a, V b) {
  return _mm_unpackhi_epi8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE V local_interleave_upper(V a, V b) {
  return _mm_unpackhi_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE V local_interleave_upper(V a, V b) {
  return _mm_unpackhi_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE V local_interleave_upper(V a, V b) {
  return _mm_unpackhi_epi64(a.v, b.v);
}

template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>, TL_IF(T::Bytes <= 16)>
TLV_INLINE Vec<T> interleave(T t, V a, V b) {
  return word::local_interleave_lower(word::bitcast(t, a), word::bitcast(t, b));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE V interleave_even(V a, V b) {
  Tag<float32_t, 4> tf;
  auto u = word::concat_even(tf, word::bitcast(tf, a), word::bitcast(tf, b));
  return word::bitcast(T(), word::local_shuf<3, 1, 2, 0>(u));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE V interleave_even(V a, V b) {
  Tag<float64_t, 2> tf;
  auto u = word::concat_even(tf, word::bitcast(tf, a), word::bitcast(tf, b));
  return word::bitcast(T(), u);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE V interleave_even(V a, V b) {
  Tag<int64_t, 2> tf;
  auto u = word::concat_even(tf, word::bitcast(tf, a), word::bitcast(tf, b));
  return word::bitcast(T(), u);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE V interleave_even(V a, V b) {
  Tag<int32_t, 4> tf;
  auto u = word::concat_even(tf, word::bitcast(tf, a), word::bitcast(tf, b));
  return word::bitcast(T(), word::local_shuf<3, 1, 2, 0>(u));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE V interleave_even(V a, V b) {
  auto u = _mm_shufflelo_epi16(b.v, _MM_SHUFFLE(2, 2, 0, 0));
  u = _mm_shufflehi_epi16(u, _MM_SHUFFLE(2, 2, 0, 0));
  return _mm_blend_epi16(a.v, u, 0b10101010);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE V interleave_even(V a, V b) {
  auto u_a = _mm_srli_epi16(_mm_slli_epi16(a.v, 8), 8); // clear high 8 bits
  auto u_b = _mm_slli_epi16(b.v, 8); // move low 8 to high 8 bits and clear low 8 bits
  return _mm_or_si128(u_a, u_b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE V interleave_odd(V a, V b) {
  Tag<float32_t, 4> tf;
  auto u = word::concat_odd(tf, word::bitcast(tf, a), word::bitcast(tf, b));
  return word::bitcast(T(), word::local_shuf<3, 1, 2, 0>(u));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE V interleave_odd(V a, V b) {
  Tag<float64_t, 2> tf;
  auto u = word::concat_odd(tf, word::bitcast(tf, a), word::bitcast(tf, b));
  return word::bitcast(T(), u);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE V interleave_odd(V a, V b) {
  Tag<int64_t, 2> tf;
  auto u = word::concat_odd(tf, word::bitcast(tf, a), word::bitcast(tf, b));
  return word::bitcast(T(), u);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE V interleave_odd(V a, V b) {
  Tag<int32_t, 4> tf;
  auto u = word::concat_odd(tf, word::bitcast(tf, a), word::bitcast(tf, b));
  return word::bitcast(T(), word::local_shuf<3, 1, 2, 0>(u));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE V interleave_odd(V a, V b) {
  auto u = _mm_shufflelo_epi16(a.v, _MM_SHUFFLE(3, 3, 1, 1));
  u = _mm_shufflehi_epi16(u, _MM_SHUFFLE(3, 3, 1, 1));
  return _mm_blend_epi16(u, b.v, 0b10101010);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE V interleave_odd(V a, V b) {
  auto u_a = _mm_srli_epi16(a.v, 8); // move high 8 to low 8 bits and clear high 8 bits
  auto u_b = _mm_slli_epi16(_mm_srli_epi16(b.v, 8), 8); // clear low 8 bits
  return _mm_or_si128(u_a, u_b);
}

#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE V local_interleave_lower(V a, V b) {
  return _mm256_unpacklo_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE V local_interleave_lower(V a, V b) {
  return _mm256_unpacklo_pd(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE V local_interleave_lower(V a, V b) {
  return _mm256_unpacklo_epi8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE V local_interleave_lower(V a, V b) {
  return _mm256_unpacklo_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE V local_interleave_lower(V a, V b) {
  return _mm256_unpacklo_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE V local_interleave_lower(V a, V b) {
  return _mm256_unpacklo_epi64(a.v, b.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE V local_interleave_upper(V a, V b) {
  return _mm256_unpackhi_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE V local_interleave_upper(V a, V b) {
  return _mm256_unpackhi_pd(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE V local_interleave_upper(V a, V b) {
  return _mm256_unpackhi_epi8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE V local_interleave_upper(V a, V b) {
  return _mm256_unpackhi_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE V local_interleave_upper(V a, V b) {
  return _mm256_unpackhi_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE V local_interleave_upper(V a, V b) {
  return _mm256_unpackhi_epi64(a.v, b.v);
}

template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>, TL_IF(T::Bytes == 32)>
TLV_INLINE Vec<T> interleave(T t, V a, V b) {
  using Tb = Tag<int64_t, 4>;
  static const Vec<Tb> idx = _mm256_set_epi64x(0, 1, 1, 0);
  auto va = word::shuf(word::bitcast(Tb{}, a), idx);
  auto vb = word::shuf(word::bitcast(Tb{}, b), idx);
  return word::local_interleave_lower(word::bitcast(t, va), word::bitcast(t, vb));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE V interleave_even(V a, V b) {
  auto u = _mm256_shuffle_ps(a.v, b.v, _MM_SHUFFLE(2, 0, 2, 0));
  return _mm256_shuffle_ps(u, u, _MM_SHUFFLE(3, 1, 2, 0));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE V interleave_even(V a, V b) {
  return _mm256_shuffle_pd(a.v, b.v, 0b0000);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE V interleave_even(V a, V b) {
  Tag<float64_t, 4> tf;
  auto u = word::interleave_even(word::bitcast(tf, a), word::bitcast(tf, b));
  return word::bitcast(T(), u);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE V interleave_even(V a, V b) {
  Tag<float32_t, 8> tf;
  auto u = word::interleave_even(word::bitcast(tf, a), word::bitcast(tf, b));
  return word::bitcast(T(), u);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE V interleave_even(V a, V b) {
  auto u = _mm256_shufflelo_epi16(b.v, _MM_SHUFFLE(2, 2, 0, 0));
  u = _mm256_shufflehi_epi16(u, _MM_SHUFFLE(2, 2, 0, 0));
  return _mm256_blend_epi16(a.v, u, 0b10101010);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE V interleave_even(V a, V b) {
  auto u_a = _mm256_srli_epi16(_mm256_slli_epi16(a.v, 8), 8); // clear high 8 bits
  auto u_b = _mm256_slli_epi16(b.v, 8); // move low 8 to high 8 bits and clear low 8 bits
  return _mm256_or_si256(u_a, u_b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE V interleave_odd(V a, V b) {
  auto u = _mm256_shuffle_ps(a.v, b.v, _MM_SHUFFLE(3, 1, 3, 1));
  return _mm256_shuffle_ps(u, u, _MM_SHUFFLE(3, 1, 2, 0));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE V interleave_odd(V a, V b) {
  return _mm256_shuffle_pd(a.v, b.v, 0b1111);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE V interleave_odd(V a, V b) {
  Tag<float64_t, 4> tf;
  auto u = word::interleave_odd(word::bitcast(tf, a), word::bitcast(tf, b));
  return word::bitcast(T(), u);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE V interleave_odd(V a, V b) {
  Tag<float32_t, 8> tf;
  auto u = word::interleave_odd(word::bitcast(tf, a), word::bitcast(tf, b));
  return word::bitcast(T(), u);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE V interleave_odd(V a, V b) {
  auto u = _mm256_shufflelo_epi16(a.v, _MM_SHUFFLE(3, 3, 1, 1));
  u = _mm256_shufflehi_epi16(u, _MM_SHUFFLE(3, 3, 1, 1));
  return _mm256_blend_epi16(u, b.v, 0b10101010);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE V interleave_odd(V a, V b) {
  auto u_a = _mm256_srli_epi16(a.v, 8); // move high 8 to low 8 bits and clear high 8 bits
  auto u_b = _mm256_slli_epi16(_mm256_srli_epi16(b.v, 8), 8); // clear low 8 bits
  return _mm256_or_si256(u_a, u_b);
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE V local_interleave_lower(V a, V b) {
  return _mm512_unpacklo_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE V local_interleave_lower(V a, V b) {
  return _mm512_unpacklo_pd(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE V local_interleave_lower(V a, V b) {
  return _mm512_unpacklo_epi8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE V local_interleave_lower(V a, V b) {
  return _mm512_unpacklo_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE V local_interleave_lower(V a, V b) {
  return _mm512_unpacklo_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE V local_interleave_lower(V a, V b) {
  return _mm512_unpacklo_epi64(a.v, b.v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE V local_interleave_upper(V a, V b) {
  return _mm512_unpackhi_ps(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE V local_interleave_upper(V a, V b) {
  return _mm512_unpackhi_pd(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE V local_interleave_upper(V a, V b) {
  return _mm512_unpackhi_epi8(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE V local_interleave_upper(V a, V b) {
  return _mm512_unpackhi_epi16(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE V local_interleave_upper(V a, V b) {
  return _mm512_unpackhi_epi32(a.v, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE V local_interleave_upper(V a, V b) {
  return _mm512_unpackhi_epi64(a.v, b.v);
}

template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>, TL_IF(T::Bytes == 64)>
TLV_INLINE Vec<T> interleave(T t, V a, V b) {
  using Tb = Tag<int64_t, 8>;
  static const Vec<Tb> idx = _mm512_set_epi64(2, 3, 3, 2, 0, 1, 1, 0);
  auto va = word::shuf(word::bitcast(Tb{}, a), idx);
  auto vb = word::shuf(word::bitcast(Tb{}, b), idx);
  return word::local_interleave_lower(word::bitcast(t, va), word::bitcast(t, vb));
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE V interleave_even(V a, V b) {
  auto u = _mm512_shuffle_ps(a.v, b.v, _MM_SHUFFLE(2, 0, 2, 0));
  return _mm512_shuffle_ps(u, u, _MM_SHUFFLE(3, 1, 2, 0));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE V interleave_even(V a, V b) {
  return _mm512_shuffle_pd(a.v, b.v, 0b00000000);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE V interleave_even(V a, V b) {
  Tag<float64_t, 8> tf;
  auto u = word::interleave_even(word::bitcast(tf, a), word::bitcast(tf, b));
  return word::bitcast(T(), u);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE V interleave_even(V a, V b) {
  Tag<float32_t, 16> tf;
  auto u = word::interleave_even(word::bitcast(tf, a), word::bitcast(tf, b));
  return word::bitcast(T(), u);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE V interleave_even(V a, V b) {
  auto u = _mm512_shufflelo_epi16(b.v, _MM_SHUFFLE(2, 2, 0, 0));
  u = _mm512_shufflehi_epi16(u, _MM_SHUFFLE(2, 2, 0, 0));
  // 0b101010...101010
  return _mm512_mask_blend_epi16(_cvtu32_mask32(0xAAAAAAAA), a.v, u);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE V interleave_even(V a, V b) {
  auto u_a = _mm512_srli_epi16(_mm512_slli_epi16(a.v, 8), 8); // clear high 8 bits
  auto u_b = _mm512_slli_epi16(b.v, 8); // move low 8 to high 8 bits and clear low 8 bits
  return _mm512_or_si512(u_a, u_b);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE V interleave_odd(V a, V b) {
  auto u = _mm512_shuffle_ps(a.v, b.v, _MM_SHUFFLE(3, 1, 3, 1));
  return _mm512_shuffle_ps(u, u, _MM_SHUFFLE(3, 1, 2, 0));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE V interleave_odd(V a, V b) {
  return _mm512_shuffle_pd(a.v, b.v, 0b11111111);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE V interleave_odd(V a, V b) {
  Tag<float64_t, 8> tf;
  auto u = word::interleave_odd(word::bitcast(tf, a), word::bitcast(tf, b));
  return word::bitcast(T(), u);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE V interleave_odd(V a, V b) {
  Tag<float32_t, 16> tf;
  auto u = word::interleave_odd(word::bitcast(tf, a), word::bitcast(tf, b));
  return word::bitcast(T(), u);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE V interleave_odd(V a, V b) {
  auto u = _mm512_shufflelo_epi16(a.v, _MM_SHUFFLE(3, 3, 1, 1));
  u = _mm512_shufflehi_epi16(u, _MM_SHUFFLE(3, 3, 1, 1));
  // 0b101010...101010
  return _mm512_mask_blend_epi16(_cvtu32_mask32(0xAAAAAAAA), u, b.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE V interleave_odd(V a, V b) {
  auto u_a = _mm512_srli_epi16(a.v, 8); // move high 8 to low 8 bits and clear high 8 bits
  auto u_b = _mm512_slli_epi16(_mm512_srli_epi16(b.v, 8), 8); // clear low 8 bits
  return _mm512_or_si512(u_a, u_b);
}
#endif // VEC_WIDTH >= 512

/* ************************************************************************** */
//                                Constructors                                //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
return _mm_set1_ps(v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
return _mm_set1_pd(v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
return _mm_set1_epi64x((long long)(v));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
return _mm_set1_epi32(int32_t(v));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
return _mm_set1_epi16(int16_t(v));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
return _mm_set1_epi8(int8_t(v));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
union { bfloat16_t b; int16_t i; } u { .b = v };
return _mm_set1_epi16(u.i);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float16_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
#ifdef HAS_AVX512_FP16
return _mm256_castph_si256(_mm256_set1_ph(v));
#else
union { float16_t b; int16_t i; } u { .b = v };
  return _mm_set1_epi16(u.i);
#endif
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
  return _mm256_set1_ps(v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
  return _mm256_set1_pd(v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
  return _mm256_set1_epi64x((long long)(v));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
  return _mm256_set1_epi32(int32_t(v));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
  return _mm256_set1_epi16(int16_t(v));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
  return _mm256_set1_epi8(int8_t(v));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
  union { bfloat16_t b; int16_t i; } u { .b = v };
  return _mm256_set1_epi16(u.i);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float16_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
  #ifdef HAS_AVX512_FP16
  return _mm256_castph_si256(_mm256_set1_ph(v));
  #else
  union { float16_t b; int16_t i; } u { .b = v };
  return _mm256_set1_epi16(u.i);
  #endif
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
  return _mm512_set1_ps(v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
  return _mm512_set1_pd(v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
  return _mm512_set1_epi64((long long)(v));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
  return _mm512_set1_epi32(int32_t(v));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
  return _mm512_set1_epi16(int16_t(v));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
  return _mm512_set1_epi8(int8_t(v));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, bfloat16_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
  union { bfloat16_t b; int16_t i; } u { .b = v };
  return _mm512_set1_epi16(u.i);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float16_t>)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v) {
  #ifdef HAS_AVX512_FP16
  return _mm512_castph_si512(_mm512_set1_ph(v));
  #else
  union { float16_t b; int16_t i; } u { .b = v };
  return _mm512_set1_epi16(u.i);
  #endif
}
#endif // VEC_WIDTH >= 512

/**
 * Masked fill
 */
template <typename T>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v, Mask<T> m, Vec<T> default_v) {
return word::blend(default_v, m, word::fill(t, v));
}
template <typename T>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> v, nint_t n, Vec<T> default_v) {
CT_ASSERT(0 <= n && n <= T::N, "%zd !in 0..%zd", n, T::N);
auto m = word::mwhilelt(t, 0, n);
return word::fill(t, v, m, default_v);
}


template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> zeros(T t) {
return _mm_setzero_ps();
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> zeros(T t) {
return _mm_setzero_pd();
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_none<TypeOf<T>, float32_t, float64_t>)>
TLV_INLINE Vec<T> zeros(T t) {
return _mm_setzero_si128();
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> zeros(T t) {
  return _mm256_setzero_ps();
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> zeros(T t) {
  return _mm256_setzero_pd();
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_none<TypeOf<T>, float32_t, float64_t>)>
TLV_INLINE Vec<T> zeros(T t) {
  return _mm256_setzero_si256();
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> zeros(T t) {
  return _mm512_setzero_ps();
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> zeros(T t) {
  return _mm512_setzero_pd();
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_none<TypeOf<T>, float32_t, float64_t>)>
TLV_INLINE Vec<T> zeros(T t) {
  return _mm512_setzero_si512();
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                            Get Vector Element                              //
/* ************************************************************************** */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE TypeOf<T> get(V v, nint_t i) {
  Rebind<Index<TypeOf<T>>, T> ti;
  auto u = word::shuf(v, Vec<decltype(ti)>{_mm_cvtsi32_si128(i)});
  return _mm_cvtss_f32(u.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE TypeOf<T> get(V v, nint_t i) {
  Rebind<Index<TypeOf<T>>, T> ti;
  auto u = word::shuf(v, Vec<decltype(ti)>{_mm_cvtsi64_si128(i)});
  return _mm_cvtsd_f64(u.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE TypeOf<T> get(V v, nint_t i) {
  Rebind<Index<TypeOf<T>>, T> ti;
  auto u = word::shuf(v, Vec<decltype(ti)>{_mm_cvtsi64_si128(i)});
  union { int64_t i; TypeOf<T> j; } U { .i = _mm_cvtsi128_si64(u.v) };
  return U.j;
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t, int16_t, uint16_t, int8_t, uint8_t, float16_t, bfloat16_t>)>
TLV_INLINE TypeOf<T> get(V v, nint_t i) {
  Rebind<Index<TypeOf<T>>, T> ti;
  auto u = word::shuf(v, Vec<decltype(ti)>{_mm_cvtsi32_si128(i)});
  union { int32_t i; TypeOf<T> j; } U { .i = _mm_cvtsi128_si32(u.v) };
  return U.j;
}

#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE TypeOf<T> get(V v, nint_t i) {
  Rebind<Index<TypeOf<T>>, T> ti; ViewAs<Index<TypeOf<T>>, T> th;
  auto u = word::shuf(v, word::bitcast(ti, Vec<decltype(th)>{_mm256_castsi128_si256(_mm_cvtsi32_si128(i))}));
  return _mm256_cvtss_f32(u.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE TypeOf<T> get(V v, nint_t i) {
  Rebind<Index<TypeOf<T>>, T> ti; ViewAs<Index<TypeOf<T>>, T> th;
  auto u = word::shuf(v, word::bitcast(ti, Vec<decltype(th)>{_mm256_castsi128_si256(_mm_cvtsi32_si128(i))}));
  return _mm256_cvtsd_f64(u.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE TypeOf<T> get(V v, nint_t i) {
  Rebind<Index<TypeOf<T>>, T> ti; ViewAs<Index<TypeOf<T>>, T> th;
  auto u = word::shuf(v, word::bitcast(ti, Vec<decltype(th)>{_mm256_castsi128_si256(_mm_cvtsi64_si128(i))}));
  union { int64_t i; TypeOf<T> j; } U { .i = _mm_cvtsi128_si64(_mm256_castsi256_si128(u.v)) };
  return U.j;
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t, int16_t, uint16_t, int8_t, uint8_t, float16_t, bfloat16_t>)>
TLV_INLINE TypeOf<T> get(V v, nint_t i) {
  Rebind<Index<TypeOf<T>>, T> ti; ViewAs<Index<TypeOf<T>>, T> th;
  auto u = word::shuf(v, word::bitcast(ti, Vec<decltype(th)>{_mm256_castsi128_si256(_mm_cvtsi32_si128(i))}));
  union { int32_t i; TypeOf<T> j; } U { .i = _mm256_cvtsi256_si32(u.v) };
  return U.j;
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE TypeOf<T> get(V v, nint_t i) {
  Rebind<Index<TypeOf<T>>, T> ti; ViewAs<Index<TypeOf<T>>, T> th;
  auto u = word::shuf(v, word::bitcast(ti, Vec<decltype(th)>{_mm512_castsi128_si512(_mm_cvtsi64_si128(i))}));
  return _mm512_cvtss_f32(u.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE TypeOf<T> get(V v, nint_t i) {
  Rebind<Index<TypeOf<T>>, T> ti; ViewAs<Index<TypeOf<T>>, T> th;
  auto u = word::shuf(v, word::bitcast(ti, Vec<decltype(th)>{_mm512_castsi128_si512(_mm_cvtsi64_si128(i))}));
  return _mm512_cvtsd_f64(u.v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE TypeOf<T> get(V v, nint_t i) {
  Rebind<Index<TypeOf<T>>, T> ti; ViewAs<Index<TypeOf<T>>, T> th;
  auto u = word::shuf(v, word::bitcast(ti, Vec<decltype(th)>{_mm512_castsi128_si512(_mm_cvtsi64_si128(i))}));
  union { int64_t i; TypeOf<T> j; } U { .i = _mm_cvtsi128_si64(_mm512_castsi512_si128(u.v)) };
  return U.j;
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t, int16_t, uint16_t, int8_t, uint8_t, float16_t, bfloat16_t>)>
TLV_INLINE TypeOf<T> get(V v, nint_t i) {
  Rebind<Index<TypeOf<T>>, T> ti; ViewAs<Index<TypeOf<T>>, T> th;
  auto u = word::shuf(v, word::bitcast(ti, Vec<decltype(th)>{_mm512_castsi128_si512(_mm_cvtsi64_si128(i))}));
  union { int32_t i; TypeOf<T> j; } U { .i = _mm512_cvtsi512_si32(u.v) };
  return U.j;
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                            Set Vector Element                              //
/* ************************************************************************** */
#ifdef HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  auto m = _cvtu32_mask8(1u << (unsigned) i);
  return _mm_mask_mov_ps(v.v, m, word::fill(T(), x).v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  auto m = _cvtu32_mask8(1u << (unsigned) i);
  return _mm_mask_mov_pd(v.v, m, word::fill(T(), x).v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  auto m = _cvtu32_mask16(1u << (unsigned) i);
  return _mm_mask_mov_epi8(v.v, m, word::fill(T(), x).v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  auto m = _cvtu32_mask8(1u << (unsigned) i);
  return _mm_mask_mov_epi16(v.v, m, word::fill(T(), x).v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  auto m = _cvtu32_mask8(1u << (unsigned) i);
  return _mm_mask_mov_epi32(v.v, m, word::fill(T(), x).v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  auto m = _cvtu32_mask8(1u << (unsigned) i);
  return _mm_mask_mov_epi64(v.v, m, word::fill(T(), x).v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  auto m = _cvtu32_mask8(1u << (unsigned) i);
  return _mm256_mask_mov_ps(v.v, m, word::fill(T(), x).v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  auto m = _cvtu32_mask8(1u << (unsigned) i);
  return _mm256_mask_mov_pd(v.v, m, word::fill(T(), x).v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  auto m = _cvtu32_mask32(1u << (unsigned) i);
  return _mm256_mask_mov_epi8(v.v, m, word::fill(T(), x).v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  auto m = _cvtu32_mask16(1u << (unsigned) i);
  return _mm256_mask_mov_epi16(v.v, m, word::fill(T(), x).v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  auto m = _cvtu32_mask8(1u << (unsigned) i);
  return _mm256_mask_mov_epi32(v.v, m, word::fill(T(), x).v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  auto m = _cvtu32_mask8(1u << (unsigned) i);
  return _mm256_mask_mov_epi64(v.v, m, word::fill(T(), x).v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  auto m = _cvtu32_mask16(1u << (unsigned) i);
  return _mm512_mask_mov_ps(v.v, m, word::fill(T(), x).v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  auto m = _cvtu32_mask8(1u << (unsigned) i);
  return _mm512_mask_mov_pd(v.v, m, word::fill(T(), x).v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  auto m = _cvtu64_mask64(1uLL << (unsigned) i);
  return _mm512_mask_mov_epi8(v.v, m, word::fill(T(), x).v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  auto m = _cvtu32_mask32(1u << (unsigned) i);
  return _mm512_mask_mov_epi16(v.v, m, word::fill(T(), x).v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  auto m = _cvtu32_mask16(1u << (unsigned) i);
  return _mm512_mask_mov_epi32(v.v, m, word::fill(T(), x).v);
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  auto m = _cvtu32_mask8(1u << (unsigned) i);
  return _mm512_mask_mov_epi64(v.v, m, word::fill(T(), x).v);
}
#else // HAS_AVX512DQ
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t, int32_t, uint32_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  static const __m128i idx = _mm_setr_epi32(0, 1, 2, 3);
  auto m = _mm_cmpeq_epi32(_mm_set1_epi32(i), idx);
  return word::blend(v, Mask<T>{m}, word::fill(T(), x));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t, int64_t, uint64_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  static const __m128i idx = _mm_setr_epi32(0, 0, 1, 1);
  auto m = _mm_cmpeq_epi32(_mm_set1_epi32(i), idx);
  return word::blend(v, Mask<T>{m}, word::fill(T(), x));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  static const __m128i idx = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
  auto m = _mm_cmpeq_epi16(_mm_set1_epi16(i), idx);
  return word::blend(v, Mask<T>{m}, word::fill(T(), x));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  static const __m128i idx = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  auto m = _mm_cmpeq_epi8(_mm_set1_epi8(i), idx);
  return word::blend(v, Mask<T>{m}, word::fill(T(), x));
}

#if VEC_WIDTH >= 256
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t, int32_t, uint32_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  static const __m256i idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  auto m = _mm256_cmpeq_epi32(_mm256_set1_epi32(i), idx);
  return word::blend(v, Mask<T>{m}, word::fill(T(), x));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t, int64_t, uint64_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  static const __m256i idx = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
  auto m = _mm256_cmpeq_epi32(_mm256_set1_epi32(i), idx);
  return word::blend(v, Mask<T>{m}, word::fill(T(), x));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  static const __m256i idx = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  auto m = _mm256_cmpeq_epi16(_mm256_set1_epi16(i), idx);
  return word::blend(v, Mask<T>{m}, word::fill(T(), x));
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE V set(V v, nint_t i, TypeOf<T> x) {
  static const __m256i idx = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
  auto m = _mm256_cmpeq_epi8(_mm256_set1_epi8(i), idx);
  return word::blend(v, Mask<T>{m}, word::fill(T(), x));
}
#endif // VEC_WIDTH >= 256
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                               Get Mask Bit                                 //
/* ************************************************************************** */
#ifdef HAS_AVX512DQ
template <TLV_DECL_MASK(M), TL_IF(M::N <= 8)>
TLV_INLINE bool get(M m, nint_t i) {
  return (_cvtmask8_u32(m.v) >> i) & 1;
}
template <TLV_DECL_MASK(M), TL_IF(M::N == 16)>
TLV_INLINE bool get(M m, nint_t i) {
  return (_cvtmask16_u32(m.v) >> i) & 1;
}
template <TLV_DECL_MASK(M), TL_IF(M::N == 32)>
TLV_INLINE bool get(M m, nint_t i) {
  return (_cvtmask32_u32(m.v) >> i) & 1;
}
template <TLV_DECL_MASK(M), TL_IF(M::N == 64)>
TLV_INLINE bool get(M m, nint_t i) {
  return (_cvtmask64_u64(m.v) >> i) & 1;
}
#else // HAS_AVX512DQ
template <TLV_DECL_MASK(M), TL_IF(M::ElSize == 1)>
TLV_INLINE bool get(M m, nint_t i) {
  Tag<int8_t, M::N> t;
  return !!get(Vec<decltype(t)>{m.v}, i);
}
template <TLV_DECL_MASK(M), TL_IF(M::ElSize == 2)>
TLV_INLINE bool get(M m, nint_t i) {
  Tag<int16_t, M::N> t;
  return !!get(Vec<decltype(t)>{m.v}, i);
}
template <TLV_DECL_MASK(M), TL_IF(M::ElSize == 4)>
TLV_INLINE bool get(M m, nint_t i) {
  Tag<int32_t, M::N> t;
  return !!get(Vec<decltype(t)>{m.v}, i);
}
template <TLV_DECL_MASK(M), TL_IF(M::ElSize == 8)>
TLV_INLINE bool get(M m, nint_t i) {
  Tag<int64_t, M::N> t;
  return !!get(Vec<decltype(t)>{m.v}, i);
}
#endif // HAS_AVX512DQ


/* ************************************************************************** */
//                               Set Mask Bit                                 //
/* ************************************************************************** */
#ifdef HAS_AVX512DQ
template <TLV_DECL_MASK(M), TL_IF(M::N <= 8)>
TLV_INLINE M set(M m, nint_t i, bool x) {
  uint32_t bits = _cvtmask8_u32(m.v);
  bits = (bits & ~(1u << i)) | ((x ? 1u : 0u) << i);
  return _cvtu32_mask8(bits);
}
template <TLV_DECL_MASK(M), TL_IF(M::N == 16)>
TLV_INLINE M set(M m, nint_t i, bool x) {
  uint32_t bits = _cvtmask16_u32(m.v);
  bits = (bits & ~(1u << i)) | ((x ? 1u : 0u) << i);
  return _cvtu32_mask16(bits);
}
template <TLV_DECL_MASK(M), TL_IF(M::N == 32)>
TLV_INLINE M set(M m, nint_t i, bool x) {
  uint32_t bits = _cvtmask32_u32(m.v);
  bits = (bits & ~(1u << i)) | ((x ? 1u : 0u) << i);
  return _cvtu32_mask32(bits);
}
template <TLV_DECL_MASK(M), TL_IF(M::N == 64)>
TLV_INLINE M set(M m, nint_t i, bool x) {
  uint64_t bits = _cvtmask64_u64(m.v);
  bits = (bits & ~(1ull << i)) | ((x ? 1ull : 0ull) << i);
  return _cvtu64_mask64(bits);
}
#else // HAS_AVX512DQ
template <TLV_DECL_MASK(M), TL_IF(M::ElSize == 1)>
TLV_INLINE M set(M m, nint_t i, bool x) {
  Tag<int8_t, M::N> t;
  return set(Vec<decltype(t)>{m.v}, i, x ? int8_t(-1) : 0).v;
}
template <TLV_DECL_MASK(M), TL_IF(M::ElSize == 2)>
TLV_INLINE M set(M m, nint_t i, bool x) {
  Tag<int16_t, M::N> t;
  return set(Vec<decltype(t)>{m.v}, i, x ? int16_t(-1) : 0).v;
}
template <TLV_DECL_MASK(M), TL_IF(M::ElSize == 4)>
TLV_INLINE M set(M m, nint_t i, bool x) {
  Tag<int32_t, M::N> t;
  return set(Vec<decltype(t)>{m.v}, i, x ? int32_t(-1) : 0).v;
}
template <TLV_DECL_MASK(M), TL_IF(M::ElSize == 8)>
TLV_INLINE M set(M m, nint_t i, bool x) {
  Tag<int64_t, M::N> t;
  return set(Vec<decltype(t)>{m.v}, i, x ? int64_t(-1) : 0).v;
}
#endif // HAS_AVX512DQ

} // namespace word
} // namespace ct::tl::vec::CPU_CAPABILITY
//@formatter:on

#endif //CTORCH_X86_BASIC_H
