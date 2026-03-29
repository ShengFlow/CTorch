//
// Created by renyz on 2026/3/25.
//

#ifndef CTORCH_X86_LOADSTORE_H
#define CTORCH_X86_LOADSTORE_H

#include "tl/cpu/impl/x86_Types.h"
#include "tl/cpu/impl/x86_Basic.h"

//@formatter:off
namespace ct::tl::vec::CPU_CAPABILITY {
namespace word {

/* ************************************************************************** */
//                              Consecutive Load                              //
/* ************************************************************************** */
namespace details {
#ifdef HAS_AVX512DQ
template <TLV_DECL_TAG(T)>
TLV_INLINE Mask<T> restrict_mask_range(T t, Mask<T> m) {
  if constexpr (T::Bytes == 16 || T::Bytes == 32 || T::Bytes == 64) {
    return m;
  }
  constexpr nint_t N = T::N;
  constexpr uint64_t M = N == 64 ? -1 : (uint64_t(1) << N) - 1;
  if constexpr (N > 32) {
    return m.v & M;
  } else {
    return m.v & uint32_t(M);
  }
}
#else // HAS_AVX512DQ
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 1)>
TLV_INLINE Mask<T> restrict_mask_range(T t, Mask<T> m) {
  return _mm_and_si128(m.v, _mm_set_epi32(0, 0, 0, 0x000000FF));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 2)>
TLV_INLINE Mask<T> restrict_mask_range(T t, Mask<T> m) {
  return _mm_and_si128(m.v, _mm_set_epi32(0, 0, 0, 0x0000FFFF));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 4)>
TLV_INLINE Mask<T> restrict_mask_range(T t, Mask<T> m) {
  return _mm_and_si128(m.v, _mm_set_epi32(0, 0, 0, (int)0xFFFFFFFF));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8)>
TLV_INLINE Mask<T> restrict_mask_range(T t, Mask<T> m) {
  return _mm_move_epi64(m.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16)>
TLV_INLINE Mask<T> restrict_mask_range(T t, Mask<T> m) {
  return m;
}
#endif // HAS_AVX512DQ
} // namespace details

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm_load_sd(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm_loadu_pd(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 4), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm_load_ss(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p) {
  Tag<float64_t, 1> t1;
  return word::bitcast(t, word::loadu(t1, (const float64_t *) p));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm_loadu_ps(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 1), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm_cvtsi32_si128((int32_t)((const uint8_t *) p)[0]);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 2), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm_cvtsi32_si128((int32_t)((const uint16_t *) p)[0]);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 4), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm_cvtsi32_si128(((const int32_t *) p)[0]);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm_cvtsi64_si128(((const int64_t *) p)[0]);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm_loadu_si128((const __m128i*) p);
}


template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> load(T t, const TypeOf<T> * p) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return _mm_load_pd(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> load(T t, const TypeOf<T> * p) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return _mm_load_ps(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
TLV_INLINE Vec<T> load(T t, const TypeOf<T> * p) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return _mm_load_si128((const __m128i*) p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes < 16)>
TLV_INLINE Vec<T> load(T t, const TypeOf<T> * p) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return word::loadu(t, p);
}


#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm256_loadu_pd(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm256_loadu_ps(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm256_loadu_si256((const __m256i*) p);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> load(T t, const TypeOf<T> * p) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return _mm256_load_pd(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> load(T t, const TypeOf<T> * p) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return _mm256_load_ps(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
TLV_INLINE Vec<T> load(T t, const TypeOf<T> * p) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return _mm256_load_si256((const __m256i*) p);
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm512_loadu_pd(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm512_loadu_ps(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p) {
  return _mm512_loadu_si512((const __m256i*) p);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> load(T t, const TypeOf<T> * p) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return _mm512_load_pd(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> load(T t, const TypeOf<T> * p) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return _mm512_load_ps(p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
TLV_INLINE Vec<T> load(T t, const TypeOf<T> * p) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return _mm512_load_si512((const __m512i*) p);
}
#endif // VEC_WIDTH >= 512


#ifdef HAS_AVX512DQ
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm_mask_loadu_ps(default_v.v, details::restrict_mask_range(t, m).v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm_mask_loadu_pd(default_v.v, details::restrict_mask_range(t, m).v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm_mask_loadu_epi8(default_v.v, details::restrict_mask_range(t, m).v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm_mask_loadu_epi16(default_v.v, details::restrict_mask_range(t, m).v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm_mask_loadu_epi32(default_v.v, details::restrict_mask_range(t, m).v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm_mask_loadu_epi64(default_v.v, details::restrict_mask_range(t, m).v, p);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm256_mask_loadu_ps(default_v.v, m.v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm256_mask_loadu_pd(default_v.v, m.v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm256_mask_loadu_epi8(default_v.v, m.v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm256_mask_loadu_epi16(default_v.v, m.v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm256_mask_loadu_epi32(default_v.v, m.v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm256_mask_loadu_epi64(default_v.v, m.v, p);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm512_mask_loadu_ps(default_v.v, m.v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm512_mask_loadu_pd(default_v.v, m.v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm512_mask_loadu_epi8(default_v.v, m.v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm512_mask_loadu_epi16(default_v.v, m.v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm512_mask_loadu_epi32(default_v.v, m.v, p);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return _mm512_mask_loadu_epi64(default_v.v, m.v, p);
}
#else // HAS_AVX512DQ
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return word::blend(default_v, m, Vec<T>{
    _mm_maskload_ps(p, details::restrict_mask_range(t, m).v)
  });
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return word::blend(default_v, m, Vec<T>{
      _mm_maskload_pd(p, details::restrict_mask_range(t, m).v)
  });
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  #ifdef HAS_AVX2
  return word::blend(default_v, m, Vec<T>{
      _mm_maskload_epi32((const int *)p, details::restrict_mask_range(t, m).v)
  });
  #else
  Rebind<float32_t, T> t1;
  return word::bitcast(t, word::loadu(t1, (const float32_t *) p, m.v, word::bitcast(t1, default_v)));
  #endif
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  #ifdef HAS_AVX2
  return word::blend(default_v, m, Vec<T>{
      _mm_maskload_epi64((const long long *)p, details::restrict_mask_range(t, m).v)
  });
  #else
  Rebind<float64_t, T> t1;
  return word::bitcast(t, word::loadu(t1, (const float64_t *) p, m.v, word::bitcast(t1, default_v)));
  #endif
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  // if p + 15 still in the same page as p
  if (((nuint_t(p) & 0xfff) + 15) <= 0xfff) {
    // excessive reads are safe
    return word::blend(default_v, m, word::loadu(t, p));
  } else {
    // Fallback to scalar implementation, TODO slow
    auto mask = details::restrict_mask_range(t, m);
    union { int16_t i[8]; __m128i m; } V{.m = default_v.v}, M{.m = mask.v};
    alignas(16) int16_t S[8];
    auto P = (const int16_t*) p;
    for (int i = 0; i < 8; ++i) S[i] = M.i[i] ? P[i] : V.i[i];
    return _mm_load_si128((const __m128i *)S);
  }
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  // if p + 15 still in the same page as p
  if (((nuint_t(p) & 0xfff) + 15) <= 0xfff) {
    // excessive reads are safe
    return word::blend(default_v, m, word::loadu(t, p));
  } else {
    // Fallback to scalar implementation, TODO slow
    auto mask = details::restrict_mask_range(t, m);
    union { int8_t i[16]; __m128i m; } V{.m = default_v.v}, M{.m = mask.v};
    alignas(16) int8_t S[16];
    auto P = (const int8_t*) p;
    for (int i = 0; i < 16; ++i) S[i] = M.i[i] ? P[i] : V.i[i];
    return _mm_load_si128((const __m128i *)S);
  }
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return word::blend(default_v, m, Vec<T>{_mm256_maskload_ps(p, m.v)});
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return word::blend(default_v, m, Vec<T>{_mm256_maskload_pd(p, m.v)});
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return word::blend(default_v, m, Vec<T>{_mm256_maskload_epi32((const int *)p, m.v)});
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  return word::blend(default_v, m, Vec<T>{_mm256_maskload_epi64((const long long *)p, m.v)});
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  // if p + 15 still in the same page as p
  if (((nuint_t(p) & 0xfff) + 31) <= 0xfff) {
    // excessive reads are safe
    return word::blend(default_v, m, word::loadu(t, p));
  } else {
    // Fallback to scalar implementation, TODO slow
    union { int16_t i[16]; __m256i m; } V{.m = default_v.v}, M{.m = m.v};
    alignas(16) int16_t S[16];
    auto P = (const int16_t*) p;
    for (int i = 0; i < 16; ++i) S[i] = M.i[i] ? P[i] : V.i[i];
    return _mm256_load_si256((const __m256i *)S);
  }
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  // if p + 31 still in the same page as p
  if (((nuint_t(p) & 0xfff) + 31) <= 0xfff) {
    // excessive reads are safe
    return word::blend(default_v, m, word::loadu(t, p));
  } else {
    // Fallback to scalar implementation, TODO slow
    union { int8_t i[32]; __m256i m; } V{.m = default_v.v}, M{.m = m.v};
    alignas(32) int8_t S[32];
    auto P = (const int8_t*) p;
    for (int i = 0; i < 32; ++i) S[i] = M.i[i] ? P[i] : V.i[i];
    return _mm256_load_si256((const __m256i *)S);
  }
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
  #error "Unreachable"
#endif // VEC_WIDTH >= 512
#endif // HAS_AVX512DQ

template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> load(T t, const TypeOf<T> * p, Mask<T> m, Vec<T> default_v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return word::loadu(t, p, m, default_v);
}

template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T> * p, nint_t n, Vec<T> default_v) {
  CT_ASSERT(0 <= n && n <= T::N, "%zd !in 0..%zd", n, T::N);
  auto m = word::mwhilelt(t, 0, n);
  return word::loadu(t, p, m, default_v);
}

template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> load(T t, const TypeOf<T> * p, nint_t n, Vec<T> default_v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  return word::loadu(t, p, n, default_v);
}


/* ************************************************************************** */
//                             Consecutive Store                              //
/* ************************************************************************** */
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm_store_sd(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm_storeu_pd(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 4), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm_store_ss(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  Tag<float64_t, 1> t1;
  word::storeu(t1, (float64_t *) p, word::bitcast(t1, v));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm_storeu_ps(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm_storeu_si128((__m128i *) p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 8), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm_storeu_si64(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 4), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm_storeu_si32(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 2), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm_storeu_si16(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 1), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  auto r = _mm_cvtsi128_si32(v.v);
  ((int8_t *) p)[0] = (int8_t)r;
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes < 16)>
TLV_INLINE void store(T t, TypeOf<T> * p, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  word::storeu(t, p, v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE void store(T t, TypeOf<T> * p, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm_store_ps(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE void store(T t, TypeOf<T> * p, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm_store_pd(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
TLV_INLINE void store(T t, TypeOf<T> * p, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm_store_si128((__m128i *) p, v.v);
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm256_storeu_pd(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm256_storeu_ps(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm256_storeu_si256((__m256i *) p, v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE void store(T t, TypeOf<T> * p, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm256_store_ps(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE void store(T t, TypeOf<T> * p, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm256_store_pd(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
TLV_INLINE void store(T t, TypeOf<T> * p, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm256_store_si256((__m256i *) p, v.v);
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm512_storeu_pd(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm512_storeu_ps(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Vec<T> v) {
  _mm512_storeu_si512((__m512i *) p, v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE void store(T t, TypeOf<T> * p, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm512_store_ps(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE void store(T t, TypeOf<T> * p, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm512_store_pd(p, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_int<TypeOf<T>> || is_small_float<TypeOf<T>>)>
TLV_INLINE void store(T t, TypeOf<T> * p, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm512_store_si512((__m512i *) p, v.v);
}
#endif// VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm_mask_storeu_ps(p, details::restrict_mask_range(t, m).v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm_mask_storeu_pd(p, details::restrict_mask_range(t, m).v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm_mask_storeu_epi8(p, details::restrict_mask_range(t, m).v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm_mask_storeu_epi16(p, details::restrict_mask_range(t, m).v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm_mask_storeu_epi32(p, details::restrict_mask_range(t, m).v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm_mask_storeu_epi64(p, details::restrict_mask_range(t, m).v, v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm256_mask_storeu_ps(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm256_mask_storeu_pd(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm256_mask_storeu_epi8(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm256_mask_storeu_epi16(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm256_mask_storeu_epi32(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm256_mask_storeu_epi64(p, m.v, v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {

  _mm512_mask_storeu_ps(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm512_mask_storeu_pd(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int8_t, uint8_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm512_mask_storeu_epi8(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm512_mask_storeu_epi16(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm512_mask_storeu_epi32(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm512_mask_storeu_epi64(p, m.v, v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes < 16)>
TLV_INLINE void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  word::storeu(t, p, m, v);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm_mask_store_ps(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm_mask_store_pd(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(sizeof(TypeOf<T>) < 4)>
TLV_INLINE void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  word::storeu(t, p, m, v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm_mask_store_epi32(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm_mask_store_epi64(p, m.v, v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm256_mask_store_ps(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm256_mask_store_pd(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm256_mask_store_epi32(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm256_mask_store_epi64(p, m.v, v.v);
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm512_mask_store_ps(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm512_mask_store_pd(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm512_mask_store_epi32(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  _mm512_mask_store_epi64(p, m.v, v.v);
}
#else // HAS_AVX512DQ
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm_maskstore_ps(p, details::restrict_mask_range(t, m).v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm_maskstore_pd(p, details::restrict_mask_range(t, m).v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  #ifdef HAS_AVX2
  _mm_maskstore_epi32((int32_t *) p, details::restrict_mask_range(t, m).v, v.v);
  #else
  Rebind<float32_t, T> t1;
  word::storeu(t1, (float32_t *) p, m.v, word::bitcast(t1, v));
  #endif
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  #ifdef HAS_AVX2
  _mm_maskstore_epi64((long long *) p, details::restrict_mask_range(t, m).v, v.v);
  #else
  Rebind<float64_t, T> t1;
  word::storeu(t1, (float64_t *) p, m.v, word::bitcast(t1, v));
  #endif
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t, int8_t, uint8_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  // TODO non-temporal memory hint?
  _mm_maskmoveu_si128(v.v, details::restrict_mask_range(t, m).v, (char *) p);
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm256_maskstore_ps(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm256_maskstore_pd(p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm256_maskstore_epi32((int32_t *) p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  _mm256_maskstore_epi64((long long *) p, m.v, v.v);
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int16_t, uint16_t, float16_t, bfloat16_t, int8_t, uint8_t>)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  ViewAs<Index<TypeOf<T>>, T> tm;
  auto lo = lower(t, v); auto lom = lower(tm, Vec<decltype(tm)>{m.v});
  auto hi = upper(t, v); auto him = upper(tm, Vec<decltype(tm)>{m.v});
  // TODO non-temporal memory hint?
  _mm_maskmoveu_si128(lo.v, lom.v, (char *) p);
  _mm_maskmoveu_si128(hi.v, him.v, (char *) p + 16);
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
  #error "Unreachable"
#endif // VEC_WIDTH >= 512

template <TLV_DECL_TAG(T)>
TLV_INLINE void store(T t, TypeOf<T> * p, Mask<T> m, Vec<T> v) {
  CT_ASSERT(is_aligned(T::Bytes, p), "Not aligned");
  word::storeu(t, p, m, v);
}
#endif // HAS_AVX512DQ

template <TLV_DECL_TAG(T)>
TLV_INLINE void storeu(T t, TypeOf<T> * p, nint_t n, Vec<T> v) {
  CT_ASSERT(0 <= n && n <= T::N, "%zd !in 0..%zd", n, T::N);
  auto m = word::mwhilelt(t, 0, n);
  word::storeu(t, p, m, v);
}

template <TLV_DECL_TAG(T)>
TLV_INLINE void store(T t, TypeOf<T> * p, nint_t n, Vec<T> v) {
  CT_ASSERT(0 <= n && n <= T::N, "%zd !in 0..%zd", n, T::N);
  auto m = word::mwhilelt(t, 0, n);
  word::store(t, p, m, v);
}


/* ************************************************************************** */
//                                  Gather                                    //
/* ************************************************************************** */
template <TLV_DECL_TAG(T)>
static TLV_INLINE Vec<T> _gather_scalar(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  alignas(T::Bytes) TypeOf<T> data[T::N];
  CT_UNROLL for (int j = 0; j < T::N; ++j) {
    data[j] = p[nint_t(word::get(i, j))];
  }
  return word::load(t, data);
}

template <TLV_DECL_TAG(T)>
static TLV_INLINE Vec<T> _mask_gather_scalar(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  alignas(T::Bytes) TypeOf<T> data[T::N];
  CT_UNROLL for (int j = 0; j < T::N; ++j) {
    data[j] = word::get(m, j) ? p[nint_t(word::get(i, j))] : word::get(default_v, j);
  }
  return word::load(t, data);
}

#ifdef HAS_AVX2
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  return _mm_i32gather_ps(p, i.v, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  return _mm_i64gather_pd(p, i.v, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  return _mm_i32gather_epi32(p, i.v, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  return _mm_i64gather_epi64(p, i.v, (int)sizeof(TypeOf<T>));
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  return _mm256_i32gather_ps(p, i.v, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  return _mm256_i64gather_pd(p, i.v, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  return _mm256_i32gather_epi32(p, i.v, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  return _mm256_i64gather_epi64(p, i.v, (int)sizeof(TypeOf<T>));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  return _mm512_i32gather_ps(i.v, p, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  return _mm512_i64gather_pd(i.v, p, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  return _mm512_i32gather_epi32(i.v, p, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  return _mm512_i64gather_epi64(i.v, p, (int)sizeof(TypeOf<T>));
}
#endif // VEC_WIDTH >= 512

#ifdef HAS_AVX512DQ
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm_mmask_i32gather_ps(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
  }
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm_mmask_i64gather_pd(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm_mmask_i32gather_epi32(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm_mmask_i64gather_epi64(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm256_mmask_i32gather_ps(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
  }
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm256_mmask_i64gather_pd(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm256_mmask_i32gather_epi32(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm256_mmask_i64gather_epi64(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm512_mask_i32gather_ps(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
  }
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm512_mask_i64gather_pd(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm512_mask_i32gather_epi32(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm512_mask_i64gather_epi64(default_v.v, m.v, i.v, p, (int)sizeof(TypeOf<T>));
}
#else // HAS_AVX512DQ
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm_mask_i32gather_ps(default_v.v, p, i.v, _mm_castsi128_ps(m.v), (int)sizeof(TypeOf<T>));
  }
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm_mask_i64gather_pd(default_v.v, p, i.v, _mm_castsi128_pd(m.v), (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm_mask_i32gather_epi32(default_v.v, p, i.v, m.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm_mask_i64gather_epi64(default_v.v, p, i.v, m.v, (int)sizeof(TypeOf<T>));
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm256_mask_i32gather_ps(default_v.v, p, i.v, _mm256_castsi256_ps(m.v), (int)sizeof(TypeOf<T>));
  }
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm256_mask_i64gather_pd(default_v.v, p, i.v, _mm256_castsi256_pd(m.v), (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm256_mask_i32gather_epi32(default_v.v, p, i.v, m.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return _mm256_mask_i64gather_epi64(default_v.v, p, i.v, m.v, (int)sizeof(TypeOf<T>));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
  #error "Unreachable"
#endif // VEC_WIDTH >= 512
#endif // HAS_AVX512DQ
#else
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  return word::_gather_scalar(t, p, i);
}
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  return word::_mask_gather_scalar(t, p, i, m, default_v);
}
#endif // HAS_AVX2

template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> gather(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, nint_t n, Vec<T> default_v) {
  CT_ASSERT(0 <= n && n <= T::N, "%zd !in 0..%zd", n, T::N);
  auto m = word::mwhilelt(t, 0, n);
  return word::gather(t, p, i, m, default_v);
}


/* ************************************************************************** */
//                                  Scatter                                   //
/* ************************************************************************** */
template <TLV_DECL_TAG(T)>
static TLV_INLINE void _scatter_scalar(T t, TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v) {
  CT_UNROLL for (int j = 0; j < T::N; ++j) {
    p[nint_t(word::get(i, j))] = word::get(v, j);
  }
}

template <TLV_DECL_TAG(T)>
static TLV_INLINE void _mask_scatter_scalar(T t, TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  CT_UNROLL for (int j = 0; j < T::N; ++j) {
    if (word::get(m, j)) {
      p[nint_t(word::get(i, j))] = word::get(v, j);
    }
  }
}

#ifdef HAS_AVX512F
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm_i32scatter_ps(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm_i64scatter_pd(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm_i32scatter_epi32(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm_i64scatter_epi64(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}

#if VEC_WIDTH >= 256
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm256_i32scatter_ps(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm256_i64scatter_pd(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm256_i32scatter_epi32(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm256_i64scatter_epi64(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}
#endif

#if VEC_WIDTH >= 512
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm512_i32scatter_ps(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm512_i64scatter_pd(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm512_i32scatter_epi32(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v) {
  _mm512_i64scatter_epi64(p, i.v, v.v, (int)sizeof(TypeOf<T>));
}
#endif

template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm_mask_i32scatter_ps(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm_mask_i64scatter_pd(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm_mask_i32scatter_epi32(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes <= 16), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm_mask_i64scatter_epi64(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm256_mask_i32scatter_ps(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm256_mask_i64scatter_pd(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm256_mask_i32scatter_epi32(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm256_mask_i64scatter_epi64(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}

template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm512_mask_i32scatter_ps(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm512_mask_i64scatter_pd(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int32_t, uint32_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm512_mask_i32scatter_epi32(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}
template <TLV_DECL_TAG(T), TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, int64_t, uint64_t>)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  _mm512_mask_i64scatter_epi64(p, m.v, i.v, v.v, (int)sizeof(TypeOf<T>));
}
#else // HAS_AVX512F
template <TLV_DECL_TAG(T)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v) {
  word::_scatter_scalar(t, p, i, v);
}
template <TLV_DECL_TAG(T)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  word::_mask_scatter_scalar(t, p, i, m, v);
}
#endif // HAS_AVX512F

template <TLV_DECL_TAG(T)>
TLV_INLINE void scatter(T t, TypeOf<T> * p, Vec<Rebind<Index<TypeOf<T>>, T>> i, nint_t n, Vec<T> v) {
  CT_ASSERT(0 <= n && n <= T::N, "%zd !in 0..%zd", n, T::N);
  auto m = word::mwhilelt(t, 0, n);
  word::scatter(t, p, i, m, v);
}

} // namespace word
} // namespace ct::tl::vec::CPU_CAPABILITY
//@formatter:on

#endif //CTORCH_X86_LOADSTORE_H
