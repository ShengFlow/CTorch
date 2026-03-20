//
// Created by renyz on 2026/3/17.
//

#ifndef CTORCH_X86_128_H
#define CTORCH_X86_128_H

//@formatter:off
#include "CoreDefs.h"
#include "tl/util/Math.h"

#ifndef ARCH_X86_FAMILY
#error "Not x86 platform"
#endif

#include <immintrin.h>
#include "tl/cpu/impl/x86_Types.h"

#ifndef HAS_AVX512DQ
#include "tl/cpu/impl/x86_MaskSupport.h"
#endif

namespace ct::tl::vec {
namespace word {

/**
 * Use in macro arguments: SOME_MACRO(a, b, X(int a, int b)) to pass an argument with comma
 * Or use inside macro definition:
 * #define SOME_MACRO(args) void fn(X args) {...}
 * Then You can define and pass args like: SOME_MACRO((int a, int b))
 */
#define X(...) __VA_ARGS__

#define _TL_XMM_DEFINE_HALVES(N, check, name, dtype, ret, params, args) \
auto name(Tag<dtype, N> t  X params) -> ret { \
    check; return word::name(Tag<dtype, 16 / sizeof(dtype)>()  X args); \
}

#define TL_CHECK_COUNT(varname) CT_ASSERT(0 <= (varname) && (varname) <= size(t), "%zd !in 0..%zd", (varname), size(t))
#define TL_CHECK_INDEX(varname) CT_ASSERT(0 <= (varname) && (varname) < size(t), "%zd !in 0..%zd", (varname), size(t))
#define TL_CHECK_ALIGN CT_ASSERT(((nuint_t)(p) & (DEFAULT_ALIGNMENT - 1)) == 0, "Not aligned");

#define TL_XMM_DEFINE_WITH_ALL_HALVES(name, dtype, ret, params, args) \
auto name(Tag<dtype, (16 / sizeof(dtype))> t, X params) -> ret; /*forward declaration*/ \
TL_XMM_APPLY_TO_ALL_HALVES(dtype, _TL_XMM_DEFINE_HALVES, , name, dtype, ret, (, X params), (, X args)) \
auto name(Tag<dtype, (16 / sizeof(dtype))> t, X params) -> ret

#define TL_XMM_DEFINE_WITH_ALL_HALVES_CHECK_COUNT(name, dtype, ret, params, args) \
auto name(Tag<dtype, (16 / sizeof(dtype))> t, X params) -> ret; /*forward declaration*/ \
TL_XMM_APPLY_TO_ALL_HALVES(dtype, _TL_XMM_DEFINE_HALVES, TL_CHECK_COUNT(n), name, dtype, ret, (, X params), (, X args)) \
auto name(Tag<dtype, (16 / sizeof(dtype))> t, X params) -> ret

#define TL_XMM_DEFINE_WITH_ALL_HALVES_CHECK_INDEX(name, dtype, ret, params, args) \
auto name(Tag<dtype, (16 / sizeof(dtype))> t, X params) -> ret; /*forward declaration*/ \
TL_XMM_APPLY_TO_ALL_HALVES(dtype, _TL_XMM_DEFINE_HALVES, TL_CHECK_INDEX(index), name, dtype, ret, (, X params), (, X args)) \
auto name(Tag<dtype, (16 / sizeof(dtype))> t, X params) -> ret

#define TL_XMM_DEFINE_WITH_ALL_HALVES_V(name, dtype, ret) \
auto name(Tag<dtype, (16 / sizeof(dtype))> t) -> ret; /*forward declaration*/ \
TL_XMM_APPLY_TO_ALL_HALVES(dtype, _TL_XMM_DEFINE_HALVES, , name, dtype, ret, (), ()) \
auto name(Tag<dtype, (16 / sizeof(dtype))> t) -> ret

/* ************************************************************************** */
//                               Constructors                                 //
/* ************************************************************************** */

// mfill, mwhilelt, mwhilege
#ifdef HAS_AVX512DQ /* mask operations */
#define TL_XMM_DEFINE_MASK_OPERATIONS(dtype, mask_size) \
TL_XMM_DEFINE_WITH_ALL_HALVES(mfill, dtype, MaskOf(t), (bool value), (value)) { \
  /* we do not guarantee that padded elements are zero */ \
  uint32_t x = value ? 0xffffffffu : 0x00; \
  return _cvtu32_mask##mask_size(x); \
} \
TL_XMM_DEFINE_WITH_ALL_HALVES(mwhilelt, dtype, MaskOf(t), (nint_t a, nint_t b), (a, b)) { \
  nint_t end = std::clamp(b - a, nint_t(0), nint_t(sizeof(unsigned) * CHAR_BIT)); \
  return _cvtu32_mask##mask_size((1u << (unsigned) end) - 1); \
} \
TL_XMM_DEFINE_WITH_ALL_HALVES(mwhilege, dtype, MaskOf(t), (nint_t a, nint_t b), (a, b)) { \
  nint_t end = std::clamp(b - a, nint_t(0), nint_t(sizeof(unsigned) * CHAR_BIT)); \
  return _cvtu32_mask##mask_size(~((1u << (unsigned) end) - 1)); \
} \

TL_XMM_DEFINE_MASK_OPERATIONS(bfloat16_t, 8)
TL_XMM_DEFINE_MASK_OPERATIONS(float16_t, 8)
TL_XMM_DEFINE_MASK_OPERATIONS(float32_t, 8)
TL_XMM_DEFINE_MASK_OPERATIONS(float64_t, 8)
TL_XMM_DEFINE_MASK_OPERATIONS(int8_t, 16)
TL_XMM_DEFINE_MASK_OPERATIONS(uint8_t, 16)
TL_XMM_DEFINE_MASK_OPERATIONS(int16_t, 8)
TL_XMM_DEFINE_MASK_OPERATIONS(uint16_t, 8)
TL_XMM_DEFINE_MASK_OPERATIONS(int32_t, 8)
TL_XMM_DEFINE_MASK_OPERATIONS(uint32_t, 8)
TL_XMM_DEFINE_MASK_OPERATIONS(int64_t, 8)
TL_XMM_DEFINE_MASK_OPERATIONS(uint64_t, 8)
#undef TL_XMM_DEFINE_MASK_OPERATIONS
#else // HAS_AVX512DQ
#define TL_XMM_DEFINE_MASK_OPERATIONS_8(dtype) \
TL_XMM_DEFINE_WITH_ALL_HALVES(mfill, dtype, MaskOf(t), (bool value), (value)) { \
  /* we do not guarantee that padded elements are zero */ \
  int64_t x = value ? int64_t(-1) : 0; \
  return _mm_set1_epi64x(x); \
} \
TL_XMM_DEFINE_WITH_ALL_HALVES(mwhilelt, dtype, MaskOf(t), (nint_t a, nint_t b), (a, b)) { \
  static const __m128i index = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0); \
  int8_t diff = (int8_t)std::clamp<nint_t>(b - a, INT8_MIN, INT8_MAX); \
  auto end = _mm_set1_epi8(diff); \
  return _mm_cmplt_epi8(index, end); \
} \
TL_XMM_DEFINE_WITH_ALL_HALVES(mwhilege, dtype, MaskOf(t), (nint_t a, nint_t b), (a, b)) { \
  static const __m128i index = _mm_set_epi8(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1); \
  int8_t diff = (int8_t)std::clamp<nint_t>(b - a, INT8_MIN, INT8_MAX); \
  auto start = _mm_set1_epi8(diff); \
  return _mm_cmplt_epi8(start, index); \
}
#define TL_XMM_DEFINE_MASK_OPERATIONS_16(dtype) \
TL_XMM_DEFINE_WITH_ALL_HALVES(mfill, dtype, MaskOf(t), (bool value), (value)) { \
  /* we do not guarantee that padded elements are zero */ \
  int64_t x = value ? int64_t(-1) : 0; \
  return _mm_set1_epi64x(x); \
} \
TL_XMM_DEFINE_WITH_ALL_HALVES(mwhilelt, dtype, MaskOf(t), (nint_t a, nint_t b), (a, b)) { \
  static const __m128i index = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0); \
  int16_t diff = (int16_t)std::clamp<nint_t>(b - a, INT16_MIN, INT16_MAX); \
  auto end = _mm_set1_epi16(diff); \
  return _mm_cmplt_epi16(index, end); \
} \
TL_XMM_DEFINE_WITH_ALL_HALVES(mwhilege, dtype, MaskOf(t), (nint_t a, nint_t b), (a, b)) { \
  static const __m128i index = _mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1); \
  int16_t diff = (int16_t)std::clamp<nint_t>(b - a, INT16_MIN, INT16_MAX); \
  auto start = _mm_set1_epi16(diff); \
  return _mm_cmplt_epi16(start, index); \
}
#define TL_XMM_DEFINE_MASK_OPERATIONS_32(dtype) \
TL_XMM_DEFINE_WITH_ALL_HALVES(mfill, dtype, MaskOf(t), (bool value), (value)) { \
  /* we do not guarantee that padded elements are zero */ \
  int64_t x = value ? int64_t(-1) : 0; \
  return _mm_set1_epi64x(x); \
} \
TL_XMM_DEFINE_WITH_ALL_HALVES(mwhilelt, dtype, MaskOf(t), (nint_t a, nint_t b), (a, b)) { \
  static const __m128i index = _mm_set_epi32(3, 2, 1, 0); \
  int32_t diff = (int32_t)std::clamp<nint_t>(b - a, INT32_MIN, INT32_MAX); \
  auto end = _mm_set1_epi32(diff); \
  return _mm_cmplt_epi32(index, end); \
} \
TL_XMM_DEFINE_WITH_ALL_HALVES(mwhilege, dtype, MaskOf(t), (nint_t a, nint_t b), (a, b)) { \
  static const __m128i index = _mm_set_epi32(4, 3, 2, 1); \
  int32_t diff = (int32_t)std::clamp<nint_t>(b - a, INT32_MIN, INT32_MAX); \
  auto start = _mm_set1_epi32(diff); \
  return _mm_cmplt_epi32(start, index); \
}
#define TL_XMM_DEFINE_MASK_OPERATIONS_64(dtype) \
TL_XMM_DEFINE_WITH_ALL_HALVES(mfill, dtype, MaskOf(t), (bool value), (value)) { \
  /* we do not guarantee that padded elements are zero */ \
  int64_t x = value ? int64_t(-1) : 0; \
  return _mm_set1_epi64x(x); \
} \
TL_XMM_DEFINE_WITH_ALL_HALVES(mwhilelt, dtype, MaskOf(t), (nint_t a, nint_t b), (a, b)) { \
  static const __m128i index = _mm_set_epi32(1, 1, 0, 0); \
  int32_t diff = (int32_t)std::clamp<nint_t>(b - a, INT32_MIN, INT32_MAX); \
  auto end = _mm_set1_epi32(diff); \
  return _mm_cmplt_epi32(index, end); \
} \
TL_XMM_DEFINE_WITH_ALL_HALVES(mwhilege, dtype, MaskOf(t), (nint_t a, nint_t b), (a, b)) { \
  static const __m128i index = _mm_set_epi32(2, 2, 1, 1); \
  int32_t diff = (int32_t)std::clamp<nint_t>(b - a, INT32_MIN, INT32_MAX); \
  auto start = _mm_set1_epi32(diff); \
  return _mm_cmplt_epi32(start, index); \
}

TL_XMM_DEFINE_MASK_OPERATIONS_16(bfloat16_t)
TL_XMM_DEFINE_MASK_OPERATIONS_16(float16_t)
TL_XMM_DEFINE_MASK_OPERATIONS_32(float32_t)
TL_XMM_DEFINE_MASK_OPERATIONS_64(float64_t)
TL_XMM_DEFINE_MASK_OPERATIONS_8(int8_t)
TL_XMM_DEFINE_MASK_OPERATIONS_8(uint8_t)
TL_XMM_DEFINE_MASK_OPERATIONS_16(int16_t)
TL_XMM_DEFINE_MASK_OPERATIONS_16(uint16_t)
TL_XMM_DEFINE_MASK_OPERATIONS_32(int32_t)
TL_XMM_DEFINE_MASK_OPERATIONS_32(uint32_t)
TL_XMM_DEFINE_MASK_OPERATIONS_64(int64_t)
TL_XMM_DEFINE_MASK_OPERATIONS_64(uint64_t)
#undef TL_XMM_DEFINE_MASK_OPERATIONS_8
#undef TL_XMM_DEFINE_MASK_OPERATIONS_16
#undef TL_XMM_DEFINE_MASK_OPERATIONS_32
#undef TL_XMM_DEFINE_MASK_OPERATIONS_64
#endif // HAS_AVX512DQ

// fill(T v)
#define TL_XMM_DEFINE_FILL(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(fill, dtype, VecOf(t), (dtype v), (v))
TL_XMM_DEFINE_FILL(bfloat16_t) {
  union { bfloat16_t b; int16_t i; } u { .b = v };
  return _mm_set1_epi16(u.i);
}
TL_XMM_DEFINE_FILL(float16_t) {
  #ifdef HAS_AVX512_FP16
  return _mm_castph_si128(_mm_set1_ph(v));
  #else
  union { float16_t b; int16_t i; } u { .b = v };
  return _mm_set1_epi16(u.i);
  #endif
}
TL_XMM_DEFINE_FILL(float32_t) { return _mm_set1_ps(v); }
TL_XMM_DEFINE_FILL(float64_t) { return _mm_set1_pd(v); }
TL_XMM_DEFINE_FILL(int8_t) { return _mm_set1_epi8(v); }
TL_XMM_DEFINE_FILL(uint8_t) { return _mm_set1_epi8((int8_t)v); }
TL_XMM_DEFINE_FILL(int16_t) { return _mm_set1_epi16(v); }
TL_XMM_DEFINE_FILL(uint16_t) { return _mm_set1_epi16((int16_t)v); }
TL_XMM_DEFINE_FILL(int32_t) { return _mm_set1_epi32(v); }
TL_XMM_DEFINE_FILL(uint32_t) { return _mm_set1_epi32((int32_t)v); }
TL_XMM_DEFINE_FILL(int64_t) { return _mm_set1_epi64x(v); }
TL_XMM_DEFINE_FILL(uint64_t) { return _mm_set1_epi64x((int64_t)v); }
#undef TL_XMM_DEFINE_FILL

// fill(T v, Mask<T> m, Vec<T> default_v)
#define TL_XMM_DEFINE_FILL(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(fill, dtype, VecOf(t), (dtype v, MaskOf(t) m, VecOf(t) default_v), (v, m, default_v))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_FILL(bfloat16_t) { return _mm_mask_mov_epi16(default_v.v, m.v, fill(t, v).v); }
TL_XMM_DEFINE_FILL(float16_t) { return _mm_mask_mov_epi16(default_v.v, m.v, fill(t, v).v); }
TL_XMM_DEFINE_FILL(float32_t) { return _mm_mask_mov_ps(default_v.v, m.v, fill(t, v).v); }
TL_XMM_DEFINE_FILL(float64_t) { return _mm_mask_mov_pd(default_v.v, m.v, fill(t, v).v); }
TL_XMM_DEFINE_FILL(int8_t) { return _mm_mask_mov_epi8(default_v.v, m.v, fill(t, v).v); }
TL_XMM_DEFINE_FILL(uint8_t) { return _mm_mask_mov_epi8(default_v.v, m.v, fill(t, v).v); }
TL_XMM_DEFINE_FILL(int16_t) { return _mm_mask_mov_epi16(default_v.v, m.v, fill(t, v).v); }
TL_XMM_DEFINE_FILL(uint16_t) { return _mm_mask_mov_epi16(default_v.v, m.v, fill(t, v).v); }
TL_XMM_DEFINE_FILL(int32_t) { return _mm_mask_mov_epi32(default_v.v, m.v, fill(t, v).v); }
TL_XMM_DEFINE_FILL(uint32_t) { return _mm_mask_mov_epi32(default_v.v, m.v, fill(t, v).v); }
TL_XMM_DEFINE_FILL(int64_t) { return _mm_mask_mov_epi64(default_v.v, m.v, fill(t, v).v); }
TL_XMM_DEFINE_FILL(uint64_t) { return _mm_mask_mov_epi64(default_v.v, m.v, fill(t, v).v); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_FILL(bfloat16_t) { return _mm_blendv_epi8(default_v.v, fill(t, v).v, m.v); }
TL_XMM_DEFINE_FILL(float16_t) { return _mm_blendv_epi8(default_v.v, fill(t, v).v, m.v); }
TL_XMM_DEFINE_FILL(float32_t) { return _mm_blendv_ps(default_v.v, fill(t, v).v, _mm_castsi128_ps(m.v)); }
TL_XMM_DEFINE_FILL(float64_t) { return _mm_blendv_pd(default_v.v, fill(t, v).v, _mm_castsi128_pd(m.v)); }
TL_XMM_DEFINE_FILL(int8_t) { return _mm_blendv_epi8(default_v.v, fill(t, v).v, m.v); }
TL_XMM_DEFINE_FILL(uint8_t) { return _mm_blendv_epi8(default_v.v, fill(t, v).v, m.v); }
TL_XMM_DEFINE_FILL(int16_t) { return _mm_blendv_epi8(default_v.v, fill(t, v).v, m.v); }
TL_XMM_DEFINE_FILL(uint16_t) { return _mm_blendv_epi8(default_v.v, fill(t, v).v, m.v); }
TL_XMM_DEFINE_FILL(int32_t) { return _mm_blendv_epi8(default_v.v, fill(t, v).v, m.v); }
TL_XMM_DEFINE_FILL(uint32_t) { return _mm_blendv_epi8(default_v.v, fill(t, v).v, m.v); }
TL_XMM_DEFINE_FILL(int64_t) { return _mm_blendv_epi8(default_v.v, fill(t, v).v, m.v); }
TL_XMM_DEFINE_FILL(uint64_t) { return _mm_blendv_epi8(default_v.v, fill(t, v).v, m.v); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_FILL

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE auto _fill_forward_mask(Tag<T, N, P> t, T v, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = mwhilelt(t, 0, n);
  return word::fill(t, v, m, default_v);
}

// fill(T v, nint_t n, Vec<T> default_v)
#define TL_XMM_DEFINE_FILL(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES_CHECK_COUNT(fill, dtype, VecOf(t), (dtype v, nint_t n, VecOf(t) default_v), (v, n, default_v))
TL_XMM_DEFINE_FILL(bfloat16_t) { return _fill_forward_mask(t, v, n, default_v); }
TL_XMM_DEFINE_FILL(float16_t) { return _fill_forward_mask(t, v, n, default_v); }
TL_XMM_DEFINE_FILL(float32_t) { return _fill_forward_mask(t, v, n, default_v); }
TL_XMM_DEFINE_FILL(float64_t) { return _fill_forward_mask(t, v, n, default_v); }
TL_XMM_DEFINE_FILL(int8_t) { return _fill_forward_mask(t, v, n, default_v); }
TL_XMM_DEFINE_FILL(uint8_t) { return _fill_forward_mask(t, v, n, default_v); }
TL_XMM_DEFINE_FILL(int16_t) { return _fill_forward_mask(t, v, n, default_v); }
TL_XMM_DEFINE_FILL(uint16_t) { return _fill_forward_mask(t, v, n, default_v); }
TL_XMM_DEFINE_FILL(int32_t) { return _fill_forward_mask(t, v, n, default_v); }
TL_XMM_DEFINE_FILL(uint32_t) { return _fill_forward_mask(t, v, n, default_v); }
TL_XMM_DEFINE_FILL(int64_t) { return _fill_forward_mask(t, v, n, default_v); }
TL_XMM_DEFINE_FILL(uint64_t) { return _fill_forward_mask(t, v, n, default_v); }
#undef TL_XMM_DEFINE_FILL

// zeros()
#define TL_XMM_DEFINE_ZEROS(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES_V(zeros, dtype, VecOf(t))
TL_XMM_DEFINE_ZEROS(bfloat16_t) { return _mm_setzero_si128(); }
TL_XMM_DEFINE_ZEROS(float16_t) { return _mm_setzero_si128(); }
TL_XMM_DEFINE_ZEROS(float32_t) { return _mm_setzero_ps(); }
TL_XMM_DEFINE_ZEROS(float64_t) { return _mm_setzero_pd(); }
TL_XMM_DEFINE_ZEROS(int8_t) { return _mm_setzero_si128(); }
TL_XMM_DEFINE_ZEROS(uint8_t) { return _mm_setzero_si128(); }
TL_XMM_DEFINE_ZEROS(int16_t) { return _mm_setzero_si128(); }
TL_XMM_DEFINE_ZEROS(uint16_t) { return _mm_setzero_si128(); }
TL_XMM_DEFINE_ZEROS(int32_t) { return _mm_setzero_si128(); }
TL_XMM_DEFINE_ZEROS(uint32_t) { return _mm_setzero_si128(); }
TL_XMM_DEFINE_ZEROS(int64_t) { return _mm_setzero_si128(); }
TL_XMM_DEFINE_ZEROS(uint64_t) { return _mm_setzero_si128(); }
#undef TL_XMM_DEFINE_ZEROS

/* ************************************************************************** */
//                          Consecutive load & store                          //
/* ************************************************************************** */

// loadu(const T* p)
#define TL_XMM_DEFINE_LOADU(dtype, N) auto loadu(Tag<dtype, N> t, const dtype* p) -> VecOf(t)
TL_XMM_DEFINE_LOADU(bfloat16_t, 8) { return _mm_castps_si128(_mm_loadu_ps((const float32_t*)p)); }
TL_XMM_DEFINE_LOADU(bfloat16_t, 4) { return _mm_castpd_si128(_mm_load_sd((const float64_t *)p)); }
TL_XMM_DEFINE_LOADU(bfloat16_t, 2) { return _mm_castps_si128(_mm_load_ss((const float32_t *)p)); }
TL_XMM_DEFINE_LOADU(float16_t, 8) {
  #ifdef HAS_AVX512_FP16
  return _mm_castph_si128(_mm_loadu_ph(p));
  #else
  return _mm_castps_si128(_mm_loadu_ps((const float32_t*) p));
  #endif
}
TL_XMM_DEFINE_LOADU(float16_t, 4) { return _mm_castpd_si128(_mm_load_sd((const float64_t *)p)); }
TL_XMM_DEFINE_LOADU(float16_t, 2) { return _mm_castps_si128(_mm_load_ss((const float32_t *)p)); }
TL_XMM_DEFINE_LOADU(float32_t, 4) { return _mm_loadu_ps(p); }
TL_XMM_DEFINE_LOADU(float32_t, 2) { return _mm_castpd_ps(_mm_load_sd((const float64_t*) p)); }
TL_XMM_DEFINE_LOADU(float64_t, 2) { return _mm_loadu_pd(p); }
TL_XMM_DEFINE_LOADU(int8_t, 16) { return _mm_loadu_si128((const __m128i *)p); }
TL_XMM_DEFINE_LOADU(uint8_t, 16) { return _mm_loadu_si128((const __m128i *)p); }
TL_XMM_DEFINE_LOADU(int8_t, 8) { return _mm_cvtsi64_si128(*((const int64_t *)p)); }
TL_XMM_DEFINE_LOADU(uint8_t, 8) { return _mm_cvtsi64_si128(*((const int64_t *)p)); }
TL_XMM_DEFINE_LOADU(int8_t, 4) { return _mm_cvtsi32_si128(*((const int32_t *)p)); }
TL_XMM_DEFINE_LOADU(uint8_t, 4) { return _mm_cvtsi32_si128(*((const int32_t *)p)); }
#ifdef HAS_AVX512_FP16
TL_XMM_DEFINE_LOADU(int8_t, 2) { return _mm_cvtsi16_si128(*((const int16_t *)p)); }
TL_XMM_DEFINE_LOADU(uint8_t, 2) { return _mm_cvtsi16_si128(*((const int16_t *)p)); }
#else
TL_XMM_DEFINE_LOADU(int8_t, 2) { return _mm_cvtsi32_si128(*((const uint16_t *)p)); }
TL_XMM_DEFINE_LOADU(uint8_t, 2) { return _mm_cvtsi32_si128(*((const uint16_t *)p)); }
#endif
TL_XMM_DEFINE_LOADU(int16_t, 8) { return _mm_loadu_si128((const __m128i *)p); }
TL_XMM_DEFINE_LOADU(uint16_t, 8) { return _mm_loadu_si128((const __m128i *)p); }
TL_XMM_DEFINE_LOADU(int16_t, 4) { return _mm_cvtsi64_si128(*((const int64_t *)p)); }
TL_XMM_DEFINE_LOADU(uint16_t, 4) { return _mm_cvtsi64_si128(*((const int64_t *)p)); }
TL_XMM_DEFINE_LOADU(int16_t, 2) { return _mm_cvtsi32_si128(*((const int32_t *)p)); }
TL_XMM_DEFINE_LOADU(uint16_t, 2) { return _mm_cvtsi32_si128(*((const int32_t *)p)); }
TL_XMM_DEFINE_LOADU(int32_t, 4) { return _mm_loadu_si128((const __m128i *)p); }
TL_XMM_DEFINE_LOADU(uint32_t, 4) { return _mm_loadu_si128((const __m128i *)p); }
TL_XMM_DEFINE_LOADU(int32_t, 2) { return _mm_cvtsi64_si128(*((const int64_t *)p)); }
TL_XMM_DEFINE_LOADU(uint32_t, 2) { return _mm_cvtsi64_si128(*((const int64_t *)p)); }
TL_XMM_DEFINE_LOADU(int64_t, 2) { return _mm_loadu_si128((const __m128i *)p); }
TL_XMM_DEFINE_LOADU(uint64_t, 2) { return _mm_loadu_si128((const __m128i *)p); }
#undef TL_XMM_DEFINE_LOADU

// load(const T* p)
#define TL_XMM_DEFINE_LOAD(dtype, N) auto load(Tag<dtype, N> t, const dtype* p) -> VecOf(t)
TL_XMM_DEFINE_LOAD(bfloat16_t, 8) { return _mm_castps_si128(_mm_load_ps((const float32_t*)p)); }
TL_XMM_DEFINE_LOAD(bfloat16_t, 4) { return _mm_castpd_si128(_mm_load_sd((const float64_t *)p)); }
TL_XMM_DEFINE_LOAD(bfloat16_t, 2) { return _mm_castps_si128(_mm_load_ss((const float32_t *)p)); }
TL_XMM_DEFINE_LOAD(float16_t, 8) {
  #ifdef HAS_AVX512_FP16
  return _mm_castph_si128(_mm_load_ph(p));
  #else
  return _mm_castps_si128(_mm_load_ps((const float32_t*) p));
  #endif
}
TL_XMM_DEFINE_LOAD(float16_t, 4) { TL_CHECK_ALIGN return _mm_castpd_si128(_mm_load_sd((const float64_t *)p)); }
TL_XMM_DEFINE_LOAD(float16_t, 2) { TL_CHECK_ALIGN return _mm_castps_si128(_mm_load_ss((const float32_t *)p)); }
TL_XMM_DEFINE_LOAD(float32_t, 4) { TL_CHECK_ALIGN return _mm_load_ps(p); }
TL_XMM_DEFINE_LOAD(float32_t, 2) { TL_CHECK_ALIGN return _mm_castpd_ps(_mm_load_sd((const float64_t*) p)); }
TL_XMM_DEFINE_LOAD(float64_t, 2) { TL_CHECK_ALIGN return _mm_load_pd(p); }
TL_XMM_DEFINE_LOAD(int8_t, 16) { TL_CHECK_ALIGN return _mm_load_si128((const __m128i *)p); }
TL_XMM_DEFINE_LOAD(uint8_t, 16) { TL_CHECK_ALIGN return _mm_load_si128((const __m128i *)p); }
TL_XMM_DEFINE_LOAD(int8_t, 8) { TL_CHECK_ALIGN return _mm_cvtsi64_si128(*((const int64_t *)p)); }
TL_XMM_DEFINE_LOAD(uint8_t, 8) { TL_CHECK_ALIGN return _mm_cvtsi64_si128(*((const int64_t *)p)); }
TL_XMM_DEFINE_LOAD(int8_t, 4) { TL_CHECK_ALIGN return _mm_cvtsi32_si128(*((const int32_t *)p)); }
TL_XMM_DEFINE_LOAD(uint8_t, 4) { TL_CHECK_ALIGN return _mm_cvtsi32_si128(*((const int32_t *)p)); }
#ifdef HAS_AVX512_FP16
TL_XMM_DEFINE_LOAD(int8_t, 2) { TL_CHECK_ALIGN return _mm_cvtsi16_si128(*((const int16_t *)p)); }
TL_XMM_DEFINE_LOAD(uint8_t, 2) { TL_CHECK_ALIGN return _mm_cvtsi16_si128(*((const int16_t *)p)); }
#else
TL_XMM_DEFINE_LOAD(int8_t, 2) { TL_CHECK_ALIGN return _mm_cvtsi32_si128(*((const uint16_t *)p)); }
TL_XMM_DEFINE_LOAD(uint8_t, 2) { TL_CHECK_ALIGN return _mm_cvtsi32_si128(*((const uint16_t *)p)); }
#endif
TL_XMM_DEFINE_LOAD(int16_t, 8) { TL_CHECK_ALIGN return _mm_load_si128((const __m128i *)p); }
TL_XMM_DEFINE_LOAD(uint16_t, 8) { TL_CHECK_ALIGN return _mm_load_si128((const __m128i *)p); }
TL_XMM_DEFINE_LOAD(int16_t, 4) { TL_CHECK_ALIGN return _mm_cvtsi64_si128(*((const int64_t *)p)); }
TL_XMM_DEFINE_LOAD(uint16_t, 4) { TL_CHECK_ALIGN return _mm_cvtsi64_si128(*((const int64_t *)p)); }
TL_XMM_DEFINE_LOAD(int16_t, 2) { TL_CHECK_ALIGN return _mm_cvtsi32_si128(*((const int32_t *)p)); }
TL_XMM_DEFINE_LOAD(uint16_t, 2) { TL_CHECK_ALIGN return _mm_cvtsi32_si128(*((const int32_t *)p)); }
TL_XMM_DEFINE_LOAD(int32_t, 4) { TL_CHECK_ALIGN return _mm_load_si128((const __m128i *)p); }
TL_XMM_DEFINE_LOAD(uint32_t, 4) { TL_CHECK_ALIGN return _mm_load_si128((const __m128i *)p); }
TL_XMM_DEFINE_LOAD(int32_t, 2) { TL_CHECK_ALIGN return _mm_cvtsi64_si128(*((const int64_t *)p)); }
TL_XMM_DEFINE_LOAD(uint32_t, 2) { TL_CHECK_ALIGN return _mm_cvtsi64_si128(*((const int64_t *)p)); }
TL_XMM_DEFINE_LOAD(int64_t, 2) { TL_CHECK_ALIGN return _mm_load_si128((const __m128i *)p); }
TL_XMM_DEFINE_LOAD(uint64_t, 2) { TL_CHECK_ALIGN return _mm_load_si128((const __m128i *)p); }
#undef TL_XMM_DEFINE_LOAD

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE auto _ensure_mask_range(Tag<T, N, P> t, MaskOf(t) m) -> MaskOf(t) {
  if constexpr (N * sizeof(T) < 16) {
    #ifdef HAS_AVX512DQ
    uint32_t x = _cvtmask8_u32(m);
    return _cvtu32_mask8(x & ((1u << N) - 1));
    #else
    if constexpr (N * sizeof(T) == 1) {
      return _mm_and_si128(m, _mm_set_epi32(0, 0, 0, 0x000000FF));
    } else if constexpr (N * sizeof(T) == 2) {
      return _mm_and_si128(m, _mm_set_epi32(0, 0, 0, 0x0000FFFF));
    } else if constexpr (N * sizeof(T) == 4) {
      return _mm_and_si128(m, _mm_set_epi32(0, 0, 0, (int)0xFFFFFFFF));
    } else if constexpr (N * sizeof(T) == 8) {
      return _mm_move_epi64(m);
    }
    #endif
  } else {
    return m;
  }
}

// loadu(const T* p, Mask<T> m, Vec<T> default_v)
#define TL_XMM_DEFINE_LOADU(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(loadu, dtype, VecOf(t), (const dtype* p, MaskOf(t) m, VecOf(t) default_v), (p, m, default_v))
// load(const T* p, Mask<T> m, Vec<T> default_v)
#define TL_XMM_DEFINE_LOAD(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(load, dtype, VecOf(t), (const dtype* p, MaskOf(t) m, VecOf(t) default_v), (p, m, default_v))

#ifdef HAS_AVX512DQ
// loadu(const T* p, Mask<T> m, Vec<T> default_v)
TL_XMM_DEFINE_LOADU(bfloat16_t) { return _mm_mask_loadu_epi16(default_v.v, _ensure_mask_range(t, m).v, p); }
TL_XMM_DEFINE_LOADU(float16_t) { return _mm_mask_loadu_epi16(default_v.v, _ensure_mask_range(t, m).v, p); }
TL_XMM_DEFINE_LOADU(float32_t) { return _mm_mask_loadu_ps(default_v.v, _ensure_mask_range(t, m).v, p); }
TL_XMM_DEFINE_LOADU(float64_t) { return _mm_mask_loadu_pd(default_v.v, _ensure_mask_range(t, m).v, p); }
TL_XMM_DEFINE_LOADU(int8_t) { return _mm_mask_loadu_epi8(default_v.v, _ensure_mask_range(t, m).v, p); }
TL_XMM_DEFINE_LOADU(uint8_t) { return _mm_mask_loadu_epi8(default_v.v, _ensure_mask_range(t, m).v, p); }
TL_XMM_DEFINE_LOADU(int16_t) { return _mm_mask_loadu_epi16(default_v.v, _ensure_mask_range(t, m).v, p); }
TL_XMM_DEFINE_LOADU(uint16_t) { return _mm_mask_loadu_epi16(default_v.v, _ensure_mask_range(t, m).v, p); }
TL_XMM_DEFINE_LOADU(int32_t) { return _mm_mask_loadu_epi32(default_v.v, _ensure_mask_range(t, m).v, p); }
TL_XMM_DEFINE_LOADU(uint32_t) { return _mm_mask_loadu_epi32(default_v.v, _ensure_mask_range(t, m).v, p); }
TL_XMM_DEFINE_LOADU(int64_t) { return _mm_mask_loadu_epi64(default_v.v, _ensure_mask_range(t, m).v, p); }
TL_XMM_DEFINE_LOADU(uint64_t) { return _mm_mask_loadu_epi64(default_v.v, _ensure_mask_range(t, m).v, p); }

// load(const T* p, Mask<T> m, Vec<T> default_v)
TL_XMM_DEFINE_LOAD(bfloat16_t) { return _mm_mask_loadu_epi16(default_v.v, _ensure_mask_range(t, m).v, p); }
TL_XMM_DEFINE_LOAD(float16_t) { return _mm_mask_loadu_epi16(default_v.v, _ensure_mask_range(t, m).v, p); }
TL_XMM_DEFINE_LOAD(float32_t) { return _mm_mask_load_ps(default_v.v, _ensure_mask_range(t, m).v, p); }
TL_XMM_DEFINE_LOAD(float64_t) { return _mm_mask_load_pd(default_v.v, _ensure_mask_range(t, m).v, p); }
TL_XMM_DEFINE_LOAD(int8_t) { return _mm_mask_loadu_epi8(default_v.v, _ensure_mask_range(t, m).v, p); }
TL_XMM_DEFINE_LOAD(uint8_t) { return _mm_mask_loadu_epi8(default_v.v, _ensure_mask_range(t, m).v, p); }
TL_XMM_DEFINE_LOAD(int16_t) { return _mm_mask_loadu_epi16(default_v.v, _ensure_mask_range(t, m).v, p); }
TL_XMM_DEFINE_LOAD(uint16_t) { return _mm_mask_loadu_epi16(default_v.v, _ensure_mask_range(t, m).v, p); }
TL_XMM_DEFINE_LOAD(int32_t) { return _mm_mask_load_epi32(default_v.v, _ensure_mask_range(t, m).v, p); }
TL_XMM_DEFINE_LOAD(uint32_t) { return _mm_mask_load_epi32(default_v.v, _ensure_mask_range(t, m).v, p); }
TL_XMM_DEFINE_LOAD(int64_t) { return _mm_mask_load_epi64(default_v.v, _ensure_mask_range(t, m).v, p); }
TL_XMM_DEFINE_LOAD(uint64_t) { return _mm_mask_load_epi64(default_v.v, _ensure_mask_range(t, m).v, p); }
#else // HAS_AVX512DQ
// Note: it's pretty hard to ensure memory boundary safety without hardware maskload for int8 and int16
// So we instead checks for page boundary to determine if a full 16-byte load is available
template <typename T, nint_t N, int POW2>
CT_ALWAYS_FORCEINLINE auto _loadu_16(Tag<T, N, POW2> t, const T* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) {
  auto mask = _ensure_mask_range(t, m).v;
  // if p + 15 still in the same page as p
  if (((nuint_t(p) & 0xfff) + 15) & 0xfff) {
    return _mm_blendv_epi8(default_v.v, _mm_castps_si128(_mm_loadu_ps((const float32_t*) p)), mask);
  } else {
    // TODO replace with word::gather call
    // Note: fallback scalar impl,
    union { int16_t i[8]; __m128i m; } V{.m = default_v.v}, M{.m = mask};
    alignas(DEFAULT_ALIGNMENT) int16_t S[8];
    auto P = (const int16_t*) p;
#pragma nounroll
    for (int i = 0; i < 8; ++i) S[i] = M.i[i] ? P[i] : V.i[i];
    return _mm_load_si128((const __m128i *)S);
  }
}

template <typename T, nint_t N, int POW2>
CT_ALWAYS_FORCEINLINE auto _loadu_8(Tag<T, N, POW2> t, const T* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) {
  auto mask = _ensure_mask_range(t, m).v;
  // if p + 15 still in the same page as p
  if (((nuint_t(p) & 0xfff) + 15) & 0xfff) {
    return _mm_blendv_epi8(default_v.v, _mm_castps_si128(_mm_loadu_ps((const float32_t*) p)), mask);
  } else {
    // TODO replace with word::gather call
    // Note: fallback scalar impl,
    union { int8_t i[16]; __m128i m; } V{.m = default_v.v}, M{.m = mask};
    alignas(DEFAULT_ALIGNMENT) int8_t S[16];
    auto P = (const int8_t*) p;
#pragma nounroll
    for (int i = 0; i < 16; ++i) S[i] = M.i[i] ? P[i] : V.i[i];
    return _mm_load_si128((const __m128i *)S);
  }
}
// loadu(const T* p, Mask<T> m, Vec<T> default_v)
TL_XMM_DEFINE_LOADU(bfloat16_t) { return _loadu_16(t, p, m, default_v); }
TL_XMM_DEFINE_LOADU(float16_t) { return _loadu_16(t, p, m, default_v); }
TL_XMM_DEFINE_LOADU(float32_t) { auto mask = _ensure_mask_range(t, m).v; return _mm_blendv_ps(default_v.v, _mm_maskload_ps(p, mask), _mm_castsi128_ps(mask)); }
TL_XMM_DEFINE_LOADU(float64_t) { auto mask = _ensure_mask_range(t, m).v; return _mm_blendv_pd(default_v.v, _mm_maskload_pd(p, mask), _mm_castsi128_pd(mask)); }
TL_XMM_DEFINE_LOADU(int8_t) { auto mask = _ensure_mask_range(t, m).v; return _loadu_8(t, p, m, default_v); }
TL_XMM_DEFINE_LOADU(uint8_t) { auto mask = _ensure_mask_range(t, m).v; return _loadu_8(t, p, m, default_v); }
TL_XMM_DEFINE_LOADU(int16_t) { auto mask = _ensure_mask_range(t, m).v; return _loadu_16(t, p, m, default_v); }
TL_XMM_DEFINE_LOADU(uint16_t) { auto mask = _ensure_mask_range(t, m).v; return _loadu_16(t, p, m, default_v); }
#ifdef HAS_AVX2
TL_XMM_DEFINE_LOADU(int32_t) { auto mask = _ensure_mask_range(t, m).v; return _mm_blendv_epi8(default_v.v, _mm_maskload_epi32(p, mask), mask); }
TL_XMM_DEFINE_LOADU(uint32_t) { auto mask = _ensure_mask_range(t, m).v; return _mm_blendv_epi8(default_v.v, _mm_maskload_epi32((const int *)p, mask), mask); }
TL_XMM_DEFINE_LOADU(int64_t) { auto mask = _ensure_mask_range(t, m).v; return _mm_blendv_epi8(default_v.v, _mm_maskload_epi64((const long long *)p, mask), mask); }
TL_XMM_DEFINE_LOADU(uint64_t) { auto mask = _ensure_mask_range(t, m).v; return _mm_blendv_epi8(default_v.v, _mm_maskload_epi64((const long long *)p, mask), mask); }
#else // HAS_AVX2
TL_XMM_DEFINE_LOADU(int32_t) { auto mask = _ensure_mask_range(t, m).v; return _mm_blendv_epi8(default_v.v, _mm_castps_si128(_mm_maskload_ps((const float32_t*)p, mask)), mask); }
TL_XMM_DEFINE_LOADU(uint32_t) { auto mask = _ensure_mask_range(t, m).v; return _mm_blendv_epi8(default_v.v, _mm_castps_si128(_mm_maskload_ps((const float32_t*)p, mask)), mask); }
TL_XMM_DEFINE_LOADU(int64_t) { auto mask = _ensure_mask_range(t, m).v; return _mm_blendv_epi8(default_v.v, _mm_castpd_si128(_mm_maskload_pd((const float64_t*)p, mask)), mask); }
TL_XMM_DEFINE_LOADU(uint64_t) { auto mask = _ensure_mask_range(t, m).v; return _mm_blendv_epi8(default_v.v, _mm_castpd_si128(_mm_maskload_pd((const float64_t*)p, mask)), mask); }
#endif // HAS_AVX2

// load(const T* p, Mask<T> m, Vec<T> default_v)
// directly forwarded to unaligned version as most of the aligned impl is identical to unaligned version.
TL_XMM_DEFINE_LOAD(bfloat16_t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
TL_XMM_DEFINE_LOAD(float16_t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
TL_XMM_DEFINE_LOAD(float32_t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
TL_XMM_DEFINE_LOAD(float64_t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
TL_XMM_DEFINE_LOAD(int8_t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
TL_XMM_DEFINE_LOAD(uint8_t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
TL_XMM_DEFINE_LOAD(int16_t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
TL_XMM_DEFINE_LOAD(uint16_t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
TL_XMM_DEFINE_LOAD(int32_t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
TL_XMM_DEFINE_LOAD(uint32_t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
TL_XMM_DEFINE_LOAD(int64_t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
TL_XMM_DEFINE_LOAD(uint64_t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_LOADU
#undef TL_XMM_DEFINE_LOAD

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE auto _loadu_forward_mask(Tag<T, N, P> t, const T * p, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = mwhilelt(t, 0, n);
  return word::loadu(t, p, m, default_v);
}

// loadu(const float32_t* p, nint_t n, Vec<T> default_v)
#define TL_XMM_DEFINE_LOADU(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES_CHECK_COUNT(loadu, dtype, VecOf(t), (const dtype* p, nint_t n, VecOf(t) default_v), (p, n, default_v))
TL_XMM_DEFINE_LOADU(bfloat16_t) { return _loadu_forward_mask(t, p, n, default_v); }
TL_XMM_DEFINE_LOADU(float16_t) { return _loadu_forward_mask(t, p, n, default_v); }
TL_XMM_DEFINE_LOADU(float32_t) { return _loadu_forward_mask(t, p, n, default_v); }
TL_XMM_DEFINE_LOADU(float64_t) { return _loadu_forward_mask(t, p, n, default_v); }
TL_XMM_DEFINE_LOADU(int8_t) { return _loadu_forward_mask(t, p, n, default_v); }
TL_XMM_DEFINE_LOADU(uint8_t) { return _loadu_forward_mask(t, p, n, default_v); }
TL_XMM_DEFINE_LOADU(int16_t) { return _loadu_forward_mask(t, p, n, default_v); }
TL_XMM_DEFINE_LOADU(uint16_t) { return _loadu_forward_mask(t, p, n, default_v); }
TL_XMM_DEFINE_LOADU(int32_t) { return _loadu_forward_mask(t, p, n, default_v); }
TL_XMM_DEFINE_LOADU(uint32_t) { return _loadu_forward_mask(t, p, n, default_v); }
TL_XMM_DEFINE_LOADU(int64_t) { return _loadu_forward_mask(t, p, n, default_v); }
TL_XMM_DEFINE_LOADU(uint64_t) { return _loadu_forward_mask(t, p, n, default_v); }
#undef TL_XMM_DEFINE_LOADU

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE auto _load_forward_mask(Tag<T, N, P> t, const T * p, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = mwhilelt(t, 0, n);
  return word::load(t, p, m, default_v);
}

// load(const float32_t* p, nint_t n, Vec<T> default_v)
#define TL_XMM_DEFINE_LOAD(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES_CHECK_COUNT(load, dtype, VecOf(t), (const dtype* p, nint_t n, VecOf(t) default_v), (p, n, default_v))
TL_XMM_DEFINE_LOAD(bfloat16_t) { return _load_forward_mask(t, p, n, default_v); }
TL_XMM_DEFINE_LOAD(float16_t) { return _load_forward_mask(t, p, n, default_v); }
TL_XMM_DEFINE_LOAD(float32_t) { return _load_forward_mask(t, p, n, default_v); }
TL_XMM_DEFINE_LOAD(float64_t) { return _load_forward_mask(t, p, n, default_v); }
TL_XMM_DEFINE_LOAD(int8_t) { return _load_forward_mask(t, p, n, default_v); }
TL_XMM_DEFINE_LOAD(uint8_t) { return _load_forward_mask(t, p, n, default_v); }
TL_XMM_DEFINE_LOAD(int16_t) { return _load_forward_mask(t, p, n, default_v); }
TL_XMM_DEFINE_LOAD(uint16_t) { return _load_forward_mask(t, p, n, default_v); }
TL_XMM_DEFINE_LOAD(int32_t) { return _load_forward_mask(t, p, n, default_v); }
TL_XMM_DEFINE_LOAD(uint32_t) { return _load_forward_mask(t, p, n, default_v); }
TL_XMM_DEFINE_LOAD(int64_t) { return _load_forward_mask(t, p, n, default_v); }
TL_XMM_DEFINE_LOAD(uint64_t) { return _load_forward_mask(t, p, n, default_v); }
#undef TL_XMM_DEFINE_LOAD

// storeu(T* p, Vec<T> v)
#define TL_XMM_DEFINE_STOREU(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(storeu, dtype, void, (dtype * p, VecOf(t) v), (p, v))
TL_XMM_DEFINE_STOREU(bfloat16_t) { _mm_storeu_ps((float32_t*)p, _mm_castsi128_ps(v.v)); }
TL_XMM_DEFINE_STOREU(float16_t) {
  #ifdef HAS_AVX512_FP16
  _mm_storeu_ph(p, _mm_castsi128_ph(v.v));
  #else
  _mm_storeu_ps((float32_t*)p, _mm_castsi128_ps(v.v));
  #endif
}
TL_XMM_DEFINE_STOREU(float32_t) { _mm_storeu_ps(p, v.v); }
TL_XMM_DEFINE_STOREU(float64_t) { _mm_storeu_pd(p, v.v); }
TL_XMM_DEFINE_STOREU(int8_t) { _mm_storeu_si128((__m128i*)p, v.v); }
TL_XMM_DEFINE_STOREU(uint8_t) { _mm_storeu_si128((__m128i*)p, v.v); }
TL_XMM_DEFINE_STOREU(int16_t) { _mm_storeu_si128((__m128i*)p, v.v); }
TL_XMM_DEFINE_STOREU(uint16_t) { _mm_storeu_si128((__m128i*)p, v.v); }
TL_XMM_DEFINE_STOREU(int32_t) { _mm_storeu_si128((__m128i*)p, v.v); }
TL_XMM_DEFINE_STOREU(uint32_t) { _mm_storeu_si128((__m128i*)p, v.v); }
TL_XMM_DEFINE_STOREU(int64_t) { _mm_storeu_si128((__m128i*)p, v.v); }
TL_XMM_DEFINE_STOREU(uint64_t) { _mm_storeu_si128((__m128i*)p, v.v); }
#undef TL_XMM_DEFINE_STOREU

// store(T* p, Vec<T> v)
#define TL_XMM_DEFINE_STORE(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(store, dtype, void, (dtype * p, VecOf(t) v), (p, v))
TL_XMM_DEFINE_STORE(bfloat16_t) { TL_CHECK_ALIGN _mm_store_ps((float32_t*)p, _mm_castsi128_ps(v.v)); }
TL_XMM_DEFINE_STORE(float16_t) {
  TL_CHECK_ALIGN
  #ifdef HAS_AVX512_FP16
  _mm_store_ph(p, _mm_castsi128_ph(v.v));
  #else
  _mm_store_ps((float32_t*)p, _mm_castsi128_ps(v.v));
  #endif
}
TL_XMM_DEFINE_STORE(float32_t) { TL_CHECK_ALIGN _mm_store_ps(p, v.v); }
TL_XMM_DEFINE_STORE(float64_t) { TL_CHECK_ALIGN _mm_store_pd(p, v.v); }
TL_XMM_DEFINE_STORE(int8_t) { TL_CHECK_ALIGN _mm_store_si128((__m128i*)p, v.v); }
TL_XMM_DEFINE_STORE(uint8_t) { TL_CHECK_ALIGN _mm_store_si128((__m128i*)p, v.v); }
TL_XMM_DEFINE_STORE(int16_t) { TL_CHECK_ALIGN _mm_store_si128((__m128i*)p, v.v); }
TL_XMM_DEFINE_STORE(uint16_t) { TL_CHECK_ALIGN _mm_store_si128((__m128i*)p, v.v); }
TL_XMM_DEFINE_STORE(int32_t) { TL_CHECK_ALIGN _mm_store_si128((__m128i*)p, v.v); }
TL_XMM_DEFINE_STORE(uint32_t) { TL_CHECK_ALIGN _mm_store_si128((__m128i*)p, v.v); }
TL_XMM_DEFINE_STORE(int64_t) { TL_CHECK_ALIGN _mm_store_si128((__m128i*)p, v.v); }
TL_XMM_DEFINE_STORE(uint64_t) { TL_CHECK_ALIGN _mm_store_si128((__m128i*)p, v.v); }
#undef TL_XMM_DEFINE_STORE

// storeu(T* p, Mask<T> m, Vec<T> v)
#define TL_XMM_DEFINE_STOREU(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(storeu, dtype, void, (dtype* p, MaskOf(t) m, VecOf(t) v), (p, m, v))
#define TL_XMM_DEFINE_STORE(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(store, dtype, void, (dtype* p, MaskOf(t) m, VecOf(t) v), (p, m, v))

#ifdef HAS_AVX512DQ
// storeu(T* p, Mask<T> m, Vec<T> v)
TL_XMM_DEFINE_STOREU(bfloat16_t) { _mm_mask_storeu_epi16(p, _ensure_mask_range(t, m).v, v.v); }
TL_XMM_DEFINE_STOREU(float16_t) { _mm_mask_storeu_epi16(p, _ensure_mask_range(t, m).v, v.v); }
TL_XMM_DEFINE_STOREU(float32_t) { _mm_mask_storeu_ps(p, _ensure_mask_range(t, m).v, v.v); }
TL_XMM_DEFINE_STOREU(float64_t) { _mm_mask_storeu_pd(p, _ensure_mask_range(t, m).v, v.v); }
TL_XMM_DEFINE_STOREU(int8_t) { _mm_mask_storeu_epi8(p, _ensure_mask_range(t, m).v, v.v); }
TL_XMM_DEFINE_STOREU(uint8_t) { _mm_mask_storeu_epi8(p, _ensure_mask_range(t, m).v, v.v); }
TL_XMM_DEFINE_STOREU(int16_t) { _mm_mask_storeu_epi16(p, _ensure_mask_range(t, m).v, v.v); }
TL_XMM_DEFINE_STOREU(uint16_t) { _mm_mask_storeu_epi16(p, _ensure_mask_range(t, m).v, v.v); }
TL_XMM_DEFINE_STOREU(int32_t) { _mm_mask_storeu_epi32(p, _ensure_mask_range(t, m).v, v.v); }
TL_XMM_DEFINE_STOREU(uint32_t) { _mm_mask_storeu_epi32(p, _ensure_mask_range(t, m).v, v.v); }
TL_XMM_DEFINE_STOREU(int64_t) { _mm_mask_storeu_epi64(p, _ensure_mask_range(t, m).v, v.v); }
TL_XMM_DEFINE_STOREU(uint64_t) { _mm_mask_storeu_epi64(p, _ensure_mask_range(t, m).v, v.v); }

// store(T* p, Mask<T> m, Vec<T> v)
TL_XMM_DEFINE_STORE(bfloat16_t) { TL_CHECK_ALIGN _mm_mask_storeu_epi16(p, _ensure_mask_range(t, m).v, v.v); }
TL_XMM_DEFINE_STORE(float16_t) { TL_CHECK_ALIGN _mm_mask_storeu_epi16(p, _ensure_mask_range(t, m).v, v.v); }
TL_XMM_DEFINE_STORE(float32_t) { TL_CHECK_ALIGN _mm_mask_store_ps(p, _ensure_mask_range(t, m).v, v.v); }
TL_XMM_DEFINE_STORE(float64_t) { TL_CHECK_ALIGN _mm_mask_store_pd(p, _ensure_mask_range(t, m).v, v.v); }
TL_XMM_DEFINE_STORE(int8_t) { TL_CHECK_ALIGN _mm_mask_storeu_epi8(p, _ensure_mask_range(t, m).v, v.v); }
TL_XMM_DEFINE_STORE(uint8_t) { TL_CHECK_ALIGN _mm_mask_storeu_epi8(p, _ensure_mask_range(t, m).v, v.v); }
TL_XMM_DEFINE_STORE(int16_t) { TL_CHECK_ALIGN _mm_mask_storeu_epi16(p, _ensure_mask_range(t, m).v, v.v); }
TL_XMM_DEFINE_STORE(uint16_t) { TL_CHECK_ALIGN _mm_mask_storeu_epi16(p, _ensure_mask_range(t, m).v, v.v); }
TL_XMM_DEFINE_STORE(int32_t) { TL_CHECK_ALIGN _mm_mask_store_epi32(p, _ensure_mask_range(t, m).v, v.v); }
TL_XMM_DEFINE_STORE(uint32_t) { TL_CHECK_ALIGN _mm_mask_store_epi32(p, _ensure_mask_range(t, m).v, v.v); }
TL_XMM_DEFINE_STORE(int64_t) { TL_CHECK_ALIGN _mm_mask_store_epi64(p, _ensure_mask_range(t, m).v, v.v); }
TL_XMM_DEFINE_STORE(uint64_t) { TL_CHECK_ALIGN _mm_mask_store_epi64(p, _ensure_mask_range(t, m).v, v.v); }
#else // HAS_AVX512DQ
// storeu(T* p, Mask<T> m, Vec<T> v)
TL_XMM_DEFINE_STOREU(bfloat16_t) { auto mask = _ensure_mask_range(t, m).v; _mm_maskmoveu_si128(v.v, mask, (char*)p); }
TL_XMM_DEFINE_STOREU(float16_t) { auto mask = _ensure_mask_range(t, m).v; _mm_maskmoveu_si128(v.v, mask, (char*)p); }
TL_XMM_DEFINE_STOREU(float32_t) { auto mask = _ensure_mask_range(t, m).v; _mm_maskstore_ps(p, mask, v.v); }
TL_XMM_DEFINE_STOREU(float64_t) { auto mask = _ensure_mask_range(t, m).v; _mm_maskstore_pd(p, mask, v.v); }
TL_XMM_DEFINE_STOREU(int8_t) { auto mask = _ensure_mask_range(t, m).v; _mm_maskmoveu_si128(v.v, mask, (char*)p); }
TL_XMM_DEFINE_STOREU(uint8_t) { auto mask = _ensure_mask_range(t, m).v; _mm_maskmoveu_si128(v.v, mask, (char*)p); }
TL_XMM_DEFINE_STOREU(int16_t) { auto mask = _ensure_mask_range(t, m).v; _mm_maskmoveu_si128(v.v, mask, (char*)p); }
TL_XMM_DEFINE_STOREU(uint16_t) { auto mask = _ensure_mask_range(t, m).v; _mm_maskmoveu_si128(v.v, mask, (char*)p); }
TL_XMM_DEFINE_STOREU(int32_t) {
  auto mask = _ensure_mask_range(t, m).v;
  #ifdef HAS_AVX2
  _mm_maskstore_epi32(p, mask, v.v);
  #else
  _mm_maskstore_ps((float32_t*)p, mask, _mm_castsi128_ps(v.v));
  #endif
}
TL_XMM_DEFINE_STOREU(uint32_t) {
  auto mask = _ensure_mask_range(t, m).v;
  #ifdef HAS_AVX2
  _mm_maskstore_epi32((int*)p, mask, v.v);
  #else
  _mm_maskstore_ps((float32_t*)p, mask, _mm_castsi128_ps(v.v));
  #endif
}
TL_XMM_DEFINE_STOREU(int64_t) {
  auto mask = _ensure_mask_range(t, m).v;
  #ifdef HAS_AVX2
  _mm_maskstore_epi64((long long*)p, mask, v.v);
  #else
  _mm_maskstore_pd((float64_t*)p, mask, _mm_castsi128_pd(v.v));
  #endif
}
TL_XMM_DEFINE_STOREU(uint64_t) {
  auto mask = _ensure_mask_range(t, m).v;
  #ifdef HAS_AVX2
  _mm_maskstore_epi64((long long*)p, mask, v.v);
  #else
  _mm_maskstore_pd((float64_t*)p, mask, _mm_castsi128_pd(v.v));
  #endif
}

// store(T* p, Mask<T> m, Vec<T> v)
// directly forwarded to unaligned version
TL_XMM_DEFINE_STORE(bfloat16_t) { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
TL_XMM_DEFINE_STORE(float16_t) { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
TL_XMM_DEFINE_STORE(float32_t) { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
TL_XMM_DEFINE_STORE(float64_t) { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
TL_XMM_DEFINE_STORE(int8_t) { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
TL_XMM_DEFINE_STORE(uint8_t) { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
TL_XMM_DEFINE_STORE(int16_t) { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
TL_XMM_DEFINE_STORE(uint16_t) { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
TL_XMM_DEFINE_STORE(int32_t) { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
TL_XMM_DEFINE_STORE(uint32_t) { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
TL_XMM_DEFINE_STORE(int64_t) { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
TL_XMM_DEFINE_STORE(uint64_t) { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_STOREU
#undef TL_XMM_DEFINE_STORE

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE auto _storeu_forward_mask(Tag<T, N, P> t, T* p, nint_t n, VecOf(t) v) -> void {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = mwhilelt(t, 0, n);
  word::storeu(t, p, m, v);
}

// storeu(T* p, nint_t n, Vec<T> v)
#define TL_XMM_DEFINE_STOREU(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES_CHECK_COUNT(storeu, dtype, void, (dtype* p, nint_t n, VecOf(t) v), (p, n, v))
TL_XMM_DEFINE_STOREU(bfloat16_t) { _storeu_forward_mask(t, p, n, v); }
TL_XMM_DEFINE_STOREU(float16_t) { _storeu_forward_mask(t, p, n, v); }
TL_XMM_DEFINE_STOREU(float32_t) { _storeu_forward_mask(t, p, n, v); }
TL_XMM_DEFINE_STOREU(float64_t) { _storeu_forward_mask(t, p, n, v); }
TL_XMM_DEFINE_STOREU(int8_t) { _storeu_forward_mask(t, p, n, v); }
TL_XMM_DEFINE_STOREU(uint8_t) { _storeu_forward_mask(t, p, n, v); }
TL_XMM_DEFINE_STOREU(int16_t) { _storeu_forward_mask(t, p, n, v); }
TL_XMM_DEFINE_STOREU(uint16_t) { _storeu_forward_mask(t, p, n, v); }
TL_XMM_DEFINE_STOREU(int32_t) { _storeu_forward_mask(t, p, n, v); }
TL_XMM_DEFINE_STOREU(uint32_t) { _storeu_forward_mask(t, p, n, v); }
TL_XMM_DEFINE_STOREU(int64_t) { _storeu_forward_mask(t, p, n, v); }
TL_XMM_DEFINE_STOREU(uint64_t) { _storeu_forward_mask(t, p, n, v); }
#undef TL_XMM_DEFINE_STOREU

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE auto _store_forward_mask(Tag<T, N, P> t, T* p, nint_t n, VecOf(t) v) -> void {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = mwhilelt(t, 0, n);
  word::store(t, p, m, v);
}

// store(T* p, nint_t n, Vec<T> v)
#define TL_XMM_DEFINE_STORE(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES_CHECK_COUNT(store, dtype, void, (dtype* p, nint_t n, VecOf(t) v), (p, n, v))
TL_XMM_DEFINE_STORE(bfloat16_t) { _store_forward_mask(t, p, n, v); }
TL_XMM_DEFINE_STORE(float16_t) { _store_forward_mask(t, p, n, v); }
TL_XMM_DEFINE_STORE(float32_t) { _store_forward_mask(t, p, n, v); }
TL_XMM_DEFINE_STORE(float64_t) { _store_forward_mask(t, p, n, v); }
TL_XMM_DEFINE_STORE(int8_t) { _store_forward_mask(t, p, n, v); }
TL_XMM_DEFINE_STORE(uint8_t) { _store_forward_mask(t, p, n, v); }
TL_XMM_DEFINE_STORE(int16_t) { _store_forward_mask(t, p, n, v); }
TL_XMM_DEFINE_STORE(uint16_t) { _store_forward_mask(t, p, n, v); }
TL_XMM_DEFINE_STORE(int32_t) { _store_forward_mask(t, p, n, v); }
TL_XMM_DEFINE_STORE(uint32_t) { _store_forward_mask(t, p, n, v); }
TL_XMM_DEFINE_STORE(int64_t) { _store_forward_mask(t, p, n, v); }
TL_XMM_DEFINE_STORE(uint64_t) { _store_forward_mask(t, p, n, v); }
#undef TL_XMM_DEFINE_STORE

/* ************************************************************************** */
//                             Get / set element                              //
/* ************************************************************************** */

// Helper: extract element from __m128i
#define TL_DEFINE_EXTRACT(dtype, postfix) static CT_ALWAYS_FORCEINLINE dtype _extract_##postfix(__m128i v, int index)
TL_DEFINE_EXTRACT(int8_t, i8) {
  alignas(16) int8_t data[16];
  _mm_store_si128((__m128i*) data, v);
  return data[index];
}
TL_DEFINE_EXTRACT(uint8_t, u8) { return (uint8_t) _extract_i8(v, index); }
TL_DEFINE_EXTRACT(int16_t, i16) {
  alignas(16) int16_t data[8];
  _mm_store_si128((__m128i*) data, v);
  return data[index];
}
TL_DEFINE_EXTRACT(uint16_t, u16) { return (uint16_t) _extract_i16(v, index); }
TL_DEFINE_EXTRACT(int32_t, i32) {
  alignas(16) int32_t data[4];
  _mm_store_si128((__m128i*) data, v);
  return data[index];
}
TL_DEFINE_EXTRACT(uint32_t, u32) { return (uint32_t) _extract_i32(v, index); }
TL_DEFINE_EXTRACT(int64_t, i64) {
  alignas(16) int64_t data[2];
  _mm_store_si128((__m128i*) data, v);
  return data[index];
}
TL_DEFINE_EXTRACT(uint64_t, u64) { return (uint64_t) _extract_i64(v, index); }
#undef TL_DEFINE_EXTRACT

// Helper: get element from mask
#ifdef HAS_AVX512DQ
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit(__mmask8 m, int index) { return (_cvtmask8_u32(m) >> index) & 1; }
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit(__mmask16 m, int index) { return (_cvtmask16_u32(m) >> index) & 1; }
#else
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit_epi8(__m128i m, int index) { return !!_extract_i8(m, index); }
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit_epi16(__m128i m, int index) { return !!_extract_i16(m, index); }
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit_epi32(__m128i m, int index) { return !!_extract_i32(m, index); }
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit_epi64(__m128i m, int index) { return !!_extract_i64(m, index); }
#endif

// Helper: set element in mask
#ifdef HAS_AVX512DQ
static CT_ALWAYS_FORCEINLINE __mmask8 _set_mask_bit(__mmask8 m, int index, bool x) {
  uint32_t bits = _cvtmask8_u32(m);
  bits = (bits & ~(1u << index)) | ((x ? 1u : 0u) << index);
  return _cvtu32_mask8(bits);
}
static CT_ALWAYS_FORCEINLINE __mmask16 _set_mask_bit(__mmask16 m, int index, bool x) {
  uint32_t bits = _cvtmask16_u32(m);
  bits = (bits & ~(1u << index)) | ((x ? 1u : 0u) << index);
  return _cvtu32_mask16(bits);
}
#else
static CT_ALWAYS_FORCEINLINE __m128i _set_mask_bit_epi8(__m128i m, int index, bool x) {
  alignas(16) int8_t data[16];
  _mm_store_si128((__m128i *) data, m);
  data[index] = x ? -1 : 0;
  return _mm_load_si128((__m128i *) data);
}
static CT_ALWAYS_FORCEINLINE __m128i _set_mask_bit_epi16(__m128i m, int index, bool x) {
  alignas(16) int16_t data[8];
  _mm_store_si128((__m128i *) data, m);
  data[index] = x ? -1 : 0;
  return _mm_load_si128((__m128i *) data);
}
static CT_ALWAYS_FORCEINLINE __m128i _set_mask_bit_epi32(__m128i m, int index, bool x) {
  alignas(16) int32_t data[4];
  _mm_store_si128((__m128i *) data, m);
  data[index] = x ? -1 : 0;
  return _mm_load_si128((__m128i *) data);
}
static CT_ALWAYS_FORCEINLINE __m128i _set_mask_bit_epi64(__m128i m, int index, bool x) {
  alignas(16) int64_t data[2];
  _mm_store_si128((__m128i *) data, m);
  data[index] = x ? -1LL : 0LL;
  return _mm_load_si128((__m128i *) data);
}
#endif

// get(Vec<T> v, nint_t index)
#define TL_XMM_DEFINE_GET(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES_CHECK_INDEX(get, dtype, dtype, (VecOf(t) v, nint_t index), (v, index))
TL_XMM_DEFINE_GET(bfloat16_t) { TL_CHECK_INDEX(index); union { bfloat16_t b; int16_t i; } u; u.i = _extract_i16(v.v, (int) index); return u.b; }
TL_XMM_DEFINE_GET(float16_t) { TL_CHECK_INDEX(index); union { float16_t h; int16_t i; } u; u.i = _extract_i16(v.v, (int) index); return u.h; }
TL_XMM_DEFINE_GET(float32_t) { TL_CHECK_INDEX(index); return _mm_cvtss_f32(_mm_permutevar_ps(v.v, _mm_cvtsi32_si128((int)index))); }
// Note: weird permutevar_pd requires second-to-the-last bit.
TL_XMM_DEFINE_GET(float64_t) { TL_CHECK_INDEX(index); return _mm_cvtsd_f64(_mm_permutevar_pd(v.v, _mm_cvtsi64_si128((int) (index << 1)))); }
TL_XMM_DEFINE_GET(int8_t) { TL_CHECK_INDEX(index); return _extract_i8(v.v, (int) index); }
TL_XMM_DEFINE_GET(uint8_t) { TL_CHECK_INDEX(index); return _extract_u8(v.v, (int) index); }
TL_XMM_DEFINE_GET(int16_t) { TL_CHECK_INDEX(index); return _extract_i16(v.v, (int) index); }
TL_XMM_DEFINE_GET(uint16_t) { TL_CHECK_INDEX(index); return _extract_u16(v.v, (int) index); }
TL_XMM_DEFINE_GET(int32_t) { TL_CHECK_INDEX(index); return _extract_i32(v.v, (int) index); }
TL_XMM_DEFINE_GET(uint32_t) { TL_CHECK_INDEX(index); return _extract_u32(v.v, (int) index); }
TL_XMM_DEFINE_GET(int64_t) { TL_CHECK_INDEX(index); return _extract_i64(v.v, (int) index); }
TL_XMM_DEFINE_GET(uint64_t) { TL_CHECK_INDEX(index); return _extract_u64(v.v, (int) index); }
#undef TL_XMM_DEFINE_GET

// get(Mask<T> m, nint_t index)
#define TL_XMM_DEFINE_GET(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES_CHECK_INDEX(get, dtype, bool, (MaskOf(t) v, nint_t index), (v, index))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_GET(bfloat16_t) { TL_CHECK_INDEX(index); return _get_mask_bit(v.v, (int) index); }
TL_XMM_DEFINE_GET(float16_t) { TL_CHECK_INDEX(index); return _get_mask_bit(v.v, (int) index); }
TL_XMM_DEFINE_GET(float32_t) { TL_CHECK_INDEX(index); return _get_mask_bit(v.v, (int) index); }
TL_XMM_DEFINE_GET(float64_t) { TL_CHECK_INDEX(index); return _get_mask_bit(v.v, (int) index); }
TL_XMM_DEFINE_GET(int8_t) { TL_CHECK_INDEX(index); return _get_mask_bit(v.v, (int) index); }
TL_XMM_DEFINE_GET(uint8_t) { TL_CHECK_INDEX(index); return _get_mask_bit(v.v, (int) index); }
TL_XMM_DEFINE_GET(int16_t) { TL_CHECK_INDEX(index); return _get_mask_bit(v.v, (int) index); }
TL_XMM_DEFINE_GET(uint16_t) { TL_CHECK_INDEX(index); return _get_mask_bit(v.v, (int) index); }
TL_XMM_DEFINE_GET(int32_t) { TL_CHECK_INDEX(index); return _get_mask_bit(v.v, (int) index); }
TL_XMM_DEFINE_GET(uint32_t) { TL_CHECK_INDEX(index); return _get_mask_bit(v.v, (int) index); }
TL_XMM_DEFINE_GET(int64_t) { TL_CHECK_INDEX(index); return _get_mask_bit(v.v, (int) index); }
TL_XMM_DEFINE_GET(uint64_t) { TL_CHECK_INDEX(index); return _get_mask_bit(v.v, (int) index); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_GET(bfloat16_t) { TL_CHECK_INDEX(index); return _get_mask_bit_epi16(v.v, (int) index); }
TL_XMM_DEFINE_GET(float16_t) { TL_CHECK_INDEX(index); return _get_mask_bit_epi16(v.v, (int) index); }
TL_XMM_DEFINE_GET(float32_t) { TL_CHECK_INDEX(index); return _get_mask_bit_epi32(v.v, (int) index); }
TL_XMM_DEFINE_GET(float64_t) { TL_CHECK_INDEX(index); return _get_mask_bit_epi64(v.v, (int) index); }
TL_XMM_DEFINE_GET(int8_t) { TL_CHECK_INDEX(index); return _get_mask_bit_epi8(v.v, (int) index); }
TL_XMM_DEFINE_GET(uint8_t) { TL_CHECK_INDEX(index); return _get_mask_bit_epi8(v.v, (int) index); }
TL_XMM_DEFINE_GET(int16_t) { TL_CHECK_INDEX(index); return _get_mask_bit_epi16(v.v, (int) index); }
TL_XMM_DEFINE_GET(uint16_t) { TL_CHECK_INDEX(index); return _get_mask_bit_epi16(v.v, (int) index); }
TL_XMM_DEFINE_GET(int32_t) { TL_CHECK_INDEX(index); return _get_mask_bit_epi32(v.v, (int) index); }
TL_XMM_DEFINE_GET(uint32_t) { TL_CHECK_INDEX(index); return _get_mask_bit_epi32(v.v, (int) index); }
TL_XMM_DEFINE_GET(int64_t) { TL_CHECK_INDEX(index); return _get_mask_bit_epi64(v.v, (int) index); }
TL_XMM_DEFINE_GET(uint64_t) { TL_CHECK_INDEX(index); return _get_mask_bit_epi64(v.v, (int) index); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_GET

// set(Vec<T> v, nint_t index, T v)
#define TL_XMM_DEFINE_SET(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES_CHECK_INDEX(set, dtype, VecOf(t), (VecOf(t) v, nint_t index, dtype x), (v, index, x))
#if defined(HAS_AVX512DQ) && defined(HAS_AVX512BW) && defined(HAS_AVX512VL)
TL_XMM_DEFINE_SET(bfloat16_t) { TL_CHECK_INDEX(index);
  union { bfloat16_t f; int16_t i; } u { .f = x };
  auto mask = _cvtu32_mask8(1u << (unsigned) index);
  return _mm_mask_mov_epi16(v.v, mask, _mm_set1_epi16(u.i));
}
TL_XMM_DEFINE_SET(float16_t) { TL_CHECK_INDEX(index);
  union { float16_t f; int16_t i; } u { .f = x };
  auto mask = _cvtu32_mask8(1u << (unsigned) index);
  return _mm_mask_mov_epi16(v.v, mask, _mm_set1_epi16(u.i));
}
TL_XMM_DEFINE_SET(float32_t) { TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask8(1u << (unsigned) index);
  return _mm_mask_mov_ps(v.v, mask, _mm_set1_ps(x));
}
TL_XMM_DEFINE_SET(float64_t) { TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask8(1u << (unsigned) index);
  return _mm_mask_mov_pd(v.v, mask, _mm_set1_pd(x));
}
TL_XMM_DEFINE_SET(int8_t) { TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask16(1u << (unsigned) index);
  return _mm_mask_mov_epi8(v.v, mask, _mm_set1_epi8(x));
}
TL_XMM_DEFINE_SET(uint8_t) { TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask16(1u << (unsigned) index);
  return _mm_mask_mov_epi8(v.v, mask, _mm_set1_epi8((int8_t)x));
}
TL_XMM_DEFINE_SET(int16_t) { TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask8(1u << (unsigned) index);
  return _mm_mask_mov_epi16(v.v, mask, _mm_set1_epi16(x));
}
TL_XMM_DEFINE_SET(uint16_t) { TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask8(1u << (unsigned) index);
  return _mm_mask_mov_epi16(v.v, mask, _mm_set1_epi16((int16_t)x));
}
TL_XMM_DEFINE_SET(int32_t) { TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask8(1u << (unsigned) index);
  return _mm_mask_mov_epi32(v.v, mask, _mm_set1_epi32(x));
}
TL_XMM_DEFINE_SET(uint32_t) { TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask8(1u << (unsigned) index);
  return _mm_mask_mov_epi32(v.v, mask, _mm_set1_epi32((int32_t)x));
}
TL_XMM_DEFINE_SET(int64_t) { TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask8(1u << (unsigned) index);
  return _mm_mask_mov_epi64(v.v, mask, _mm_set1_epi64x(x));
}
TL_XMM_DEFINE_SET(uint64_t) { TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask8(1u << (unsigned) index);
  return _mm_mask_mov_epi64(v.v, mask, _mm_set1_epi64x((int64_t)x));
}
#else // defined(HAS_AVX512DQ) && defined(HAS_AVX512BW) && defined(HAS_AVX512VL)
TL_XMM_DEFINE_SET(bfloat16_t) { TL_CHECK_INDEX(index);
  alignas(16) bfloat16_t data[8];
  _mm_store_si128((__m128i *)data, v.v);
  data[index] = x;
  return _mm_load_si128((__m128i *)data);
}
TL_XMM_DEFINE_SET(float16_t) { TL_CHECK_INDEX(index);
  alignas(16) float16_t data[8];
  _mm_store_si128((__m128i *)data, v.v);
  data[index] = x;
  return _mm_load_si128((__m128i *)data);
}
TL_XMM_DEFINE_SET(float32_t) { TL_CHECK_INDEX(index);
  alignas(16) float32_t data[4];
  _mm_store_ps(data, v.v);
  data[index] = x;
  return _mm_load_ps(data);
}
TL_XMM_DEFINE_SET(float64_t) { TL_CHECK_INDEX(index);
  alignas(16) float64_t data[2];
  _mm_store_pd(data, v.v);
  data[index] = x;
  return _mm_load_pd(data);
}
TL_XMM_DEFINE_SET(int8_t) { TL_CHECK_INDEX(index);
  alignas(16) int8_t data[16];
  _mm_store_si128((__m128i *)data, v.v);
  data[index] = x;
  return _mm_load_si128((__m128i *)data);
}
TL_XMM_DEFINE_SET(uint8_t) { TL_CHECK_INDEX(index);
  alignas(16) uint8_t data[16];
  _mm_store_si128((__m128i *)data, v.v);
  data[index] = x;
  return _mm_load_si128((__m128i *)data);
}
TL_XMM_DEFINE_SET(int16_t) { TL_CHECK_INDEX(index);
  alignas(16) int16_t data[8];
  _mm_store_si128((__m128i *)data, v.v);
  data[index] = x;
  return _mm_load_si128((__m128i *)data);
}
TL_XMM_DEFINE_SET(uint16_t) { TL_CHECK_INDEX(index);
  alignas(16) uint16_t data[8];
  _mm_store_si128((__m128i *)data, v.v);
  data[index] = x;
  return _mm_load_si128((__m128i *)data);
}
TL_XMM_DEFINE_SET(int32_t) { TL_CHECK_INDEX(index);
  alignas(16) int32_t data[4];
  _mm_store_si128((__m128i *)data, v.v);
  data[index] = x;
  return _mm_load_si128((__m128i *)data);
}
TL_XMM_DEFINE_SET(uint32_t) { TL_CHECK_INDEX(index);
  alignas(16) uint32_t data[4];
  _mm_store_si128((__m128i *)data, v.v);
  data[index] = x;
  return _mm_load_si128((__m128i *)data);
}
TL_XMM_DEFINE_SET(int64_t) { TL_CHECK_INDEX(index);
  alignas(16) int64_t data[2];
  _mm_store_si128((__m128i *)data, v.v);
  data[index] = x;
  return _mm_load_si128((__m128i *)data);
}
TL_XMM_DEFINE_SET(uint64_t) { TL_CHECK_INDEX(index);
  alignas(16) uint64_t data[2];
  _mm_store_si128((__m128i *)data, v.v);
  data[index] = x;
  return _mm_load_si128((__m128i *)data);
}
#endif // defined(HAS_AVX512DQ) && defined(HAS_AVX512BW) && defined(HAS_AVX512VL)
#undef TL_XMM_DEFINE_SET

// set(Mask<T> m, nint_t index, bool v)
#define TL_XMM_DEFINE_SET(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES_CHECK_INDEX(set, dtype, MaskOf(t), (MaskOf(t) v, nint_t index, bool x), (v, index, x))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_SET(bfloat16_t) { TL_CHECK_INDEX(index); return _set_mask_bit(v.v, (int) index, x); }
TL_XMM_DEFINE_SET(float16_t) { TL_CHECK_INDEX(index); return _set_mask_bit(v.v, (int) index, x); }
TL_XMM_DEFINE_SET(float32_t) { TL_CHECK_INDEX(index); return _set_mask_bit(v.v, (int) index, x); }
TL_XMM_DEFINE_SET(float64_t) { TL_CHECK_INDEX(index); return _set_mask_bit(v.v, (int) index, x); }
TL_XMM_DEFINE_SET(int8_t) { TL_CHECK_INDEX(index); return _set_mask_bit(v.v, (int) index, x); }
TL_XMM_DEFINE_SET(uint8_t) { TL_CHECK_INDEX(index); return _set_mask_bit(v.v, (int) index, x); }
TL_XMM_DEFINE_SET(int16_t) { TL_CHECK_INDEX(index); return _set_mask_bit(v.v, (int) index, x); }
TL_XMM_DEFINE_SET(uint16_t) { TL_CHECK_INDEX(index); return _set_mask_bit(v.v, (int) index, x); }
TL_XMM_DEFINE_SET(int32_t) { TL_CHECK_INDEX(index); return _set_mask_bit(v.v, (int) index, x); }
TL_XMM_DEFINE_SET(uint32_t) { TL_CHECK_INDEX(index); return _set_mask_bit(v.v, (int) index, x); }
TL_XMM_DEFINE_SET(int64_t) { TL_CHECK_INDEX(index); return _set_mask_bit(v.v, (int) index, x); }
TL_XMM_DEFINE_SET(uint64_t) { TL_CHECK_INDEX(index); return _set_mask_bit(v.v, (int) index, x); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_SET(bfloat16_t) { TL_CHECK_INDEX(index); return _set_mask_bit_epi16(v.v, (int) index, x); }
TL_XMM_DEFINE_SET(float16_t) { TL_CHECK_INDEX(index); return _set_mask_bit_epi16(v.v, (int) index, x); }
TL_XMM_DEFINE_SET(float32_t) { TL_CHECK_INDEX(index); return _set_mask_bit_epi32(v.v, (int) index, x); }
TL_XMM_DEFINE_SET(float64_t) { TL_CHECK_INDEX(index); return _set_mask_bit_epi64(v.v, (int) index, x); }
TL_XMM_DEFINE_SET(int8_t) { TL_CHECK_INDEX(index); return _set_mask_bit_epi8(v.v, (int) index, x); }
TL_XMM_DEFINE_SET(uint8_t) { TL_CHECK_INDEX(index); return _set_mask_bit_epi8(v.v, (int) index, x); }
TL_XMM_DEFINE_SET(int16_t) { TL_CHECK_INDEX(index); return _set_mask_bit_epi16(v.v, (int) index, x); }
TL_XMM_DEFINE_SET(uint16_t) { TL_CHECK_INDEX(index); return _set_mask_bit_epi16(v.v, (int) index, x); }
TL_XMM_DEFINE_SET(int32_t) { TL_CHECK_INDEX(index); return _set_mask_bit_epi32(v.v, (int) index, x); }
TL_XMM_DEFINE_SET(uint32_t) { TL_CHECK_INDEX(index); return _set_mask_bit_epi32(v.v, (int) index, x); }
TL_XMM_DEFINE_SET(int64_t) { TL_CHECK_INDEX(index); return _set_mask_bit_epi64(v.v, (int) index, x); }
TL_XMM_DEFINE_SET(uint64_t) { TL_CHECK_INDEX(index); return _set_mask_bit_epi64(v.v, (int) index, x); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_SET

#undef _TL_XMM_DEFINE_HALVES
#undef _TL_XMM_DEFINE_HALVES_CHECK_RANGE
#undef _TL_XMM_DEFINE_HALVES_CHECK_INDEX
#undef TL_XMM_DEFINE_WITH_ALL_HALVES
#undef TL_XMM_DEFINE_WITH_ALL_HALVES_CHECK_COUNT
#undef TL_XMM_DEFINE_WITH_ALL_HALVES_V
#undef TL_CHECK_COUNT
#undef TL_CHECK_INDEX
#undef TL_CHECK_ALIGN
} // namespace word
} // namespace ct::tl::vec
//@formatter:on

#endif //CTORCH_X86_128_H
