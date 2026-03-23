//
// Created by renyz on 2026/3/17.
//

#ifndef CTORCH_X86_128_H
#define CTORCH_X86_128_H

//@formatter:off
#include <cmath>
#include "CoreDefs.h"
#include "tl/util/Math.h"

#ifndef ARCH_X86_FAMILY
  #error "Not x86 platform"
#endif

#ifndef HAS_CPU_CAPABILITY_AVX
  #error "AVX instruction set required"
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
CT_ALWAYS_FORCEINLINE auto name(Tag<dtype, N> t  X params) -> ret { \
    check; return word::name(Tag<dtype, 16 / sizeof(dtype)>()  X args); \
}

#define TL_CHECK_COUNT(varname) CT_ASSERT(0 <= (varname) && (varname) <= size(t), "%zd !in 0..%zd", (varname), size(t))
#define TL_CHECK_INDEX(varname) CT_ASSERT(0 <= (varname) && (varname) < size(t), "%zd !in 0..%zd", (varname), size(t))
#define TL_CHECK_ALIGN CT_ASSERT(((nuint_t)(p) & (16 - 1)) == 0, "Not aligned");

#define TL_XMM_DEFINE_WITH_ALL_HALVES(name, dtype, ret, params, args) \
auto name(Tag<dtype, (16 / sizeof(dtype))> t, X params) -> ret; /*forward declaration*/ \
TL_XMM_APPLY_TO_ALL_HALVES(dtype, _TL_XMM_DEFINE_HALVES, , name, dtype, ret, (, X params), (, X args)) \
CT_ALWAYS_FORCEINLINE auto name(Tag<dtype, (16 / sizeof(dtype))> t, X params) -> ret

#define TL_XMM_DEFINE_WITH_ALL_HALVES_CHECK_COUNT(name, dtype, ret, params, args) \
auto name(Tag<dtype, (16 / sizeof(dtype))> t, X params) -> ret; /*forward declaration*/ \
TL_XMM_APPLY_TO_ALL_HALVES(dtype, _TL_XMM_DEFINE_HALVES, TL_CHECK_COUNT(n), name, dtype, ret, (, X params), (, X args)) \
CT_ALWAYS_FORCEINLINE auto name(Tag<dtype, (16 / sizeof(dtype))> t, X params) -> ret

#define TL_XMM_DEFINE_WITH_ALL_HALVES_CHECK_INDEX(name, dtype, ret, params, args) \
auto name(Tag<dtype, (16 / sizeof(dtype))> t, X params) -> ret; /*forward declaration*/ \
TL_XMM_APPLY_TO_ALL_HALVES(dtype, _TL_XMM_DEFINE_HALVES, TL_CHECK_INDEX(index), name, dtype, ret, (, X params), (, X args)) \
CT_ALWAYS_FORCEINLINE auto name(Tag<dtype, (16 / sizeof(dtype))> t, X params) -> ret

#define TL_XMM_DEFINE_WITH_ALL_HALVES_V(name, dtype, ret) \
auto name(Tag<dtype, (16 / sizeof(dtype))> t) -> ret; /*forward declaration*/ \
TL_XMM_APPLY_TO_ALL_HALVES(dtype, _TL_XMM_DEFINE_HALVES, , name, dtype, ret, (), ()) \
CT_ALWAYS_FORCEINLINE auto name(Tag<dtype, (16 / sizeof(dtype))> t) -> ret



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
CT_ALWAYS_FORCEINLINE auto _fill_fwd_mask_128(Tag<T, N, P> t, T v, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = word::mwhilelt(t, 0, n);
  return word::fill(t, v, m, default_v);
}

// fill(T v, nint_t n, Vec<T> default_v)
#define TL_XMM_DEFINE_FILL(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES_CHECK_COUNT(fill, dtype, VecOf(t), (dtype v, nint_t n, VecOf(t) default_v), (v, n, default_v))
TL_XMM_DEFINE_FILL(bfloat16_t) { return _fill_fwd_mask_128(t, v, n, default_v); }
TL_XMM_DEFINE_FILL(float16_t) { return _fill_fwd_mask_128(t, v, n, default_v); }
TL_XMM_DEFINE_FILL(float32_t) { return _fill_fwd_mask_128(t, v, n, default_v); }
TL_XMM_DEFINE_FILL(float64_t) { return _fill_fwd_mask_128(t, v, n, default_v); }
TL_XMM_DEFINE_FILL(int8_t) { return _fill_fwd_mask_128(t, v, n, default_v); }
TL_XMM_DEFINE_FILL(uint8_t) { return _fill_fwd_mask_128(t, v, n, default_v); }
TL_XMM_DEFINE_FILL(int16_t) { return _fill_fwd_mask_128(t, v, n, default_v); }
TL_XMM_DEFINE_FILL(uint16_t) { return _fill_fwd_mask_128(t, v, n, default_v); }
TL_XMM_DEFINE_FILL(int32_t) { return _fill_fwd_mask_128(t, v, n, default_v); }
TL_XMM_DEFINE_FILL(uint32_t) { return _fill_fwd_mask_128(t, v, n, default_v); }
TL_XMM_DEFINE_FILL(int64_t) { return _fill_fwd_mask_128(t, v, n, default_v); }
TL_XMM_DEFINE_FILL(uint64_t) { return _fill_fwd_mask_128(t, v, n, default_v); }
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
CT_ALWAYS_FORCEINLINE auto _ensure_mask_range_128(Tag<T, N, P> t, MaskOf(t) m) -> MaskOf(t) {
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
TL_XMM_DEFINE_LOADU(bfloat16_t) { return _mm_mask_loadu_epi16(default_v.v, _ensure_mask_range_128(t, m).v, p); }
TL_XMM_DEFINE_LOADU(float16_t) { return _mm_mask_loadu_epi16(default_v.v, _ensure_mask_range_128(t, m).v, p); }
TL_XMM_DEFINE_LOADU(float32_t) { return _mm_mask_loadu_ps(default_v.v, _ensure_mask_range_128(t, m).v, p); }
TL_XMM_DEFINE_LOADU(float64_t) { return _mm_mask_loadu_pd(default_v.v, _ensure_mask_range_128(t, m).v, p); }
TL_XMM_DEFINE_LOADU(int8_t) { return _mm_mask_loadu_epi8(default_v.v, _ensure_mask_range_128(t, m).v, p); }
TL_XMM_DEFINE_LOADU(uint8_t) { return _mm_mask_loadu_epi8(default_v.v, _ensure_mask_range_128(t, m).v, p); }
TL_XMM_DEFINE_LOADU(int16_t) { return _mm_mask_loadu_epi16(default_v.v, _ensure_mask_range_128(t, m).v, p); }
TL_XMM_DEFINE_LOADU(uint16_t) { return _mm_mask_loadu_epi16(default_v.v, _ensure_mask_range_128(t, m).v, p); }
TL_XMM_DEFINE_LOADU(int32_t) { return _mm_mask_loadu_epi32(default_v.v, _ensure_mask_range_128(t, m).v, p); }
TL_XMM_DEFINE_LOADU(uint32_t) { return _mm_mask_loadu_epi32(default_v.v, _ensure_mask_range_128(t, m).v, p); }
TL_XMM_DEFINE_LOADU(int64_t) { return _mm_mask_loadu_epi64(default_v.v, _ensure_mask_range_128(t, m).v, p); }
TL_XMM_DEFINE_LOADU(uint64_t) { return _mm_mask_loadu_epi64(default_v.v, _ensure_mask_range_128(t, m).v, p); }

// load(const T* p, Mask<T> m, Vec<T> default_v)
TL_XMM_DEFINE_LOAD(bfloat16_t) { return _mm_mask_loadu_epi16(default_v.v, _ensure_mask_range_128(t, m).v, p); }
TL_XMM_DEFINE_LOAD(float16_t) { return _mm_mask_loadu_epi16(default_v.v, _ensure_mask_range_128(t, m).v, p); }
TL_XMM_DEFINE_LOAD(float32_t) { return _mm_mask_load_ps(default_v.v, _ensure_mask_range_128(t, m).v, p); }
TL_XMM_DEFINE_LOAD(float64_t) { return _mm_mask_load_pd(default_v.v, _ensure_mask_range_128(t, m).v, p); }
TL_XMM_DEFINE_LOAD(int8_t) { return _mm_mask_loadu_epi8(default_v.v, _ensure_mask_range_128(t, m).v, p); }
TL_XMM_DEFINE_LOAD(uint8_t) { return _mm_mask_loadu_epi8(default_v.v, _ensure_mask_range_128(t, m).v, p); }
TL_XMM_DEFINE_LOAD(int16_t) { return _mm_mask_loadu_epi16(default_v.v, _ensure_mask_range_128(t, m).v, p); }
TL_XMM_DEFINE_LOAD(uint16_t) { return _mm_mask_loadu_epi16(default_v.v, _ensure_mask_range_128(t, m).v, p); }
TL_XMM_DEFINE_LOAD(int32_t) { return _mm_mask_load_epi32(default_v.v, _ensure_mask_range_128(t, m).v, p); }
TL_XMM_DEFINE_LOAD(uint32_t) { return _mm_mask_load_epi32(default_v.v, _ensure_mask_range_128(t, m).v, p); }
TL_XMM_DEFINE_LOAD(int64_t) { return _mm_mask_load_epi64(default_v.v, _ensure_mask_range_128(t, m).v, p); }
TL_XMM_DEFINE_LOAD(uint64_t) { return _mm_mask_load_epi64(default_v.v, _ensure_mask_range_128(t, m).v, p); }
#else // HAS_AVX512DQ
// Note: it's pretty hard to ensure memory boundary safety without hardware maskload for int8 and int16
// So we instead checks for page boundary to determine if a full 16-byte load is available
template <typename T, nint_t N, int POW2>
CT_ALWAYS_FORCEINLINE auto _loadu_8_128(Tag<T, N, POW2> t, const T* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) {
  auto mask = _ensure_mask_range_128(t, m).v;
  // if p + 15 still in the same page as p
  if (((nuint_t(p) & 0xfff) + 15) <= 0xfff) {
    return _mm_blendv_epi8(default_v.v, _mm_castps_si128(_mm_loadu_ps((const float32_t*) p)), mask);
  } else {
    // TODO slow
    // Fallback to scalar implementation
    union { int8_t i[16]; __m128i m; } V{.m = default_v.v}, M{.m = mask};
    alignas(16) int8_t S[16];
    auto P = (const int8_t*) p;
    for (int i = 0; i < 16; ++i) S[i] = M.i[i] ? P[i] : V.i[i];
    return _mm_load_si128((const __m128i *)S);
  }
}

template <typename T, nint_t N, int POW2>
CT_ALWAYS_FORCEINLINE auto _loadu_16_128(Tag<T, N, POW2> t, const T* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) {
  auto mask = _ensure_mask_range_128(t, m).v;
  // if p + 15 still in the same page as p
  if (((nuint_t(p) & 0xfff) + 15) <= 0xfff) {
    return _mm_blendv_epi8(default_v.v, _mm_castps_si128(_mm_loadu_ps((const float32_t*) p)), mask);
  } else {
    // TODO slow
    // Fallback to scalar implementation
    union { int16_t i[8]; __m128i m; } V{.m = default_v.v}, M{.m = mask};
    alignas(16) int16_t S[8];
    auto P = (const int16_t*) p;
    for (int i = 0; i < 8; ++i) S[i] = M.i[i] ? P[i] : V.i[i];
    return _mm_load_si128((const __m128i *)S);
  }
}
// loadu(const T* p, Mask<T> m, Vec<T> default_v)
TL_XMM_DEFINE_LOADU(bfloat16_t) { return _loadu_16_128(t, p, m, default_v); }
TL_XMM_DEFINE_LOADU(float16_t) { return _loadu_16_128(t, p, m, default_v); }
TL_XMM_DEFINE_LOADU(float32_t) { auto mask = _ensure_mask_range_128(t, m).v; return _mm_blendv_ps(default_v.v, _mm_maskload_ps(p, mask), _mm_castsi128_ps(mask)); }
TL_XMM_DEFINE_LOADU(float64_t) { auto mask = _ensure_mask_range_128(t, m).v; return _mm_blendv_pd(default_v.v, _mm_maskload_pd(p, mask), _mm_castsi128_pd(mask)); }
TL_XMM_DEFINE_LOADU(int8_t) { return _loadu_8_128(t, p, m, default_v); }
TL_XMM_DEFINE_LOADU(uint8_t) { return _loadu_8_128(t, p, m, default_v); }
TL_XMM_DEFINE_LOADU(int16_t) { return _loadu_16_128(t, p, m, default_v); }
TL_XMM_DEFINE_LOADU(uint16_t) { return _loadu_16_128(t, p, m, default_v); }
#ifdef HAS_AVX2
TL_XMM_DEFINE_LOADU(int32_t) { auto mask = _ensure_mask_range_128(t, m).v; return _mm_blendv_epi8(default_v.v, _mm_maskload_epi32(p, mask), mask); }
TL_XMM_DEFINE_LOADU(uint32_t) { auto mask = _ensure_mask_range_128(t, m).v; return _mm_blendv_epi8(default_v.v, _mm_maskload_epi32((const int *)p, mask), mask); }
TL_XMM_DEFINE_LOADU(int64_t) { auto mask = _ensure_mask_range_128(t, m).v; return _mm_blendv_epi8(default_v.v, _mm_maskload_epi64((const long long *)p, mask), mask); }
TL_XMM_DEFINE_LOADU(uint64_t) { auto mask = _ensure_mask_range_128(t, m).v; return _mm_blendv_epi8(default_v.v, _mm_maskload_epi64((const long long *)p, mask), mask); }
#else // HAS_AVX2
TL_XMM_DEFINE_LOADU(int32_t) { auto mask = _ensure_mask_range_128(t, m).v; return _mm_blendv_epi8(default_v.v, _mm_castps_si128(_mm_maskload_ps((const float32_t*)p, mask)), mask); }
TL_XMM_DEFINE_LOADU(uint32_t) { auto mask = _ensure_mask_range_128(t, m).v; return _mm_blendv_epi8(default_v.v, _mm_castps_si128(_mm_maskload_ps((const float32_t*)p, mask)), mask); }
TL_XMM_DEFINE_LOADU(int64_t) { auto mask = _ensure_mask_range_128(t, m).v; return _mm_blendv_epi8(default_v.v, _mm_castpd_si128(_mm_maskload_pd((const float64_t*)p, mask)), mask); }
TL_XMM_DEFINE_LOADU(uint64_t) { auto mask = _ensure_mask_range_128(t, m).v; return _mm_blendv_epi8(default_v.v, _mm_castpd_si128(_mm_maskload_pd((const float64_t*)p, mask)), mask); }
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
CT_ALWAYS_FORCEINLINE auto _loadu_fwd_mask_128(Tag<T, N, P> t, const T * p, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = mwhilelt(t, 0, n);
  return word::loadu(t, p, m, default_v);
}

// loadu(const float32_t* p, nint_t n, Vec<T> default_v)
#define TL_XMM_DEFINE_LOADU(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES_CHECK_COUNT(loadu, dtype, VecOf(t), (const dtype* p, nint_t n, VecOf(t) default_v), (p, n, default_v))
TL_XMM_DEFINE_LOADU(bfloat16_t) { return _loadu_fwd_mask_128(t, p, n, default_v); }
TL_XMM_DEFINE_LOADU(float16_t) { return _loadu_fwd_mask_128(t, p, n, default_v); }
TL_XMM_DEFINE_LOADU(float32_t) { return _loadu_fwd_mask_128(t, p, n, default_v); }
TL_XMM_DEFINE_LOADU(float64_t) { return _loadu_fwd_mask_128(t, p, n, default_v); }
TL_XMM_DEFINE_LOADU(int8_t) { return _loadu_fwd_mask_128(t, p, n, default_v); }
TL_XMM_DEFINE_LOADU(uint8_t) { return _loadu_fwd_mask_128(t, p, n, default_v); }
TL_XMM_DEFINE_LOADU(int16_t) { return _loadu_fwd_mask_128(t, p, n, default_v); }
TL_XMM_DEFINE_LOADU(uint16_t) { return _loadu_fwd_mask_128(t, p, n, default_v); }
TL_XMM_DEFINE_LOADU(int32_t) { return _loadu_fwd_mask_128(t, p, n, default_v); }
TL_XMM_DEFINE_LOADU(uint32_t) { return _loadu_fwd_mask_128(t, p, n, default_v); }
TL_XMM_DEFINE_LOADU(int64_t) { return _loadu_fwd_mask_128(t, p, n, default_v); }
TL_XMM_DEFINE_LOADU(uint64_t) { return _loadu_fwd_mask_128(t, p, n, default_v); }
#undef TL_XMM_DEFINE_LOADU

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE auto _load_fwd_mask_128(Tag<T, N, P> t, const T * p, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = mwhilelt(t, 0, n);
  return word::load(t, p, m, default_v);
}

// load(const float32_t* p, nint_t n, Vec<T> default_v)
#define TL_XMM_DEFINE_LOAD(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES_CHECK_COUNT(load, dtype, VecOf(t), (const dtype* p, nint_t n, VecOf(t) default_v), (p, n, default_v))
TL_XMM_DEFINE_LOAD(bfloat16_t) { return _load_fwd_mask_128(t, p, n, default_v); }
TL_XMM_DEFINE_LOAD(float16_t) { return _load_fwd_mask_128(t, p, n, default_v); }
TL_XMM_DEFINE_LOAD(float32_t) { return _load_fwd_mask_128(t, p, n, default_v); }
TL_XMM_DEFINE_LOAD(float64_t) { return _load_fwd_mask_128(t, p, n, default_v); }
TL_XMM_DEFINE_LOAD(int8_t) { return _load_fwd_mask_128(t, p, n, default_v); }
TL_XMM_DEFINE_LOAD(uint8_t) { return _load_fwd_mask_128(t, p, n, default_v); }
TL_XMM_DEFINE_LOAD(int16_t) { return _load_fwd_mask_128(t, p, n, default_v); }
TL_XMM_DEFINE_LOAD(uint16_t) { return _load_fwd_mask_128(t, p, n, default_v); }
TL_XMM_DEFINE_LOAD(int32_t) { return _load_fwd_mask_128(t, p, n, default_v); }
TL_XMM_DEFINE_LOAD(uint32_t) { return _load_fwd_mask_128(t, p, n, default_v); }
TL_XMM_DEFINE_LOAD(int64_t) { return _load_fwd_mask_128(t, p, n, default_v); }
TL_XMM_DEFINE_LOAD(uint64_t) { return _load_fwd_mask_128(t, p, n, default_v); }
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
TL_XMM_DEFINE_STOREU(bfloat16_t) { _mm_mask_storeu_epi16(p, _ensure_mask_range_128(t, m).v, v.v); }
TL_XMM_DEFINE_STOREU(float16_t) { _mm_mask_storeu_epi16(p, _ensure_mask_range_128(t, m).v, v.v); }
TL_XMM_DEFINE_STOREU(float32_t) { _mm_mask_storeu_ps(p, _ensure_mask_range_128(t, m).v, v.v); }
TL_XMM_DEFINE_STOREU(float64_t) { _mm_mask_storeu_pd(p, _ensure_mask_range_128(t, m).v, v.v); }
TL_XMM_DEFINE_STOREU(int8_t) { _mm_mask_storeu_epi8(p, _ensure_mask_range_128(t, m).v, v.v); }
TL_XMM_DEFINE_STOREU(uint8_t) { _mm_mask_storeu_epi8(p, _ensure_mask_range_128(t, m).v, v.v); }
TL_XMM_DEFINE_STOREU(int16_t) { _mm_mask_storeu_epi16(p, _ensure_mask_range_128(t, m).v, v.v); }
TL_XMM_DEFINE_STOREU(uint16_t) { _mm_mask_storeu_epi16(p, _ensure_mask_range_128(t, m).v, v.v); }
TL_XMM_DEFINE_STOREU(int32_t) { _mm_mask_storeu_epi32(p, _ensure_mask_range_128(t, m).v, v.v); }
TL_XMM_DEFINE_STOREU(uint32_t) { _mm_mask_storeu_epi32(p, _ensure_mask_range_128(t, m).v, v.v); }
TL_XMM_DEFINE_STOREU(int64_t) { _mm_mask_storeu_epi64(p, _ensure_mask_range_128(t, m).v, v.v); }
TL_XMM_DEFINE_STOREU(uint64_t) { _mm_mask_storeu_epi64(p, _ensure_mask_range_128(t, m).v, v.v); }

// store(T* p, Mask<T> m, Vec<T> v)
TL_XMM_DEFINE_STORE(bfloat16_t) { TL_CHECK_ALIGN _mm_mask_storeu_epi16(p, _ensure_mask_range_128(t, m).v, v.v); }
TL_XMM_DEFINE_STORE(float16_t) { TL_CHECK_ALIGN _mm_mask_storeu_epi16(p, _ensure_mask_range_128(t, m).v, v.v); }
TL_XMM_DEFINE_STORE(float32_t) { TL_CHECK_ALIGN _mm_mask_store_ps(p, _ensure_mask_range_128(t, m).v, v.v); }
TL_XMM_DEFINE_STORE(float64_t) { TL_CHECK_ALIGN _mm_mask_store_pd(p, _ensure_mask_range_128(t, m).v, v.v); }
TL_XMM_DEFINE_STORE(int8_t) { TL_CHECK_ALIGN _mm_mask_storeu_epi8(p, _ensure_mask_range_128(t, m).v, v.v); }
TL_XMM_DEFINE_STORE(uint8_t) { TL_CHECK_ALIGN _mm_mask_storeu_epi8(p, _ensure_mask_range_128(t, m).v, v.v); }
TL_XMM_DEFINE_STORE(int16_t) { TL_CHECK_ALIGN _mm_mask_storeu_epi16(p, _ensure_mask_range_128(t, m).v, v.v); }
TL_XMM_DEFINE_STORE(uint16_t) { TL_CHECK_ALIGN _mm_mask_storeu_epi16(p, _ensure_mask_range_128(t, m).v, v.v); }
TL_XMM_DEFINE_STORE(int32_t) { TL_CHECK_ALIGN _mm_mask_store_epi32(p, _ensure_mask_range_128(t, m).v, v.v); }
TL_XMM_DEFINE_STORE(uint32_t) { TL_CHECK_ALIGN _mm_mask_store_epi32(p, _ensure_mask_range_128(t, m).v, v.v); }
TL_XMM_DEFINE_STORE(int64_t) { TL_CHECK_ALIGN _mm_mask_store_epi64(p, _ensure_mask_range_128(t, m).v, v.v); }
TL_XMM_DEFINE_STORE(uint64_t) { TL_CHECK_ALIGN _mm_mask_store_epi64(p, _ensure_mask_range_128(t, m).v, v.v); }
#else // HAS_AVX512DQ
// storeu(T* p, Mask<T> m, Vec<T> v)
TL_XMM_DEFINE_STOREU(bfloat16_t) { auto mask = _ensure_mask_range_128(t, m).v; _mm_maskmoveu_si128(v.v, mask, (char*)p); }
TL_XMM_DEFINE_STOREU(float16_t) { auto mask = _ensure_mask_range_128(t, m).v; _mm_maskmoveu_si128(v.v, mask, (char*)p); }
TL_XMM_DEFINE_STOREU(float32_t) { auto mask = _ensure_mask_range_128(t, m).v; _mm_maskstore_ps(p, mask, v.v); }
TL_XMM_DEFINE_STOREU(float64_t) { auto mask = _ensure_mask_range_128(t, m).v; _mm_maskstore_pd(p, mask, v.v); }
TL_XMM_DEFINE_STOREU(int8_t) { auto mask = _ensure_mask_range_128(t, m).v; _mm_maskmoveu_si128(v.v, mask, (char*)p); }
TL_XMM_DEFINE_STOREU(uint8_t) { auto mask = _ensure_mask_range_128(t, m).v; _mm_maskmoveu_si128(v.v, mask, (char*)p); }
TL_XMM_DEFINE_STOREU(int16_t) { auto mask = _ensure_mask_range_128(t, m).v; _mm_maskmoveu_si128(v.v, mask, (char*)p); }
TL_XMM_DEFINE_STOREU(uint16_t) { auto mask = _ensure_mask_range_128(t, m).v; _mm_maskmoveu_si128(v.v, mask, (char*)p); }
TL_XMM_DEFINE_STOREU(int32_t) {
  auto mask = _ensure_mask_range_128(t, m).v;
  #ifdef HAS_AVX2
  _mm_maskstore_epi32(p, mask, v.v);
  #else
  _mm_maskstore_ps((float32_t*)p, mask, _mm_castsi128_ps(v.v));
  #endif
}
TL_XMM_DEFINE_STOREU(uint32_t) {
  auto mask = _ensure_mask_range_128(t, m).v;
  #ifdef HAS_AVX2
  _mm_maskstore_epi32((int*)p, mask, v.v);
  #else
  _mm_maskstore_ps((float32_t*)p, mask, _mm_castsi128_ps(v.v));
  #endif
}
TL_XMM_DEFINE_STOREU(int64_t) {
  auto mask = _ensure_mask_range_128(t, m).v;
  #ifdef HAS_AVX2
  _mm_maskstore_epi64((long long*)p, mask, v.v);
  #else
  _mm_maskstore_pd((float64_t*)p, mask, _mm_castsi128_pd(v.v));
  #endif
}
TL_XMM_DEFINE_STOREU(uint64_t) {
  auto mask = _ensure_mask_range_128(t, m).v;
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
CT_ALWAYS_FORCEINLINE auto _storeu_fwd_mask_128(Tag<T, N, P> t, T* p, nint_t n, VecOf(t) v) -> void {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = mwhilelt(t, 0, n);
  word::storeu(t, p, m, v);
}

// storeu(T* p, nint_t n, Vec<T> v)
#define TL_XMM_DEFINE_STOREU(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES_CHECK_COUNT(storeu, dtype, void, (dtype* p, nint_t n, VecOf(t) v), (p, n, v))
TL_XMM_DEFINE_STOREU(bfloat16_t) { _storeu_fwd_mask_128(t, p, n, v); }
TL_XMM_DEFINE_STOREU(float16_t) { _storeu_fwd_mask_128(t, p, n, v); }
TL_XMM_DEFINE_STOREU(float32_t) { _storeu_fwd_mask_128(t, p, n, v); }
TL_XMM_DEFINE_STOREU(float64_t) { _storeu_fwd_mask_128(t, p, n, v); }
TL_XMM_DEFINE_STOREU(int8_t) { _storeu_fwd_mask_128(t, p, n, v); }
TL_XMM_DEFINE_STOREU(uint8_t) { _storeu_fwd_mask_128(t, p, n, v); }
TL_XMM_DEFINE_STOREU(int16_t) { _storeu_fwd_mask_128(t, p, n, v); }
TL_XMM_DEFINE_STOREU(uint16_t) { _storeu_fwd_mask_128(t, p, n, v); }
TL_XMM_DEFINE_STOREU(int32_t) { _storeu_fwd_mask_128(t, p, n, v); }
TL_XMM_DEFINE_STOREU(uint32_t) { _storeu_fwd_mask_128(t, p, n, v); }
TL_XMM_DEFINE_STOREU(int64_t) { _storeu_fwd_mask_128(t, p, n, v); }
TL_XMM_DEFINE_STOREU(uint64_t) { _storeu_fwd_mask_128(t, p, n, v); }
#undef TL_XMM_DEFINE_STOREU

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE auto _store_fwd_mask_128(Tag<T, N, P> t, T* p, nint_t n, VecOf(t) v) -> void {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = mwhilelt(t, 0, n);
  word::store(t, p, m, v);
}

// store(T* p, nint_t n, Vec<T> v)
#define TL_XMM_DEFINE_STORE(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES_CHECK_COUNT(store, dtype, void, (dtype* p, nint_t n, VecOf(t) v), (p, n, v))
TL_XMM_DEFINE_STORE(bfloat16_t) { _store_fwd_mask_128(t, p, n, v); }
TL_XMM_DEFINE_STORE(float16_t) { _store_fwd_mask_128(t, p, n, v); }
TL_XMM_DEFINE_STORE(float32_t) { _store_fwd_mask_128(t, p, n, v); }
TL_XMM_DEFINE_STORE(float64_t) { _store_fwd_mask_128(t, p, n, v); }
TL_XMM_DEFINE_STORE(int8_t) { _store_fwd_mask_128(t, p, n, v); }
TL_XMM_DEFINE_STORE(uint8_t) { _store_fwd_mask_128(t, p, n, v); }
TL_XMM_DEFINE_STORE(int16_t) { _store_fwd_mask_128(t, p, n, v); }
TL_XMM_DEFINE_STORE(uint16_t) { _store_fwd_mask_128(t, p, n, v); }
TL_XMM_DEFINE_STORE(int32_t) { _store_fwd_mask_128(t, p, n, v); }
TL_XMM_DEFINE_STORE(uint32_t) { _store_fwd_mask_128(t, p, n, v); }
TL_XMM_DEFINE_STORE(int64_t) { _store_fwd_mask_128(t, p, n, v); }
TL_XMM_DEFINE_STORE(uint64_t) { _store_fwd_mask_128(t, p, n, v); }
#undef TL_XMM_DEFINE_STORE



/* ************************************************************************** */
//                             Get / set element                              //
/* ************************************************************************** */

// Helper: extract element from __m128i
#define TL_DEFINE_EXTRACT_128(dtype, postfix) static CT_ALWAYS_FORCEINLINE dtype _extract128_##postfix(__m128i v, int index)
#ifdef HAS_AVX512F
TL_DEFINE_EXTRACT_128(int8_t, i8) {
return (int8_t)_mm_cvtsi128_si32(_mm_permutex2var_epi8(v, _mm_cvtsi32_si128(int(index)), v)) & 0xff;
}
TL_DEFINE_EXTRACT_128(uint8_t, u8) { return (uint8_t) _extract128_i8(v, index); }
TL_DEFINE_EXTRACT_128(int16_t, i16) {
return (int16_t)_mm_cvtsi128_si32(_mm_permutex2var_epi16(v, _mm_cvtsi32_si128(int(index)), v));
}
TL_DEFINE_EXTRACT_128(uint16_t, u16) { return (uint16_t) _extract128_i16(v, index); }
TL_DEFINE_EXTRACT_128(int32_t, i32) {
return _mm_cvtsi128_si32(_mm_permutex2var_epi32(v, _mm_cvtsi32_si128(int(index)), v));
}
TL_DEFINE_EXTRACT_128(uint32_t, u32) { return (uint32_t) _extract128_i32(v, index); }
TL_DEFINE_EXTRACT_128(int64_t, i64) {
return _mm_cvtsi128_si64(_mm_permutex2var_epi64(v, _mm_cvtsi32_si128(int(index)), v));
}
TL_DEFINE_EXTRACT_128(uint64_t, u64) { return (uint64_t) _extract128_i64(v, index); }
#else // HAS_AVX512F
TL_DEFINE_EXTRACT_128(int8_t, i8) {
  alignas(16) int8_t data[16];
  _mm_store_si128((__m128i*) data, v);
  return data[index];
}
TL_DEFINE_EXTRACT_128(uint8_t, u8) { return (uint8_t) _extract128_i8(v, index); }
TL_DEFINE_EXTRACT_128(int16_t, i16) {
  alignas(16) int16_t data[8];
  _mm_store_si128((__m128i*) data, v);
  return data[index];
}
TL_DEFINE_EXTRACT_128(uint16_t, u16) { return (uint16_t) _extract128_i16(v, index); }
TL_DEFINE_EXTRACT_128(int32_t, i32) {
  alignas(16) int32_t data[4];
  _mm_store_si128((__m128i*) data, v);
  return data[index];
}
TL_DEFINE_EXTRACT_128(uint32_t, u32) { return (uint32_t) _extract128_i32(v, index); }
TL_DEFINE_EXTRACT_128(int64_t, i64) {
  alignas(16) int64_t data[2];
  _mm_store_si128((__m128i*) data, v);
  return data[index];
}
TL_DEFINE_EXTRACT_128(uint64_t, u64) { return (uint64_t) _extract128_i64(v, index); }
#endif // HAS_AVX512F
#undef TL_DEFINE_EXTRACT_128

// Helper: get element from mask
#ifdef HAS_AVX512DQ
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit(__mmask8 m, int index) { return (_cvtmask8_u32(m) >> index) & 1; }
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit(__mmask16 m, int index) { return (_cvtmask16_u32(m) >> index) & 1; }
#else
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit_epi8(__m128i m, int index) { return !!_extract128_i8(m, index); }
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit_epi16(__m128i m, int index) { return !!_extract128_i16(m, index); }
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit_epi32(__m128i m, int index) { return !!_extract128_i32(m, index); }
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit_epi64(__m128i m, int index) { return !!_extract128_i64(m, index); }
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
TL_XMM_DEFINE_GET(bfloat16_t) { TL_CHECK_INDEX(index); union { bfloat16_t b; int16_t i; } u; u.i = _extract128_i16(v.v, (int) index); return u.b; }
TL_XMM_DEFINE_GET(float16_t) { TL_CHECK_INDEX(index); union { float16_t h; int16_t i; } u; u.i = _extract128_i16(v.v, (int) index); return u.h; }
TL_XMM_DEFINE_GET(float32_t) { TL_CHECK_INDEX(index); return _mm_cvtss_f32(_mm_permutevar_ps(v.v, _mm_cvtsi32_si128((int)index))); }
// Note: weird permutevar_pd requires second-to-the-last bit.
TL_XMM_DEFINE_GET(float64_t) { TL_CHECK_INDEX(index); return _mm_cvtsd_f64(_mm_permutevar_pd(v.v, _mm_cvtsi64_si128((int) (index << 1)))); }
TL_XMM_DEFINE_GET(int8_t) { TL_CHECK_INDEX(index); return _extract128_i8(v.v, (int) index); }
TL_XMM_DEFINE_GET(uint8_t) { TL_CHECK_INDEX(index); return _extract128_u8(v.v, (int) index); }
TL_XMM_DEFINE_GET(int16_t) { TL_CHECK_INDEX(index); return _extract128_i16(v.v, (int) index); }
TL_XMM_DEFINE_GET(uint16_t) { TL_CHECK_INDEX(index); return _extract128_u16(v.v, (int) index); }
TL_XMM_DEFINE_GET(int32_t) { TL_CHECK_INDEX(index); return _extract128_i32(v.v, (int) index); }
TL_XMM_DEFINE_GET(uint32_t) { TL_CHECK_INDEX(index); return _extract128_u32(v.v, (int) index); }
TL_XMM_DEFINE_GET(int64_t) { TL_CHECK_INDEX(index); return _extract128_i64(v.v, (int) index); }
TL_XMM_DEFINE_GET(uint64_t) { TL_CHECK_INDEX(index); return _extract128_u64(v.v, (int) index); }
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



/* ************************************************************************** */
//                       Basic arithmetic operations                          //
/* ************************************************************************** */
// TODO arithmetics for float16_t and bfloat16_t


// add(Vec<T> a, Vec<T> b)
#define TL_XMM_DEFINE_ADD(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(add, dtype, VecOf(t), (VecOf(t) a, VecOf(t) b), (a, b))
TL_XMM_DEFINE_ADD(float32_t) { return _mm_add_ps(a.v, b.v); }
TL_XMM_DEFINE_ADD(float64_t) { return _mm_add_pd(a.v, b.v); }
TL_XMM_DEFINE_ADD(int8_t) { return _mm_add_epi8(a.v, b.v); }
TL_XMM_DEFINE_ADD(uint8_t) { return _mm_add_epi8(a.v, b.v); }
TL_XMM_DEFINE_ADD(int16_t) { return _mm_add_epi16(a.v, b.v); }
TL_XMM_DEFINE_ADD(uint16_t) { return _mm_add_epi16(a.v, b.v); }
TL_XMM_DEFINE_ADD(int32_t) { return _mm_add_epi32(a.v, b.v); }
TL_XMM_DEFINE_ADD(uint32_t) { return _mm_add_epi32(a.v, b.v); }
TL_XMM_DEFINE_ADD(int64_t) { return _mm_add_epi64(a.v, b.v); }
TL_XMM_DEFINE_ADD(uint64_t) { return _mm_add_epi64(a.v, b.v); }
#undef TL_XMM_DEFINE_ADD

// add(Vec<T> a, Vec<T> b, Mask<T> m)
#define TL_XMM_DEFINE_ADD(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(add, dtype, VecOf(t), (VecOf(t) a, VecOf(t) b, MaskOf(t) m), (a, b, m))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_ADD(float32_t) { return _mm_mask_add_ps(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_ADD(float64_t) { return _mm_mask_add_pd(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_ADD(int8_t) { return _mm_mask_add_epi8(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_ADD(uint8_t) { return _mm_mask_add_epi8(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_ADD(int16_t) { return _mm_mask_add_epi16(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_ADD(uint16_t) { return _mm_mask_add_epi16(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_ADD(int32_t) { return _mm_mask_add_epi32(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_ADD(uint32_t) { return _mm_mask_add_epi32(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_ADD(int64_t) { return _mm_mask_add_epi64(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_ADD(uint64_t) { return _mm_mask_add_epi64(a.v, m.v, a.v, b.v); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_ADD(float32_t) { return _mm_blendv_ps(a.v, add(t, a.v, b.v).v, _mm_castsi128_ps(m.v)); }
TL_XMM_DEFINE_ADD(float64_t) { return _mm_blendv_pd(a.v, add(t, a.v, b.v).v, _mm_castsi128_pd(m.v)); }
TL_XMM_DEFINE_ADD(int8_t) { return _mm_blendv_epi8(a.v, add(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_ADD(uint8_t) { return _mm_blendv_epi8(a.v, add(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_ADD(int16_t) { return _mm_blendv_epi8(a.v, add(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_ADD(uint16_t) { return _mm_blendv_epi8(a.v, add(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_ADD(int32_t) { return _mm_blendv_epi8(a.v, add(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_ADD(uint32_t) { return _mm_blendv_epi8(a.v, add(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_ADD(int64_t) { return _mm_blendv_epi8(a.v, add(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_ADD(uint64_t) { return _mm_blendv_epi8(a.v, add(t, a.v, b.v).v, m.v); }
#endif //HAS_AVX512DQ
#undef TL_XMM_DEFINE_ADD



// sub(Vec<T> a, Vec<T> b)
#define TL_XMM_DEFINE_SUB(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(sub, dtype, VecOf(t), (VecOf(t) a, VecOf(t) b), (a, b))
TL_XMM_DEFINE_SUB(float32_t) { return _mm_sub_ps(a.v, b.v); }
TL_XMM_DEFINE_SUB(float64_t) { return _mm_sub_pd(a.v, b.v); }
TL_XMM_DEFINE_SUB(int8_t) { return _mm_sub_epi8(a.v, b.v); }
TL_XMM_DEFINE_SUB(uint8_t) { return _mm_sub_epi8(a.v, b.v); }
TL_XMM_DEFINE_SUB(int16_t) { return _mm_sub_epi16(a.v, b.v); }
TL_XMM_DEFINE_SUB(uint16_t) { return _mm_sub_epi16(a.v, b.v); }
TL_XMM_DEFINE_SUB(int32_t) { return _mm_sub_epi32(a.v, b.v); }
TL_XMM_DEFINE_SUB(uint32_t) { return _mm_sub_epi32(a.v, b.v); }
TL_XMM_DEFINE_SUB(int64_t) { return _mm_sub_epi64(a.v, b.v); }
TL_XMM_DEFINE_SUB(uint64_t) { return _mm_sub_epi64(a.v, b.v); }
#undef TL_XMM_DEFINE_SUB

// sub(Vec<T> a, Vec<T> b, Mask<T> m)
#define TL_XMM_DEFINE_SUB(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(sub, dtype, VecOf(t), (VecOf(t) a, VecOf(t) b, MaskOf(t) m), (a, b, m))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_SUB(float32_t) { return _mm_mask_sub_ps(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_SUB(float64_t) { return _mm_mask_sub_pd(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_SUB(int8_t) { return _mm_mask_sub_epi8(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_SUB(uint8_t) { return _mm_mask_sub_epi8(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_SUB(int16_t) { return _mm_mask_sub_epi16(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_SUB(uint16_t) { return _mm_mask_sub_epi16(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_SUB(int32_t) { return _mm_mask_sub_epi32(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_SUB(uint32_t) { return _mm_mask_sub_epi32(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_SUB(int64_t) { return _mm_mask_sub_epi64(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_SUB(uint64_t) { return _mm_mask_sub_epi64(a.v, m.v, a.v, b.v); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_SUB(float32_t) { return _mm_blendv_ps(a.v, sub(t, a.v, b.v).v, _mm_castsi128_ps(m.v)); }
TL_XMM_DEFINE_SUB(float64_t) { return _mm_blendv_pd(a.v, sub(t, a.v, b.v).v, _mm_castsi128_pd(m.v)); }
TL_XMM_DEFINE_SUB(int8_t) { return _mm_blendv_epi8(a.v, sub(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_SUB(uint8_t) { return _mm_blendv_epi8(a.v, sub(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_SUB(int16_t) { return _mm_blendv_epi8(a.v, sub(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_SUB(uint16_t) { return _mm_blendv_epi8(a.v, sub(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_SUB(int32_t) { return _mm_blendv_epi8(a.v, sub(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_SUB(uint32_t) { return _mm_blendv_epi8(a.v, sub(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_SUB(int64_t) { return _mm_blendv_epi8(a.v, sub(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_SUB(uint64_t) { return _mm_blendv_epi8(a.v, sub(t, a.v, b.v).v, m.v); }
#endif //HAS_AVX512DQ
#undef TL_XMM_DEFINE_SUB



// mul(Vec<T> a, Vec<T> b)
#define TL_XMM_DEFINE_MUL(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(mul, dtype, VecOf(t), (VecOf(t) a, VecOf(t) b), (a, b))
TL_XMM_DEFINE_MUL(float32_t) { return _mm_mul_ps(a.v, b.v); }
TL_XMM_DEFINE_MUL(float64_t) { return _mm_mul_pd(a.v, b.v); }
TL_XMM_DEFINE_MUL(int8_t) {
  auto even = _mm_mullo_epi16(a.v, b.v);
  auto odd = _mm_mullo_epi16(_mm_srli_epi16(a.v, 8), _mm_srli_epi16(b.v, 8));
  return _mm_or_si128(_mm_slli_epi16(odd, 8), _mm_and_si128(even, _mm_set1_epi16(0xFF)));
}
TL_XMM_DEFINE_MUL(uint8_t) { return mul(Tag<int8_t, 16>(), a.v, b.v).v; }
TL_XMM_DEFINE_MUL(int16_t) { return _mm_mullo_epi16(a.v, b.v); }
TL_XMM_DEFINE_MUL(uint16_t) { return _mm_mullo_epi16(a.v, b.v); } // signed mul & unsigned mul behaves same in low 16-bit result, same to 8, 32 and 64.
TL_XMM_DEFINE_MUL(int32_t) { return _mm_mullo_epi32(a.v, b.v); }
TL_XMM_DEFINE_MUL(uint32_t) { return _mm_mullo_epi32(a.v, b.v); }
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_MUL(int64_t) { return _mm_mullo_epi64(a.v, b.v); }
TL_XMM_DEFINE_MUL(uint64_t) { return _mm_mullo_epi64(a.v, b.v); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_MUL(int64_t) {
  auto lo_lo = _mm_mul_epu32(a.v, b.v);
  auto a_hi = _mm_shuffle_epi32(a.v, _MM_SHUFFLE(3, 3, 1, 1));
  auto b_hi = _mm_shuffle_epi32(b.v, _MM_SHUFFLE(3, 3, 1, 1));
  auto hi_lo = _mm_mul_epu32(a_hi, b.v); // a_hi × b_lo
  auto lo_hi = _mm_mul_epu32(a.v, b_hi); // a_lo × b_hi
  __m128i cross = _mm_add_epi64(hi_lo, lo_hi);
  cross = _mm_slli_epi64(cross, 32);
  return _mm_add_epi64(lo_lo, cross);
}
TL_XMM_DEFINE_MUL(uint64_t) {
  return mul(Tag<int64_t, 2>(), a.v, b.v).v;
}
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_MUL

// mul(Vec<T> a, Vec<T> b, Mask<T> m)
#define TL_XMM_DEFINE_MUL(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(mul, dtype, VecOf(t), (VecOf(t) a, VecOf(t) b, MaskOf(t) m), (a, b, m))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_MUL(float32_t) { return _mm_mask_mul_ps(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MUL(float64_t) { return _mm_mask_mul_pd(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MUL(int8_t) { return _mm_mask_blend_epi8(m.v, a.v, mul(t, a.v, b.v).v); }
TL_XMM_DEFINE_MUL(uint8_t) { return mul(Tag<int8_t, 16>(), a.v, b.v, m.v).v; }
TL_XMM_DEFINE_MUL(int16_t) { return _mm_mask_mullo_epi16(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MUL(uint16_t) { return _mm_mask_mullo_epi16(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MUL(int32_t) { return _mm_mask_mullo_epi32(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MUL(uint32_t) { return _mm_mask_mullo_epi32(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MUL(int64_t) { return _mm_mask_mullo_epi64(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MUL(uint64_t) { return _mm_mask_mullo_epi64(a.v, m.v, a.v, b.v); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_MUL(float32_t) { return _mm_blendv_ps(a.v, mul(t, a.v, b.v).v, _mm_castsi128_ps(m.v)); }
TL_XMM_DEFINE_MUL(float64_t) { return _mm_blendv_pd(a.v, mul(t, a.v, b.v).v, _mm_castsi128_pd(m.v)); }
TL_XMM_DEFINE_MUL(int8_t) { return _mm_blendv_epi8(a.v, mul(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_MUL(uint8_t) { return _mm_blendv_epi8(a.v, mul(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_MUL(int16_t) { return _mm_blendv_epi8(a.v, mul(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_MUL(uint16_t) { return _mm_blendv_epi8(a.v, mul(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_MUL(int32_t) { return _mm_blendv_epi8(a.v, mul(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_MUL(uint32_t) { return _mm_blendv_epi8(a.v, mul(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_MUL(int64_t) { return _mm_blendv_epi8(a.v, mul(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_MUL(uint64_t) { return _mm_blendv_epi8(a.v, mul(t, a.v, b.v).v, m.v); }
#endif //HAS_AVX512DQ
#undef TL_XMM_DEFINE_MUL



// div(Vec<T> a, Vec<T> b)
#define TL_XMM_DEFINE_DIV(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(div, dtype, VecOf(t), (VecOf(t) a, VecOf(t) b), (a, b))
TL_XMM_DEFINE_DIV(float32_t) { return _mm_div_ps(a.v, b.v); }
TL_XMM_DEFINE_DIV(float64_t) { return _mm_div_pd(a.v, b.v); }
#undef TL_XMM_DEFINE_DIV

// div(Vec<T> a, Vec<T> b, Mask<T> m)
#define TL_XMM_DEFINE_DIV(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(div, dtype, VecOf(t), (VecOf(t) a, VecOf(t) b, MaskOf(t) m), (a, b, m))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_DIV(float32_t) { return _mm_mask_div_ps(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_DIV(float64_t) { return _mm_mask_div_pd(a.v, m.v, a.v, b.v); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_DIV(float32_t) { return _mm_blendv_ps(a.v, div(t, a, b).v, _mm_castsi128_ps(m.v)); }
TL_XMM_DEFINE_DIV(float64_t) { return _mm_blendv_pd(a.v, div(t, a, b).v, _mm_castsi128_pd(m.v)); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_DIV



// rcp(Vec<T> v)
// Note: AVX512 uses rcp14 which gives higher accuracy than rcp used not in AVX512
#define TL_XMM_DEFINE_RCP(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(rcp, dtype, VecOf(t), (VecOf(t) v), (v))
#ifdef HAS_AVX512F
TL_XMM_DEFINE_RCP(float32_t) { return _mm_rcp14_ps(v.v); }
TL_XMM_DEFINE_RCP(float64_t) { return _mm_rcp14_pd(v.v); }
#else // HAS_AVX512F
TL_XMM_DEFINE_RCP(float32_t) { return _mm_rcp_ps(v.v); }
TL_XMM_DEFINE_RCP(float64_t) { return div(t, fill(t, 1), v.v); }
#endif // HAS_AVX512F
#undef TL_XMM_DEFINE_RCP

// rcp(Vec<T> v, Mask<T> m, Vec<T> default_v)
#define TL_XMM_DEFINE_RCP(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(rcp, dtype, VecOf(t), (VecOf(t) v, MaskOf(t) m, VecOf(t) default_v), (v, m, default_v))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_RCP(float32_t) { return _mm_mask_rcp14_ps(default_v.v, m.v, v.v); }
TL_XMM_DEFINE_RCP(float64_t) { return _mm_mask_rcp14_pd(default_v.v, m.v, v.v); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_RCP(float32_t) { return _mm_blendv_ps(default_v.v, rcp(t, v).v, _mm_castsi128_ps(m.v)); }
TL_XMM_DEFINE_RCP(float64_t) { return _mm_blendv_pd(default_v.v, rcp(t, v).v, _mm_castsi128_pd(m.v)); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_RCP



// max(Vec<T> a, Vec<T> b)
#define TL_XMM_DEFINE_MAX(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(max, dtype, VecOf(t), (VecOf(t) a, VecOf(t) b), (a, b))
TL_XMM_DEFINE_MAX(float32_t) { return _mm_max_ps(a.v, b.v); }
TL_XMM_DEFINE_MAX(float64_t) { return _mm_max_pd(a.v, b.v); }
TL_XMM_DEFINE_MAX(int8_t) { return _mm_max_epi8(a.v, b.v); }
TL_XMM_DEFINE_MAX(uint8_t) { return _mm_max_epu8(a.v, b.v); }
TL_XMM_DEFINE_MAX(int16_t) { return _mm_max_epi16(a.v, b.v); }
TL_XMM_DEFINE_MAX(uint16_t) { return _mm_max_epu16(a.v, b.v); }
TL_XMM_DEFINE_MAX(int32_t) { return _mm_max_epi32(a.v, b.v); }
TL_XMM_DEFINE_MAX(uint32_t) { return _mm_max_epu32(a.v, b.v); }
#ifdef HAS_AVX512F
TL_XMM_DEFINE_MAX(int64_t) { return _mm_max_epi64(a.v, b.v); }
TL_XMM_DEFINE_MAX(uint64_t) { return _mm_max_epu64(a.v, b.v); }
#else
TL_XMM_DEFINE_MAX(int64_t) { return _mm_blendv_epi8(b.v, a.v, _mm_cmpgt_epi64(a.v, b.v)); }
TL_XMM_DEFINE_MAX(uint64_t) {
  static const __m128i sign_bit = _mm_set1_epi64x((int64_t)0x8000000000000000LL);
  auto a_flip = _mm_xor_si128(a.v, sign_bit);
  auto b_flip = _mm_xor_si128(b.v, sign_bit);
  return _mm_blendv_epi8(b.v, a.v, _mm_cmpgt_epi64(a_flip, b_flip));
}
#endif
#undef TL_XMM_DEFINE_MAX

// max(Vec<T> a, Vec<T> b, Mask<T> m)
#define TL_XMM_DEFINE_MAX(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(max, dtype, VecOf(t), (VecOf(t) a, VecOf(t) b, MaskOf(t) m), (a, b, m))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_MAX(float32_t) { return _mm_mask_max_ps(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MAX(float64_t) { return _mm_mask_max_pd(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MAX(int8_t) { return _mm_mask_max_epi8(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MAX(uint8_t) { return _mm_mask_max_epu8(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MAX(int16_t) { return _mm_mask_max_epi16(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MAX(uint16_t) { return _mm_mask_max_epu16(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MAX(int32_t) { return _mm_mask_max_epi32(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MAX(uint32_t) { return _mm_mask_max_epu32(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MAX(int64_t) { return _mm_mask_max_epi64(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MAX(uint64_t) { return _mm_mask_max_epu64(a.v, m.v, a.v, b.v); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_MAX(float32_t) { return _mm_blendv_ps(a.v, max(t, a.v, b.v).v, _mm_castsi128_ps(m.v)); }
TL_XMM_DEFINE_MAX(float64_t) { return _mm_blendv_pd(a.v, max(t, a.v, b.v).v, _mm_castsi128_pd(m.v)); }
TL_XMM_DEFINE_MAX(int8_t) { return _mm_blendv_epi8(a.v, max(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_MAX(uint8_t) { return _mm_blendv_epi8(a.v, max(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_MAX(int16_t) { return _mm_blendv_epi8(a.v, max(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_MAX(uint16_t) { return _mm_blendv_epi8(a.v, max(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_MAX(int32_t) { return _mm_blendv_epi8(a.v, max(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_MAX(uint32_t) { return _mm_blendv_epi8(a.v, max(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_MAX(int64_t) { return _mm_blendv_epi8(a.v, max(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_MAX(uint64_t) { return _mm_blendv_epi8(a.v, max(t, a.v, b.v).v, m.v); }
#endif //HAS_AVX512DQ
#undef TL_XMM_DEFINE_MAX



// min(Vec<T> a, Vec<T> b)
#define TL_XMM_DEFINE_MIN(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(min, dtype, VecOf(t), (VecOf(t) a, VecOf(t) b), (a, b))
TL_XMM_DEFINE_MIN(float32_t) { return _mm_min_ps(a.v, b.v); }
TL_XMM_DEFINE_MIN(float64_t) { return _mm_min_pd(a.v, b.v); }
TL_XMM_DEFINE_MIN(int8_t) { return _mm_min_epi8(a.v, b.v); }
TL_XMM_DEFINE_MIN(uint8_t) { return _mm_min_epu8(a.v, b.v); }
TL_XMM_DEFINE_MIN(int16_t) { return _mm_min_epi16(a.v, b.v); }
TL_XMM_DEFINE_MIN(uint16_t) { return _mm_min_epu16(a.v, b.v); }
TL_XMM_DEFINE_MIN(int32_t) { return _mm_min_epi32(a.v, b.v); }
TL_XMM_DEFINE_MIN(uint32_t) { return _mm_min_epu32(a.v, b.v); }
#ifdef HAS_AVX512F
TL_XMM_DEFINE_MIN(int64_t) { return _mm_min_epi64(a.v, b.v); }
TL_XMM_DEFINE_MIN(uint64_t) { return _mm_min_epu64(a.v, b.v); }
#else
TL_XMM_DEFINE_MIN(int64_t) { return _mm_blendv_epi8(b.v, a.v, _mm_cmpgt_epi64(b.v, a.v)); }
TL_XMM_DEFINE_MIN(uint64_t) {
  static const __m128i flip = _mm_set1_epi64x((int64_t)0x8000000000000000LL);
  __m128i a_flip = _mm_xor_si128(a.v, flip);
  __m128i b_flip = _mm_xor_si128(b.v, flip);
  return _mm_blendv_epi8(b.v, a.v, _mm_cmpgt_epi64(b_flip, a_flip));
}
#endif
#undef TL_XMM_DEFINE_MIN

// min(Vec<T> a, Vec<T> b, Mask<T> m)
#define TL_XMM_DEFINE_MIN(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(min, dtype, VecOf(t), (VecOf(t) a, VecOf(t) b, MaskOf(t) m), (a, b, m))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_MIN(float32_t) { return _mm_mask_min_ps(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MIN(float64_t) { return _mm_mask_min_pd(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MIN(int8_t) { return _mm_mask_min_epi8(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MIN(uint8_t) { return _mm_mask_min_epu8(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MIN(int16_t) { return _mm_mask_min_epi16(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MIN(uint16_t) { return _mm_mask_min_epu16(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MIN(int32_t) { return _mm_mask_min_epi32(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MIN(uint32_t) { return _mm_mask_min_epu32(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MIN(int64_t) { return _mm_mask_min_epi64(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_MIN(uint64_t) { return _mm_mask_min_epu64(a.v, m.v, a.v, b.v); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_MIN(float32_t) { return _mm_blendv_ps(a.v, min(t, a.v, b.v).v, _mm_castsi128_ps(m.v)); }
TL_XMM_DEFINE_MIN(float64_t) { return _mm_blendv_pd(a.v, min(t, a.v, b.v).v, _mm_castsi128_pd(m.v)); }
TL_XMM_DEFINE_MIN(int8_t) { return _mm_blendv_epi8(a.v, min(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_MIN(uint8_t) { return _mm_blendv_epi8(a.v, min(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_MIN(int16_t) { return _mm_blendv_epi8(a.v, min(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_MIN(uint16_t) { return _mm_blendv_epi8(a.v, min(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_MIN(int32_t) { return _mm_blendv_epi8(a.v, min(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_MIN(uint32_t) { return _mm_blendv_epi8(a.v, min(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_MIN(int64_t) { return _mm_blendv_epi8(a.v, min(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_MIN(uint64_t) { return _mm_blendv_epi8(a.v, min(t, a.v, b.v).v, m.v); }
#endif //HAS_AVX512DQ
#undef TL_XMM_DEFINE_MIN



// bit_and(Vec<T> a, Vec<T> b)
#define TL_XMM_DEFINE_BIT_AND(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(bit_and, dtype, VecOf(t), (VecOf(t) a, VecOf(t) b), (a, b))
TL_XMM_DEFINE_BIT_AND(int8_t) { return _mm_and_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_AND(uint8_t) { return _mm_and_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_AND(int16_t) { return _mm_and_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_AND(uint16_t) { return _mm_and_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_AND(int32_t) { return _mm_and_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_AND(uint32_t) { return _mm_and_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_AND(int64_t) { return _mm_and_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_AND(uint64_t) { return _mm_and_si128(a.v, b.v); }
#undef TL_XMM_DEFINE_BIT_AND

// bit_and(Vec<T> a, Vec<T> b, Mask<T> m)
#define TL_XMM_DEFINE_BIT_AND(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(bit_and, dtype, VecOf(t), (VecOf(t) a, VecOf(t) b, MaskOf(t) m), (a, b, m))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_BIT_AND(int8_t) { return _mm_mask_blend_epi8(m.v, a.v, _mm_and_si128(a.v, b.v)); }
TL_XMM_DEFINE_BIT_AND(uint8_t) { return _mm_mask_blend_epi8(m.v, a.v, _mm_and_si128(a.v, b.v)); }
TL_XMM_DEFINE_BIT_AND(int16_t) { return _mm_mask_blend_epi16(m.v, a.v, _mm_and_si128(a.v, b.v)); }
TL_XMM_DEFINE_BIT_AND(uint16_t) { return _mm_mask_blend_epi16(m.v, a.v, _mm_and_si128(a.v, b.v)); }
TL_XMM_DEFINE_BIT_AND(int32_t) { return _mm_mask_and_epi32(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_BIT_AND(uint32_t) { return _mm_mask_and_epi32(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_BIT_AND(int64_t) { return _mm_mask_and_epi64(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_BIT_AND(uint64_t) { return _mm_mask_and_epi64(a.v, m.v, a.v, b.v); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_BIT_AND(int8_t) { return _mm_blendv_epi8(a.v, bit_and(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_AND(uint8_t) { return _mm_blendv_epi8(a.v, bit_and(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_AND(int16_t) { return _mm_blendv_epi8(a.v, bit_and(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_AND(uint16_t) { return _mm_blendv_epi8(a.v, bit_and(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_AND(int32_t) { return _mm_blendv_epi8(a.v, bit_and(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_AND(uint32_t) { return _mm_blendv_epi8(a.v, bit_and(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_AND(int64_t) { return _mm_blendv_epi8(a.v, bit_and(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_AND(uint64_t) { return _mm_blendv_epi8(a.v, bit_and(t, a.v, b.v).v, m.v); }
#endif //HAS_AVX512DQ
#undef TL_XMM_DEFINE_BIT_AND



// bit_or(Vec<T> a, Vec<T> b)
#define TL_XMM_DEFINE_BIT_OR(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(bit_or, dtype, VecOf(t), (VecOf(t) a, VecOf(t) b), (a, b))
TL_XMM_DEFINE_BIT_OR(int8_t) { return _mm_or_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_OR(uint8_t) { return _mm_or_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_OR(int16_t) { return _mm_or_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_OR(uint16_t) { return _mm_or_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_OR(int32_t) { return _mm_or_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_OR(uint32_t) { return _mm_or_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_OR(int64_t) { return _mm_or_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_OR(uint64_t) { return _mm_or_si128(a.v, b.v); }
#undef TL_XMM_DEFINE_BIT_OR

// bit_or(Vec<T> a, Vec<T> b, Mask<T> m)
#define TL_XMM_DEFINE_BIT_OR(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(bit_or, dtype, VecOf(t), (VecOf(t) a, VecOf(t) b, MaskOf(t) m), (a, b, m))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_BIT_OR(int8_t) { return _mm_mask_blend_epi8(m.v, a.v, _mm_or_si128(a.v, b.v)); }
TL_XMM_DEFINE_BIT_OR(uint8_t) { return _mm_mask_blend_epi8(m.v, a.v, _mm_or_si128(a.v, b.v)); }
TL_XMM_DEFINE_BIT_OR(int16_t) { return _mm_mask_blend_epi16(m.v, a.v, _mm_or_si128(a.v, b.v)); }
TL_XMM_DEFINE_BIT_OR(uint16_t) { return _mm_mask_blend_epi16(m.v, a.v, _mm_or_si128(a.v, b.v)); }
TL_XMM_DEFINE_BIT_OR(int32_t) { return _mm_mask_or_epi32(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_BIT_OR(uint32_t) { return _mm_mask_or_epi32(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_BIT_OR(int64_t) { return _mm_mask_or_epi64(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_BIT_OR(uint64_t) { return _mm_mask_or_epi64(a.v, m.v, a.v, b.v); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_BIT_OR(int8_t) { return _mm_blendv_epi8(a.v, bit_or(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_OR(uint8_t) { return _mm_blendv_epi8(a.v, bit_or(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_OR(int16_t) { return _mm_blendv_epi8(a.v, bit_or(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_OR(uint16_t) { return _mm_blendv_epi8(a.v, bit_or(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_OR(int32_t) { return _mm_blendv_epi8(a.v, bit_or(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_OR(uint32_t) { return _mm_blendv_epi8(a.v, bit_or(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_OR(int64_t) { return _mm_blendv_epi8(a.v, bit_or(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_OR(uint64_t) { return _mm_blendv_epi8(a.v, bit_or(t, a.v, b.v).v, m.v); }
#endif //HAS_AVX512DQ
#undef TL_XMM_DEFINE_BIT_OR



// bit_xor(Vec<T> a, Vec<T> b)
#define TL_XMM_DEFINE_BIT_XOR(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(bit_xor, dtype, VecOf(t), (VecOf(t) a, VecOf(t) b), (a, b))
TL_XMM_DEFINE_BIT_XOR(int8_t) { return _mm_xor_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_XOR(uint8_t) { return _mm_xor_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_XOR(int16_t) { return _mm_xor_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_XOR(uint16_t) { return _mm_xor_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_XOR(int32_t) { return _mm_xor_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_XOR(uint32_t) { return _mm_xor_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_XOR(int64_t) { return _mm_xor_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_XOR(uint64_t) { return _mm_xor_si128(a.v, b.v); }
#undef TL_XMM_DEFINE_BIT_XOR

// bit_xor(Vec<T> a, Vec<T> b, Mask<T> m)
#define TL_XMM_DEFINE_BIT_XOR(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(bit_xor, dtype, VecOf(t), (VecOf(t) a, VecOf(t) b, MaskOf(t) m), (a, b, m))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_BIT_XOR(int8_t) { return _mm_mask_blend_epi8(m.v, a.v, _mm_xor_si128(a.v, b.v)); }
TL_XMM_DEFINE_BIT_XOR(uint8_t) { return _mm_mask_blend_epi8(m.v, a.v, _mm_xor_si128(a.v, b.v)); }
TL_XMM_DEFINE_BIT_XOR(int16_t) { return _mm_mask_blend_epi16(m.v, a.v, _mm_xor_si128(a.v, b.v)); }
TL_XMM_DEFINE_BIT_XOR(uint16_t) { return _mm_mask_blend_epi16(m.v, a.v, _mm_xor_si128(a.v, b.v)); }
TL_XMM_DEFINE_BIT_XOR(int32_t) { return _mm_mask_xor_epi32(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_BIT_XOR(uint32_t) { return _mm_mask_xor_epi32(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_BIT_XOR(int64_t) { return _mm_mask_xor_epi64(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_BIT_XOR(uint64_t) { return _mm_mask_xor_epi64(a.v, m.v, a.v, b.v); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_BIT_XOR(int8_t) { return _mm_blendv_epi8(a.v, bit_xor(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_XOR(uint8_t) { return _mm_blendv_epi8(a.v, bit_xor(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_XOR(int16_t) { return _mm_blendv_epi8(a.v, bit_xor(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_XOR(uint16_t) { return _mm_blendv_epi8(a.v, bit_xor(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_XOR(int32_t) { return _mm_blendv_epi8(a.v, bit_xor(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_XOR(uint32_t) { return _mm_blendv_epi8(a.v, bit_xor(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_XOR(int64_t) { return _mm_blendv_epi8(a.v, bit_xor(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_XOR(uint64_t) { return _mm_blendv_epi8(a.v, bit_xor(t, a.v, b.v).v, m.v); }
#endif //HAS_AVX512DQ
#undef TL_XMM_DEFINE_BIT_XOR



// bit_andnot(Vec<T> a, Vec<T> b)
#define TL_XMM_DEFINE_BIT_ANDNOT(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(bit_andnot, dtype, VecOf(t), (VecOf(t) a, VecOf(t) b), (a, b))
TL_XMM_DEFINE_BIT_ANDNOT(int8_t) { return _mm_andnot_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_ANDNOT(uint8_t) { return _mm_andnot_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_ANDNOT(int16_t) { return _mm_andnot_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_ANDNOT(uint16_t) { return _mm_andnot_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_ANDNOT(int32_t) { return _mm_andnot_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_ANDNOT(uint32_t) { return _mm_andnot_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_ANDNOT(int64_t) { return _mm_andnot_si128(a.v, b.v); }
TL_XMM_DEFINE_BIT_ANDNOT(uint64_t) { return _mm_andnot_si128(a.v, b.v); }
#undef TL_XMM_DEFINE_BIT_ANDNOT

// bit_andnot(Vec<T> a, Vec<T> b, Mask<T> m)
#define TL_XMM_DEFINE_BIT_ANDNOT(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(bit_andnot, dtype, VecOf(t), (VecOf(t) a, VecOf(t) b, MaskOf(t) m), (a, b, m))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_BIT_ANDNOT(int8_t) { return _mm_mask_blend_epi8(m.v, a.v, _mm_andnot_si128(a.v, b.v)); }
TL_XMM_DEFINE_BIT_ANDNOT(uint8_t) { return _mm_mask_blend_epi8(m.v, a.v, _mm_andnot_si128(a.v, b.v)); }
TL_XMM_DEFINE_BIT_ANDNOT(int16_t) { return _mm_mask_blend_epi16(m.v, a.v, _mm_andnot_si128(a.v, b.v)); }
TL_XMM_DEFINE_BIT_ANDNOT(uint16_t) { return _mm_mask_blend_epi16(m.v, a.v, _mm_andnot_si128(a.v, b.v)); }
TL_XMM_DEFINE_BIT_ANDNOT(int32_t) { return _mm_mask_andnot_epi32(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_BIT_ANDNOT(uint32_t) { return _mm_mask_andnot_epi32(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_BIT_ANDNOT(int64_t) { return _mm_mask_andnot_epi64(a.v, m.v, a.v, b.v); }
TL_XMM_DEFINE_BIT_ANDNOT(uint64_t) { return _mm_mask_andnot_epi64(a.v, m.v, a.v, b.v); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_BIT_ANDNOT(int8_t) { return _mm_blendv_epi8(a.v, bit_andnot(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_ANDNOT(uint8_t) { return _mm_blendv_epi8(a.v, bit_andnot(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_ANDNOT(int16_t) { return _mm_blendv_epi8(a.v, bit_andnot(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_ANDNOT(uint16_t) { return _mm_blendv_epi8(a.v, bit_andnot(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_ANDNOT(int32_t) { return _mm_blendv_epi8(a.v, bit_andnot(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_ANDNOT(uint32_t) { return _mm_blendv_epi8(a.v, bit_andnot(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_ANDNOT(int64_t) { return _mm_blendv_epi8(a.v, bit_andnot(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_BIT_ANDNOT(uint64_t) { return _mm_blendv_epi8(a.v, bit_andnot(t, a.v, b.v).v, m.v); }
#endif //HAS_AVX512DQ
#undef TL_XMM_DEFINE_BIT_ANDNOT



// bit_not(Vec<T> v)
#define TL_XMM_DEFINE_BIT_NOT(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(bit_not, dtype, VecOf(t), (VecOf(t) v), (v))
  #if defined(HAS_AVX512VL)
    #define _tlmm_vec_not128(a)  _mm_ternarylogic_epi32((a), (a), (a), 0x33)
  #else
    #define _tlmm_vec_not128(a)  _mm_xor_si128((a), _mm_set1_epi32(-1))
  #endif
TL_XMM_DEFINE_BIT_NOT(int8_t) { return _tlmm_vec_not128(v.v); }
TL_XMM_DEFINE_BIT_NOT(uint8_t) { return _tlmm_vec_not128(v.v); }
TL_XMM_DEFINE_BIT_NOT(int16_t) { return _tlmm_vec_not128(v.v); }
TL_XMM_DEFINE_BIT_NOT(uint16_t) { return _tlmm_vec_not128(v.v); }
TL_XMM_DEFINE_BIT_NOT(int32_t) { return _tlmm_vec_not128(v.v); }
TL_XMM_DEFINE_BIT_NOT(uint32_t) { return _tlmm_vec_not128(v.v); }
TL_XMM_DEFINE_BIT_NOT(int64_t) { return _tlmm_vec_not128(v.v); }
TL_XMM_DEFINE_BIT_NOT(uint64_t) { return _tlmm_vec_not128(v.v); }
#undef TL_XMM_DEFINE_BIT_NOT

// bit_not(Vec<T> v, Mask<T> m, Vec<T> default_v)
#define TL_XMM_DEFINE_BIT_NOT(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(bit_not, dtype, VecOf(t), (VecOf(t) v, MaskOf(t) m, VecOf(t) default_v), (v, m, default_v))
#ifdef HAS_AVX512DQ
  #define _tlmm_mask_vec_not_epi32(src, mask, a)  _mm_mask_ternarylogic_epi32((src), (mask), (a), (a), 0x33)
  #define _tlmm_mask_vec_not_epi64(src, mask, a)  _mm_mask_ternarylogic_epi64((src), (mask), (a), (a), 0x33)
TL_XMM_DEFINE_BIT_NOT(int8_t) { return _mm_mask_blend_epi8(m.v, default_v.v, _tlmm_vec_not128(v.v)); }
TL_XMM_DEFINE_BIT_NOT(uint8_t) { return _mm_mask_blend_epi8(m.v, default_v.v, _tlmm_vec_not128(v.v)); }
TL_XMM_DEFINE_BIT_NOT(int16_t) { return _mm_mask_blend_epi16(m.v, default_v.v, _tlmm_vec_not128(v.v)); }
TL_XMM_DEFINE_BIT_NOT(uint16_t) { return _mm_mask_blend_epi16(m.v, default_v.v, _tlmm_vec_not128(v.v)); }
TL_XMM_DEFINE_BIT_NOT(int32_t) { return _tlmm_mask_vec_not_epi32(default_v.v, m.v, v.v); }
TL_XMM_DEFINE_BIT_NOT(uint32_t) { return _tlmm_mask_vec_not_epi32(default_v.v, m.v, v.v); }
TL_XMM_DEFINE_BIT_NOT(int64_t) { return _tlmm_mask_vec_not_epi64(default_v.v, m.v, v.v); }
TL_XMM_DEFINE_BIT_NOT(uint64_t) { return _tlmm_mask_vec_not_epi64(default_v.v, m.v, v.v); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_BIT_NOT(int8_t) { return _mm_blendv_epi8(default_v.v, bit_not(t, v.v).v, m.v); }
TL_XMM_DEFINE_BIT_NOT(uint8_t) { return _mm_blendv_epi8(default_v.v, bit_not(t, v.v).v, m.v); }
TL_XMM_DEFINE_BIT_NOT(int16_t) { return _mm_blendv_epi8(default_v.v, bit_not(t, v.v).v, m.v); }
TL_XMM_DEFINE_BIT_NOT(uint16_t) { return _mm_blendv_epi8(default_v.v, bit_not(t, v.v).v, m.v); }
TL_XMM_DEFINE_BIT_NOT(int32_t) { return _mm_blendv_epi8(default_v.v, bit_not(t, v.v).v, m.v); }
TL_XMM_DEFINE_BIT_NOT(uint32_t) { return _mm_blendv_epi8(default_v.v, bit_not(t, v.v).v, m.v); }
TL_XMM_DEFINE_BIT_NOT(int64_t) { return _mm_blendv_epi8(default_v.v, bit_not(t, v.v).v, m.v); }
TL_XMM_DEFINE_BIT_NOT(uint64_t) { return _mm_blendv_epi8(default_v.v, bit_not(t, v.v).v, m.v); }
#endif //HAS_AVX512DQ
#undef TL_XMM_DEFINE_BIT_NOT



// neg(Vec<T> v)
#define TL_XMM_DEFINE_NEG(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(neg, dtype, VecOf(t), (VecOf(t) v), (v))
TL_XMM_DEFINE_NEG(float32_t) { return _mm_sub_ps(_mm_setzero_ps(), v.v); }
TL_XMM_DEFINE_NEG(float64_t) { return _mm_sub_pd(_mm_setzero_pd(), v.v); }
TL_XMM_DEFINE_NEG(int8_t) { return _mm_sub_epi8(_mm_setzero_si128(), v.v); }
TL_XMM_DEFINE_NEG(uint8_t) { return _mm_sub_epi8(_mm_setzero_si128(), v.v); }
TL_XMM_DEFINE_NEG(int16_t) { return _mm_sub_epi16(_mm_setzero_si128(), v.v); }
TL_XMM_DEFINE_NEG(uint16_t) { return _mm_sub_epi16(_mm_setzero_si128(), v.v); }
TL_XMM_DEFINE_NEG(int32_t) { return _mm_sub_epi32(_mm_setzero_si128(), v.v); }
TL_XMM_DEFINE_NEG(uint32_t) { return _mm_sub_epi32(_mm_setzero_si128(), v.v); }
TL_XMM_DEFINE_NEG(int64_t) { return _mm_sub_epi64(_mm_setzero_si128(), v.v); }
TL_XMM_DEFINE_NEG(uint64_t) { return _mm_sub_epi64(_mm_setzero_si128(), v.v); }
#undef TL_XMM_DEFINE_NEG

// neg(Vec<T> a, Vec<T> b, Mask<T> m)
#define TL_XMM_DEFINE_NEG(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(neg, dtype, VecOf(t), (VecOf(t) v, MaskOf(t) m, VecOf(t) default_v), (v, m, default_v))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_NEG(float32_t) { return _mm_mask_sub_ps(default_v.v, m.v, _mm_setzero_ps(), v.v); }
TL_XMM_DEFINE_NEG(float64_t) { return _mm_mask_sub_pd(default_v.v, m.v, _mm_setzero_pd(), v.v); }
TL_XMM_DEFINE_NEG(int8_t) { return _mm_mask_sub_epi8(default_v.v, m.v, _mm_setzero_si128(), v.v); }
TL_XMM_DEFINE_NEG(uint8_t) { return _mm_mask_sub_epi8(default_v.v, m.v, _mm_setzero_si128(), v.v); }
TL_XMM_DEFINE_NEG(int16_t) { return _mm_mask_sub_epi16(default_v.v, m.v, _mm_setzero_si128(), v.v); }
TL_XMM_DEFINE_NEG(uint16_t) { return _mm_mask_sub_epi16(default_v.v, m.v, _mm_setzero_si128(), v.v); }
TL_XMM_DEFINE_NEG(int32_t) { return _mm_mask_sub_epi32(default_v.v, m.v, _mm_setzero_si128(), v.v); }
TL_XMM_DEFINE_NEG(uint32_t) { return _mm_mask_sub_epi32(default_v.v, m.v, _mm_setzero_si128(), v.v); }
TL_XMM_DEFINE_NEG(int64_t) { return _mm_mask_sub_epi64(default_v.v, m.v, _mm_setzero_si128(), v.v); }
TL_XMM_DEFINE_NEG(uint64_t) { return _mm_mask_sub_epi64(default_v.v, m.v, _mm_setzero_si128(), v.v); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_NEG(float32_t) { return _mm_blendv_ps(default_v.v, neg(t, v.v).v, _mm_castsi128_ps(m.v)); }
TL_XMM_DEFINE_NEG(float64_t) { return _mm_blendv_pd(default_v.v, neg(t, v.v).v, _mm_castsi128_pd(m.v)); }
TL_XMM_DEFINE_NEG(int8_t) { return _mm_blendv_epi8(default_v.v, neg(t, v.v).v, m.v); }
TL_XMM_DEFINE_NEG(uint8_t) { return _mm_blendv_epi8(default_v.v, neg(t, v.v).v, m.v); }
TL_XMM_DEFINE_NEG(int16_t) { return _mm_blendv_epi8(default_v.v, neg(t, v.v).v, m.v); }
TL_XMM_DEFINE_NEG(uint16_t) { return _mm_blendv_epi8(default_v.v, neg(t, v.v).v, m.v); }
TL_XMM_DEFINE_NEG(int32_t) { return _mm_blendv_epi8(default_v.v, neg(t, v.v).v, m.v); }
TL_XMM_DEFINE_NEG(uint32_t) { return _mm_blendv_epi8(default_v.v, neg(t, v.v).v, m.v); }
TL_XMM_DEFINE_NEG(int64_t) { return _mm_blendv_epi8(default_v.v, neg(t, v.v).v, m.v); }
TL_XMM_DEFINE_NEG(uint64_t) { return _mm_blendv_epi8(default_v.v, neg(t, v.v).v, m.v); }
#endif //HAS_AVX512DQ
#undef TL_XMM_DEFINE_NEG



// abs(Vec<T> v)
#define TL_XMM_DEFINE_ABS(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(abs, dtype, VecOf(t), (VecOf(t) v), (v))
TL_XMM_DEFINE_ABS(float32_t) { return _mm_and_ps(_mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)), v.v); }
TL_XMM_DEFINE_ABS(float64_t) { return _mm_and_pd(_mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)), v.v); }
TL_XMM_DEFINE_ABS(int8_t) { return _mm_abs_epi8(v.v); }
TL_XMM_DEFINE_ABS(uint8_t) { return v.v; }
TL_XMM_DEFINE_ABS(int16_t) { return _mm_abs_epi16(v.v); }
TL_XMM_DEFINE_ABS(uint16_t) { return v.v; }
TL_XMM_DEFINE_ABS(int32_t) { return _mm_abs_epi32(v.v); }
TL_XMM_DEFINE_ABS(uint32_t) { return v.v; }
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_ABS(int64_t) { return _mm_abs_epi64(v.v); }
#else
TL_XMM_DEFINE_ABS(int64_t) {
  auto high32    = _mm_shuffle_epi32(v.v, _MM_SHUFFLE(3, 3, 1, 1));
  auto sign_mask = _mm_srai_epi32(high32, 31);
  auto xored     = _mm_xor_si128(v.v, sign_mask);
  return _mm_sub_epi64(xored, sign_mask);
}
#endif
TL_XMM_DEFINE_ABS(uint64_t) { return v.v; }
#undef TL_XMM_DEFINE_ABS

// abs(Vec<T> a, Vec<T> b, Mask<T> m)
#define TL_XMM_DEFINE_ABS(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(abs, dtype, VecOf(t), (VecOf(t) v, MaskOf(t) m, VecOf(t) default_v), (v, m, default_v))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_ABS(float32_t) { return _mm_mask_and_ps(default_v.v, m.v, _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)), v.v); }
TL_XMM_DEFINE_ABS(float64_t) { return _mm_mask_and_pd(default_v.v, m.v, _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)), v.v); }
TL_XMM_DEFINE_ABS(int8_t) { return _mm_mask_abs_epi8(default_v.v, m.v, v.v); }
TL_XMM_DEFINE_ABS(uint8_t) { return _mm_mask_blend_epi8(m.v, default_v.v, v.v); }
TL_XMM_DEFINE_ABS(int16_t) { return _mm_mask_abs_epi16(default_v.v, m.v, v.v); }
TL_XMM_DEFINE_ABS(uint16_t) { return _mm_mask_blend_epi16(m.v, default_v.v, v.v); }
TL_XMM_DEFINE_ABS(int32_t) { return _mm_mask_abs_epi32(default_v.v, m.v, v.v); }
TL_XMM_DEFINE_ABS(uint32_t) { return _mm_mask_blend_epi32(m.v, default_v.v, v.v); }
TL_XMM_DEFINE_ABS(int64_t) { return _mm_mask_abs_epi64(default_v.v, m.v, v.v); }
TL_XMM_DEFINE_ABS(uint64_t) { return _mm_mask_blend_epi64(m.v, default_v.v, v.v); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_ABS(float32_t) { return _mm_blendv_ps(default_v.v, abs(t, v.v).v, _mm_castsi128_ps(m.v)); }
TL_XMM_DEFINE_ABS(float64_t) { return _mm_blendv_pd(default_v.v, abs(t, v.v).v, _mm_castsi128_pd(m.v)); }
TL_XMM_DEFINE_ABS(int8_t) { return _mm_blendv_epi8(default_v.v, abs(t, v.v).v, m.v); }
TL_XMM_DEFINE_ABS(uint8_t) { return _mm_blendv_epi8(default_v.v, v.v, m.v); }
TL_XMM_DEFINE_ABS(int16_t) { return _mm_blendv_epi8(default_v.v, abs(t, v.v).v, m.v); }
TL_XMM_DEFINE_ABS(uint16_t) { return _mm_blendv_epi8(default_v.v, v.v, m.v); }
TL_XMM_DEFINE_ABS(int32_t) { return _mm_blendv_epi8(default_v.v, abs(t, v.v).v, m.v); }
TL_XMM_DEFINE_ABS(uint32_t) { return _mm_blendv_epi8(default_v.v, v.v, m.v); }
TL_XMM_DEFINE_ABS(int64_t) { return _mm_blendv_epi8(default_v.v, abs(t, v.v).v, m.v); }
TL_XMM_DEFINE_ABS(uint64_t) { return _mm_blendv_epi8(default_v.v, v.v, m.v); }
#endif //HAS_AVX512DQ
#undef TL_XMM_DEFINE_ABS



// sqrt(Vec<T> v)
#define TL_XMM_DEFINE_SQRT(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(sqrt, dtype, VecOf(t), (VecOf(t) v), (v))
TL_XMM_DEFINE_SQRT(float32_t) { return _mm_sqrt_ps(v.v); }
TL_XMM_DEFINE_SQRT(float64_t) { return _mm_sqrt_pd(v.v); }
#undef TL_XMM_DEFINE_SQRT

// sqrt(Vec<T> a, Vec<T> b, Mask<T> m)
#define TL_XMM_DEFINE_SQRT(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(sqrt, dtype, VecOf(t), (VecOf(t) v, MaskOf(t) m, VecOf(t) default_v), (v, m, default_v))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_SQRT(float32_t) { return _mm_mask_sqrt_ps(default_v.v, m.v, v.v); }
TL_XMM_DEFINE_SQRT(float64_t) { return _mm_mask_sqrt_pd(default_v.v, m.v, v.v); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_SQRT(float32_t) { return _mm_blendv_ps(default_v.v, sqrt(t, v.v).v, _mm_castsi128_ps(m.v)); }
TL_XMM_DEFINE_SQRT(float64_t) { return _mm_blendv_pd(default_v.v, sqrt(t, v.v).v, _mm_castsi128_pd(m.v)); }
#endif //HAS_AVX512DQ
#undef TL_XMM_DEFINE_SQRT



// rsqrt(Vec<T> v)
// Note: AVX512 uses rsqrt14 which gives higher accuracy than rsqrt used not in AVX512
#define TL_XMM_DEFINE_RSQRT(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(rsqrt, dtype, VecOf(t), (VecOf(t) v), (v))
#ifdef HAS_AVX512F
TL_XMM_DEFINE_RSQRT(float32_t) { return _mm_rsqrt14_ps(v.v); }
TL_XMM_DEFINE_RSQRT(float64_t) { return _mm_rsqrt14_pd(v.v); }
#else // HAS_AVX512F
TL_XMM_DEFINE_RSQRT(float32_t) { return _mm_rsqrt_ps(v.v); }
TL_XMM_DEFINE_RSQRT(float64_t) { return div(t, fill(t, 1), sqrt(t, v.v)); }
#endif // HAS_AVX512F
#undef TL_XMM_DEFINE_RSQRT

// rsqrt(Vec<T> a, Vec<T> b, Mask<T> m)
#define TL_XMM_DEFINE_RSQRT(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(rsqrt, dtype, VecOf(t), (VecOf(t) v, MaskOf(t) m, VecOf(t) default_v), (v, m, default_v))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_RSQRT(float32_t) { return _mm_mask_rsqrt14_ps(default_v.v, m.v, v.v); }
TL_XMM_DEFINE_RSQRT(float64_t) { return _mm_mask_rsqrt14_pd(default_v.v, m.v, v.v); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_RSQRT(float32_t) { return _mm_blendv_ps(default_v.v, rsqrt(t, v.v).v, _mm_castsi128_ps(m.v)); }
TL_XMM_DEFINE_RSQRT(float64_t) { return _mm_blendv_pd(default_v.v, rsqrt(t, v.v).v, _mm_castsi128_pd(m.v)); }
#endif //HAS_AVX512DQ
#undef TL_XMM_DEFINE_RSQRT



// cmpeq(Vec<T> a, Vec<T> b)
#define TL_XMM_DEFINE_CMPEQ(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(cmpeq, dtype, MaskOf(t), (VecOf(t) a, VecOf(t) b), (a, b))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_CMPEQ(float32_t) { return _mm_cmp_ps_mask(a.v, b.v, _CMP_EQ_OQ); }
TL_XMM_DEFINE_CMPEQ(float64_t) { return _mm_cmp_pd_mask(a.v, b.v, _CMP_EQ_OQ); }
TL_XMM_DEFINE_CMPEQ(int8_t) { return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_EQ); }
TL_XMM_DEFINE_CMPEQ(uint8_t) { return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_EQ); }
TL_XMM_DEFINE_CMPEQ(int16_t) { return _mm_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_EQ); }
TL_XMM_DEFINE_CMPEQ(uint16_t) { return _mm_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_EQ); }
TL_XMM_DEFINE_CMPEQ(int32_t) { return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_EQ); }
TL_XMM_DEFINE_CMPEQ(uint32_t) { return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_EQ); }
TL_XMM_DEFINE_CMPEQ(int64_t) { return _mm_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_EQ); }
TL_XMM_DEFINE_CMPEQ(uint64_t) { return _mm_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_EQ); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_CMPEQ(float32_t) { return _mm_castps_si128(_mm_cmpeq_ps(a.v, b.v)); }
TL_XMM_DEFINE_CMPEQ(float64_t) { return _mm_castpd_si128(_mm_cmpeq_pd(a.v, b.v)); }
TL_XMM_DEFINE_CMPEQ(int8_t) { return _mm_cmpeq_epi8(a.v, b.v); }
TL_XMM_DEFINE_CMPEQ(uint8_t) { return _mm_cmpeq_epi8(a.v, b.v); }
TL_XMM_DEFINE_CMPEQ(int16_t) { return _mm_cmpeq_epi16(a.v, b.v); }
TL_XMM_DEFINE_CMPEQ(uint16_t) { return _mm_cmpeq_epi16(a.v, b.v); }
TL_XMM_DEFINE_CMPEQ(int32_t) { return _mm_cmpeq_epi32(a.v, b.v); }
TL_XMM_DEFINE_CMPEQ(uint32_t) { return _mm_cmpeq_epi32(a.v, b.v); }
TL_XMM_DEFINE_CMPEQ(int64_t) { return _mm_cmpeq_epi64(a.v, b.v); }
TL_XMM_DEFINE_CMPEQ(uint64_t) { return _mm_cmpeq_epi64(a.v, b.v); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_CMPEQ

// cmpeq(Vec<T> a, Vec<T> b, Mask<T> m)
#define TL_XMM_DEFINE_CMPEQ(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(cmpeq, dtype, MaskOf(t), (VecOf(t) a, VecOf(t) b, MaskOf(t) m), (a, b, m))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_CMPEQ(float32_t) { return _mm_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_EQ_OQ); }
TL_XMM_DEFINE_CMPEQ(float64_t) { return _mm_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_EQ_OQ); }
TL_XMM_DEFINE_CMPEQ(int8_t) { return _mm_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
TL_XMM_DEFINE_CMPEQ(uint8_t) { return _mm_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
TL_XMM_DEFINE_CMPEQ(int16_t) { return _mm_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
TL_XMM_DEFINE_CMPEQ(uint16_t) { return _mm_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
TL_XMM_DEFINE_CMPEQ(int32_t) { return _mm_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
TL_XMM_DEFINE_CMPEQ(uint32_t) { return _mm_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
TL_XMM_DEFINE_CMPEQ(int64_t) { return _mm_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
TL_XMM_DEFINE_CMPEQ(uint64_t) { return _mm_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_CMPEQ(float32_t) { return _mm_and_si128(cmpeq(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPEQ(float64_t) { return _mm_and_si128(cmpeq(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPEQ(int8_t) { return _mm_and_si128(cmpeq(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPEQ(uint8_t) { return _mm_and_si128(cmpeq(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPEQ(int16_t) { return _mm_and_si128(cmpeq(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPEQ(uint16_t) { return _mm_and_si128(cmpeq(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPEQ(int32_t) { return _mm_and_si128(cmpeq(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPEQ(uint32_t) { return _mm_and_si128(cmpeq(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPEQ(int64_t) { return _mm_and_si128(cmpeq(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPEQ(uint64_t) { return _mm_and_si128(cmpeq(t, a.v, b.v).v, m.v); }
#endif //HAS_AVX512DQ
#undef TL_XMM_DEFINE_CMPEQ



// cmpne(Vec<T> a, Vec<T> b)
#define TL_XMM_DEFINE_CMPNE(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(cmpne, dtype, MaskOf(t), (VecOf(t) a, VecOf(t) b), (a, b))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_CMPNE(float32_t) { return _mm_cmp_ps_mask(a.v, b.v, _CMP_NEQ_OQ); }
TL_XMM_DEFINE_CMPNE(float64_t) { return _mm_cmp_pd_mask(a.v, b.v, _CMP_NEQ_OQ); }
TL_XMM_DEFINE_CMPNE(int8_t) { return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NE); }
TL_XMM_DEFINE_CMPNE(uint8_t) { return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NE); }
TL_XMM_DEFINE_CMPNE(int16_t) { return _mm_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NE); }
TL_XMM_DEFINE_CMPNE(uint16_t) { return _mm_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NE); }
TL_XMM_DEFINE_CMPNE(int32_t) { return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NE); }
TL_XMM_DEFINE_CMPNE(uint32_t) { return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NE); }
TL_XMM_DEFINE_CMPNE(int64_t) { return _mm_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NE); }
TL_XMM_DEFINE_CMPNE(uint64_t) { return _mm_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NE); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_CMPNE(float32_t) { return _mm_castps_si128(_mm_cmpneq_ps(a.v, b.v)); }
TL_XMM_DEFINE_CMPNE(float64_t) { return _mm_castpd_si128(_mm_cmpneq_pd(a.v, b.v)); }
TL_XMM_DEFINE_CMPNE(int8_t) { return bit_not(Tag<int32_t, 4>(), cmpeq(t, a.v, b.v).v).v; }
TL_XMM_DEFINE_CMPNE(uint8_t) { return bit_not(Tag<int32_t, 4>(), cmpeq(t, a.v, b.v).v).v; }
TL_XMM_DEFINE_CMPNE(int16_t) { return bit_not(Tag<int32_t, 4>(), cmpeq(t, a.v, b.v).v).v; }
TL_XMM_DEFINE_CMPNE(uint16_t) { return bit_not(Tag<int32_t, 4>(), cmpeq(t, a.v, b.v).v).v; }
TL_XMM_DEFINE_CMPNE(int32_t) { return bit_not(Tag<int32_t, 4>(), cmpeq(t, a.v, b.v).v).v; }
TL_XMM_DEFINE_CMPNE(uint32_t) { return bit_not(Tag<int32_t, 4>(), cmpeq(t, a.v, b.v).v).v; }
TL_XMM_DEFINE_CMPNE(int64_t) { return bit_not(Tag<int32_t, 4>(), cmpeq(t, a.v, b.v).v).v; }
TL_XMM_DEFINE_CMPNE(uint64_t) { return bit_not(Tag<int32_t, 4>(), cmpeq(t, a.v, b.v).v).v; }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_CMPNE

// cmpne(Vec<T> a, Vec<T> b, Mask<T> m)
#define TL_XMM_DEFINE_CMPNE(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(cmpne, dtype, MaskOf(t), (VecOf(t) a, VecOf(t) b, MaskOf(t) m), (a, b, m))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_CMPNE(float32_t) { return _mm_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_NEQ_OQ); }
TL_XMM_DEFINE_CMPNE(float64_t) { return _mm_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_NEQ_OQ); }
TL_XMM_DEFINE_CMPNE(int8_t) { return _mm_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
TL_XMM_DEFINE_CMPNE(uint8_t) { return _mm_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
TL_XMM_DEFINE_CMPNE(int16_t) { return _mm_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
TL_XMM_DEFINE_CMPNE(uint16_t) { return _mm_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
TL_XMM_DEFINE_CMPNE(int32_t) { return _mm_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
TL_XMM_DEFINE_CMPNE(uint32_t) { return _mm_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
TL_XMM_DEFINE_CMPNE(int64_t) { return _mm_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
TL_XMM_DEFINE_CMPNE(uint64_t) { return _mm_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_CMPNE(float32_t) { return _mm_and_si128(cmpne(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPNE(float64_t) { return _mm_and_si128(cmpne(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPNE(int8_t) { return _mm_and_si128(cmpne(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPNE(uint8_t) { return _mm_and_si128(cmpne(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPNE(int16_t) { return _mm_and_si128(cmpne(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPNE(uint16_t) { return _mm_and_si128(cmpne(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPNE(int32_t) { return _mm_and_si128(cmpne(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPNE(uint32_t) { return _mm_and_si128(cmpne(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPNE(int64_t) { return _mm_and_si128(cmpne(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPNE(uint64_t) { return _mm_and_si128(cmpne(t, a.v, b.v).v, m.v); }
#endif //HAS_AVX512DQ
#undef TL_XMM_DEFINE_CMPNE



// cmplt(Vec<T> a, Vec<T> b)
#define TL_XMM_DEFINE_CMPLT(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(cmplt, dtype, MaskOf(t), (VecOf(t) a, VecOf(t) b), (a, b))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_CMPLT(float32_t) { return _mm_cmp_ps_mask(a.v, b.v, _CMP_LT_OQ); }
TL_XMM_DEFINE_CMPLT(float64_t) { return _mm_cmp_pd_mask(a.v, b.v, _CMP_LT_OQ); }
TL_XMM_DEFINE_CMPLT(int8_t)    { return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LT); }
TL_XMM_DEFINE_CMPLT(uint8_t)   { return _mm_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_LT); }
TL_XMM_DEFINE_CMPLT(int16_t)   { return _mm_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LT); }
TL_XMM_DEFINE_CMPLT(uint16_t)  { return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LT); }
TL_XMM_DEFINE_CMPLT(int32_t)   { return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LT); }
TL_XMM_DEFINE_CMPLT(uint32_t)  { return _mm_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_LT); }
TL_XMM_DEFINE_CMPLT(int64_t)   { return _mm_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LT); }
TL_XMM_DEFINE_CMPLT(uint64_t)  { return _mm_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_LT); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_CMPLT(float32_t) { return _mm_castps_si128(_mm_cmplt_ps(a.v, b.v)); }
TL_XMM_DEFINE_CMPLT(float64_t) { return _mm_castpd_si128(_mm_cmplt_pd(a.v, b.v)); }
TL_XMM_DEFINE_CMPLT(int8_t)    { return _mm_cmplt_epi8(a.v, b.v); }
TL_XMM_DEFINE_CMPLT(uint8_t)   { return _mm_cmplt_epi8(_mm_xor_si128(a.v, _mm_set1_epi8((char)0x80)), _mm_xor_si128(b.v, _mm_set1_epi8((char)0x80))); }
TL_XMM_DEFINE_CMPLT(int16_t)   { return _mm_cmplt_epi16(a.v, b.v); }
TL_XMM_DEFINE_CMPLT(uint16_t)  { return _mm_cmplt_epi16(_mm_xor_si128(a.v, _mm_set1_epi16((short)0x8000)), _mm_xor_si128(b.v, _mm_set1_epi16((short)0x8000))); }
TL_XMM_DEFINE_CMPLT(int32_t)   { return _mm_cmplt_epi32(a.v, b.v); }
TL_XMM_DEFINE_CMPLT(uint32_t)  { return _mm_cmplt_epi32(_mm_xor_si128(a.v, _mm_set1_epi32((int)0x80000000u)), _mm_xor_si128(b.v, _mm_set1_epi32((int)0x80000000u))); }
TL_XMM_DEFINE_CMPLT(int64_t)   { return _mm_cmpgt_epi64(b.v, a.v); }
TL_XMM_DEFINE_CMPLT(uint64_t)  { return _mm_cmpgt_epi64(_mm_xor_si128(b.v, _mm_set1_epi64x((int64_t)0x8000000000000000ull)), _mm_xor_si128(a.v, _mm_set1_epi64x((int64_t)0x8000000000000000ull))); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_CMPLT

// cmplt(Vec<T> a, Vec<T> b, Mask<T> m)
#define TL_XMM_DEFINE_CMPLT(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(cmplt, dtype, MaskOf(t), (VecOf(t) a, VecOf(t) b, MaskOf(t) m), (a, b, m))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_CMPLT(float32_t) { return _mm_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_LT_OQ); }
TL_XMM_DEFINE_CMPLT(float64_t) { return _mm_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_LT_OQ); }
TL_XMM_DEFINE_CMPLT(int8_t)    { return _mm_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
TL_XMM_DEFINE_CMPLT(uint8_t)   { return _mm_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
TL_XMM_DEFINE_CMPLT(int16_t)   { return _mm_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
TL_XMM_DEFINE_CMPLT(uint16_t)  { return _mm_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
TL_XMM_DEFINE_CMPLT(int32_t)   { return _mm_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
TL_XMM_DEFINE_CMPLT(uint32_t)  { return _mm_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
TL_XMM_DEFINE_CMPLT(int64_t)   { return _mm_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
TL_XMM_DEFINE_CMPLT(uint64_t)  { return _mm_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_CMPLT(float32_t) { return _mm_and_si128(cmplt(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPLT(float64_t) { return _mm_and_si128(cmplt(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPLT(int8_t)    { return _mm_and_si128(cmplt(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPLT(uint8_t)   { return _mm_and_si128(cmplt(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPLT(int16_t)   { return _mm_and_si128(cmplt(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPLT(uint16_t)  { return _mm_and_si128(cmplt(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPLT(int32_t)   { return _mm_and_si128(cmplt(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPLT(uint32_t)  { return _mm_and_si128(cmplt(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPLT(int64_t)   { return _mm_and_si128(cmplt(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPLT(uint64_t)  { return _mm_and_si128(cmplt(t, a.v, b.v).v, m.v); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_CMPLT



// cmpgt(Vec<T> a, Vec<T> b)
#define TL_XMM_DEFINE_CMPGT(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(cmpgt, dtype, MaskOf(t), (VecOf(t) a, VecOf(t) b), (a, b))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_CMPGT(float32_t) { return _mm_cmp_ps_mask(a.v, b.v, _CMP_GT_OQ); }
TL_XMM_DEFINE_CMPGT(float64_t) { return _mm_cmp_pd_mask(a.v, b.v, _CMP_GT_OQ); }
TL_XMM_DEFINE_CMPGT(int8_t)    { return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NLE); }
TL_XMM_DEFINE_CMPGT(uint8_t)   { return _mm_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NLE); }
TL_XMM_DEFINE_CMPGT(int16_t)   { return _mm_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NLE); }
TL_XMM_DEFINE_CMPGT(uint16_t)  { return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NLE); }
TL_XMM_DEFINE_CMPGT(int32_t)   { return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLE); }
TL_XMM_DEFINE_CMPGT(uint32_t)  { return _mm_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NLE); }
TL_XMM_DEFINE_CMPGT(int64_t)   { return _mm_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NLE); }
TL_XMM_DEFINE_CMPGT(uint64_t)  { return _mm_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NLE); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_CMPGT(float32_t) { return _mm_castps_si128(_mm_cmpgt_ps(a.v, b.v)); }
TL_XMM_DEFINE_CMPGT(float64_t) { return _mm_castpd_si128(_mm_cmpgt_pd(a.v, b.v)); }
TL_XMM_DEFINE_CMPGT(int8_t)    { return _mm_cmpgt_epi8(a.v, b.v); }
TL_XMM_DEFINE_CMPGT(uint8_t)   { return _mm_cmpgt_epi8(_mm_xor_si128(a.v, _mm_set1_epi8((char)0x80)), _mm_xor_si128(b.v, _mm_set1_epi8((char)0x80))); }
TL_XMM_DEFINE_CMPGT(int16_t)   { return _mm_cmpgt_epi16(a.v, b.v); }
TL_XMM_DEFINE_CMPGT(uint16_t)  { return _mm_cmpgt_epi16(_mm_xor_si128(a.v, _mm_set1_epi16((short)0x8000)), _mm_xor_si128(b.v, _mm_set1_epi16((short)0x8000))); }
TL_XMM_DEFINE_CMPGT(int32_t)   { return _mm_cmpgt_epi32(a.v, b.v); }
TL_XMM_DEFINE_CMPGT(uint32_t)  { return _mm_cmpgt_epi32(_mm_xor_si128(a.v, _mm_set1_epi32((int)0x80000000u)), _mm_xor_si128(b.v, _mm_set1_epi32((int)0x80000000u))); }
TL_XMM_DEFINE_CMPGT(int64_t)   { return _mm_cmpgt_epi64(a.v, b.v); }
TL_XMM_DEFINE_CMPGT(uint64_t)  { return _mm_cmpgt_epi64(_mm_xor_si128(a.v, _mm_set1_epi64x((int64_t)0x8000000000000000ull)), _mm_xor_si128(b.v, _mm_set1_epi64x((int64_t)0x8000000000000000ull))); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_CMPGT

// cmpgt(Vec<T> a, Vec<T> b, Mask<T> m)
#define TL_XMM_DEFINE_CMPGT(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(cmpgt, dtype, MaskOf(t), (VecOf(t) a, VecOf(t) b, MaskOf(t) m), (a, b, m))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_CMPGT(float32_t) { return _mm_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_GT_OQ); }
TL_XMM_DEFINE_CMPGT(float64_t) { return _mm_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_GT_OQ); }
TL_XMM_DEFINE_CMPGT(int8_t)    { return _mm_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
TL_XMM_DEFINE_CMPGT(uint8_t)   { return _mm_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
TL_XMM_DEFINE_CMPGT(int16_t)   { return _mm_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
TL_XMM_DEFINE_CMPGT(uint16_t)  { return _mm_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
TL_XMM_DEFINE_CMPGT(int32_t)   { return _mm_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
TL_XMM_DEFINE_CMPGT(uint32_t)  { return _mm_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
TL_XMM_DEFINE_CMPGT(int64_t)   { return _mm_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
TL_XMM_DEFINE_CMPGT(uint64_t)  { return _mm_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_CMPGT(float32_t) { return _mm_and_si128(cmpgt(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPGT(float64_t) { return _mm_and_si128(cmpgt(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPGT(int8_t)    { return _mm_and_si128(cmpgt(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPGT(uint8_t)   { return _mm_and_si128(cmpgt(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPGT(int16_t)   { return _mm_and_si128(cmpgt(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPGT(uint16_t)  { return _mm_and_si128(cmpgt(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPGT(int32_t)   { return _mm_and_si128(cmpgt(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPGT(uint32_t)  { return _mm_and_si128(cmpgt(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPGT(int64_t)   { return _mm_and_si128(cmpgt(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPGT(uint64_t)  { return _mm_and_si128(cmpgt(t, a.v, b.v).v, m.v); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_CMPGT



// cmple(Vec<T> a, Vec<T> b)
#define TL_XMM_DEFINE_CMPLE(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(cmple, dtype, MaskOf(t), (VecOf(t) a, VecOf(t) b), (a, b))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_CMPLE(float32_t) { return _mm_cmp_ps_mask(a.v, b.v, _CMP_LE_OQ); }
TL_XMM_DEFINE_CMPLE(float64_t) { return _mm_cmp_pd_mask(a.v, b.v, _CMP_LE_OQ); }
TL_XMM_DEFINE_CMPLE(int8_t)    { return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LE); }
TL_XMM_DEFINE_CMPLE(uint8_t)   { return _mm_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_LE); }
TL_XMM_DEFINE_CMPLE(int16_t)   { return _mm_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LE); }
TL_XMM_DEFINE_CMPLE(uint16_t)  { return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LE); }
TL_XMM_DEFINE_CMPLE(int32_t)   { return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LE); }
TL_XMM_DEFINE_CMPLE(uint32_t)  { return _mm_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_LE); }
TL_XMM_DEFINE_CMPLE(int64_t)   { return _mm_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LE); }
TL_XMM_DEFINE_CMPLE(uint64_t)  { return _mm_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_LE); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_CMPLE(float32_t) { return _mm_castps_si128(_mm_cmple_ps(a.v, b.v)); }
TL_XMM_DEFINE_CMPLE(float64_t) { return _mm_castpd_si128(_mm_cmple_pd(a.v, b.v)); }
TL_XMM_DEFINE_CMPLE(int8_t)    { return bit_not(Tag<int32_t, 4>(), cmpgt(t, a.v, b.v).v).v; }
TL_XMM_DEFINE_CMPLE(uint8_t)   { return bit_not(Tag<int32_t, 4>(), cmpgt(t, a.v, b.v).v).v; }
TL_XMM_DEFINE_CMPLE(int16_t)   { return bit_not(Tag<int32_t, 4>(), cmpgt(t, a.v, b.v).v).v; }
TL_XMM_DEFINE_CMPLE(uint16_t)  { return bit_not(Tag<int32_t, 4>(), cmpgt(t, a.v, b.v).v).v; }
TL_XMM_DEFINE_CMPLE(int32_t)   { return bit_not(Tag<int32_t, 4>(), cmpgt(t, a.v, b.v).v).v; }
TL_XMM_DEFINE_CMPLE(uint32_t)  { return bit_not(Tag<int32_t, 4>(), cmpgt(t, a.v, b.v).v).v; }
TL_XMM_DEFINE_CMPLE(int64_t)   { return bit_not(Tag<int32_t, 4>(), cmpgt(t, a.v, b.v).v).v; }
TL_XMM_DEFINE_CMPLE(uint64_t)  { return bit_not(Tag<int32_t, 4>(), cmpgt(t, a.v, b.v).v).v; }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_CMPLE

// cmple(Vec<T> a, Vec<T> b, Mask<T> m)
#define TL_XMM_DEFINE_CMPLE(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(cmple, dtype, MaskOf(t), (VecOf(t) a, VecOf(t) b, MaskOf(t) m), (a, b, m))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_CMPLE(float32_t) { return _mm_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_LE_OQ); }
TL_XMM_DEFINE_CMPLE(float64_t) { return _mm_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_LE_OQ); }
TL_XMM_DEFINE_CMPLE(int8_t)    { return _mm_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
TL_XMM_DEFINE_CMPLE(uint8_t)   { return _mm_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
TL_XMM_DEFINE_CMPLE(int16_t)   { return _mm_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
TL_XMM_DEFINE_CMPLE(uint16_t)  { return _mm_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
TL_XMM_DEFINE_CMPLE(int32_t)   { return _mm_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
TL_XMM_DEFINE_CMPLE(uint32_t)  { return _mm_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
TL_XMM_DEFINE_CMPLE(int64_t)   { return _mm_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
TL_XMM_DEFINE_CMPLE(uint64_t)  { return _mm_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_CMPLE(float32_t) { return _mm_and_si128(cmple(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPLE(float64_t) { return _mm_and_si128(cmple(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPLE(int8_t)    { return _mm_and_si128(cmple(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPLE(uint8_t)   { return _mm_and_si128(cmple(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPLE(int16_t)   { return _mm_and_si128(cmple(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPLE(uint16_t)  { return _mm_and_si128(cmple(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPLE(int32_t)   { return _mm_and_si128(cmple(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPLE(uint32_t)  { return _mm_and_si128(cmple(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPLE(int64_t)   { return _mm_and_si128(cmple(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPLE(uint64_t)  { return _mm_and_si128(cmple(t, a.v, b.v).v, m.v); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_CMPLE



// cmpge(Vec<T> a, Vec<T> b)
#define TL_XMM_DEFINE_CMPGE(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(cmpge, dtype, MaskOf(t), (VecOf(t) a, VecOf(t) b), (a, b))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_CMPGE(float32_t) { return _mm_cmp_ps_mask(a.v, b.v, _CMP_GE_OQ); }
TL_XMM_DEFINE_CMPGE(float64_t) { return _mm_cmp_pd_mask(a.v, b.v, _CMP_GE_OQ); }
TL_XMM_DEFINE_CMPGE(int8_t)    { return _mm_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NLT); }
TL_XMM_DEFINE_CMPGE(uint8_t)   { return _mm_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NLT); }
TL_XMM_DEFINE_CMPGE(int16_t)   { return _mm_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NLT); }
TL_XMM_DEFINE_CMPGE(uint16_t)  { return _mm_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NLT); }
TL_XMM_DEFINE_CMPGE(int32_t)   { return _mm_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLT); }
TL_XMM_DEFINE_CMPGE(uint32_t)  { return _mm_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NLT); }
TL_XMM_DEFINE_CMPGE(int64_t)   { return _mm_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NLT); }
TL_XMM_DEFINE_CMPGE(uint64_t)  { return _mm_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NLT); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_CMPGE(float32_t) { return _mm_castps_si128(_mm_cmpge_ps(a.v, b.v)); }
TL_XMM_DEFINE_CMPGE(float64_t) { return _mm_castpd_si128(_mm_cmpge_pd(a.v, b.v)); }
TL_XMM_DEFINE_CMPGE(int8_t)    { return bit_not(Tag<int32_t, 4>(), cmplt(t, a.v, b.v).v).v; }
TL_XMM_DEFINE_CMPGE(uint8_t)   { return bit_not(Tag<int32_t, 4>(), cmplt(t, a.v, b.v).v).v; }
TL_XMM_DEFINE_CMPGE(int16_t)   { return bit_not(Tag<int32_t, 4>(), cmplt(t, a.v, b.v).v).v; }
TL_XMM_DEFINE_CMPGE(uint16_t)  { return bit_not(Tag<int32_t, 4>(), cmplt(t, a.v, b.v).v).v; }
TL_XMM_DEFINE_CMPGE(int32_t)   { return bit_not(Tag<int32_t, 4>(), cmplt(t, a.v, b.v).v).v; }
TL_XMM_DEFINE_CMPGE(uint32_t)  { return bit_not(Tag<int32_t, 4>(), cmplt(t, a.v, b.v).v).v; }
TL_XMM_DEFINE_CMPGE(int64_t)   { return bit_not(Tag<int32_t, 4>(), cmplt(t, a.v, b.v).v).v; }
TL_XMM_DEFINE_CMPGE(uint64_t)  { return bit_not(Tag<int32_t, 4>(), cmplt(t, a.v, b.v).v).v; }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_CMPGE

// cmpge(Vec<T> a, Vec<T> b, Mask<T> m)
#define TL_XMM_DEFINE_CMPGE(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(cmpge, dtype, MaskOf(t), (VecOf(t) a, VecOf(t) b, MaskOf(t) m), (a, b, m))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_CMPGE(float32_t) { return _mm_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_GE_OQ); }
TL_XMM_DEFINE_CMPGE(float64_t) { return _mm_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_GE_OQ); }
TL_XMM_DEFINE_CMPGE(int8_t)    { return _mm_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
TL_XMM_DEFINE_CMPGE(uint8_t)   { return _mm_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
TL_XMM_DEFINE_CMPGE(int16_t)   { return _mm_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
TL_XMM_DEFINE_CMPGE(uint16_t)  { return _mm_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
TL_XMM_DEFINE_CMPGE(int32_t)   { return _mm_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
TL_XMM_DEFINE_CMPGE(uint32_t)  { return _mm_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
TL_XMM_DEFINE_CMPGE(int64_t)   { return _mm_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
TL_XMM_DEFINE_CMPGE(uint64_t)  { return _mm_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_CMPGE(float32_t) { return _mm_and_si128(cmpge(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPGE(float64_t) { return _mm_and_si128(cmpge(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPGE(int8_t)    { return _mm_and_si128(cmpge(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPGE(uint8_t)   { return _mm_and_si128(cmpge(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPGE(int16_t)   { return _mm_and_si128(cmpge(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPGE(uint16_t)  { return _mm_and_si128(cmpge(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPGE(int32_t)   { return _mm_and_si128(cmpge(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPGE(uint32_t)  { return _mm_and_si128(cmpge(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPGE(int64_t)   { return _mm_and_si128(cmpge(t, a.v, b.v).v, m.v); }
TL_XMM_DEFINE_CMPGE(uint64_t)  { return _mm_and_si128(cmpge(t, a.v, b.v).v, m.v); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_CMPGE



// isnan(Vec<T> v)
#define TL_XMM_DEFINE_ISNAN(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(isnan, dtype, MaskOf(t), (VecOf(t) v), (v))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_ISNAN(float32_t) { return _mm_cmp_ps_mask(v.v, v.v, _CMP_UNORD_Q); }
TL_XMM_DEFINE_ISNAN(float64_t) { return _mm_cmp_pd_mask(v.v, v.v, _CMP_UNORD_Q); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_ISNAN(float32_t) { return _mm_castps_si128(_mm_cmpunord_ps(v.v, v.v)); }
TL_XMM_DEFINE_ISNAN(float64_t) { return _mm_castpd_si128(_mm_cmpunord_pd(v.v, v.v)); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_ISNAN

// isnan(Vec<T> v, Mask<T> m)
#define TL_XMM_DEFINE_ISNAN(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(isnan, dtype, MaskOf(t), (VecOf(t) v, MaskOf(t) m), (v, m))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_ISNAN(float32_t) { return _mm_mask_cmp_ps_mask(m.v, v.v, v.v, _CMP_UNORD_Q); }
TL_XMM_DEFINE_ISNAN(float64_t) { return _mm_mask_cmp_pd_mask(m.v, v.v, v.v, _CMP_UNORD_Q); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_ISNAN(float32_t) { return _mm_and_si128(isnan(t, v.v).v, m.v); }
TL_XMM_DEFINE_ISNAN(float64_t) { return _mm_and_si128(isnan(t, v.v).v, m.v); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_ISNAN



// isposinf(Vec<T> v)
#define TL_XMM_DEFINE_ISPOSINF(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(isposinf, dtype, MaskOf(t), (VecOf(t) v), (v))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_ISPOSINF(float32_t) { return _mm_cmp_ps_mask(v.v, _mm_set1_ps(INFINITY), _CMP_EQ_OQ); }
TL_XMM_DEFINE_ISPOSINF(float64_t) { return _mm_cmp_pd_mask(v.v, _mm_set1_pd(INFINITY), _CMP_EQ_OQ); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_ISPOSINF(float32_t) { return _mm_castps_si128(_mm_cmpeq_ps(v.v, _mm_set1_ps(INFINITY))); }
TL_XMM_DEFINE_ISPOSINF(float64_t) { return _mm_castpd_si128(_mm_cmpeq_pd(v.v, _mm_set1_pd(INFINITY))); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_ISPOSINF

// isposinf(Vec<T> v, Mask<T> m)
#define TL_XMM_DEFINE_ISPOSINF(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(isposinf, dtype, MaskOf(t), (VecOf(t) v, MaskOf(t) m), (v, m))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_ISPOSINF(float32_t) { return _mm_mask_cmp_ps_mask(m.v, v.v, _mm_set1_ps(INFINITY), _CMP_EQ_OQ); }
TL_XMM_DEFINE_ISPOSINF(float64_t) { return _mm_mask_cmp_pd_mask(m.v, v.v, _mm_set1_pd(INFINITY), _CMP_EQ_OQ); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_ISPOSINF(float32_t) { return _mm_and_si128(isposinf(t, v.v).v, m.v); }
TL_XMM_DEFINE_ISPOSINF(float64_t) { return _mm_and_si128(isposinf(t, v.v).v, m.v); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_ISPOSINF

// ============================================================================
// isneginf(Vec<T> v)
// ============================================================================
#define TL_XMM_DEFINE_ISNEGINF(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(isneginf, dtype, MaskOf(t), (VecOf(t) v), (v))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_ISNEGINF(float32_t) { return _mm_cmp_ps_mask(v.v, _mm_set1_ps(-INFINITY), _CMP_EQ_OQ); }
TL_XMM_DEFINE_ISNEGINF(float64_t) { return _mm_cmp_pd_mask(v.v, _mm_set1_pd(-INFINITY), _CMP_EQ_OQ); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_ISNEGINF(float32_t) { return _mm_castps_si128(_mm_cmpeq_ps(v.v, _mm_set1_ps(-INFINITY))); }
TL_XMM_DEFINE_ISNEGINF(float64_t) { return _mm_castpd_si128(_mm_cmpeq_pd(v.v, _mm_set1_pd(-INFINITY))); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_ISNEGINF

// isneginf(Vec<T> v, Mask<T> m)
#define TL_XMM_DEFINE_ISNEGINF(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(isneginf, dtype, MaskOf(t), (VecOf(t) v, MaskOf(t) m), (v, m))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_ISNEGINF(float32_t) { return _mm_mask_cmp_ps_mask(m.v, v.v, _mm_set1_ps(-INFINITY), _CMP_EQ_OQ); }
TL_XMM_DEFINE_ISNEGINF(float64_t) { return _mm_mask_cmp_pd_mask(m.v, v.v, _mm_set1_pd(-INFINITY), _CMP_EQ_OQ); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_ISNEGINF(float32_t) { return _mm_and_si128(isneginf(t, v.v).v, m.v); }
TL_XMM_DEFINE_ISNEGINF(float64_t) { return _mm_and_si128(isneginf(t, v.v).v, m.v); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_ISNEGINF



// isinf(Vec<T> v)   — must be defined after isposinf and isneginf
#define TL_XMM_DEFINE_ISINF(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(isinf, dtype, MaskOf(t), (VecOf(t) v), (v))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_ISINF(float32_t) { return isposinf(t, abs(t, v)).v; }
TL_XMM_DEFINE_ISINF(float64_t) { return isposinf(t, abs(t, v)).v; }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_ISINF(float32_t) { return _mm_or_si128(isposinf(t, v.v).v, isneginf(t, v.v).v); }
TL_XMM_DEFINE_ISINF(float64_t) { return _mm_or_si128(isposinf(t, v.v).v, isneginf(t, v.v).v); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_ISINF

// isinf(Vec<T> v, Mask<T> m)
#define TL_XMM_DEFINE_ISINF(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(isinf, dtype, MaskOf(t), (VecOf(t) v, MaskOf(t) m), (v, m))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_ISINF(float32_t) { return isinf(t, v).v & m.v; }
TL_XMM_DEFINE_ISINF(float64_t) { return isinf(t, v).v & m.v; }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_ISINF(float32_t) { return _mm_and_si128(isinf(t, v.v).v, m.v); }
TL_XMM_DEFINE_ISINF(float64_t) { return _mm_and_si128(isinf(t, v.v).v, m.v); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_ISINF



/* ************************************************************************** */
//                            Bit shift operations                            //
/* ************************************************************************** */

// Helper for 8-bit left shift (no native _mm_sll_epi8)
static CT_ALWAYS_FORCEINLINE __m128i _bit_shl_epi8_128(__m128i v, int count) {
  // Expand 8-bit values to 16-bit with zero extension for proper shift behavior
  auto zero = _mm_setzero_si128();
  auto lo = _mm_unpacklo_epi8(v, zero);
  auto hi = _mm_unpackhi_epi8(v, zero);
  auto count_vec = _mm_cvtsi32_si128(count);
  lo = _mm_sll_epi16(lo, count_vec);
  hi = _mm_sll_epi16(hi, count_vec);
  // Use pack with mask to truncate back to 8-bit (modulo 256 behavior, not saturation)
  // _mm_packus_epi16 saturates, but we want truncation
  // Mask to keep only low 8 bits, then combine
  auto mask = _mm_set1_epi16(0xFF);
  lo = _mm_and_si128(lo, mask);
  hi = _mm_and_si128(hi, mask);
  // Pack: lo[i] -> result[2*i], hi[i] -> result[2*i+1]
  return _mm_packus_epi16(lo, hi);
}

// Helper for 8-bit logical right shift (no native _mm_srl_epi8)
static CT_ALWAYS_FORCEINLINE __m128i _bit_srl_epi8_128(__m128i v, int count) {
  auto zero = _mm_setzero_si128();
  auto lo = _mm_unpacklo_epi8(zero, v);  // high byte = 0, low byte = original
  auto hi = _mm_unpackhi_epi8(zero, v);
  lo = _mm_srli_epi16(lo, 8);  // align to low byte
  hi = _mm_srli_epi16(hi, 8);
  auto count_vec = _mm_cvtsi32_si128(count);
  lo = _mm_srl_epi16(lo, count_vec);
  hi = _mm_srl_epi16(hi, count_vec);
  return _mm_packus_epi16(lo, hi);
}

// Helper for 8-bit arithmetic right shift (no native _mm_sra_epi8)
static CT_ALWAYS_FORCEINLINE __m128i _bit_sra_epi8_128(__m128i v, int count) {
  // Sign extend to 16-bit, shift, then saturate pack back
  auto zero = _mm_setzero_si128();
  auto signs = _mm_cmplt_epi8(v, zero);
  auto lo = _mm_unpacklo_epi8(v, signs);
  auto hi = _mm_unpackhi_epi8(v, signs);
  auto count_vec = _mm_cvtsi32_si128(count);
  lo = _mm_sra_epi16(lo, count_vec);
  hi = _mm_sra_epi16(hi, count_vec);
  return _mm_packs_epi16(lo, hi);
}

// Helper for 64-bit arithmetic right shift (no native _mm_sra_epi64)
static CT_ALWAYS_FORCEINLINE __m128i _bit_sra_epi64_128(__m128i v, int count) {
  // Emulate using sign extension and 32-bit shifts
  // Get sign bits
  auto signs = _mm_srai_epi32(v, 31);
  auto sign_hi = _mm_shuffle_epi32(signs, _MM_SHUFFLE(3, 3, 1, 1));
  if (count >= 64) {
    return sign_hi;
  }
  if (count >= 32) {
    // For count >= 32:
    // - The original low 32 bits are completely shifted out
    // - The new low 32 bits come from original high 32 bits (shifted right by count-32)
    // - The new high 32 bits are all sign bits
    
    // Step 1: Get original high 32 bits and shift them right (arithmetic) by count-32
    // Use shuffle to extract high 32-bit parts: [Hi1, Hi1, Hi2, Hi2]
    auto hi_parts = _mm_shuffle_epi32(v, _MM_SHUFFLE(3, 3, 1, 1));
    // Arithmetic right shift by (count - 32)
    auto lo_result = _mm_srai_epi32(hi_parts, count - 32);
    
    // Step 2: The high 32 bits of each 64-bit element should be sign bits
    // sign_hi is [Sign1, Sign1, Sign2, Sign2] where Sign = 0xFFFFFFFF or 0
    // We need to combine: high 32 bits = sign, low 32 bits = lo_result
    // sign_hi << 32 gives us the sign bits in the high 32-bit positions
    auto hi_result = _mm_slli_epi64(sign_hi, 32);
    
    // Combine: low 32 bits from lo_result, high 32 bits from hi_result
    // lo_result has the correct low 32 bits (and incorrect high 32 bits)
    // hi_result has the correct high 32 bits (and zero low 32 bits)
    // We need to mask out the high 32 bits of lo_result and OR with hi_result
    auto mask_lo = _mm_set1_epi64x(0xFFFFFFFF);
    lo_result = _mm_and_si128(lo_result, mask_lo);
    return _mm_or_si128(lo_result, hi_result);
  }
  // count < 32
  // For arithmetic right shift, we need to:
  // 1. Do logical right shift to get the low bits
  // 2. Fill the high 'count' bits with sign bits
  auto lo = _mm_srl_epi64(v, _mm_cvtsi32_si128(count));
  // Create mask for high 'count' bits: (0xFFFFFFFFFFFFFFFF << (64 - count))
  // But we need to handle the case where count could be 0
  auto all_ones = _mm_set1_epi64x(-1);
  auto mask = _mm_slli_epi64(all_ones, 64 - count);
  // For negative numbers, fill high bits with 1s; for positive, fill with 0s
  auto sign_part = _mm_and_si128(sign_hi, mask);
  return _mm_or_si128(lo, sign_part);
}

// bit_shl(Vec<T> v, int count) -> Vec<T>
#define TL_XMM_DEFINE_BIT_SHL(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(bit_shl, dtype, VecOf(t), (VecOf(t) v, int count), (v, count))
TL_XMM_DEFINE_BIT_SHL(int8_t) { return _bit_shl_epi8_128(v.v, count); }
TL_XMM_DEFINE_BIT_SHL(uint8_t) { return _bit_shl_epi8_128(v.v, count); }
TL_XMM_DEFINE_BIT_SHL(int16_t) { return _mm_sll_epi16(v.v, _mm_cvtsi32_si128(count)); }
TL_XMM_DEFINE_BIT_SHL(uint16_t) { return _mm_sll_epi16(v.v, _mm_cvtsi32_si128(count)); }
TL_XMM_DEFINE_BIT_SHL(int32_t) { return _mm_sll_epi32(v.v, _mm_cvtsi32_si128(count)); }
TL_XMM_DEFINE_BIT_SHL(uint32_t) { return _mm_sll_epi32(v.v, _mm_cvtsi32_si128(count)); }
TL_XMM_DEFINE_BIT_SHL(int64_t) { return _mm_sll_epi64(v.v, _mm_cvtsi32_si128(count)); }
TL_XMM_DEFINE_BIT_SHL(uint64_t) { return _mm_sll_epi64(v.v, _mm_cvtsi32_si128(count)); }
#undef TL_XMM_DEFINE_BIT_SHL

// bit_shl(Vec<T> v, int count, Mask<T> m) -> Vec<T>
#define TL_XMM_DEFINE_BIT_SHL(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(bit_shl, dtype, VecOf(t), (VecOf(t) v, int count, MaskOf(t) m), (v, count, m))
#ifdef HAS_AVX512DQ
TL_XMM_DEFINE_BIT_SHL(int8_t) { return _mm_mask_blend_epi8(m.v, v.v, _bit_shl_epi8_128(v.v, count)); }
TL_XMM_DEFINE_BIT_SHL(uint8_t) { return _mm_mask_blend_epi8(m.v, v.v, _bit_shl_epi8_128(v.v, count)); }
TL_XMM_DEFINE_BIT_SHL(int16_t) { return _mm_mask_sll_epi16(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
TL_XMM_DEFINE_BIT_SHL(uint16_t) { return _mm_mask_sll_epi16(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
TL_XMM_DEFINE_BIT_SHL(int32_t) { return _mm_mask_sll_epi32(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
TL_XMM_DEFINE_BIT_SHL(uint32_t) { return _mm_mask_sll_epi32(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
TL_XMM_DEFINE_BIT_SHL(int64_t) { return _mm_mask_sll_epi64(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
TL_XMM_DEFINE_BIT_SHL(uint64_t) { return _mm_mask_sll_epi64(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
#else // HAS_AVX512DQ
TL_XMM_DEFINE_BIT_SHL(int8_t) { return _mm_blendv_epi8(v.v, _bit_shl_epi8_128(v.v, count), m.v); }
TL_XMM_DEFINE_BIT_SHL(uint8_t) { return _mm_blendv_epi8(v.v, _bit_shl_epi8_128(v.v, count), m.v); }
TL_XMM_DEFINE_BIT_SHL(int16_t) { return _mm_blendv_epi8(v.v, _mm_sll_epi16(v.v, _mm_cvtsi32_si128(count)), m.v); }
TL_XMM_DEFINE_BIT_SHL(uint16_t) { return _mm_blendv_epi8(v.v, _mm_sll_epi16(v.v, _mm_cvtsi32_si128(count)), m.v); }
TL_XMM_DEFINE_BIT_SHL(int32_t) { return _mm_blendv_epi8(v.v, _mm_sll_epi32(v.v, _mm_cvtsi32_si128(count)), m.v); }
TL_XMM_DEFINE_BIT_SHL(uint32_t) { return _mm_blendv_epi8(v.v, _mm_sll_epi32(v.v, _mm_cvtsi32_si128(count)), m.v); }
TL_XMM_DEFINE_BIT_SHL(int64_t) { return _mm_blendv_epi8(v.v, _mm_sll_epi64(v.v, _mm_cvtsi32_si128(count)), m.v); }
TL_XMM_DEFINE_BIT_SHL(uint64_t) { return _mm_blendv_epi8(v.v, _mm_sll_epi64(v.v, _mm_cvtsi32_si128(count)), m.v); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_BIT_SHL



// bit_shr(Vec<T> v, int count) -> Vec<T>
// Signed: arithmetic shift (sign extension), Unsigned: logical shift (zero fill)
#define TL_XMM_DEFINE_BIT_SHR(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(bit_shr, dtype, VecOf(t), (VecOf(t) v, int count), (v, count))
// Signed types - arithmetic right shift
TL_XMM_DEFINE_BIT_SHR(int8_t) { return _bit_sra_epi8_128(v.v, count); }
TL_XMM_DEFINE_BIT_SHR(int16_t) { return _mm_sra_epi16(v.v, _mm_cvtsi32_si128(count)); }
TL_XMM_DEFINE_BIT_SHR(int32_t) { return _mm_sra_epi32(v.v, _mm_cvtsi32_si128(count)); }
#ifdef HAS_AVX512VL
TL_XMM_DEFINE_BIT_SHR(int64_t) { return _mm_sra_epi64(v.v, _mm_cvtsi32_si128(count)); }
#else
TL_XMM_DEFINE_BIT_SHR(int64_t) { return _bit_sra_epi64_128(v.v, count); }
#endif
// Unsigned types - logical right shift
TL_XMM_DEFINE_BIT_SHR(uint8_t) { return _bit_srl_epi8_128(v.v, count); }
TL_XMM_DEFINE_BIT_SHR(uint16_t) { return _mm_srl_epi16(v.v, _mm_cvtsi32_si128(count)); }
TL_XMM_DEFINE_BIT_SHR(uint32_t) { return _mm_srl_epi32(v.v, _mm_cvtsi32_si128(count)); }
TL_XMM_DEFINE_BIT_SHR(uint64_t) { return _mm_srl_epi64(v.v, _mm_cvtsi32_si128(count)); }
#undef TL_XMM_DEFINE_BIT_SHR

// bit_shr(Vec<T> v, int count, Mask<T> m) -> Vec<T>
#define TL_XMM_DEFINE_BIT_SHR(dtype) TL_XMM_DEFINE_WITH_ALL_HALVES(bit_shr, dtype, VecOf(t), (VecOf(t) v, int count, MaskOf(t) m), (v, count, m))
#ifdef HAS_AVX512DQ
// Signed types - arithmetic right shift
TL_XMM_DEFINE_BIT_SHR(int8_t) { return _mm_mask_blend_epi8(m.v, v.v, _bit_sra_epi8_128(v.v, count)); }
TL_XMM_DEFINE_BIT_SHR(int16_t) { return _mm_mask_sra_epi16(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
TL_XMM_DEFINE_BIT_SHR(int32_t) { return _mm_mask_sra_epi32(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
#ifdef HAS_AVX512VL
TL_XMM_DEFINE_BIT_SHR(int64_t) { return _mm_mask_sra_epi64(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
#else
TL_XMM_DEFINE_BIT_SHR(int64_t) { return _mm_mask_blend_epi8(m.v, v.v, _bit_sra_epi64_128(v.v, count)); }
#endif
// Unsigned types - logical right shift
TL_XMM_DEFINE_BIT_SHR(uint8_t) { return _mm_mask_blend_epi8(m.v, v.v, _bit_srl_epi8_128(v.v, count)); }
TL_XMM_DEFINE_BIT_SHR(uint16_t) { return _mm_mask_srl_epi16(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
TL_XMM_DEFINE_BIT_SHR(uint32_t) { return _mm_mask_srl_epi32(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
TL_XMM_DEFINE_BIT_SHR(uint64_t) { return _mm_mask_srl_epi64(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
#else // HAS_AVX512DQ
// Signed types - arithmetic right shift
TL_XMM_DEFINE_BIT_SHR(int8_t) { return _mm_blendv_epi8(v.v, _bit_sra_epi8_128(v.v, count), m.v); }
TL_XMM_DEFINE_BIT_SHR(int16_t) { return _mm_blendv_epi8(v.v, _mm_sra_epi16(v.v, _mm_cvtsi32_si128(count)), m.v); }
TL_XMM_DEFINE_BIT_SHR(int32_t) { return _mm_blendv_epi8(v.v, _mm_sra_epi32(v.v, _mm_cvtsi32_si128(count)), m.v); }
TL_XMM_DEFINE_BIT_SHR(int64_t) { return _mm_blendv_epi8(v.v, _bit_sra_epi64_128(v.v, count), m.v); }
// Unsigned types - logical right shift
TL_XMM_DEFINE_BIT_SHR(uint8_t) { return _mm_blendv_epi8(v.v, _bit_srl_epi8_128(v.v, count), m.v); }
TL_XMM_DEFINE_BIT_SHR(uint16_t) { return _mm_blendv_epi8(v.v, _mm_srl_epi16(v.v, _mm_cvtsi32_si128(count)), m.v); }
TL_XMM_DEFINE_BIT_SHR(uint32_t) { return _mm_blendv_epi8(v.v, _mm_srl_epi32(v.v, _mm_cvtsi32_si128(count)), m.v); }
TL_XMM_DEFINE_BIT_SHR(uint64_t) { return _mm_blendv_epi8(v.v, _mm_srl_epi64(v.v, _mm_cvtsi32_si128(count)), m.v); }
#endif // HAS_AVX512DQ
#undef TL_XMM_DEFINE_BIT_SHR



#undef _TL_XMM_DEFINE_HALVES
#undef _TL_XMM_DEFINE_HALVES_CHECK_RANGE
#undef _TL_XMM_DEFINE_HALVES_CHECK_INDEX
#undef TL_XMM_DEFINE_WITH_ALL_HALVES
#undef TL_XMM_DEFINE_WITH_ALL_HALVES_CHECK_COUNT
#undef TL_XMM_DEFINE_WITH_ALL_HALVES_V
#undef TL_CHECK_COUNT
#undef TL_CHECK_INDEX
#undef TL_CHECK_ALIGN
#undef X
} // namespace word
} // namespace ct::tl::vec
//@formatter:on

#endif //CTORCH_X86_128_H
