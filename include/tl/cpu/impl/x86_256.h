//
// Created by renyz on 2026/3/20.
//

#ifndef CTORCH_X86_256_H
#define CTORCH_X86_256_H

//@formatter:off
#include <cmath>
#include "CoreDefs.h"
#include "tl/util/Math.h"
#include "tl/cpu/VecBase.h"

#ifndef ARCH_X86_FAMILY
#error "Not x86 platform"
#endif

#ifndef HAS_CPU_CAPABILITY_AVX2
  #error "AVX2 instruction set required"
#endif

#include <immintrin.h>
#include "tl/cpu/impl/x86_128.h"
#include "tl/cpu/impl/x86_Types.h"

#ifndef HAS_AVX512DQ
#include "tl/cpu/impl/x86_MaskSupport.h"
#endif

namespace ct::tl::vec {
namespace word {

#define TL_CHECK_COUNT(varname) CT_ASSERT(0 <= (varname) && (varname) <= size(t), "%zd !in 0..%zd", (varname), size(t))
#define TL_CHECK_INDEX(varname) CT_ASSERT(0 <= (varname) && (varname) < size(t), "%zd !in 0..%zd", (varname), size(t))
#define TL_CHECK_ALIGN CT_ASSERT(((nuint_t)(p) & (DEFAULT_ALIGNMENT - 1)) == 0, "Not aligned");



/* ************************************************************************** */
//                               Constructors                                 //
/* ************************************************************************** */

// mfill, mwhilelt, mwhilege
#ifdef HAS_AVX512DQ /* mask operations */
#define TL_YMM_DEFINE_MASK_OPERATIONS(dtype, mask_size) \
CT_ALWAYS_FORCEINLINE auto mfill(Tag<dtype, (32 / sizeof(dtype))> t, bool value) -> MaskOf(t) { \
  uint32_t x = value ? 0xffffffffu : 0x00; \
  return _cvtu32_mask##mask_size(x); \
} \
CT_ALWAYS_FORCEINLINE auto mwhilelt(Tag<dtype, (32 / sizeof(dtype))> t, nint_t a, nint_t b) -> MaskOf(t) { \
  nint_t end = std::clamp(b - a, nint_t(0), nint_t(sizeof(unsigned) * CHAR_BIT)); \
  return _cvtu32_mask##mask_size(uint32_t((nint_t(1) << end) - 1)); \
} \
CT_ALWAYS_FORCEINLINE auto mwhilege(Tag<dtype, (32 / sizeof(dtype))> t, nint_t a, nint_t b) -> MaskOf(t) { \
  nint_t end = std::clamp(b - a, nint_t(0), nint_t(sizeof(nint_t) * CHAR_BIT)); \
  return _cvtu32_mask##mask_size(~uint32_t((nint_t(1) << end) - 1)); \
}

TL_YMM_DEFINE_MASK_OPERATIONS(bfloat16_t, 16)
TL_YMM_DEFINE_MASK_OPERATIONS(float16_t, 16)
TL_YMM_DEFINE_MASK_OPERATIONS(float32_t, 8)
TL_YMM_DEFINE_MASK_OPERATIONS(float64_t, 8)
TL_YMM_DEFINE_MASK_OPERATIONS(int8_t, 32)
TL_YMM_DEFINE_MASK_OPERATIONS(uint8_t, 32)
TL_YMM_DEFINE_MASK_OPERATIONS(int16_t, 16)
TL_YMM_DEFINE_MASK_OPERATIONS(uint16_t, 16)
TL_YMM_DEFINE_MASK_OPERATIONS(int32_t, 8)
TL_YMM_DEFINE_MASK_OPERATIONS(uint32_t, 8)
TL_YMM_DEFINE_MASK_OPERATIONS(int64_t, 8)
TL_YMM_DEFINE_MASK_OPERATIONS(uint64_t, 8)
#undef TL_YMM_DEFINE_MASK_OPERATIONS

#else // HAS_AVX512DQ
#define TL_YMM_DEFINE_MASK_OPERATIONS_8(dtype) \
CT_ALWAYS_FORCEINLINE auto mfill(Tag<dtype, (32 / sizeof(dtype))> t, bool value) -> MaskOf(t) { \
  int64_t x = value ? int64_t(-1) : 0; \
  return _mm256_set1_epi64x(x); \
} \
CT_ALWAYS_FORCEINLINE auto mwhilelt(Tag<dtype, (32 / sizeof(dtype))> t, nint_t a, nint_t b) -> MaskOf(t) { \
  static const __m256i index = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, \
                                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31); \
  int8_t diff = (int8_t)std::clamp<nint_t>(b - a, INT8_MIN, INT8_MAX); \
  auto end = _mm256_set1_epi8(diff); \
  return _mm256_cmpgt_epi8(end, index); \
} \
CT_ALWAYS_FORCEINLINE auto mwhilege(Tag<dtype, (32 / sizeof(dtype))> t, nint_t a, nint_t b) -> MaskOf(t) { \
  static const __m256i index = _mm256_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, \
                                                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32); \
  int8_t diff = (int8_t)std::clamp<nint_t>(b - a, INT8_MIN, INT8_MAX); \
  auto start = _mm256_set1_epi8(diff); \
  return _mm256_cmpgt_epi8(index, start); \
}
#define TL_YMM_DEFINE_MASK_OPERATIONS_16(dtype) \
CT_ALWAYS_FORCEINLINE auto mfill(Tag<dtype, (32 / sizeof(dtype))> t, bool value) -> MaskOf(t) { \
  int64_t x = value ? int64_t(-1) : 0; \
  return _mm256_set1_epi64x(x); \
} \
CT_ALWAYS_FORCEINLINE auto mwhilelt(Tag<dtype, (32 / sizeof(dtype))> t, nint_t a, nint_t b) -> MaskOf(t) { \
  static const __m256i index = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15); \
  int16_t diff = (int16_t)std::clamp<nint_t>(b - a, INT16_MIN, INT16_MAX); \
  auto end = _mm256_set1_epi16(diff); \
  return _mm256_cmpgt_epi16(end, index); \
} \
CT_ALWAYS_FORCEINLINE auto mwhilege(Tag<dtype, (32 / sizeof(dtype))> t, nint_t a, nint_t b) -> MaskOf(t) { \
  static const __m256i index = _mm256_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16); \
  int16_t diff = (int16_t)std::clamp<nint_t>(b - a, INT16_MIN, INT16_MAX); \
  auto start = _mm256_set1_epi16(diff); \
  return _mm256_cmpgt_epi16(index, start); \
}
#define TL_YMM_DEFINE_MASK_OPERATIONS_32(dtype) \
CT_ALWAYS_FORCEINLINE auto mfill(Tag<dtype, (32 / sizeof(dtype))> t, bool value) -> MaskOf(t) { \
  int64_t x = value ? int64_t(-1) : 0; \
  return _mm256_set1_epi64x(x); \
} \
CT_ALWAYS_FORCEINLINE auto mwhilelt(Tag<dtype, (32 / sizeof(dtype))> t, nint_t a, nint_t b) -> MaskOf(t) { \
  static const __m256i index = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7); \
  int32_t diff = (int32_t)std::clamp<nint_t>(b - a, INT32_MIN, INT32_MAX); \
  auto end = _mm256_set1_epi32(diff); \
  return _mm256_cmpgt_epi32(end, index); \
} \
CT_ALWAYS_FORCEINLINE auto mwhilege(Tag<dtype, (32 / sizeof(dtype))> t, nint_t a, nint_t b) -> MaskOf(t) { \
  static const __m256i index = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8); \
  int32_t diff = (int32_t)std::clamp<nint_t>(b - a, INT32_MIN, INT32_MAX); \
  auto start = _mm256_set1_epi32(diff); \
  return _mm256_cmpgt_epi32(index, start); \
}
#define TL_YMM_DEFINE_MASK_OPERATIONS_64(dtype) \
CT_ALWAYS_FORCEINLINE auto mfill(Tag<dtype, (32 / sizeof(dtype))> t, bool value) -> MaskOf(t) { \
  int64_t x = value ? int64_t(-1) : 0; \
  return _mm256_set1_epi64x(x); \
} \
CT_ALWAYS_FORCEINLINE auto mwhilelt(Tag<dtype, (32 / sizeof(dtype))> t, nint_t a, nint_t b) -> MaskOf(t) { \
  static const __m256i index = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3); \
  int32_t diff = (int32_t)std::clamp<nint_t>(b - a, INT32_MIN, INT32_MAX); \
  auto end = _mm256_set1_epi32(diff); \
  return _mm256_cmpgt_epi32(end, index); \
} \
CT_ALWAYS_FORCEINLINE auto mwhilege(Tag<dtype, (32 / sizeof(dtype))> t, nint_t a, nint_t b) -> MaskOf(t) { \
  static const __m256i index = _mm256_setr_epi32(1, 1, 2, 2, 3, 3, 4, 4); \
  int32_t diff = (int32_t)std::clamp<nint_t>(b - a, INT32_MIN, INT32_MAX); \
  auto start = _mm256_set1_epi32(diff); \
  return _mm256_cmpgt_epi32(index, start); \
}

TL_YMM_DEFINE_MASK_OPERATIONS_16(bfloat16_t)
TL_YMM_DEFINE_MASK_OPERATIONS_16(float16_t)
TL_YMM_DEFINE_MASK_OPERATIONS_32(float32_t)
TL_YMM_DEFINE_MASK_OPERATIONS_64(float64_t)
TL_YMM_DEFINE_MASK_OPERATIONS_8(int8_t)
TL_YMM_DEFINE_MASK_OPERATIONS_8(uint8_t)
TL_YMM_DEFINE_MASK_OPERATIONS_16(int16_t)
TL_YMM_DEFINE_MASK_OPERATIONS_16(uint16_t)
TL_YMM_DEFINE_MASK_OPERATIONS_32(int32_t)
TL_YMM_DEFINE_MASK_OPERATIONS_32(uint32_t)
TL_YMM_DEFINE_MASK_OPERATIONS_64(int64_t)
TL_YMM_DEFINE_MASK_OPERATIONS_64(uint64_t)
#undef TL_YMM_DEFINE_MASK_OPERATIONS_8
#undef TL_YMM_DEFINE_MASK_OPERATIONS_16
#undef TL_YMM_DEFINE_MASK_OPERATIONS_32
#undef TL_YMM_DEFINE_MASK_OPERATIONS_64
#endif // HAS_AVX512DQ



// fill(T v)
CT_ALWAYS_FORCEINLINE auto fill(Tag<bfloat16_t, 16> t, bfloat16_t v) -> VecOf(t) {
  union { bfloat16_t b; int16_t i; } u { .b = v };
  return _mm256_set1_epi16(u.i);
}
CT_ALWAYS_FORCEINLINE auto fill(Tag<float16_t, 16> t, float16_t v) -> VecOf(t) {
  #ifdef HAS_AVX512_FP16
  return _mm256_castph_si256(_mm256_set1_ph(v));
  #else
  union { float16_t b; int16_t i; } u { .b = v };
  return _mm256_set1_epi16(u.i);
  #endif
}
CT_ALWAYS_FORCEINLINE auto fill(Tag<float32_t, 8> t, float32_t v) -> VecOf(t) { return _mm256_set1_ps(v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<float64_t, 4> t, float64_t v) -> VecOf(t) { return _mm256_set1_pd(v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int8_t, 32> t, int8_t v) -> VecOf(t) { return _mm256_set1_epi8(v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint8_t, 32> t, uint8_t v) -> VecOf(t) { return _mm256_set1_epi8((int8_t)v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int16_t, 16> t, int16_t v) -> VecOf(t) { return _mm256_set1_epi16(v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint16_t, 16> t, uint16_t v) -> VecOf(t) { return _mm256_set1_epi16((int16_t)v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int32_t, 8> t, int32_t v) -> VecOf(t) { return _mm256_set1_epi32(v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint32_t, 8> t, uint32_t v) -> VecOf(t) { return _mm256_set1_epi32((int32_t)v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int64_t, 4> t, int64_t v) -> VecOf(t) { return _mm256_set1_epi64x(v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint64_t, 4> t, uint64_t v) -> VecOf(t) { return _mm256_set1_epi64x((int64_t)v); }

// fill(T v, Mask<T> m, Vec<T> default_v)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto fill(Tag<bfloat16_t, 16> t, bfloat16_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_mov_epi16(default_v.v, m.v, fill(t, v).v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<float16_t, 16> t, float16_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_mov_epi16(default_v.v, m.v, fill(t, v).v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<float32_t, 8> t, float32_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_mov_ps(default_v.v, m.v, fill(t, v).v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<float64_t, 4> t, float64_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_mov_pd(default_v.v, m.v, fill(t, v).v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int8_t, 32> t, int8_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_mov_epi8(default_v.v, m.v, fill(t, v).v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint8_t, 32> t, uint8_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_mov_epi8(default_v.v, m.v, fill(t, v).v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int16_t, 16> t, int16_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_mov_epi16(default_v.v, m.v, fill(t, v).v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint16_t, 16> t, uint16_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_mov_epi16(default_v.v, m.v, fill(t, v).v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int32_t, 8> t, int32_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_mov_epi32(default_v.v, m.v, fill(t, v).v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint32_t, 8> t, uint32_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_mov_epi32(default_v.v, m.v, fill(t, v).v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int64_t, 4> t, int64_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_mov_epi64(default_v.v, m.v, fill(t, v).v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint64_t, 4> t, uint64_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_mov_epi64(default_v.v, m.v, fill(t, v).v); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto fill(Tag<bfloat16_t, 16> t, bfloat16_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, fill(t, v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<float16_t, 16> t, float16_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, fill(t, v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<float32_t, 8> t, float32_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_ps(default_v.v, fill(t, v).v, _mm256_castsi256_ps(m.v)); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<float64_t, 4> t, float64_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_pd(default_v.v, fill(t, v).v, _mm256_castsi256_pd(m.v)); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int8_t, 32> t, int8_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, fill(t, v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint8_t, 32> t, uint8_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, fill(t, v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int16_t, 16> t, int16_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, fill(t, v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint16_t, 16> t, uint16_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, fill(t, v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int32_t, 8> t, int32_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, fill(t, v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint32_t, 8> t, uint32_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, fill(t, v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int64_t, 4> t, int64_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, fill(t, v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint64_t, 4> t, uint64_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, fill(t, v).v, m.v); }
#endif // HAS_AVX512DQ

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE auto fill_fwd_mask_256(Tag<T, N, P> t, T v, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = word::mwhilelt(t, 0, n);
  return word::fill(t, v, m, default_v);
}

// fill(T v, nint_t n, Vec<T> default_v)
CT_ALWAYS_FORCEINLINE auto fill(Tag<bfloat16_t, 16> t, bfloat16_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_256(t, v, n, default_v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<float16_t, 16> t, float16_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_256(t, v, n, default_v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<float32_t, 8> t, float32_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_256(t, v, n, default_v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<float64_t, 4> t, float64_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_256(t, v, n, default_v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int8_t, 32> t, int8_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_256(t, v, n, default_v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint8_t, 32> t, uint8_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_256(t, v, n, default_v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int16_t, 16> t, int16_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_256(t, v, n, default_v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint16_t, 16> t, uint16_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_256(t, v, n, default_v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int32_t, 8> t, int32_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_256(t, v, n, default_v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint32_t, 8> t, uint32_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_256(t, v, n, default_v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int64_t, 4> t, int64_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_256(t, v, n, default_v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint64_t, 4> t, uint64_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_256(t, v, n, default_v); }

// zeros()
CT_ALWAYS_FORCEINLINE auto zeros(Tag<bfloat16_t, 16> t) -> VecOf(t) { return _mm256_setzero_si256(); }
CT_ALWAYS_FORCEINLINE auto zeros(Tag<float16_t, 16> t) -> VecOf(t) { return _mm256_setzero_si256(); }
CT_ALWAYS_FORCEINLINE auto zeros(Tag<float32_t, 8> t) -> VecOf(t) { return _mm256_setzero_ps(); }
CT_ALWAYS_FORCEINLINE auto zeros(Tag<float64_t, 4> t) -> VecOf(t) { return _mm256_setzero_pd(); }
CT_ALWAYS_FORCEINLINE auto zeros(Tag<int8_t, 32> t) -> VecOf(t) { return _mm256_setzero_si256(); }
CT_ALWAYS_FORCEINLINE auto zeros(Tag<uint8_t, 32> t) -> VecOf(t) { return _mm256_setzero_si256(); }
CT_ALWAYS_FORCEINLINE auto zeros(Tag<int16_t, 16> t) -> VecOf(t) { return _mm256_setzero_si256(); }
CT_ALWAYS_FORCEINLINE auto zeros(Tag<uint16_t, 16> t) -> VecOf(t) { return _mm256_setzero_si256(); }
CT_ALWAYS_FORCEINLINE auto zeros(Tag<int32_t, 8> t) -> VecOf(t) { return _mm256_setzero_si256(); }
CT_ALWAYS_FORCEINLINE auto zeros(Tag<uint32_t, 8> t) -> VecOf(t) { return _mm256_setzero_si256(); }
CT_ALWAYS_FORCEINLINE auto zeros(Tag<int64_t, 4> t) -> VecOf(t) { return _mm256_setzero_si256(); }
CT_ALWAYS_FORCEINLINE auto zeros(Tag<uint64_t, 4> t) -> VecOf(t) { return _mm256_setzero_si256(); }



/* ************************************************************************** */
//                          Consecutive load & store                          //
/* ************************************************************************** */

// loadu(const T* p)
CT_ALWAYS_FORCEINLINE auto loadu(Tag<bfloat16_t, 16> t, const bfloat16_t* p) -> VecOf(t) { return _mm256_loadu_si256((const __m256i *)p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<float16_t, 16> t, const float16_t* p) -> VecOf(t) {
  #ifdef HAS_AVX512_FP16
  return _mm256_castph_si256(_mm256_loadu_ph(p));
  #else
  return _mm256_loadu_si256((const __m256i *)p);
  #endif
}
CT_ALWAYS_FORCEINLINE auto loadu(Tag<float32_t, 8> t, const float32_t* p) -> VecOf(t) { return _mm256_loadu_ps(p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<float64_t, 4> t, const float64_t* p) -> VecOf(t) { return _mm256_loadu_pd(p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int8_t, 32> t, const int8_t* p) -> VecOf(t) { return _mm256_loadu_si256((const __m256i *)p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint8_t, 32> t, const uint8_t* p) -> VecOf(t) { return _mm256_loadu_si256((const __m256i *)p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int16_t, 16> t, const int16_t* p) -> VecOf(t) { return _mm256_loadu_si256((const __m256i *)p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint16_t, 16> t, const uint16_t* p) -> VecOf(t) { return _mm256_loadu_si256((const __m256i *)p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int32_t, 8> t, const int32_t* p) -> VecOf(t) { return _mm256_loadu_si256((const __m256i *)p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint32_t, 8> t, const uint32_t* p) -> VecOf(t) { return _mm256_loadu_si256((const __m256i *)p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int64_t, 4> t, const int64_t* p) -> VecOf(t) { return _mm256_loadu_si256((const __m256i *)p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint64_t, 4> t, const uint64_t* p) -> VecOf(t) { return _mm256_loadu_si256((const __m256i *)p); }

// load(const T* p)
CT_ALWAYS_FORCEINLINE auto load(Tag<bfloat16_t, 16> t, const bfloat16_t* p) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_load_si256((const __m256i *)p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<float16_t, 16> t, const float16_t* p) -> VecOf(t) {
  TL_CHECK_ALIGN
  #ifdef HAS_AVX512_FP16
  return _mm256_castph_si256(_mm256_load_ph(p));
  #else
  return _mm256_load_si256((const __m256i *)p);
  #endif
}
CT_ALWAYS_FORCEINLINE auto load(Tag<float32_t, 8> t, const float32_t* p) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_load_ps(p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<float64_t, 4> t, const float64_t* p) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_load_pd(p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int8_t, 32> t, const int8_t* p) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_load_si256((const __m256i *)p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint8_t, 32> t, const uint8_t* p) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_load_si256((const __m256i *)p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int16_t, 16> t, const int16_t* p) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_load_si256((const __m256i *)p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint16_t, 16> t, const uint16_t* p) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_load_si256((const __m256i *)p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int32_t, 8> t, const int32_t* p) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_load_si256((const __m256i *)p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint32_t, 8> t, const uint32_t* p) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_load_si256((const __m256i *)p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int64_t, 4> t, const int64_t* p) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_load_si256((const __m256i *)p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint64_t, 4> t, const uint64_t* p) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_load_si256((const __m256i *)p); }

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE auto _ensure_mask_range_256(Tag<T, N, P> t, MaskOf(t) m) -> MaskOf(t) {
  if constexpr (N * sizeof(T) < 32) {
    #ifdef HAS_AVX512DQ
    uint32_t x = _cvtmask8_u32(m);
    return _cvtu32_mask8(x & uint32_t((nint_t(1) << N) - 1));
    #else
    if constexpr (N * sizeof(T) == 1) {
      return _mm256_and_si256(m, _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0x000000FF));
    } else if constexpr (N * sizeof(T) == 2) {
      return _mm256_and_si256(m, _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0x0000FFFF));
    } else if constexpr (N * sizeof(T) == 4) {
      return _mm256_and_si256(m, _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, (int)0xFFFFFFFF));
    } else if constexpr (N * sizeof(T) == 8) {
      return _mm256_and_si256(m, _mm256_set_epi64x(0, 0, 0, (long long)0xFFFFFFFF'FFFFFFFFLL));
    }
    #endif
  } else {
    return m;
  }
}

// loadu(const T* p, Mask<T> m, Vec<T> default_v)
// load(const T* p, Mask<T> m, Vec<T> default_v)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto loadu(Tag<bfloat16_t, 16> t, const bfloat16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_loadu_epi16(default_v.v, _ensure_mask_range_256(t, m).v, p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<float16_t, 16> t, const float16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_loadu_epi16(default_v.v, _ensure_mask_range_256(t, m).v, p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<float32_t, 8> t, const float32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_loadu_ps(default_v.v, _ensure_mask_range_256(t, m).v, p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<float64_t, 4> t, const float64_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_loadu_pd(default_v.v, _ensure_mask_range_256(t, m).v, p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int8_t, 32> t, const int8_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_loadu_epi8(default_v.v, _ensure_mask_range_256(t, m).v, p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint8_t, 32> t, const uint8_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_loadu_epi8(default_v.v, _ensure_mask_range_256(t, m).v, p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int16_t, 16> t, const int16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_loadu_epi16(default_v.v, _ensure_mask_range_256(t, m).v, p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint16_t, 16> t, const uint16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_loadu_epi16(default_v.v, _ensure_mask_range_256(t, m).v, p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int32_t, 8> t, const int32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_loadu_epi32(default_v.v, _ensure_mask_range_256(t, m).v, p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint32_t, 8> t, const uint32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_loadu_epi32(default_v.v, _ensure_mask_range_256(t, m).v, p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int64_t, 4> t, const int64_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_loadu_epi64(default_v.v, _ensure_mask_range_256(t, m).v, p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint64_t, 4> t, const uint64_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_loadu_epi64(default_v.v, _ensure_mask_range_256(t, m).v, p); }
// No aligned mask load for epi8, epi16
CT_ALWAYS_FORCEINLINE auto load(Tag<bfloat16_t, 16> t, const bfloat16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_mask_loadu_epi16(default_v.v, _ensure_mask_range_256(t, m).v, p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<float16_t, 16> t, const float16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_mask_loadu_epi16(default_v.v, _ensure_mask_range_256(t, m).v, p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<float32_t, 8> t, const float32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_mask_load_ps(default_v.v, _ensure_mask_range_256(t, m).v, p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<float64_t, 4> t, const float64_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_mask_load_pd(default_v.v, _ensure_mask_range_256(t, m).v, p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int8_t, 32> t, const int8_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_mask_loadu_epi8(default_v.v, _ensure_mask_range_256(t, m).v, p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint8_t, 32> t, const uint8_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_mask_loadu_epi8(default_v.v, _ensure_mask_range_256(t, m).v, p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int16_t, 16> t, const int16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_mask_loadu_epi16(default_v.v, _ensure_mask_range_256(t, m).v, p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint16_t, 16> t, const uint16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_mask_loadu_epi16(default_v.v, _ensure_mask_range_256(t, m).v, p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int32_t, 8> t, const int32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_mask_load_epi32(default_v.v, _ensure_mask_range_256(t, m).v, p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint32_t, 8> t, const uint32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_mask_load_epi32(default_v.v, _ensure_mask_range_256(t, m).v, p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int64_t, 4> t, const int64_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_mask_load_epi64(default_v.v, _ensure_mask_range_256(t, m).v, p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint64_t, 4> t, const uint64_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm256_mask_load_epi64(default_v.v, _ensure_mask_range_256(t, m).v, p); }
#else // HAS_AVX512DQ
// AVX2 fallback implementations for masked load
CT_ALWAYS_FORCEINLINE auto loadu(Tag<float32_t, 8> t, const float32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { auto mask = _ensure_mask_range_256(t, m).v; return _mm256_blendv_ps(default_v.v, _mm256_maskload_ps(p, mask), _mm256_castsi256_ps(mask)); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<float64_t, 4> t, const float64_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { auto mask = _ensure_mask_range_256(t, m).v; return _mm256_blendv_pd(default_v.v, _mm256_maskload_pd(p, mask), _mm256_castsi256_pd(mask)); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int32_t, 8> t, const int32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { auto mask = _ensure_mask_range_256(t, m).v; return _mm256_blendv_epi8(default_v.v, _mm256_maskload_epi32(p, mask), mask); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint32_t, 8> t, const uint32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { auto mask = _ensure_mask_range_256(t, m).v; return _mm256_blendv_epi8(default_v.v, _mm256_maskload_epi32((const int*)p, mask), mask); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int64_t, 4> t, const int64_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { auto mask = _ensure_mask_range_256(t, m).v; return _mm256_blendv_epi8(default_v.v, _mm256_maskload_epi64((const long long*)p, mask), mask); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint64_t, 4> t, const uint64_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { auto mask = _ensure_mask_range_256(t, m).v; return _mm256_blendv_epi8(default_v.v, _mm256_maskload_epi64((const long long*)p, mask), mask); }

// For 8-bit and 16-bit types, fallback to scalar or use page boundary check
template <typename T, nint_t N, int POW2>
CT_ALWAYS_FORCEINLINE auto _loadu_8_256(Tag<T, N, POW2> t, const T* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) {
  auto mask = _ensure_mask_range_256(t, m).v;
  // For 8-bit types, check page boundary and use full load if safe
  if (((nuint_t(p) & 0xfff) + 31) <= 0xfff) {
    return _mm256_blendv_epi8(default_v.v, _mm256_loadu_si256((const __m256i *)p), mask);
  } else {
    // Fallback to scalar implementation
    union { int8_t i[32]; __m256i m; } V{.m = default_v.v}, M{.m = mask};
    alignas(32) int8_t S[32];
    auto P = (const int8_t*) p;
    for (int i = 0; i < 32; ++i) S[i] = M.i[i] ? P[i] : V.i[i];
    return _mm256_load_si256((const __m256i *)S);
  }
}
template <typename T, nint_t N, int POW2>
CT_ALWAYS_FORCEINLINE auto _loadu_16_256(Tag<T, N, POW2> t, const T* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) {
  auto mask = _ensure_mask_range_256(t, m).v;
  if (((nuint_t(p) & 0xfff) + 31) <= 0xfff) {
    return _mm256_blendv_epi8(default_v.v, _mm256_loadu_si256((const __m256i *)p), mask);
  } else {
    // TODO slow
    // Fallback to scalar implementation
    union { int16_t i[8]; __m256i m; } V{.m = default_v.v}, M{.m = mask};
    alignas(32) int16_t S[8];
    auto P = (const int16_t *) p;
    for (int i = 0; i < 8; ++i) S[i] = M.i[i] ? P[i] : V.i[i];
    return _mm256_load_si256((const __m256i *)S);
  }
}
CT_ALWAYS_FORCEINLINE auto loadu(Tag<bfloat16_t, 16> t, const bfloat16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _loadu_16_256(t, p, m, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<float16_t, 16> t, const float16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _loadu_16_256(t, p, m, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int8_t, 32> t, const int8_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _loadu_8_256(t, p, m, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint8_t, 32> t, const uint8_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _loadu_8_256(t, p, m, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int16_t, 16> t, const int16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _loadu_16_256(t, p, m, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint16_t, 16> t, const uint16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _loadu_16_256(t, p, m, default_v); }

// load(const T* p, Mask<T> m, Vec<T> default_v) - aligned version
CT_ALWAYS_FORCEINLINE auto load(Tag<bfloat16_t, 16> t, const bfloat16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<float16_t, 16> t, const float16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<float32_t, 8> t, const float32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<float64_t, 4> t, const float64_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int8_t, 32> t, const int8_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint8_t, 32> t, const uint8_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int16_t, 16> t, const int16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint16_t, 16> t, const uint16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int32_t, 8> t, const int32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint32_t, 8> t, const uint32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int64_t, 4> t, const int64_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint64_t, 4> t, const uint64_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return word::loadu(t, p, m, default_v); }
#endif // HAS_AVX512DQ

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE auto _loadu_fwd_mask_256(Tag<T, N, P> t, const T* p, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = mwhilelt(t, 0, n);
  return word::loadu(t, p, m, default_v);
}

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE auto _load_fwd_mask_256(Tag<T, N, P> t, const T* p, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = mwhilelt(t, 0, n);
  return word::load(t, p, m, default_v);
}

// loadu(const T* p, nint_t n, Vec<T> default_v)
CT_ALWAYS_FORCEINLINE auto loadu(Tag<bfloat16_t, 16> t, const bfloat16_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_256(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<float16_t, 16> t, const float16_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_256(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<float32_t, 8> t, const float32_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_256(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<float64_t, 4> t, const float64_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_256(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int8_t, 32> t, const int8_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_256(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint8_t, 32> t, const uint8_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_256(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int16_t, 16> t, const int16_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_256(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint16_t, 16> t, const uint16_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_256(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int32_t, 8> t, const int32_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_256(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint32_t, 8> t, const uint32_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_256(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int64_t, 4> t, const int64_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_256(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint64_t, 4> t, const uint64_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_256(t, p, n, default_v); }

// load(const T* p, nint_t n, Vec<T> default_v)
CT_ALWAYS_FORCEINLINE auto load(Tag<bfloat16_t, 16> t, const bfloat16_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_256(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<float16_t, 16> t, const float16_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_256(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<float32_t, 8> t, const float32_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_256(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<float64_t, 4> t, const float64_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_256(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int8_t, 32> t, const int8_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_256(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint8_t, 32> t, const uint8_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_256(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int16_t, 16> t, const int16_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_256(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint16_t, 16> t, const uint16_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_256(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int32_t, 8> t, const int32_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_256(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint32_t, 8> t, const uint32_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_256(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int64_t, 4> t, const int64_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_256(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint64_t, 4> t, const uint64_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_256(t, p, n, default_v); }



// storeu(T* p, Vec<T> v)
CT_ALWAYS_FORCEINLINE auto storeu(Tag<bfloat16_t, 16> t, bfloat16_t* p, VecOf(t) v) -> void { _mm256_storeu_si256((__m256i *)p, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<float16_t, 16> t, float16_t* p, VecOf(t) v) -> void {
  #ifdef HAS_AVX512_FP16
  _mm256_storeu_ph(p, _mm256_castsi256_ph(v.v));
  #else
  _mm256_storeu_si256((__m256i *)p, v.v);
  #endif
}
CT_ALWAYS_FORCEINLINE auto storeu(Tag<float32_t, 8> t, float32_t* p, VecOf(t) v) -> void { _mm256_storeu_ps(p, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<float64_t, 4> t, float64_t* p, VecOf(t) v) -> void { _mm256_storeu_pd(p, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int8_t, 32> t, int8_t* p, VecOf(t) v) -> void { _mm256_storeu_si256((__m256i *)p, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint8_t, 32> t, uint8_t* p, VecOf(t) v) -> void { _mm256_storeu_si256((__m256i *)p, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int16_t, 16> t, int16_t* p, VecOf(t) v) -> void { _mm256_storeu_si256((__m256i *)p, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint16_t, 16> t, uint16_t* p, VecOf(t) v) -> void { _mm256_storeu_si256((__m256i *)p, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int32_t, 8> t, int32_t* p, VecOf(t) v) -> void { _mm256_storeu_si256((__m256i *)p, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint32_t, 8> t, uint32_t* p, VecOf(t) v) -> void { _mm256_storeu_si256((__m256i *)p, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int64_t, 4> t, int64_t* p, VecOf(t) v) -> void { _mm256_storeu_si256((__m256i *)p, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint64_t, 4> t, uint64_t* p, VecOf(t) v) -> void { _mm256_storeu_si256((__m256i *)p, v.v); }

// store(T* p, Vec<T> v)
CT_ALWAYS_FORCEINLINE auto store(Tag<bfloat16_t, 16> t, bfloat16_t* p, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_store_si256((__m256i *)p, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<float16_t, 16> t, float16_t* p, VecOf(t) v) -> void {
  TL_CHECK_ALIGN
  #ifdef HAS_AVX512_FP16
  _mm256_store_ph(p, _mm256_castsi256_ph(v.v));
  #else
  _mm256_store_si256((__m256i *)p, v.v);
  #endif
}
CT_ALWAYS_FORCEINLINE auto store(Tag<float32_t, 8> t, float32_t* p, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_store_ps(p, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<float64_t, 4> t, float64_t* p, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_store_pd(p, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int8_t, 32> t, int8_t* p, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_store_si256((__m256i *)p, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint8_t, 32> t, uint8_t* p, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_store_si256((__m256i *)p, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int16_t, 16> t, int16_t* p, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_store_si256((__m256i *)p, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint16_t, 16> t, uint16_t* p, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_store_si256((__m256i *)p, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int32_t, 8> t, int32_t* p, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_store_si256((__m256i *)p, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint32_t, 8> t, uint32_t* p, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_store_si256((__m256i *)p, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int64_t, 4> t, int64_t* p, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_store_si256((__m256i *)p, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint64_t, 4> t, uint64_t* p, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_store_si256((__m256i *)p, v.v); }

// storeu(T* p, Mask<T> m, Vec<T> v)
// store(T* p, Mask<T> m, Vec<T> v)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto storeu(Tag<bfloat16_t, 16> t, bfloat16_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm256_mask_storeu_epi16(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<float16_t, 16> t, float16_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm256_mask_storeu_epi16(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<float32_t, 8> t, float32_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm256_mask_storeu_ps(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<float64_t, 4> t, float64_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm256_mask_storeu_pd(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int8_t, 32> t, int8_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm256_mask_storeu_epi8(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint8_t, 32> t, uint8_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm256_mask_storeu_epi8(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int16_t, 16> t, int16_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm256_mask_storeu_epi16(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint16_t, 16> t, uint16_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm256_mask_storeu_epi16(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int32_t, 8> t, int32_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm256_mask_storeu_epi32(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint32_t, 8> t, uint32_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm256_mask_storeu_epi32(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int64_t, 4> t, int64_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm256_mask_storeu_epi64(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint64_t, 4> t, uint64_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm256_mask_storeu_epi64(p, m.v, v.v); }
// Note: no aligned store for epi8 and epi16
CT_ALWAYS_FORCEINLINE auto store(Tag<bfloat16_t, 16> t, bfloat16_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_mask_storeu_epi16(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<float16_t, 16> t, float16_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_mask_storeu_epi16(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<float32_t, 8> t, float32_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_mask_store_ps(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<float64_t, 4> t, float64_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_mask_store_pd(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int8_t, 32> t, int8_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_mask_storeu_epi8(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint8_t, 32> t, uint8_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_mask_storeu_epi8(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int16_t, 16> t, int16_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_mask_storeu_epi16(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint16_t, 16> t, uint16_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_mask_storeu_epi16(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int32_t, 8> t, int32_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_mask_store_epi32(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint32_t, 8> t, uint32_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_mask_store_epi32(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int64_t, 4> t, int64_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_mask_store_epi64(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint64_t, 4> t, uint64_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm256_mask_store_epi64(p, m.v, v.v); }
#else // HAS_AVX512DQ
// split to 2 maskmoveu call
CT_ALWAYS_FORCEINLINE auto storeu(Tag<bfloat16_t, 16> t, bfloat16_t* p, MaskOf(t) m, VecOf(t) v) -> void {
  _mm_maskmoveu_si128(_mm256_castsi256_si128(v.v), _mm256_castsi256_si128(m.v), (char *)p);
  _mm_maskmoveu_si128(_mm256_extractf128_si256(v.v, 1), _mm256_extractf128_si256(m.v, 1), (char *)p + 16);
}
CT_ALWAYS_FORCEINLINE auto storeu(Tag<float16_t, 16> t, float16_t* p, MaskOf(t) m, VecOf(t) v) -> void {
  _mm_maskmoveu_si128(_mm256_castsi256_si128(v.v), _mm256_castsi256_si128(m.v), (char *)p);
  _mm_maskmoveu_si128(_mm256_extractf128_si256(v.v, 1), _mm256_extractf128_si256(m.v, 1), (char *)p + 16);
}
// AVX2 fallback implementations for masked store
CT_ALWAYS_FORCEINLINE auto storeu(Tag<float32_t, 8> t, float32_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm256_maskstore_ps(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<float64_t, 4> t, float64_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm256_maskstore_pd(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int8_t, 32> t, int8_t* p, MaskOf(t) m, VecOf(t) v) -> void {
  _mm_maskmoveu_si128(_mm256_castsi256_si128(v.v), _mm256_castsi256_si128(m.v), (char *)p);
  _mm_maskmoveu_si128(_mm256_extractf128_si256(v.v, 1), _mm256_extractf128_si256(m.v, 1), (char *)p + 16);
}
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint8_t, 32> t, uint8_t* p, MaskOf(t) m, VecOf(t) v) -> void {
  _mm_maskmoveu_si128(_mm256_castsi256_si128(v.v), _mm256_castsi256_si128(m.v), (char *)p);
  _mm_maskmoveu_si128(_mm256_extractf128_si256(v.v, 1), _mm256_extractf128_si256(m.v, 1), (char *)p + 16);
}
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int16_t, 16> t, int16_t* p, MaskOf(t) m, VecOf(t) v) -> void {
  _mm_maskmoveu_si128(_mm256_castsi256_si128(v.v), _mm256_castsi256_si128(m.v), (char *)p);
  _mm_maskmoveu_si128(_mm256_extractf128_si256(v.v, 1), _mm256_extractf128_si256(m.v, 1), (char *)p + 16);
}
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint16_t, 16> t, uint16_t* p, MaskOf(t) m, VecOf(t) v) -> void {
  _mm_maskmoveu_si128(_mm256_castsi256_si128(v.v), _mm256_castsi256_si128(m.v), (char *)p);
  _mm_maskmoveu_si128(_mm256_extractf128_si256(v.v, 1), _mm256_extractf128_si256(m.v, 1), (char *)p + 16);
}
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int32_t, 8> t, int32_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm256_maskstore_epi32(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint32_t, 8> t, uint32_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm256_maskstore_epi32((int*)p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int64_t, 4> t, int64_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm256_maskstore_epi64((long long*)p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint64_t, 4> t, uint64_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm256_maskstore_epi64((long long*)p, m.v, v.v); }

// store aligned - forward to unaligned
CT_ALWAYS_FORCEINLINE auto store(Tag<bfloat16_t, 16> t, bfloat16_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<float16_t, 16> t, float16_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<float32_t, 8> t, float32_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<float64_t, 4> t, float64_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int8_t, 32> t, int8_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint8_t, 32> t, uint8_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int16_t, 16> t, int16_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint16_t, 16> t, uint16_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int32_t, 8> t, int32_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint32_t, 8> t, uint32_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int64_t, 4> t, int64_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint64_t, 4> t, uint64_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN word::storeu(t, p, m, v); }
#endif // HAS_AVX512DQ

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE auto _storeu_fwd_mask_256(Tag<T, N, P> t, T* p, nint_t n, VecOf(t) v) -> void {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = mwhilelt(t, 0, n);
  word::storeu(t, p, m, v);
}

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE auto _store_fwd_mask_256(Tag<T, N, P> t, T* p, nint_t n, VecOf(t) v) -> void {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = mwhilelt(t, 0, n);
  word::store(t, p, m, v);
}

// storeu(T* p, nint_t n, Vec<T> v)
CT_ALWAYS_FORCEINLINE auto storeu(Tag<bfloat16_t, 16> t, bfloat16_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_256(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<float16_t, 16> t, float16_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_256(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<float32_t, 8> t, float32_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_256(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<float64_t, 4> t, float64_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_256(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int8_t, 32> t, int8_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_256(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint8_t, 32> t, uint8_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_256(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int16_t, 16> t, int16_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_256(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint16_t, 16> t, uint16_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_256(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int32_t, 8> t, int32_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_256(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint32_t, 8> t, uint32_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_256(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int64_t, 4> t, int64_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_256(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint64_t, 4> t, uint64_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_256(t, p, n, v); }

// store(T* p, nint_t n, Vec<T> v)
CT_ALWAYS_FORCEINLINE auto store(Tag<bfloat16_t, 16> t, bfloat16_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_256(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<float16_t, 16> t, float16_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_256(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<float32_t, 8> t, float32_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_256(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<float64_t, 4> t, float64_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_256(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int8_t, 32> t, int8_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_256(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint8_t, 32> t, uint8_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_256(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int16_t, 16> t, int16_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_256(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint16_t, 16> t, uint16_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_256(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int32_t, 8> t, int32_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_256(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint32_t, 8> t, uint32_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_256(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int64_t, 4> t, int64_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_256(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint64_t, 4> t, uint64_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_256(t, p, n, v); }



/* ************************************************************************** */
//                             Get / set element                              //
/* ************************************************************************** */

// Helper: extract element from __m256i
#define TL_DEFINE_EXTRACT_256(dtype, postfix) static CT_ALWAYS_FORCEINLINE dtype _extract256_##postfix(__m256i v, int index)
#ifdef HAS_AVX512F
TL_DEFINE_EXTRACT_256(int8_t, i8) {
return (int8_t)_mm256_cvtsi256_si32(_mm256_permutex2var_epi8(v, _mm256_castsi128_si256(_mm_cvtsi32_si128(int(index))), v)) & 0xff;
}
TL_DEFINE_EXTRACT_256(uint8_t, u8) { return (uint8_t) _extract256_i8(v, index); }
TL_DEFINE_EXTRACT_256(int16_t, i16) {
return (int16_t)_mm256_cvtsi256_si32(_mm256_permutex2var_epi16(v, _mm256_castsi128_si256(_mm_cvtsi32_si128(int(index))), v));
}
TL_DEFINE_EXTRACT_256(uint16_t, u16) { return (uint16_t) _extract256_i16(v, index); }
TL_DEFINE_EXTRACT_256(int32_t, i32) {
return _mm256_cvtsi256_si32(_mm256_permutex2var_epi32(v, _mm256_castsi128_si256(_mm_cvtsi32_si128(int(index))), v));
}
TL_DEFINE_EXTRACT_256(uint32_t, u32) { return (uint32_t) _extract256_i32(v, index); }
TL_DEFINE_EXTRACT_256(int64_t, i64) {
return _mm_cvtsi128_si64(_mm256_castsi256_si128(_mm256_permutex2var_epi64(v, _mm256_castsi128_si256(_mm_cvtsi32_si128(int(index))), v)));
}
TL_DEFINE_EXTRACT_256(uint64_t, u64) { return (uint64_t) _extract256_i64(v, index); }
#else // HAS_AVX512F
TL_DEFINE_EXTRACT_256(int8_t, i8) {
  alignas(32) int8_t data[32];
  _mm256_store_si256((__m256i *) data, v);
  return data[index];
}
TL_DEFINE_EXTRACT_256(uint8_t, u8) { return (uint8_t) _extract256_i8(v, index); }
TL_DEFINE_EXTRACT_256(int16_t, i16) {
  alignas(32) int16_t data[16];
  _mm256_store_si256((__m256i *) data, v);
  return data[index];
}
TL_DEFINE_EXTRACT_256(uint16_t, u16) { return (uint16_t) _extract256_i16(v, index); }
TL_DEFINE_EXTRACT_256(int32_t, i32) {
  alignas(32) int32_t data[8];
  _mm256_store_si256((__m256i *) data, v);
  return data[index];
}
TL_DEFINE_EXTRACT_256(uint32_t, u32) { return (uint32_t) _extract256_i32(v, index); }
TL_DEFINE_EXTRACT_256(int64_t, i64) {
  alignas(32) int64_t data[4];
  _mm256_store_si256((__m256i *) data, v);
  return data[index];
}
TL_DEFINE_EXTRACT_256(uint64_t, u64) { return (uint64_t) _extract256_i64(v, index); }
#endif // HAS_AVX512F
#undef TL_DEFINE_EXTRACT_256

// Helper: get element from mask
#ifdef HAS_AVX512DQ
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit_256(__mmask8 m, int index) { return (_cvtmask8_u32(m) >> index) & 1; }
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit_256(__mmask16 m, int index) { return (_cvtmask16_u32(m) >> index) & 1; }
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit_256(__mmask32 m, int index) { return (_cvtmask32_u32(m) >> index) & 1; }
#else
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit_256_epi8(__m256i m, int index) { return !!_extract256_i8(m, index); }
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit_256_epi16(__m256i m, int index) { return !!_extract256_i16(m, index); }
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit_256_epi32(__m256i m, int index) { return !!_extract256_i32(m, index); }
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit_256_epi64(__m256i m, int index) { return !!_extract256_i64(m, index); }
#endif

// Helper: set element in mask
#ifdef HAS_AVX512DQ
static CT_ALWAYS_FORCEINLINE __mmask8 _set_mask_bit_256(__mmask8 m, int index, bool x) {
  uint32_t bits = _cvtmask8_u32(m);
  bits = (bits & ~(1u << index)) | ((x ? 1u : 0u) << index);
  return _cvtu32_mask8(bits);
}
static CT_ALWAYS_FORCEINLINE __mmask16 _set_mask_bit_256(__mmask16 m, int index, bool x) {
  uint32_t bits = _cvtmask16_u32(m);
  bits = (bits & ~(1u << index)) | ((x ? 1u : 0u) << index);
  return _cvtu32_mask16(bits);
}
static CT_ALWAYS_FORCEINLINE __mmask32 _set_mask_bit_256(__mmask32 m, int index, bool x) {
  uint32_t bits = _cvtmask32_u32(m);
  bits = (bits & ~(1u << index)) | ((x ? 1u : 0u) << index);
  return _cvtu32_mask32(bits);
}
#else
static CT_ALWAYS_FORCEINLINE __m256i _set_mask_bit_256_epi8(__m256i m, int index, bool x) {
  alignas(32) int8_t data[32];
  _mm256_store_si256((__m256i *) data, m);
  data[index] = x ? -1 : 0;
  return _mm256_load_si256((__m256i *) data);
}
static CT_ALWAYS_FORCEINLINE __m256i _set_mask_bit_256_epi16(__m256i m, int index, bool x) {
  alignas(32) int16_t data[16];
  _mm256_store_si256((__m256i *) data, m);
  data[index] = x ? -1 : 0;
  return _mm256_load_si256((__m256i *) data);
}
static CT_ALWAYS_FORCEINLINE __m256i _set_mask_bit_256_epi32(__m256i m, int index, bool x) {
  alignas(32) int32_t data[8];
  _mm256_store_si256((__m256i *) data, m);
  data[index] = x ? -1 : 0;
  return _mm256_load_si256((__m256i *) data);
}
static CT_ALWAYS_FORCEINLINE __m256i _set_mask_bit_256_epi64(__m256i m, int index, bool x) {
  alignas(32) int64_t data[4];
  _mm256_store_si256((__m256i *) data, m);
  data[index] = x ? -1LL : 0LL;
  return _mm256_load_si256((__m256i *) data);
}
#endif



// get(Vec<T> v, nint_t index)
CT_ALWAYS_FORCEINLINE auto get(Tag<bfloat16_t, 16> t, VecOf(t) v, nint_t index) -> bfloat16_t { TL_CHECK_INDEX(index); union { bfloat16_t b; int16_t i; } u; u.i = _extract256_i16(v.v, (int) index); return u.b; }
CT_ALWAYS_FORCEINLINE auto get(Tag<float16_t, 16> t, VecOf(t) v, nint_t index) -> float16_t { TL_CHECK_INDEX(index); union { float16_t h; int16_t i; } u; u.i = _extract256_i16(v.v, (int) index); return u.h; }
CT_ALWAYS_FORCEINLINE auto get(Tag<float32_t, 8> t, VecOf(t) v, nint_t index) -> float32_t { TL_CHECK_INDEX(index); return _mm256_cvtss_f32(_mm256_permutevar8x32_ps(v.v, _mm256_set1_epi32((int)index))); }
CT_ALWAYS_FORCEINLINE auto get(Tag<float64_t, 4> t, VecOf(t) v, nint_t index) -> float64_t { TL_CHECK_INDEX(index);
  auto two_index = index << 1;
  auto idx_vec = _mm256_castsi128_si256(_mm_cvtsi64_si128(((two_index | 1) << 32) | two_index));
  __m256 perm = _mm256_permutevar8x32_ps(_mm256_castpd_ps(v.v), idx_vec);
  __m128d result = _mm_castps_pd(_mm256_castps256_ps128(perm));
  return _mm_cvtsd_f64(result);
}
CT_ALWAYS_FORCEINLINE auto get(Tag<int8_t, 32> t, VecOf(t) v, nint_t index) -> int8_t { TL_CHECK_INDEX(index); return _extract256_i8(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<uint8_t, 32> t, VecOf(t) v, nint_t index) -> uint8_t { TL_CHECK_INDEX(index); return _extract256_u8(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<int16_t, 16> t, VecOf(t) v, nint_t index) -> int16_t { TL_CHECK_INDEX(index); return _extract256_i16(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<uint16_t, 16> t, VecOf(t) v, nint_t index) -> uint16_t { TL_CHECK_INDEX(index); return _extract256_u16(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<int32_t, 8> t, VecOf(t) v, nint_t index) -> int32_t { TL_CHECK_INDEX(index); return _extract256_i32(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<uint32_t, 8> t, VecOf(t) v, nint_t index) -> uint32_t { TL_CHECK_INDEX(index); return _extract256_u32(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<int64_t, 4> t, VecOf(t) v, nint_t index) -> int64_t { TL_CHECK_INDEX(index); return _extract256_i64(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<uint64_t, 4> t, VecOf(t) v, nint_t index) -> uint64_t { TL_CHECK_INDEX(index); return _extract256_u64(v.v, (int) index); }

// get(Mask<T> m, nint_t index)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto get(Tag<bfloat16_t, 16> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<float16_t, 16> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<float32_t, 8> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<float64_t, 4> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<int8_t, 32> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<uint8_t, 32> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<int16_t, 16> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<uint16_t, 16> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<int32_t, 8> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<uint32_t, 8> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<int64_t, 4> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<uint64_t, 4> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256(v.v, (int) index); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto get(Tag<bfloat16_t, 16> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256_epi16(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<float16_t, 16> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256_epi16(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<float32_t, 8> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256_epi32(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<float64_t, 4> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256_epi64(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<int8_t, 32> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256_epi8(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<uint8_t, 32> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256_epi8(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<int16_t, 16> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256_epi16(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<uint16_t, 16> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256_epi16(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<int32_t, 8> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256_epi32(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<uint32_t, 8> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256_epi32(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<int64_t, 4> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256_epi64(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<uint64_t, 4> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_256_epi64(v.v, (int) index); }
#endif // HAS_AVX512DQ



// set(Vec<T> v, nint_t index, T x)
#if defined(HAS_AVX512DQ) && defined(HAS_AVX512BW) && defined(HAS_AVX512VL)
CT_ALWAYS_FORCEINLINE auto set(Tag<bfloat16_t, 16> t, VecOf(t) v, nint_t index, bfloat16_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  union { bfloat16_t f; int16_t i; } u { .f = x };
  auto mask = _cvtu32_mask16(1u << (unsigned) index);
  return _mm256_mask_mov_epi16(v.v, mask, _mm256_set1_epi16(u.i));
}
CT_ALWAYS_FORCEINLINE auto set(Tag<float16_t, 16> t, VecOf(t) v, nint_t index, float16_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  union { float16_t f; int16_t i; } u { .f = x };
  auto mask = _cvtu32_mask16(1u << (unsigned) index);
  return _mm256_mask_mov_epi16(v.v, mask, _mm256_set1_epi16(u.i));
}
CT_ALWAYS_FORCEINLINE auto set(Tag<float32_t, 8> t, VecOf(t) v, nint_t index, float32_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask8(1u << (unsigned) index);
  return _mm256_mask_mov_ps(v.v, mask, _mm256_set1_ps(x));
}
CT_ALWAYS_FORCEINLINE auto set(Tag<float64_t, 4> t, VecOf(t) v, nint_t index, float64_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask8(1u << (unsigned) index);
  return _mm256_mask_mov_pd(v.v, mask, _mm256_set1_pd(x));
}
CT_ALWAYS_FORCEINLINE auto set(Tag<int8_t, 32> t, VecOf(t) v, nint_t index, int8_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask32(1u << (unsigned) index);
  return _mm256_mask_mov_epi8(v.v, mask, _mm256_set1_epi8(x));
}
CT_ALWAYS_FORCEINLINE auto set(Tag<uint8_t, 32> t, VecOf(t) v, nint_t index, uint8_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask32(1u << (unsigned) index);
  return _mm256_mask_mov_epi8(v.v, mask, _mm256_set1_epi8((int8_t)x));
}
CT_ALWAYS_FORCEINLINE auto set(Tag<int16_t, 16> t, VecOf(t) v, nint_t index, int16_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask16(1u << (unsigned) index);
  return _mm256_mask_mov_epi16(v.v, mask, _mm256_set1_epi16(x));
}
CT_ALWAYS_FORCEINLINE auto set(Tag<uint16_t, 16> t, VecOf(t) v, nint_t index, uint16_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask16(1u << (unsigned) index);
  return _mm256_mask_mov_epi16(v.v, mask, _mm256_set1_epi16((int16_t)x));
}
CT_ALWAYS_FORCEINLINE auto set(Tag<int32_t, 8> t, VecOf(t) v, nint_t index, int32_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask8(1u << (unsigned) index);
  return _mm256_mask_mov_epi32(v.v, mask, _mm256_set1_epi32(x));
}
CT_ALWAYS_FORCEINLINE auto set(Tag<uint32_t, 8> t, VecOf(t) v, nint_t index, uint32_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask8(1u << (unsigned) index);
  return _mm256_mask_mov_epi32(v.v, mask, _mm256_set1_epi32((int32_t)x));
}
CT_ALWAYS_FORCEINLINE auto set(Tag<int64_t, 4> t, VecOf(t) v, nint_t index, int64_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask8(1u << (unsigned) index);
  return _mm256_mask_mov_epi64(v.v, mask, _mm256_set1_epi64x(x));
}
CT_ALWAYS_FORCEINLINE auto set(Tag<uint64_t, 4> t, VecOf(t) v, nint_t index, uint64_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask8(1u << (unsigned) index);
  return _mm256_mask_mov_epi64(v.v, mask, _mm256_set1_epi64x((int64_t)x));
}
#else // defined(HAS_AVX512DQ) && defined(HAS_AVX512BW) && defined(HAS_AVX512VL)
// Scalar fallback for set
CT_ALWAYS_FORCEINLINE auto set(Tag<bfloat16_t, 16> t, VecOf(t) v, nint_t index, bfloat16_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  alignas(32) bfloat16_t data[16];
  _mm256_store_si256((__m256i *)data, v.v);
  data[index] = x;
  return _mm256_load_si256((__m256i *)data);
}
CT_ALWAYS_FORCEINLINE auto set(Tag<float16_t, 16> t, VecOf(t) v, nint_t index, float16_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  alignas(32) float16_t data[16];
  _mm256_store_si256((__m256i *)data, v.v);
  data[index] = x;
  return _mm256_load_si256((__m256i *)data);
}
CT_ALWAYS_FORCEINLINE auto set(Tag<float32_t, 8> t, VecOf(t) v, nint_t index, float32_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  alignas(32) float32_t data[8];
  _mm256_store_ps(data, v.v);
  data[index] = x;
  return _mm256_load_ps(data);
}
CT_ALWAYS_FORCEINLINE auto set(Tag<float64_t, 4> t, VecOf(t) v, nint_t index, float64_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  alignas(32) float64_t data[4];
  _mm256_store_pd(data, v.v);
  data[index] = x;
  return _mm256_load_pd(data);
}
CT_ALWAYS_FORCEINLINE auto set(Tag<int8_t, 32> t, VecOf(t) v, nint_t index, int8_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  alignas(32) int8_t data[32];
  _mm256_store_si256((__m256i *)data, v.v);
  data[index] = x;
  return _mm256_load_si256((__m256i *)data);
}
CT_ALWAYS_FORCEINLINE auto set(Tag<uint8_t, 32> t, VecOf(t) v, nint_t index, uint8_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  alignas(32) uint8_t data[32];
  _mm256_store_si256((__m256i *)data, v.v);
  data[index] = x;
  return _mm256_load_si256((__m256i *)data);
}
CT_ALWAYS_FORCEINLINE auto set(Tag<int16_t, 16> t, VecOf(t) v, nint_t index, int16_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  alignas(32) int16_t data[16];
  _mm256_store_si256((__m256i *)data, v.v);
  data[index] = x;
  return _mm256_load_si256((__m256i *)data);
}
CT_ALWAYS_FORCEINLINE auto set(Tag<uint16_t, 16> t, VecOf(t) v, nint_t index, uint16_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  alignas(32) uint16_t data[16];
  _mm256_store_si256((__m256i *)data, v.v);
  data[index] = x;
  return _mm256_load_si256((__m256i *)data);
}
CT_ALWAYS_FORCEINLINE auto set(Tag<int32_t, 8> t, VecOf(t) v, nint_t index, int32_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  alignas(32) int32_t data[8];
  _mm256_store_si256((__m256i *)data, v.v);
  data[index] = x;
  return _mm256_load_si256((__m256i *)data);
}
CT_ALWAYS_FORCEINLINE auto set(Tag<uint32_t, 8> t, VecOf(t) v, nint_t index, uint32_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  alignas(32) uint32_t data[8];
  _mm256_store_si256((__m256i *)data, v.v);
  data[index] = x;
  return _mm256_load_si256((__m256i *)data);
}
CT_ALWAYS_FORCEINLINE auto set(Tag<int64_t, 4> t, VecOf(t) v, nint_t index, int64_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  alignas(32) int64_t data[4];
  _mm256_store_si256((__m256i *)data, v.v);
  data[index] = x;
  return _mm256_load_si256((__m256i *)data);
}
CT_ALWAYS_FORCEINLINE auto set(Tag<uint64_t, 4> t, VecOf(t) v, nint_t index, uint64_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  alignas(32) uint64_t data[4];
  _mm256_store_si256((__m256i *)data, v.v);
  data[index] = x;
  return _mm256_load_si256((__m256i *)data);
}
#endif // defined(HAS_AVX512DQ) && defined(HAS_AVX512BW) && defined(HAS_AVX512VL)

// set(Mask<T> m, nint_t index, bool x)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto set(Tag<bfloat16_t, 16> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<float16_t, 16> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<float32_t, 8> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<float64_t, 4> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<int8_t, 32> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<uint8_t, 32> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<int16_t, 16> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<uint16_t, 16> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<int32_t, 8> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<uint32_t, 8> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<int64_t, 4> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<uint64_t, 4> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256(v.v, (int) index, x); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto set(Tag<bfloat16_t, 16> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256_epi16(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<float16_t, 16> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256_epi16(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<float32_t, 8> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256_epi32(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<float64_t, 4> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256_epi64(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<int8_t, 32> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256_epi8(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<uint8_t, 32> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256_epi8(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<int16_t, 16> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256_epi16(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<uint16_t, 16> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256_epi16(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<int32_t, 8> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256_epi32(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<uint32_t, 8> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256_epi32(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<int64_t, 4> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256_epi64(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<uint64_t, 4> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_256_epi64(v.v, (int) index, x); }
#endif // HAS_AVX512DQ



/* ************************************************************************** */
//                       Basic arithmetic operations                          //
/* ************************************************************************** */
// TODO arithmetics for float16_t and bfloat16_t


// add(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto add(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_add_ps(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_add_pd(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_add_epi8(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_add_epi8(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_add_epi16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_add_epi16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_add_epi32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_add_epi32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_add_epi64(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_add_epi64(a.v, b.v); }

// add(Vec<T> a, Vec<T> b, Mask<T> m)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto add(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_add_ps(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_add_pd(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_add_epi8(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_add_epi8(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_add_epi16(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_add_epi16(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_add_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_add_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_add_epi64(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_add_epi64(a.v, m.v, a.v, b.v); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto add(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_ps(a.v, add(t, a.v, b.v).v, _mm256_castsi256_ps(m.v)); }
CT_ALWAYS_FORCEINLINE auto add(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_pd(a.v, add(t, a.v, b.v).v, _mm256_castsi256_pd(m.v)); }
CT_ALWAYS_FORCEINLINE auto add(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, add(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, add(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, add(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, add(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, add(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, add(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, add(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, add(t, a.v, b.v).v, m.v); }
#endif //HAS_AVX512DQ



// sub(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto sub(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_sub_ps(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_sub_pd(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_sub_epi8(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_sub_epi8(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_sub_epi16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_sub_epi16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_sub_epi32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_sub_epi32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_sub_epi64(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_sub_epi64(a.v, b.v); }

// sub(Vec<T> a, Vec<T> b, Mask<T> m)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto sub(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_sub_ps(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_sub_pd(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_sub_epi8(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_sub_epi8(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_sub_epi16(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_sub_epi16(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_sub_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_sub_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_sub_epi64(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_sub_epi64(a.v, m.v, a.v, b.v); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto sub(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_ps(a.v, sub(t, a.v, b.v).v, _mm256_castsi256_ps(m.v)); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_pd(a.v, sub(t, a.v, b.v).v, _mm256_castsi256_pd(m.v)); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, sub(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, sub(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, sub(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, sub(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, sub(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, sub(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, sub(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, sub(t, a.v, b.v).v, m.v); }
#endif //HAS_AVX512DQ



// mul(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto mul(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_mul_ps(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_mul_pd(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) {
  // 8-bit multiplication using 16-bit intermediate
  auto zero = _mm256_setzero_si256();
  auto a_lo = _mm256_unpacklo_epi8(a.v, zero);
  auto a_hi = _mm256_unpackhi_epi8(a.v, zero);
  auto b_lo = _mm256_unpacklo_epi8(b.v, zero);
  auto b_hi = _mm256_unpackhi_epi8(b.v, zero);
  a_lo = _mm256_mullo_epi16(a_lo, b_lo);
  a_hi = _mm256_mullo_epi16(a_hi, b_hi);
  return _mm256_packus_epi16(a_lo, a_hi);
}
CT_ALWAYS_FORCEINLINE auto mul(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return mul(Tag<int8_t, 32>(), a.v, b.v).v; }
CT_ALWAYS_FORCEINLINE auto mul(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_mullo_epi16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_mullo_epi16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_mullo_epi32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_mullo_epi32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_mullo_epi64(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_mullo_epi64(a.v, b.v); }

// mul(Vec<T> a, Vec<T> b, Mask<T> m)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto mul(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_mul_ps(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_mul_pd(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_blend_epi8(m.v, a.v, mul(t, a.v, b.v).v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_blend_epi8(m.v, a.v, mul(t, a.v, b.v).v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_mullo_epi16(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_mullo_epi16(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_mullo_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_mullo_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_mullo_epi64(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_mullo_epi64(a.v, m.v, a.v, b.v); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto mul(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_ps(a.v, mul(t, a.v, b.v).v, _mm256_castsi256_ps(m.v)); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_pd(a.v, mul(t, a.v, b.v).v, _mm256_castsi256_pd(m.v)); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, mul(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, mul(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, mul(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, mul(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, mul(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, mul(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, mul(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, mul(t, a.v, b.v).v, m.v); }
#endif //HAS_AVX512DQ



// div(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto div(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_div_ps(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto div(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_div_pd(a.v, b.v); }

// div(Vec<T> a, Vec<T> b, Mask<T> m)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto div(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_div_ps(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto div(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_div_pd(a.v, m.v, a.v, b.v); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto div(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_ps(a.v, div(t, a, b).v, _mm256_castsi256_ps(m.v)); }
CT_ALWAYS_FORCEINLINE auto div(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_pd(a.v, div(t, a, b).v, _mm256_castsi256_pd(m.v)); }
#endif // HAS_AVX512DQ



// rcp(Vec<T> v)
#ifdef HAS_AVX512F
CT_ALWAYS_FORCEINLINE auto rcp(Tag<float32_t, 8> t, VecOf(t) v) -> VecOf(t) { return _mm256_rcp14_ps(v.v); }
CT_ALWAYS_FORCEINLINE auto rcp(Tag<float64_t, 4> t, VecOf(t) v) -> VecOf(t) { return _mm256_rcp14_pd(v.v); }
#else // HAS_AVX512F
CT_ALWAYS_FORCEINLINE auto rcp(Tag<float32_t, 8> t, VecOf(t) v) -> VecOf(t) { return _mm256_rcp_ps(v.v); }
CT_ALWAYS_FORCEINLINE auto rcp(Tag<float64_t, 4> t, VecOf(t) v) -> VecOf(t) { return div(t, fill(t, 1), v.v); }
#endif // HAS_AVX512F

// rcp(Vec<T> v, Mask<T> m, Vec<T> default_v)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto rcp(Tag<float32_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_rcp14_ps(default_v.v, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto rcp(Tag<float64_t, 4> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_rcp14_pd(default_v.v, m.v, v.v); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto rcp(Tag<float32_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_ps(default_v.v, rcp(t, v).v, _mm256_castsi256_ps(m.v)); }
CT_ALWAYS_FORCEINLINE auto rcp(Tag<float64_t, 4> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_pd(default_v.v, rcp(t, v).v, _mm256_castsi256_pd(m.v)); }
#endif // HAS_AVX512DQ



// max(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto max(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_max_ps(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_max_pd(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_max_epi8(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_max_epu8(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_max_epi16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_max_epu16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_max_epi32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_max_epu32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_max_epi64(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_max_epu64(a.v, b.v); }

// max(Vec<T> a, Vec<T> b, Mask<T> m)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto max(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_max_ps(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_max_pd(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_max_epi8(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_max_epu8(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_max_epi16(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_max_epu16(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_max_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_max_epu32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_max_epi64(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_max_epu64(a.v, m.v, a.v, b.v); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto max(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_ps(a.v, max(t, a.v, b.v).v, _mm256_castsi256_ps(m.v)); }
CT_ALWAYS_FORCEINLINE auto max(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_pd(a.v, max(t, a.v, b.v).v, _mm256_castsi256_pd(m.v)); }
CT_ALWAYS_FORCEINLINE auto max(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, max(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, max(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, max(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, max(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, max(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, max(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, max(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, max(t, a.v, b.v).v, m.v); }
#endif //HAS_AVX512DQ



// min(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto min(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_min_ps(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_min_pd(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_min_epi8(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_min_epu8(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_min_epi16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_min_epu16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_min_epi32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_min_epu32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_min_epi64(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_min_epu64(a.v, b.v); }

// min(Vec<T> a, Vec<T> b, Mask<T> m)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto min(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_min_ps(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_min_pd(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_min_epi8(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_min_epu8(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_min_epi16(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_min_epu16(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_min_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_min_epu32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_min_epi64(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_min_epu64(a.v, m.v, a.v, b.v); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto min(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_ps(a.v, min(t, a.v, b.v).v, _mm256_castsi256_ps(m.v)); }
CT_ALWAYS_FORCEINLINE auto min(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_pd(a.v, min(t, a.v, b.v).v, _mm256_castsi256_pd(m.v)); }
CT_ALWAYS_FORCEINLINE auto min(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, min(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, min(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, min(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, min(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, min(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, min(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, min(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, min(t, a.v, b.v).v, m.v); }
#endif //HAS_AVX512DQ



// bit_and(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_and_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_and_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_and_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_and_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_and_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_and_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_and_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_and_si256(a.v, b.v); }

// bit_and(Vec<T> a, Vec<T> b, Mask<T> m)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_blend_epi8(m.v, a.v, _mm256_and_si256(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_blend_epi8(m.v, a.v, _mm256_and_si256(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_blend_epi16(m.v, a.v, _mm256_and_si256(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_blend_epi16(m.v, a.v, _mm256_and_si256(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_and_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_and_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_and_epi64(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_and_epi64(a.v, m.v, a.v, b.v); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_and(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_and(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_and(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_and(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_and(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_and(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_and(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_and(t, a.v, b.v).v, m.v); }
#endif //HAS_AVX512DQ



// bit_or(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_or_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_or_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_or_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_or_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_or_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_or_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_or_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_or_si256(a.v, b.v); }

// bit_or(Vec<T> a, Vec<T> b, Mask<T> m)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_blend_epi8(m.v, a.v, _mm256_or_si256(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_blend_epi8(m.v, a.v, _mm256_or_si256(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_blend_epi16(m.v, a.v, _mm256_or_si256(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_blend_epi16(m.v, a.v, _mm256_or_si256(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_or_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_or_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_or_epi64(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_or_epi64(a.v, m.v, a.v, b.v); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_or(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_or(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_or(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_or(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_or(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_or(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_or(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_or(t, a.v, b.v).v, m.v); }
#endif //HAS_AVX512DQ



// bit_xor(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_xor_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_xor_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_xor_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_xor_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_xor_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_xor_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_xor_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_xor_si256(a.v, b.v); }

// bit_xor(Vec<T> a, Vec<T> b, Mask<T> m)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_blend_epi8(m.v, a.v, _mm256_xor_si256(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_blend_epi8(m.v, a.v, _mm256_xor_si256(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_blend_epi16(m.v, a.v, _mm256_xor_si256(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_blend_epi16(m.v, a.v, _mm256_xor_si256(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_xor_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_xor_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_xor_epi64(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_xor_epi64(a.v, m.v, a.v, b.v); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_xor(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_xor(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_xor(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_xor(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_xor(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_xor(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_xor(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_xor(t, a.v, b.v).v, m.v); }
#endif //HAS_AVX512DQ



// bit_andnot(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_andnot_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_andnot_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_andnot_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_andnot_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_andnot_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_andnot_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_andnot_si256(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm256_andnot_si256(a.v, b.v); }

// bit_andnot(Vec<T> a, Vec<T> b, Mask<T> m)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_blend_epi8(m.v, a.v, _mm256_andnot_si256(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_blend_epi8(m.v, a.v, _mm256_andnot_si256(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_blend_epi16(m.v, a.v, _mm256_andnot_si256(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_blend_epi16(m.v, a.v, _mm256_andnot_si256(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_andnot_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_andnot_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_andnot_epi64(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_andnot_epi64(a.v, m.v, a.v, b.v); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_andnot(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_andnot(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_andnot(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_andnot(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_andnot(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_andnot(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_andnot(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(a.v, bit_andnot(t, a.v, b.v).v, m.v); }
#endif //HAS_AVX512DQ



// bit_not(Vec<T> v)
#define _tlmm_vec_not256(a) _mm256_xor_si256((a), _mm256_set1_epi32(-1))
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<int8_t, 32> t, VecOf(t) v) -> VecOf(t) { return _tlmm_vec_not256(v.v); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<uint8_t, 32> t, VecOf(t) v) -> VecOf(t) { return _tlmm_vec_not256(v.v); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<int16_t, 16> t, VecOf(t) v) -> VecOf(t) { return _tlmm_vec_not256(v.v); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<uint16_t, 16> t, VecOf(t) v) -> VecOf(t) { return _tlmm_vec_not256(v.v); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<int32_t, 8> t, VecOf(t) v) -> VecOf(t) { return _tlmm_vec_not256(v.v); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<uint32_t, 8> t, VecOf(t) v) -> VecOf(t) { return _tlmm_vec_not256(v.v); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<int64_t, 4> t, VecOf(t) v) -> VecOf(t) { return _tlmm_vec_not256(v.v); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<uint64_t, 4> t, VecOf(t) v) -> VecOf(t) { return _tlmm_vec_not256(v.v); }
#undef _tlmm_vec_not256

// bit_not(Vec<T> v, Mask<T> m, Vec<T> default_v)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<int8_t, 32> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_blend_epi8(m.v, default_v.v, _mm256_xor_si256(v.v, _mm256_set1_epi32(-1))); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<uint8_t, 32> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_blend_epi8(m.v, default_v.v, _mm256_xor_si256(v.v, _mm256_set1_epi32(-1))); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<int16_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_blend_epi16(m.v, default_v.v, _mm256_xor_si256(v.v, _mm256_set1_epi32(-1))); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<uint16_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_blend_epi16(m.v, default_v.v, _mm256_xor_si256(v.v, _mm256_set1_epi32(-1))); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<int32_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_xor_epi32(default_v.v, m.v, v.v, _mm256_set1_epi32(-1)); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<uint32_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_xor_epi32(default_v.v, m.v, v.v, _mm256_set1_epi32(-1)); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<int64_t, 4> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_xor_epi64(default_v.v, m.v, v.v, _mm256_set1_epi64x(-1)); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<uint64_t, 4> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_xor_epi64(default_v.v, m.v, v.v, _mm256_set1_epi64x(-1)); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<int8_t, 32> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, bit_not(t, v.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<uint8_t, 32> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, bit_not(t, v.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<int16_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, bit_not(t, v.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<uint16_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, bit_not(t, v.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<int32_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, bit_not(t, v.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<uint32_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, bit_not(t, v.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<int64_t, 4> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, bit_not(t, v.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<uint64_t, 4> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, bit_not(t, v.v).v, m.v); }
#endif //HAS_AVX512DQ



// neg(Vec<T> v)
CT_ALWAYS_FORCEINLINE auto neg(Tag<float32_t, 8> t, VecOf(t) v) -> VecOf(t) { return _mm256_sub_ps(_mm256_setzero_ps(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<float64_t, 4> t, VecOf(t) v) -> VecOf(t) { return _mm256_sub_pd(_mm256_setzero_pd(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<int8_t, 32> t, VecOf(t) v) -> VecOf(t) { return _mm256_sub_epi8(_mm256_setzero_si256(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<uint8_t, 32> t, VecOf(t) v) -> VecOf(t) { return _mm256_sub_epi8(_mm256_setzero_si256(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<int16_t, 16> t, VecOf(t) v) -> VecOf(t) { return _mm256_sub_epi16(_mm256_setzero_si256(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<uint16_t, 16> t, VecOf(t) v) -> VecOf(t) { return _mm256_sub_epi16(_mm256_setzero_si256(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<int32_t, 8> t, VecOf(t) v) -> VecOf(t) { return _mm256_sub_epi32(_mm256_setzero_si256(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<uint32_t, 8> t, VecOf(t) v) -> VecOf(t) { return _mm256_sub_epi32(_mm256_setzero_si256(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<int64_t, 4> t, VecOf(t) v) -> VecOf(t) { return _mm256_sub_epi64(_mm256_setzero_si256(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<uint64_t, 4> t, VecOf(t) v) -> VecOf(t) { return _mm256_sub_epi64(_mm256_setzero_si256(), v.v); }

// neg(Vec<T> v, Mask<T> m, Vec<T> default_v)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto neg(Tag<float32_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_sub_ps(default_v.v, m.v, _mm256_setzero_ps(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<float64_t, 4> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_sub_pd(default_v.v, m.v, _mm256_setzero_pd(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<int8_t, 32> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_sub_epi8(default_v.v, m.v, _mm256_setzero_si256(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<uint8_t, 32> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_sub_epi8(default_v.v, m.v, _mm256_setzero_si256(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<int16_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_sub_epi16(default_v.v, m.v, _mm256_setzero_si256(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<uint16_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_sub_epi16(default_v.v, m.v, _mm256_setzero_si256(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<int32_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_sub_epi32(default_v.v, m.v, _mm256_setzero_si256(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<uint32_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_sub_epi32(default_v.v, m.v, _mm256_setzero_si256(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<int64_t, 4> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_sub_epi64(default_v.v, m.v, _mm256_setzero_si256(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<uint64_t, 4> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_sub_epi64(default_v.v, m.v, _mm256_setzero_si256(), v.v); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto neg(Tag<float32_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_ps(default_v.v, neg(t, v.v).v, _mm256_castsi256_ps(m.v)); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<float64_t, 4> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_pd(default_v.v, neg(t, v.v).v, _mm256_castsi256_pd(m.v)); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<int8_t, 32> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, neg(t, v.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<uint8_t, 32> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, neg(t, v.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<int16_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, neg(t, v.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<uint16_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, neg(t, v.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<int32_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, neg(t, v.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<uint32_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, neg(t, v.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<int64_t, 4> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, neg(t, v.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<uint64_t, 4> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, neg(t, v.v).v, m.v); }
#endif //HAS_AVX512DQ



// abs(Vec<T> v)
CT_ALWAYS_FORCEINLINE auto abs(Tag<float32_t, 8> t, VecOf(t) v) -> VecOf(t) { return _mm256_and_ps(_mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)), v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<float64_t, 4> t, VecOf(t) v) -> VecOf(t) { return _mm256_and_pd(_mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)), v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<int8_t, 32> t, VecOf(t) v) -> VecOf(t) { return _mm256_abs_epi8(v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<uint8_t, 32> t, VecOf(t) v) -> VecOf(t) { return v.v; }
CT_ALWAYS_FORCEINLINE auto abs(Tag<int16_t, 16> t, VecOf(t) v) -> VecOf(t) { return _mm256_abs_epi16(v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<uint16_t, 16> t, VecOf(t) v) -> VecOf(t) { return v.v; }
CT_ALWAYS_FORCEINLINE auto abs(Tag<int32_t, 8> t, VecOf(t) v) -> VecOf(t) { return _mm256_abs_epi32(v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<uint32_t, 8> t, VecOf(t) v) -> VecOf(t) { return v.v; }
CT_ALWAYS_FORCEINLINE auto abs(Tag<int64_t, 4> t, VecOf(t) v) -> VecOf(t) { return _mm256_abs_epi64(v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<uint64_t, 4> t, VecOf(t) v) -> VecOf(t) { return v.v; }

// abs(Vec<T> v, Mask<T> m, Vec<T> default_v)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto abs(Tag<float32_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_and_ps(default_v.v, m.v, _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)), v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<float64_t, 4> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_and_pd(default_v.v, m.v, _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL)), v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<int8_t, 32> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_abs_epi8(default_v.v, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<uint8_t, 32> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_blend_epi8(m.v, default_v.v, v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<int16_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_abs_epi16(default_v.v, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<uint16_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_blend_epi16(m.v, default_v.v, v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<int32_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_abs_epi32(default_v.v, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<uint32_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_blend_epi32(m.v, default_v.v, v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<int64_t, 4> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_abs_epi64(default_v.v, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<uint64_t, 4> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_blend_epi64(m.v, default_v.v, v.v); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto abs(Tag<float32_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_ps(default_v.v, abs(t, v.v).v, _mm256_castsi256_ps(m.v)); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<float64_t, 4> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_pd(default_v.v, abs(t, v.v).v, _mm256_castsi256_pd(m.v)); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<int8_t, 32> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, abs(t, v.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<uint8_t, 32> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, v.v, m.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<int16_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, abs(t, v.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<uint16_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, v.v, m.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<int32_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, abs(t, v.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<uint32_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, v.v, m.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<int64_t, 4> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, abs(t, v.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<uint64_t, 4> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_epi8(default_v.v, v.v, m.v); }
#endif //HAS_AVX512DQ



// sqrt(Vec<T> v)
CT_ALWAYS_FORCEINLINE auto sqrt(Tag<float32_t, 8> t, VecOf(t) v) -> VecOf(t) { return _mm256_sqrt_ps(v.v); }
CT_ALWAYS_FORCEINLINE auto sqrt(Tag<float64_t, 4> t, VecOf(t) v) -> VecOf(t) { return _mm256_sqrt_pd(v.v); }

// sqrt(Vec<T> v, Mask<T> m, Vec<T> default_v)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto sqrt(Tag<float32_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_sqrt_ps(default_v.v, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto sqrt(Tag<float64_t, 4> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_sqrt_pd(default_v.v, m.v, v.v); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto sqrt(Tag<float32_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_ps(default_v.v, sqrt(t, v.v).v, _mm256_castsi256_ps(m.v)); }
CT_ALWAYS_FORCEINLINE auto sqrt(Tag<float64_t, 4> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_pd(default_v.v, sqrt(t, v.v).v, _mm256_castsi256_pd(m.v)); }
#endif // HAS_AVX512DQ



// rsqrt(Vec<T> v)
#ifdef HAS_AVX512F
CT_ALWAYS_FORCEINLINE auto rsqrt(Tag<float32_t, 8> t, VecOf(t) v) -> VecOf(t) { return _mm256_rsqrt14_ps(v.v); }
CT_ALWAYS_FORCEINLINE auto rsqrt(Tag<float64_t, 4> t, VecOf(t) v) -> VecOf(t) { return _mm256_rsqrt14_pd(v.v); }
#else // HAS_AVX512F
CT_ALWAYS_FORCEINLINE auto rsqrt(Tag<float32_t, 8> t, VecOf(t) v) -> VecOf(t) { return _mm256_rsqrt_ps(v.v); }
CT_ALWAYS_FORCEINLINE auto rsqrt(Tag<float64_t, 4> t, VecOf(t) v) -> VecOf(t) { return div(t, fill(t, 1), sqrt(t, v.v)); }
#endif // HAS_AVX512F

// rsqrt(Vec<T> v, Mask<T> m, Vec<T> default_v)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto rsqrt(Tag<float32_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_rsqrt14_ps(default_v.v, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto rsqrt(Tag<float64_t, 4> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_mask_rsqrt14_pd(default_v.v, m.v, v.v); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto rsqrt(Tag<float32_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_ps(default_v.v, rsqrt(t, v.v).v, _mm256_castsi256_ps(m.v)); }
CT_ALWAYS_FORCEINLINE auto rsqrt(Tag<float64_t, 4> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm256_blendv_pd(default_v.v, rsqrt(t, v.v).v, _mm256_castsi256_pd(m.v)); }
#endif // HAS_AVX512DQ



// cmpeq(Vec<T> a, Vec<T> b)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_ps_mask(a.v, b.v, _CMP_EQ_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_pd_mask(a.v, b.v, _CMP_EQ_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_EQ); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_EQ_OQ)); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_EQ_OQ)); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpeq_epi8(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpeq_epi8(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpeq_epi16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpeq_epi16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpeq_epi32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpeq_epi32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpeq_epi64(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpeq_epi64(a.v, b.v); }
#endif // HAS_AVX512DQ

// cmpeq(Vec<T> a, Vec<T> b, Mask<T> m)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_EQ_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_EQ_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpeq(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpeq(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpeq(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpeq(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpeq(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpeq(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpeq(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpeq(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpeq(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpeq(t, a.v, b.v).v, m.v); }
#endif //HAS_AVX512DQ



// cmpne(Vec<T> a, Vec<T> b)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_ps_mask(a.v, b.v, _CMP_NEQ_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_pd_mask(a.v, b.v, _CMP_NEQ_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NE); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_NEQ_OQ)); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_NEQ_OQ)); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmpeq(t, a.v, b.v).v).v; }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmpeq(t, a.v, b.v).v).v; }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmpeq(t, a.v, b.v).v).v; }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmpeq(t, a.v, b.v).v).v; }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmpeq(t, a.v, b.v).v).v; }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmpeq(t, a.v, b.v).v).v; }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmpeq(t, a.v, b.v).v).v; }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmpeq(t, a.v, b.v).v).v; }
#endif // HAS_AVX512DQ

// cmpne(Vec<T> a, Vec<T> b, Mask<T> m)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_NEQ_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_NEQ_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpne(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpne(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpne(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpne(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpne(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpne(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpne(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpne(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpne(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpne(t, a.v, b.v).v, m.v); }
#endif //HAS_AVX512DQ



// cmplt(Vec<T> a, Vec<T> b)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_ps_mask(a.v, b.v, _CMP_LT_OQ); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_pd_mask(a.v, b.v, _CMP_LT_OQ); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_LT); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_LT_OQ)); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_LT_OQ)); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpgt_epi8(b.v, a.v); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpgt_epi8(_mm256_xor_si256(b.v, _mm256_set1_epi8((char)0x80)), _mm256_xor_si256(a.v, _mm256_set1_epi8((char)0x80))); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpgt_epi16(b.v, a.v); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpgt_epi16(_mm256_xor_si256(b.v, _mm256_set1_epi16((short)0x8000)), _mm256_xor_si256(a.v, _mm256_set1_epi16((short)0x8000))); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpgt_epi32(b.v, a.v); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpgt_epi32(_mm256_xor_si256(b.v, _mm256_set1_epi32((int)0x80000000u)), _mm256_xor_si256(a.v, _mm256_set1_epi32((int)0x80000000u))); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpgt_epi64(b.v, a.v); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpgt_epi64(_mm256_xor_si256(b.v, _mm256_set1_epi64x((int64_t)0x8000000000000000ull)), _mm256_xor_si256(a.v, _mm256_set1_epi64x((int64_t)0x8000000000000000ull))); }
#endif // HAS_AVX512DQ

// cmplt(Vec<T> a, Vec<T> b, Mask<T> m)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_LT_OQ); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_LT_OQ); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmplt(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmplt(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmplt(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmplt(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmplt(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmplt(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmplt(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmplt(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmplt(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmplt(t, a.v, b.v).v, m.v); }
#endif // HAS_AVX512DQ



// cmpgt(Vec<T> a, Vec<T> b)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_ps_mask(a.v, b.v, _CMP_GT_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_pd_mask(a.v, b.v, _CMP_GT_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NLE); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_GT_OQ)); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_GT_OQ)); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpgt_epi8(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpgt_epi8(_mm256_xor_si256(a.v, _mm256_set1_epi8((char)0x80)), _mm256_xor_si256(b.v, _mm256_set1_epi8((char)0x80))); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpgt_epi16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpgt_epi16(_mm256_xor_si256(a.v, _mm256_set1_epi16((short)0x8000)), _mm256_xor_si256(b.v, _mm256_set1_epi16((short)0x8000))); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpgt_epi32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpgt_epi32(_mm256_xor_si256(a.v, _mm256_set1_epi32((int)0x80000000u)), _mm256_xor_si256(b.v, _mm256_set1_epi32((int)0x80000000u))); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpgt_epi64(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmpgt_epi64(_mm256_xor_si256(a.v, _mm256_set1_epi64x((int64_t)0x8000000000000000ull)), _mm256_xor_si256(b.v, _mm256_set1_epi64x((int64_t)0x8000000000000000ull))); }
#endif // HAS_AVX512DQ

// cmpgt(Vec<T> a, Vec<T> b, Mask<T> m)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_GT_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_GT_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpgt(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpgt(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpgt(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpgt(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpgt(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpgt(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpgt(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpgt(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpgt(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpgt(t, a.v, b.v).v, m.v); }
#endif // HAS_AVX512DQ



// cmple(Vec<T> a, Vec<T> b)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmple(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_ps_mask(a.v, b.v, _CMP_LE_OQ); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_pd_mask(a.v, b.v, _CMP_LE_OQ); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_LE); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmple(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_LE_OQ)); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_LE_OQ)); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmpgt(t, a.v, b.v).v).v; }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmpgt(t, a.v, b.v).v).v; }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmpgt(t, a.v, b.v).v).v; }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmpgt(t, a.v, b.v).v).v; }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmpgt(t, a.v, b.v).v).v; }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmpgt(t, a.v, b.v).v).v; }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmpgt(t, a.v, b.v).v).v; }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmpgt(t, a.v, b.v).v).v; }
#endif // HAS_AVX512DQ

// cmple(Vec<T> a, Vec<T> b, Mask<T> m)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmple(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_LE_OQ); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_LE_OQ); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmple(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmple(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmple(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmple(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmple(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmple(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmple(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmple(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmple(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmple(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmple(t, a.v, b.v).v, m.v); }
#endif // HAS_AVX512DQ



// cmpge(Vec<T> a, Vec<T> b)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_ps_mask(a.v, b.v, _CMP_GE_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_pd_mask(a.v, b.v, _CMP_GE_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NLT); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_castps_si256(_mm256_cmp_ps(a.v, b.v, _CMP_GE_OQ)); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm256_castpd_si256(_mm256_cmp_pd(a.v, b.v, _CMP_GE_OQ)); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmplt(t, a.v, b.v).v).v; }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmplt(t, a.v, b.v).v).v; }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmplt(t, a.v, b.v).v).v; }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmplt(t, a.v, b.v).v).v; }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmplt(t, a.v, b.v).v).v; }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmplt(t, a.v, b.v).v).v; }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmplt(t, a.v, b.v).v).v; }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return bit_not(Tag<int32_t, 8>(), cmplt(t, a.v, b.v).v).v; }
#endif // HAS_AVX512DQ

// cmpge(Vec<T> a, Vec<T> b, Mask<T> m)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_GE_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_GE_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<float32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpge(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<float64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpge(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpge(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint8_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpge(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpge(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint16_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpge(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpge(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint32_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpge(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpge(t, a.v, b.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint64_t, 4> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(cmpge(t, a.v, b.v).v, m.v); }
#endif // HAS_AVX512DQ



// isnan(Vec<T> v)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto isnan(Tag<float32_t, 8> t, VecOf(t) v) -> MaskOf(t) { return _mm256_cmp_ps_mask(v.v, v.v, _CMP_UNORD_Q); }
CT_ALWAYS_FORCEINLINE auto isnan(Tag<float64_t, 4> t, VecOf(t) v) -> MaskOf(t) { return _mm256_cmp_pd_mask(v.v, v.v, _CMP_UNORD_Q); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto isnan(Tag<float32_t, 8> t, VecOf(t) v) -> MaskOf(t) { return _mm256_castps_si256(_mm256_cmp_ps(v.v, v.v, _CMP_UNORD_Q)); }
CT_ALWAYS_FORCEINLINE auto isnan(Tag<float64_t, 4> t, VecOf(t) v) -> MaskOf(t) { return _mm256_castpd_si256(_mm256_cmp_pd(v.v, v.v, _CMP_UNORD_Q)); }
#endif // HAS_AVX512DQ

// isnan(Vec<T> v, Mask<T> m)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto isnan(Tag<float32_t, 8> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_ps_mask(m.v, v.v, v.v, _CMP_UNORD_Q); }
CT_ALWAYS_FORCEINLINE auto isnan(Tag<float64_t, 4> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_pd_mask(m.v, v.v, v.v, _CMP_UNORD_Q); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto isnan(Tag<float32_t, 8> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(isnan(t, v.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto isnan(Tag<float64_t, 4> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(isnan(t, v.v).v, m.v); }
#endif // HAS_AVX512DQ



// isposinf(Vec<T> v)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto isposinf(Tag<float32_t, 8> t, VecOf(t) v) -> MaskOf(t) { return _mm256_cmp_ps_mask(v.v, _mm256_set1_ps(INFINITY), _CMP_EQ_OQ); }
CT_ALWAYS_FORCEINLINE auto isposinf(Tag<float64_t, 4> t, VecOf(t) v) -> MaskOf(t) { return _mm256_cmp_pd_mask(v.v, _mm256_set1_pd(INFINITY), _CMP_EQ_OQ); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto isposinf(Tag<float32_t, 8> t, VecOf(t) v) -> MaskOf(t) { return _mm256_castps_si256(_mm256_cmp_ps(v.v, _mm256_set1_ps(INFINITY), _CMP_EQ_OQ)); }
CT_ALWAYS_FORCEINLINE auto isposinf(Tag<float64_t, 4> t, VecOf(t) v) -> MaskOf(t) { return _mm256_castpd_si256(_mm256_cmp_pd(v.v, _mm256_set1_pd(INFINITY), _CMP_EQ_OQ)); }
#endif // HAS_AVX512DQ

// isposinf(Vec<T> v, Mask<T> m)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto isposinf(Tag<float32_t, 8> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_ps_mask(m.v, v.v, _mm256_set1_ps(INFINITY), _CMP_EQ_OQ); }
CT_ALWAYS_FORCEINLINE auto isposinf(Tag<float64_t, 4> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_pd_mask(m.v, v.v, _mm256_set1_pd(INFINITY), _CMP_EQ_OQ); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto isposinf(Tag<float32_t, 8> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(isposinf(t, v.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto isposinf(Tag<float64_t, 4> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(isposinf(t, v.v).v, m.v); }
#endif // HAS_AVX512DQ



// isneginf(Vec<T> v)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto isneginf(Tag<float32_t, 8> t, VecOf(t) v) -> MaskOf(t) { return _mm256_cmp_ps_mask(v.v, _mm256_set1_ps(-INFINITY), _CMP_EQ_OQ); }
CT_ALWAYS_FORCEINLINE auto isneginf(Tag<float64_t, 4> t, VecOf(t) v) -> MaskOf(t) { return _mm256_cmp_pd_mask(v.v, _mm256_set1_pd(-INFINITY), _CMP_EQ_OQ); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto isneginf(Tag<float32_t, 8> t, VecOf(t) v) -> MaskOf(t) { return _mm256_castps_si256(_mm256_cmp_ps(v.v, _mm256_set1_ps(-INFINITY), _CMP_EQ_OQ)); }
CT_ALWAYS_FORCEINLINE auto isneginf(Tag<float64_t, 4> t, VecOf(t) v) -> MaskOf(t) { return _mm256_castpd_si256(_mm256_cmp_pd(v.v, _mm256_set1_pd(-INFINITY), _CMP_EQ_OQ)); }
#endif // HAS_AVX512DQ

// isneginf(Vec<T> v, Mask<T> m)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto isneginf(Tag<float32_t, 8> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_ps_mask(m.v, v.v, _mm256_set1_ps(-INFINITY), _CMP_EQ_OQ); }
CT_ALWAYS_FORCEINLINE auto isneginf(Tag<float64_t, 4> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return _mm256_mask_cmp_pd_mask(m.v, v.v, _mm256_set1_pd(-INFINITY), _CMP_EQ_OQ); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto isneginf(Tag<float32_t, 8> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(isneginf(t, v.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto isneginf(Tag<float64_t, 4> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(isneginf(t, v.v).v, m.v); }
#endif // HAS_AVX512DQ



// isinf(Vec<T> v)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto isinf(Tag<float32_t, 8> t, VecOf(t) v) -> MaskOf(t) { return isposinf(t, abs(t, v)).v; }
CT_ALWAYS_FORCEINLINE auto isinf(Tag<float64_t, 4> t, VecOf(t) v) -> MaskOf(t) { return isposinf(t, abs(t, v)).v; }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto isinf(Tag<float32_t, 8> t, VecOf(t) v) -> MaskOf(t) { return _mm256_or_si256(isposinf(t, v.v).v, isneginf(t, v.v).v); }
CT_ALWAYS_FORCEINLINE auto isinf(Tag<float64_t, 4> t, VecOf(t) v) -> MaskOf(t) { return _mm256_or_si256(isposinf(t, v.v).v, isneginf(t, v.v).v); }
#endif // HAS_AVX512DQ

// isinf(Vec<T> v, Mask<T> m)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto isinf(Tag<float32_t, 8> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return isinf(t, v).v & m.v; }
CT_ALWAYS_FORCEINLINE auto isinf(Tag<float64_t, 4> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return isinf(t, v).v & m.v; }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto isinf(Tag<float32_t, 8> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(isinf(t, v.v).v, m.v); }
CT_ALWAYS_FORCEINLINE auto isinf(Tag<float64_t, 4> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return _mm256_and_si256(isinf(t, v.v).v, m.v); }
#endif // HAS_AVX512DQ



/* ************************************************************************** */
//                            Bit shift operations                            */
/* ************************************************************************** */

// Helper for 8-bit left shift (no native _mm256_sll_epi8)
static CT_ALWAYS_FORCEINLINE __m256i _bit_shl_epi8_256(__m256i v, int count) {
  auto zero = _mm256_setzero_si256();
  auto lo = _mm256_unpacklo_epi8(v, zero);
  auto hi = _mm256_unpackhi_epi8(v, zero);
  auto count_vec = _mm_cvtsi32_si128(count);
  lo = _mm256_sll_epi16(lo, count_vec);
  hi = _mm256_sll_epi16(hi, count_vec);
  auto mask = _mm256_set1_epi16(0xFF);
  lo = _mm256_and_si256(lo, mask);
  hi = _mm256_and_si256(hi, mask);
  return _mm256_packus_epi16(lo, hi);
}

// Helper for 8-bit logical right shift
static CT_ALWAYS_FORCEINLINE __m256i _bit_srl_epi8_256(__m256i v, int count) {
  auto zero = _mm256_setzero_si256();
  auto lo = _mm256_unpacklo_epi8(zero, v);
  auto hi = _mm256_unpackhi_epi8(zero, v);
  lo = _mm256_srli_epi16(lo, 8);
  hi = _mm256_srli_epi16(hi, 8);
  auto count_vec = _mm_cvtsi32_si128(count);
  lo = _mm256_srl_epi16(lo, count_vec);
  hi = _mm256_srl_epi16(hi, count_vec);
  return _mm256_packus_epi16(lo, hi);
}

// Helper for 8-bit arithmetic right shift
static CT_ALWAYS_FORCEINLINE __m256i _bit_sra_epi8_256(__m256i v, int count) {
  auto zero = _mm256_setzero_si256();
  auto signs = _mm256_cmpgt_epi8(zero, v);
  auto lo = _mm256_unpacklo_epi8(v, signs);
  auto hi = _mm256_unpackhi_epi8(v, signs);
  auto count_vec = _mm_cvtsi32_si128(count);
  lo = _mm256_sra_epi16(lo, count_vec);
  hi = _mm256_sra_epi16(hi, count_vec);
  return _mm256_packs_epi16(lo, hi);
}

// Helper for 64-bit arithmetic right shift on AVX2 (no native _mm256_sra_epi64)
static CT_ALWAYS_FORCEINLINE __m256i _bit_sra_epi64_256(__m256i v, int count) {
  // Emulate using sign extension and 32-bit shifts
  // Get sign bits
  auto signs = _mm256_srai_epi32(v, 31);
  auto sign_hi = _mm256_shuffle_epi32(signs, _MM_SHUFFLE(3, 3, 1, 1));

  if (count >= 64) {
    return sign_hi;
  }

  if (count >= 32) {
    // For count >= 32:
    // - The original low 32 bits are completely shifted out
    // - The new low 32 bits come from original high 32 bits (shifted right by count-32)
    // - The new high 32 bits are all sign bits

    // Step 1: Get original high 32 bits and shift them right (arithmetic) by count-32
    // Use shuffle to extract high 32-bit parts: [Hi0, Hi0, Hi1, Hi1, Hi2, Hi2, Hi3, Hi3]
    auto hi_parts = _mm256_shuffle_epi32(v, _MM_SHUFFLE(3, 3, 1, 1));
    // Arithmetic right shift by (count - 32)
    auto lo_result = _mm256_srav_epi32(hi_parts, _mm256_set1_epi32(count - 32));

    // Step 2: The high 32 bits of each 64-bit element should be sign bits
    // sign_hi is [Sign0, Sign0, Sign1, Sign1, Sign2, Sign2, Sign3, Sign3]
    // where Sign = 0xFFFFFFFF or 0
    // sign_hi << 32 gives us the sign bits in the high 32-bit positions
    auto hi_result = _mm256_slli_epi64(sign_hi, 32);

    // Combine: low 32 bits from lo_result, high 32 bits from hi_result
    // lo_result has the correct low 32 bits (and incorrect high 32 bits)
    // hi_result has the correct high 32 bits (and zero low 32 bits)
    // We need to mask out the high 32 bits of lo_result and OR with hi_result
    auto mask_lo = _mm256_set1_epi64x(0xFFFFFFFF);
    lo_result = _mm256_and_si256(lo_result, mask_lo);
    return _mm256_or_si256(lo_result, hi_result);
  }

  // count < 32
  // For arithmetic right shift, we need to:
  // 1. Do logical right shift to get the low bits
  // 2. Fill the high 'count' bits with sign bits
  auto lo = _mm256_srl_epi64(v, _mm_cvtsi32_si128(count));
  // Create mask for high 'count' bits: (0xFFFFFFFFFFFFFFFF << (64 - count))
  // But we need to handle the case where count could be 0
  auto all_ones = _mm256_set1_epi64x(-1);
  auto mask = _mm256_sllv_epi64(all_ones, _mm256_set1_epi64x(64 - count));
  // For negative numbers, fill high bits with 1s; for positive, fill with 0s
  auto sign_part = _mm256_and_si256(sign_hi, mask);
  return _mm256_or_si256(lo, sign_part);
}


// bit_shl(Vec<T> v, int count)
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<int8_t, 32> t, VecOf(t) v, int count) -> VecOf(t) { return _bit_shl_epi8_256(v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<uint8_t, 32> t, VecOf(t) v, int count) -> VecOf(t) { return _bit_shl_epi8_256(v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<int16_t, 16> t, VecOf(t) v, int count) -> VecOf(t) { return _mm256_sll_epi16(v.v, _mm_cvtsi32_si128(count)); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<uint16_t, 16> t, VecOf(t) v, int count) -> VecOf(t) { return _mm256_sll_epi16(v.v, _mm_cvtsi32_si128(count)); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<int32_t, 8> t, VecOf(t) v, int count) -> VecOf(t) { return _mm256_sll_epi32(v.v, _mm_cvtsi32_si128(count)); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<uint32_t, 8> t, VecOf(t) v, int count) -> VecOf(t) { return _mm256_sll_epi32(v.v, _mm_cvtsi32_si128(count)); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<int64_t, 4> t, VecOf(t) v, int count) -> VecOf(t) { return _mm256_sll_epi64(v.v, _mm_cvtsi32_si128(count)); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<uint64_t, 4> t, VecOf(t) v, int count) -> VecOf(t) { return _mm256_sll_epi64(v.v, _mm_cvtsi32_si128(count)); }

// bit_shl(Vec<T> v, int count, Mask<T> m)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<int8_t, 32> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_blend_epi8(m.v, v.v, _bit_shl_epi8_256(v.v, count)); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<uint8_t, 32> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_blend_epi8(m.v, v.v, _bit_shl_epi8_256(v.v, count)); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<int16_t, 16> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_sll_epi16(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<uint16_t, 16> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_sll_epi16(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<int32_t, 8> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_sll_epi32(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<uint32_t, 8> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_sll_epi32(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<int64_t, 4> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_sll_epi64(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<uint64_t, 4> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_sll_epi64(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<int8_t, 32> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(v.v, _bit_shl_epi8_256(v.v, count), m.v); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<uint8_t, 32> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(v.v, _bit_shl_epi8_256(v.v, count), m.v); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<int16_t, 16> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(v.v, _mm256_sll_epi16(v.v, _mm_cvtsi32_si128(count)), m.v); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<uint16_t, 16> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(v.v, _mm256_sll_epi16(v.v, _mm_cvtsi32_si128(count)), m.v); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<int32_t, 8> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(v.v, _mm256_sll_epi32(v.v, _mm_cvtsi32_si128(count)), m.v); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<uint32_t, 8> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(v.v, _mm256_sll_epi32(v.v, _mm_cvtsi32_si128(count)), m.v); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<int64_t, 4> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(v.v, _mm256_sll_epi64(v.v, _mm_cvtsi32_si128(count)), m.v); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<uint64_t, 4> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(v.v, _mm256_sll_epi64(v.v, _mm_cvtsi32_si128(count)), m.v); }
#endif // HAS_AVX512DQ



// bit_shr(Vec<T> v, int count) - Signed: arithmetic shift, Unsigned: logical shift
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<int8_t, 32> t, VecOf(t) v, int count) -> VecOf(t) { return _bit_sra_epi8_256(v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<int16_t, 16> t, VecOf(t) v, int count) -> VecOf(t) { return _mm256_sra_epi16(v.v, _mm_cvtsi32_si128(count)); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<int32_t, 8> t, VecOf(t) v, int count) -> VecOf(t) { return _mm256_sra_epi32(v.v, _mm_cvtsi32_si128(count)); }
#ifdef HAS_AVX512VL
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<int64_t, 4> t, VecOf(t) v, int count) -> VecOf(t) { return _mm256_sra_epi64(v.v, _mm_cvtsi32_si128(count)); }
#else
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<int64_t, 4> t, VecOf(t) v, int count) -> VecOf(t) { return _bit_sra_epi64_256(v.v, count); }
#endif
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<uint8_t, 32> t, VecOf(t) v, int count) -> VecOf(t) { return _bit_srl_epi8_256(v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<uint16_t, 16> t, VecOf(t) v, int count) -> VecOf(t) { return _mm256_srl_epi16(v.v, _mm_cvtsi32_si128(count)); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<uint32_t, 8> t, VecOf(t) v, int count) -> VecOf(t) { return _mm256_srl_epi32(v.v, _mm_cvtsi32_si128(count)); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<uint64_t, 4> t, VecOf(t) v, int count) -> VecOf(t) { return _mm256_srl_epi64(v.v, _mm_cvtsi32_si128(count)); }

// bit_shr(Vec<T> v, int count, Mask<T> m)
#ifdef HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<int8_t, 32> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_blend_epi8(m.v, v.v, _bit_sra_epi8_256(v.v, count)); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<int16_t, 16> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_sra_epi16(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<int32_t, 8> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_sra_epi32(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<int64_t, 4> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_sra_epi64(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<uint8_t, 32> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_blend_epi8(m.v, v.v, _bit_srl_epi8_256(v.v, count)); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<uint16_t, 16> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_srl_epi16(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<uint32_t, 8> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_srl_epi32(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<uint64_t, 4> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_mask_srl_epi64(v.v, m.v, v.v, _mm_cvtsi32_si128(count)); }
#else // HAS_AVX512DQ
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<int8_t, 32> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(v.v, _bit_sra_epi8_256(v.v, count), m.v); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<int16_t, 16> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(v.v, _mm256_sra_epi16(v.v, _mm_cvtsi32_si128(count)), m.v); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<int32_t, 8> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(v.v, _mm256_sra_epi32(v.v, _mm_cvtsi32_si128(count)), m.v); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<int64_t, 4> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(v.v, _mm256_sra_epi64(v.v, _mm_cvtsi32_si128(count)), m.v); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<uint8_t, 32> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(v.v, _bit_srl_epi8_256(v.v, count), m.v); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<uint16_t, 16> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(v.v, _mm256_srl_epi16(v.v, _mm_cvtsi32_si128(count)), m.v); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<uint32_t, 8> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(v.v, _mm256_srl_epi32(v.v, _mm_cvtsi32_si128(count)), m.v); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<uint64_t, 4> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm256_blendv_epi8(v.v, _mm256_srl_epi64(v.v, _mm_cvtsi32_si128(count)), m.v); }
#endif // HAS_AVX512DQ



#undef TL_CHECK_COUNT
#undef TL_CHECK_INDEX
#undef TL_CHECK_ALIGN
} // namespace word
} // namespace ct::tl::vec
//@formatter:on

#endif //CTORCH_X86_256_H
