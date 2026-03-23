//
// Created by renyz on 2026/3/21.
//

#ifndef CTORCH_X86_512_H
#define CTORCH_X86_512_H

//@formatter:off
#include <cmath>
#include "CoreDefs.h"
#include "tl/util/Math.h"
#include "tl/cpu/VecBase.h"

#ifndef ARCH_X86_FAMILY
#error "Not x86 platform"
#endif

#ifndef HAS_CPU_CAPABILITY_AVX512
  #error "AVX512F, AVX512CD, AVX512BW, AVX512DQ instruction set required"
#endif

#include <immintrin.h>
#include "tl/cpu/impl/x86_128.h"
#include "tl/cpu/impl/x86_256.h"
#include "tl/cpu/impl/x86_Types.h"

namespace ct::tl::vec {
namespace word {

#define TL_CHECK_COUNT(varname) CT_ASSERT(0 <= (varname) && (varname) <= size(t), "%zd !in 0..%zd", (varname), size(t))
#define TL_CHECK_INDEX(varname) CT_ASSERT(0 <= (varname) && (varname) < size(t), "%zd !in 0..%zd", (varname), size(t))
#define TL_CHECK_ALIGN CT_ASSERT(((nuint_t)(p) & (DEFAULT_ALIGNMENT - 1)) == 0, "Not aligned");



/* ************************************************************************** */
//                               Constructors                                 //
/* ************************************************************************** */

// mfill, mwhilelt, mwhilege
// AVX512 with VL+BW+DQ always available, so masks are native __mmask types.
// For 512-bit: int8/uint8 → __mmask64, int16/uint16 → __mmask32,
// int32/uint32 → __mmask16, int64/uint64 → __mmask8,
// float32 → __mmask16, float64 → __mmask8,
// bfloat16/float16 → __mmask32

#define TL_ZMM_DEFINE_MASK_OPERATIONS(dtype, mask_size) \
CT_ALWAYS_FORCEINLINE auto mfill(Tag<dtype, (64 / sizeof(dtype))> t, bool value) -> MaskOf(t) { \
  uint64_t x = value ? 0xFFFFFFFFFFFFFFFFull : 0x00; \
  if constexpr (64 / sizeof(dtype) <= 8) return _cvtu32_mask8((uint32_t)x); \
  if constexpr (64 / sizeof(dtype) <= 16) return _cvtu32_mask16((uint32_t)x); \
  if constexpr (64 / sizeof(dtype) <= 32) return _cvtu32_mask32((uint32_t)x); \
  return _cvtu64_mask64(x); \
} \
CT_ALWAYS_FORCEINLINE auto mwhilelt(Tag<dtype, (64 / sizeof(dtype))> t, nint_t a, nint_t b) -> MaskOf(t) { \
  nint_t end = std::clamp(b - a, nint_t(0), nint_t(64 / sizeof(dtype))); \
  uint64_t x = end == 64 ? -1LL : uint64_t((nuint_t(1) << end) - 1); \
  if constexpr (64 / sizeof(dtype) <= 8) return _cvtu32_mask8((uint32_t)x); \
  if constexpr (64 / sizeof(dtype) <= 16) return _cvtu32_mask16((uint32_t)x); \
  if constexpr (64 / sizeof(dtype) <= 32) return _cvtu32_mask32((uint32_t)x); \
  return _cvtu64_mask64(x); \
} \
CT_ALWAYS_FORCEINLINE auto mwhilege(Tag<dtype, (64 / sizeof(dtype))> t, nint_t a, nint_t b) -> MaskOf(t) { \
  nint_t end = std::clamp(b - a, nint_t(0), nint_t(64 / sizeof(dtype))); \
  uint64_t x = end == 64 ? 0LL : ~uint64_t((nuint_t(1) << end) - 1); \
  if constexpr (64 / sizeof(dtype) <= 8) return _cvtu32_mask8((uint32_t)x); \
  if constexpr (64 / sizeof(dtype) <= 16) return _cvtu32_mask16((uint32_t)x); \
  if constexpr (64 / sizeof(dtype) <= 32) return _cvtu32_mask32((uint32_t)x); \
  return _cvtu64_mask64(x); \
}

TL_ZMM_DEFINE_MASK_OPERATIONS(bfloat16_t, 32)
TL_ZMM_DEFINE_MASK_OPERATIONS(float16_t, 32)
TL_ZMM_DEFINE_MASK_OPERATIONS(float32_t, 16)
TL_ZMM_DEFINE_MASK_OPERATIONS(float64_t, 8)
TL_ZMM_DEFINE_MASK_OPERATIONS(int8_t, 64)
TL_ZMM_DEFINE_MASK_OPERATIONS(uint8_t, 64)
TL_ZMM_DEFINE_MASK_OPERATIONS(int16_t, 32)
TL_ZMM_DEFINE_MASK_OPERATIONS(uint16_t, 32)
TL_ZMM_DEFINE_MASK_OPERATIONS(int32_t, 16)
TL_ZMM_DEFINE_MASK_OPERATIONS(uint32_t, 16)
TL_ZMM_DEFINE_MASK_OPERATIONS(int64_t, 8)
TL_ZMM_DEFINE_MASK_OPERATIONS(uint64_t, 8)
#undef TL_ZMM_DEFINE_MASK_OPERATIONS



// fill(T v)
CT_ALWAYS_FORCEINLINE auto fill(Tag<bfloat16_t, 32> t, bfloat16_t v) -> VecOf(t) {
  union { bfloat16_t b; int16_t i; } u { .b = v };
  return _mm512_set1_epi16(u.i);
}
CT_ALWAYS_FORCEINLINE auto fill(Tag<float16_t, 32> t, float16_t v) -> VecOf(t) {
  #ifdef HAS_AVX512_FP16
  return _mm512_castph_si512(_mm512_set1_ph(v));
  #else
  union { float16_t h; int16_t i; } u { .h = v };
  return _mm512_set1_epi16(u.i);
  #endif
}
CT_ALWAYS_FORCEINLINE auto fill(Tag<float32_t, 16> t, float32_t v) -> VecOf(t) { return _mm512_set1_ps(v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<float64_t, 8> t, float64_t v) -> VecOf(t) { return _mm512_set1_pd(v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int8_t, 64> t, int8_t v) -> VecOf(t) { return _mm512_set1_epi8(v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint8_t, 64> t, uint8_t v) -> VecOf(t) { return _mm512_set1_epi8((int8_t)v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int16_t, 32> t, int16_t v) -> VecOf(t) { return _mm512_set1_epi16(v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint16_t, 32> t, uint16_t v) -> VecOf(t) { return _mm512_set1_epi16((int16_t)v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int32_t, 16> t, int32_t v) -> VecOf(t) { return _mm512_set1_epi32(v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint32_t, 16> t, uint32_t v) -> VecOf(t) { return _mm512_set1_epi32((int32_t)v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int64_t, 8> t, int64_t v) -> VecOf(t) { return _mm512_set1_epi64(v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint64_t, 8> t, uint64_t v) -> VecOf(t) { return _mm512_set1_epi64((int64_t)v); }

// fill(T v, Mask<T> m, Vec<T> default_v)
CT_ALWAYS_FORCEINLINE auto fill(Tag<bfloat16_t, 32> t, bfloat16_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_mov_epi16(default_v.v, m.v, fill(t, v).v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<float16_t, 32> t, float16_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_mov_epi16(default_v.v, m.v, fill(t, v).v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<float32_t, 16> t, float32_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_mov_ps(default_v.v, m.v, fill(t, v).v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<float64_t, 8> t, float64_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_mov_pd(default_v.v, m.v, fill(t, v).v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int8_t, 64> t, int8_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_mov_epi8(default_v.v, m.v, fill(t, v).v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint8_t, 64> t, uint8_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_mov_epi8(default_v.v, m.v, fill(t, v).v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int16_t, 32> t, int16_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_mov_epi16(default_v.v, m.v, fill(t, v).v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint16_t, 32> t, uint16_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_mov_epi16(default_v.v, m.v, fill(t, v).v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int32_t, 16> t, int32_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_mov_epi32(default_v.v, m.v, fill(t, v).v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint32_t, 16> t, uint32_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_mov_epi32(default_v.v, m.v, fill(t, v).v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int64_t, 8> t, int64_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_mov_epi64(default_v.v, m.v, fill(t, v).v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint64_t, 8> t, uint64_t v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_mov_epi64(default_v.v, m.v, fill(t, v).v); }

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE auto fill_fwd_mask_512(Tag<T, N, P> t, T v, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = word::mwhilelt(t, 0, n);
  return word::fill(t, v, m, default_v);
}

// fill(T v, nint_t n, Vec<T> default_v)
CT_ALWAYS_FORCEINLINE auto fill(Tag<bfloat16_t, 32> t, bfloat16_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_512(t, v, n, default_v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<float16_t, 32> t, float16_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_512(t, v, n, default_v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<float32_t, 16> t, float32_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_512(t, v, n, default_v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<float64_t, 8> t, float64_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_512(t, v, n, default_v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int8_t, 64> t, int8_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_512(t, v, n, default_v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint8_t, 64> t, uint8_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_512(t, v, n, default_v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int16_t, 32> t, int16_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_512(t, v, n, default_v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint16_t, 32> t, uint16_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_512(t, v, n, default_v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int32_t, 16> t, int32_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_512(t, v, n, default_v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint32_t, 16> t, uint32_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_512(t, v, n, default_v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<int64_t, 8> t, int64_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_512(t, v, n, default_v); }
CT_ALWAYS_FORCEINLINE auto fill(Tag<uint64_t, 8> t, uint64_t v, nint_t n, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_COUNT(n); return fill_fwd_mask_512(t, v, n, default_v); }

// zeros()
CT_ALWAYS_FORCEINLINE auto zeros(Tag<bfloat16_t, 32> t) -> VecOf(t) { return _mm512_setzero_si512(); }
CT_ALWAYS_FORCEINLINE auto zeros(Tag<float16_t, 32> t) -> VecOf(t) { return _mm512_setzero_si512(); }
CT_ALWAYS_FORCEINLINE auto zeros(Tag<float32_t, 16> t) -> VecOf(t) { return _mm512_setzero_ps(); }
CT_ALWAYS_FORCEINLINE auto zeros(Tag<float64_t, 8> t) -> VecOf(t) { return _mm512_setzero_pd(); }
CT_ALWAYS_FORCEINLINE auto zeros(Tag<int8_t, 64> t) -> VecOf(t) { return _mm512_setzero_si512(); }
CT_ALWAYS_FORCEINLINE auto zeros(Tag<uint8_t, 64> t) -> VecOf(t) { return _mm512_setzero_si512(); }
CT_ALWAYS_FORCEINLINE auto zeros(Tag<int16_t, 32> t) -> VecOf(t) { return _mm512_setzero_si512(); }
CT_ALWAYS_FORCEINLINE auto zeros(Tag<uint16_t, 32> t) -> VecOf(t) { return _mm512_setzero_si512(); }
CT_ALWAYS_FORCEINLINE auto zeros(Tag<int32_t, 16> t) -> VecOf(t) { return _mm512_setzero_si512(); }
CT_ALWAYS_FORCEINLINE auto zeros(Tag<uint32_t, 16> t) -> VecOf(t) { return _mm512_setzero_si512(); }
CT_ALWAYS_FORCEINLINE auto zeros(Tag<int64_t, 8> t) -> VecOf(t) { return _mm512_setzero_si512(); }
CT_ALWAYS_FORCEINLINE auto zeros(Tag<uint64_t, 8> t) -> VecOf(t) { return _mm512_setzero_si512(); }



/* ************************************************************************** */
//                          Consecutive load & store                          //
/* ************************************************************************** */

// loadu(const T* p)
CT_ALWAYS_FORCEINLINE auto loadu(Tag<bfloat16_t, 32> t, const bfloat16_t* p) -> VecOf(t) { return _mm512_loadu_si512((const void*)p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<float16_t, 32> t, const float16_t* p) -> VecOf(t) {
  #ifdef HAS_AVX512_FP16
  return _mm512_castph_si512(_mm512_loadu_ph(p));
  #else
  return _mm512_loadu_si512((const void*)p);
  #endif
}
CT_ALWAYS_FORCEINLINE auto loadu(Tag<float32_t, 16> t, const float32_t* p) -> VecOf(t) { return _mm512_loadu_ps(p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<float64_t, 8> t, const float64_t* p) -> VecOf(t) { return _mm512_loadu_pd(p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int8_t, 64> t, const int8_t* p) -> VecOf(t) { return _mm512_loadu_si512((const void*)p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint8_t, 64> t, const uint8_t* p) -> VecOf(t) { return _mm512_loadu_si512((const void*)p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int16_t, 32> t, const int16_t* p) -> VecOf(t) { return _mm512_loadu_si512((const void*)p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint16_t, 32> t, const uint16_t* p) -> VecOf(t) { return _mm512_loadu_si512((const void*)p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int32_t, 16> t, const int32_t* p) -> VecOf(t) { return _mm512_loadu_si512((const void*)p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint32_t, 16> t, const uint32_t* p) -> VecOf(t) { return _mm512_loadu_si512((const void*)p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int64_t, 8> t, const int64_t* p) -> VecOf(t) { return _mm512_loadu_si512((const void*)p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint64_t, 8> t, const uint64_t* p) -> VecOf(t) { return _mm512_loadu_si512((const void*)p); }

// load(const T* p)
CT_ALWAYS_FORCEINLINE auto load(Tag<bfloat16_t, 32> t, const bfloat16_t* p) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_load_si512((const void*)p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<float16_t, 32> t, const float16_t* p) -> VecOf(t) {
  TL_CHECK_ALIGN
  #ifdef HAS_AVX512_FP16
  return _mm512_castph_si512(_mm512_load_ph(p));
  #else
  return _mm512_load_si512((const void*)p);
  #endif
}
CT_ALWAYS_FORCEINLINE auto load(Tag<float32_t, 16> t, const float32_t* p) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_load_ps(p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<float64_t, 8> t, const float64_t* p) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_load_pd(p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int8_t, 64> t, const int8_t* p) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_load_si512((const void*)p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint8_t, 64> t, const uint8_t* p) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_load_si512((const void*)p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int16_t, 32> t, const int16_t* p) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_load_si512((const void*)p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint16_t, 32> t, const uint16_t* p) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_load_si512((const void*)p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int32_t, 16> t, const int32_t* p) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_load_si512((const void*)p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint32_t, 16> t, const uint32_t* p) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_load_si512((const void*)p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int64_t, 8> t, const int64_t* p) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_load_si512((const void*)p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint64_t, 8> t, const uint64_t* p) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_load_si512((const void*)p); }

// loadu(const T* p, Mask<T> m, Vec<T> default_v)
CT_ALWAYS_FORCEINLINE auto loadu(Tag<bfloat16_t, 32> t, const bfloat16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_loadu_epi16(default_v.v, m.v, p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<float16_t, 32> t, const float16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_loadu_epi16(default_v.v, m.v, p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<float32_t, 16> t, const float32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_loadu_ps(default_v.v, m.v, p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<float64_t, 8> t, const float64_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_loadu_pd(default_v.v, m.v, p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int8_t, 64> t, const int8_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_loadu_epi8(default_v.v, m.v, p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint8_t, 64> t, const uint8_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_loadu_epi8(default_v.v, m.v, p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int16_t, 32> t, const int16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_loadu_epi16(default_v.v, m.v, p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint16_t, 32> t, const uint16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_loadu_epi16(default_v.v, m.v, p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int32_t, 16> t, const int32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_loadu_epi32(default_v.v, m.v, p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint32_t, 16> t, const uint32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_loadu_epi32(default_v.v, m.v, p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int64_t, 8> t, const int64_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_loadu_epi64(default_v.v, m.v, p); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint64_t, 8> t, const uint64_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_loadu_epi64(default_v.v, m.v, p); }

// load(const T* p, Mask<T> m, Vec<T> default_v)
// For epi8 and epi16, AVX512 has no dedicated aligned mask load intrinsics; use unaligned.
CT_ALWAYS_FORCEINLINE auto load(Tag<bfloat16_t, 32> t, const bfloat16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_mask_loadu_epi16(default_v.v, m.v, p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<float16_t, 32> t, const float16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_mask_loadu_epi16(default_v.v, m.v, p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<float32_t, 16> t, const float32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_mask_load_ps(default_v.v, m.v, p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<float64_t, 8> t, const float64_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_mask_load_pd(default_v.v, m.v, p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int8_t, 64> t, const int8_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_mask_loadu_epi8(default_v.v, m.v, p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint8_t, 64> t, const uint8_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_mask_loadu_epi8(default_v.v, m.v, p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int16_t, 32> t, const int16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_mask_loadu_epi16(default_v.v, m.v, p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint16_t, 32> t, const uint16_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_mask_loadu_epi16(default_v.v, m.v, p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int32_t, 16> t, const int32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_mask_load_epi32(default_v.v, m.v, p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint32_t, 16> t, const uint32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_mask_load_epi32(default_v.v, m.v, p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int64_t, 8> t, const int64_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_mask_load_epi64(default_v.v, m.v, p); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint64_t, 8> t, const uint64_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { TL_CHECK_ALIGN return _mm512_mask_load_epi64(default_v.v, m.v, p); }

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE auto _loadu_fwd_mask_512(Tag<T, N, P> t, const T* p, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = mwhilelt(t, 0, n);
  return word::loadu(t, p, m, default_v);
}

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE auto _load_fwd_mask_512(Tag<T, N, P> t, const T* p, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = mwhilelt(t, 0, n);
  return word::load(t, p, m, default_v);
}

// loadu(const T* p, nint_t n, Vec<T> default_v)
CT_ALWAYS_FORCEINLINE auto loadu(Tag<bfloat16_t, 32> t, const bfloat16_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_512(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<float16_t, 32> t, const float16_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_512(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<float32_t, 16> t, const float32_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_512(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<float64_t, 8> t, const float64_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_512(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int8_t, 64> t, const int8_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_512(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint8_t, 64> t, const uint8_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_512(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int16_t, 32> t, const int16_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_512(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint16_t, 32> t, const uint16_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_512(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int32_t, 16> t, const int32_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_512(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint32_t, 16> t, const uint32_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_512(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<int64_t, 8> t, const int64_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_512(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto loadu(Tag<uint64_t, 8> t, const uint64_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _loadu_fwd_mask_512(t, p, n, default_v); }

// load(const T* p, nint_t n, Vec<T> default_v)
CT_ALWAYS_FORCEINLINE auto load(Tag<bfloat16_t, 32> t, const bfloat16_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_512(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<float16_t, 32> t, const float16_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_512(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<float32_t, 16> t, const float32_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_512(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<float64_t, 8> t, const float64_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_512(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int8_t, 64> t, const int8_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_512(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint8_t, 64> t, const uint8_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_512(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int16_t, 32> t, const int16_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_512(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint16_t, 32> t, const uint16_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_512(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int32_t, 16> t, const int32_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_512(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint32_t, 16> t, const uint32_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_512(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<int64_t, 8> t, const int64_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_512(t, p, n, default_v); }
CT_ALWAYS_FORCEINLINE auto load(Tag<uint64_t, 8> t, const uint64_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) { return _load_fwd_mask_512(t, p, n, default_v); }



// storeu(T* p, Vec<T> v)
CT_ALWAYS_FORCEINLINE auto storeu(Tag<bfloat16_t, 32> t, bfloat16_t* p, VecOf(t) v) -> void { _mm512_storeu_si512((void*)p, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<float16_t, 32> t, float16_t* p, VecOf(t) v) -> void {
  #ifdef HAS_AVX512_FP16
  _mm512_storeu_ph(p, _mm512_castsi512_ph(v.v));
  #else
  _mm512_storeu_si512((void*)p, v.v);
  #endif
}
CT_ALWAYS_FORCEINLINE auto storeu(Tag<float32_t, 16> t, float32_t* p, VecOf(t) v) -> void { _mm512_storeu_ps(p, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<float64_t, 8> t, float64_t* p, VecOf(t) v) -> void { _mm512_storeu_pd(p, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int8_t, 64> t, int8_t* p, VecOf(t) v) -> void { _mm512_storeu_si512((void*)p, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint8_t, 64> t, uint8_t* p, VecOf(t) v) -> void { _mm512_storeu_si512((void*)p, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int16_t, 32> t, int16_t* p, VecOf(t) v) -> void { _mm512_storeu_si512((void*)p, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint16_t, 32> t, uint16_t* p, VecOf(t) v) -> void { _mm512_storeu_si512((void*)p, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int32_t, 16> t, int32_t* p, VecOf(t) v) -> void { _mm512_storeu_si512((void*)p, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint32_t, 16> t, uint32_t* p, VecOf(t) v) -> void { _mm512_storeu_si512((void*)p, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int64_t, 8> t, int64_t* p, VecOf(t) v) -> void { _mm512_storeu_si512((void*)p, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint64_t, 8> t, uint64_t* p, VecOf(t) v) -> void { _mm512_storeu_si512((void*)p, v.v); }

// store(T* p, Vec<T> v)
CT_ALWAYS_FORCEINLINE auto store(Tag<bfloat16_t, 32> t, bfloat16_t* p, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_store_si512((void*)p, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<float16_t, 32> t, float16_t* p, VecOf(t) v) -> void {
  TL_CHECK_ALIGN
  #ifdef HAS_AVX512_FP16
  _mm512_store_ph(p, _mm512_castsi512_ph(v.v));
  #else
  _mm512_store_si512((void*)p, v.v);
  #endif
}
CT_ALWAYS_FORCEINLINE auto store(Tag<float32_t, 16> t, float32_t* p, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_store_ps(p, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<float64_t, 8> t, float64_t* p, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_store_pd(p, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int8_t, 64> t, int8_t* p, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_store_si512((void*)p, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint8_t, 64> t, uint8_t* p, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_store_si512((void*)p, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int16_t, 32> t, int16_t* p, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_store_si512((void*)p, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint16_t, 32> t, uint16_t* p, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_store_si512((void*)p, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int32_t, 16> t, int32_t* p, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_store_si512((void*)p, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint32_t, 16> t, uint32_t* p, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_store_si512((void*)p, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int64_t, 8> t, int64_t* p, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_store_si512((void*)p, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint64_t, 8> t, uint64_t* p, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_store_si512((void*)p, v.v); }

// storeu(T* p, Mask<T> m, Vec<T> v)
CT_ALWAYS_FORCEINLINE auto storeu(Tag<bfloat16_t, 32> t, bfloat16_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm512_mask_storeu_epi16(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<float16_t, 32> t, float16_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm512_mask_storeu_epi16(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<float32_t, 16> t, float32_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm512_mask_storeu_ps(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<float64_t, 8> t, float64_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm512_mask_storeu_pd(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int8_t, 64> t, int8_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm512_mask_storeu_epi8(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint8_t, 64> t, uint8_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm512_mask_storeu_epi8(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int16_t, 32> t, int16_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm512_mask_storeu_epi16(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint16_t, 32> t, uint16_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm512_mask_storeu_epi16(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int32_t, 16> t, int32_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm512_mask_storeu_epi32(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint32_t, 16> t, uint32_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm512_mask_storeu_epi32(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int64_t, 8> t, int64_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm512_mask_storeu_epi64(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint64_t, 8> t, uint64_t* p, MaskOf(t) m, VecOf(t) v) -> void { _mm512_mask_storeu_epi64(p, m.v, v.v); }

// store(T* p, Mask<T> m, Vec<T> v)
// Note: no aligned store for epi8 and epi16
CT_ALWAYS_FORCEINLINE auto store(Tag<bfloat16_t, 32> t, bfloat16_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_mask_storeu_epi16(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<float16_t, 32> t, float16_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_mask_storeu_epi16(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<float32_t, 16> t, float32_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_mask_store_ps(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<float64_t, 8> t, float64_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_mask_store_pd(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int8_t, 64> t, int8_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_mask_storeu_epi8(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint8_t, 64> t, uint8_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_mask_storeu_epi8(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int16_t, 32> t, int16_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_mask_storeu_epi16(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint16_t, 32> t, uint16_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_mask_storeu_epi16(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int32_t, 16> t, int32_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_mask_store_epi32(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint32_t, 16> t, uint32_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_mask_store_epi32(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int64_t, 8> t, int64_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_mask_store_epi64(p, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint64_t, 8> t, uint64_t* p, MaskOf(t) m, VecOf(t) v) -> void { TL_CHECK_ALIGN _mm512_mask_store_epi64(p, m.v, v.v); }

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE auto _storeu_fwd_mask_512(Tag<T, N, P> t, T* p, nint_t n, VecOf(t) v) -> void {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = mwhilelt(t, 0, n);
  word::storeu(t, p, m, v);
}

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE auto _store_fwd_mask_512(Tag<T, N, P> t, T* p, nint_t n, VecOf(t) v) -> void {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto m = mwhilelt(t, 0, n);
  word::store(t, p, m, v);
}

// storeu(T* p, nint_t n, Vec<T> v)
CT_ALWAYS_FORCEINLINE auto storeu(Tag<bfloat16_t, 32> t, bfloat16_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_512(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<float16_t, 32> t, float16_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_512(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<float32_t, 16> t, float32_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_512(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<float64_t, 8> t, float64_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_512(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int8_t, 64> t, int8_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_512(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint8_t, 64> t, uint8_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_512(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int16_t, 32> t, int16_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_512(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint16_t, 32> t, uint16_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_512(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int32_t, 16> t, int32_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_512(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint32_t, 16> t, uint32_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_512(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<int64_t, 8> t, int64_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_512(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto storeu(Tag<uint64_t, 8> t, uint64_t* p, nint_t n, VecOf(t) v) -> void { _storeu_fwd_mask_512(t, p, n, v); }

// store(T* p, nint_t n, Vec<T> v)
CT_ALWAYS_FORCEINLINE auto store(Tag<bfloat16_t, 32> t, bfloat16_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_512(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<float16_t, 32> t, float16_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_512(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<float32_t, 16> t, float32_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_512(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<float64_t, 8> t, float64_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_512(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int8_t, 64> t, int8_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_512(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint8_t, 64> t, uint8_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_512(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int16_t, 32> t, int16_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_512(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint16_t, 32> t, uint16_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_512(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int32_t, 16> t, int32_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_512(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint32_t, 16> t, uint32_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_512(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<int64_t, 8> t, int64_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_512(t, p, n, v); }
CT_ALWAYS_FORCEINLINE auto store(Tag<uint64_t, 8> t, uint64_t* p, nint_t n, VecOf(t) v) -> void { _store_fwd_mask_512(t, p, n, v); }



/* ************************************************************************** */
//                             Get / set element                              //
/* ************************************************************************** */

// Helper: extract element from __m512i
#define TL_DEFINE_EXTRACT_512(dtype, postfix) static CT_ALWAYS_FORCEINLINE dtype _extract512_##postfix(__m512i v, int index)
TL_DEFINE_EXTRACT_512(int8_t, i8) {
  return _mm512_cvtsi512_si32(_mm512_permutex2var_epi8(v, _mm512_castsi128_si512(_mm_cvtsi32_si128(int(index))), v)) & 0xff;
}
TL_DEFINE_EXTRACT_512(uint8_t, u8) { return (uint8_t) _extract512_i8(v, index); }
TL_DEFINE_EXTRACT_512(int16_t, i16) {
  return _mm512_cvtsi512_si32(_mm512_permutex2var_epi16(v, _mm512_castsi128_si512(_mm_cvtsi32_si128(int(index))), v)) & 0xffff;
}
TL_DEFINE_EXTRACT_512(uint16_t, u16) { return (uint16_t) _extract512_i16(v, index); }
TL_DEFINE_EXTRACT_512(int32_t, i32) {
  return _mm512_cvtsi512_si32(_mm512_permutex2var_epi32(v, _mm512_castsi128_si512(_mm_cvtsi32_si128(int(index))), v));
}
TL_DEFINE_EXTRACT_512(uint32_t, u32) { return (uint32_t) _extract512_i32(v, index); }
TL_DEFINE_EXTRACT_512(int64_t, i64) {
  return _mm_cvtsi128_si64(_mm512_castsi512_si128(_mm512_permutex2var_epi64(v, _mm512_castsi128_si512(_mm_cvtsi32_si128(int(index))), v)));
}
TL_DEFINE_EXTRACT_512(uint64_t, u64) { return (uint64_t) _extract512_i64(v, index); }
#undef TL_DEFINE_EXTRACT_512

// Helper: get element from mask (native __mmask types)
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit_512(__mmask8 m, int index) { return (_cvtmask8_u32(m) >> index) & 1; }
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit_512(__mmask16 m, int index) { return (_cvtmask16_u32(m) >> index) & 1; }
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit_512(__mmask32 m, int index) { return (_cvtmask32_u32(m) >> index) & 1; }
static CT_ALWAYS_FORCEINLINE bool _get_mask_bit_512(__mmask64 m, int index) { return (_cvtmask64_u64(m) >> index) & 1; }

// Helper: set element in mask (native __mmask types)
static CT_ALWAYS_FORCEINLINE __mmask8 _set_mask_bit_512(__mmask8 m, int index, bool x) {
  uint32_t bits = _cvtmask8_u32(m);
  bits = (bits & ~(1u << index)) | ((x ? 1u : 0u) << index);
  return _cvtu32_mask8(bits);
}
static CT_ALWAYS_FORCEINLINE __mmask16 _set_mask_bit_512(__mmask16 m, int index, bool x) {
  uint32_t bits = _cvtmask16_u32(m);
  bits = (bits & ~(1u << index)) | ((x ? 1u : 0u) << index);
  return _cvtu32_mask16(bits);
}
static CT_ALWAYS_FORCEINLINE __mmask32 _set_mask_bit_512(__mmask32 m, int index, bool x) {
  uint32_t bits = _cvtmask32_u32(m);
  bits = (bits & ~(1u << index)) | ((x ? 1u : 0u) << index);
  return _cvtu32_mask32(bits);
}
static CT_ALWAYS_FORCEINLINE __mmask64 _set_mask_bit_512(__mmask64 m, int index, bool x) {
  uint64_t bits = _cvtmask64_u64(m);
  bits = (bits & ~(1ull << index)) | ((x ? 1ull : 0ull) << index);
  return _cvtu64_mask64(bits);
}



// get(Vec<T> v, nint_t index)
CT_ALWAYS_FORCEINLINE auto get(Tag<bfloat16_t, 32> t, VecOf(t) v, nint_t index) -> bfloat16_t { TL_CHECK_INDEX(index); union { bfloat16_t b; int16_t i; } u; u.i = _extract512_i16(v.v, (int) index); return u.b; }
CT_ALWAYS_FORCEINLINE auto get(Tag<float16_t, 32> t, VecOf(t) v, nint_t index) -> float16_t { TL_CHECK_INDEX(index); union { float16_t h; int16_t i; } u; u.i = _extract512_i16(v.v, (int) index); return u.h; }
CT_ALWAYS_FORCEINLINE auto get(Tag<float32_t, 16> t, VecOf(t) v, nint_t index) -> float32_t { TL_CHECK_INDEX(index);
  return _mm512_cvtss_f32(_mm512_permutex2var_ps(v.v, _mm512_castsi128_si512(_mm_cvtsi32_si128(int(index))), v.v));
}
CT_ALWAYS_FORCEINLINE auto get(Tag<float64_t, 8> t, VecOf(t) v, nint_t index) -> float64_t { TL_CHECK_INDEX(index);
  return _mm512_cvtsd_f64(_mm512_permutex2var_pd(v.v, _mm512_castsi128_si512(_mm_cvtsi32_si128(int(index))), v.v));
}
CT_ALWAYS_FORCEINLINE auto get(Tag<int8_t, 64> t, VecOf(t) v, nint_t index) -> int8_t { TL_CHECK_INDEX(index); return _extract512_i8(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<uint8_t, 64> t, VecOf(t) v, nint_t index) -> uint8_t { TL_CHECK_INDEX(index); return _extract512_u8(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<int16_t, 32> t, VecOf(t) v, nint_t index) -> int16_t { TL_CHECK_INDEX(index); return _extract512_i16(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<uint16_t, 32> t, VecOf(t) v, nint_t index) -> uint16_t { TL_CHECK_INDEX(index); return _extract512_u16(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<int32_t, 16> t, VecOf(t) v, nint_t index) -> int32_t { TL_CHECK_INDEX(index); return _extract512_i32(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<uint32_t, 16> t, VecOf(t) v, nint_t index) -> uint32_t { TL_CHECK_INDEX(index); return _extract512_u32(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<int64_t, 8> t, VecOf(t) v, nint_t index) -> int64_t { TL_CHECK_INDEX(index); return _extract512_i64(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<uint64_t, 8> t, VecOf(t) v, nint_t index) -> uint64_t { TL_CHECK_INDEX(index); return _extract512_u64(v.v, (int) index); }

// get(Mask<T> m, nint_t index)
CT_ALWAYS_FORCEINLINE auto get(Tag<bfloat16_t, 32> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_512(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<float16_t, 32> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_512(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<float32_t, 16> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_512(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<float64_t, 8> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_512(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<int8_t, 64> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_512(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<uint8_t, 64> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_512(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<int16_t, 32> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_512(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<uint16_t, 32> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_512(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<int32_t, 16> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_512(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<uint32_t, 16> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_512(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<int64_t, 8> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_512(v.v, (int) index); }
CT_ALWAYS_FORCEINLINE auto get(Tag<uint64_t, 8> t, MaskOf(t) v, nint_t index) -> bool { TL_CHECK_INDEX(index); return _get_mask_bit_512(v.v, (int) index); }



// set(Vec<T> v, nint_t index, T x)
CT_ALWAYS_FORCEINLINE auto set(Tag<bfloat16_t, 32> t, VecOf(t) v, nint_t index, bfloat16_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  union { bfloat16_t f; int16_t i; } u { .f = x };
  auto mask = _cvtu32_mask32(1u << (unsigned) index);
  return _mm512_mask_mov_epi16(v.v, mask, _mm512_set1_epi16(u.i));
}
CT_ALWAYS_FORCEINLINE auto set(Tag<float16_t, 32> t, VecOf(t) v, nint_t index, float16_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  union { float16_t f; int16_t i; } u { .f = x };
  auto mask = _cvtu32_mask32(1u << (unsigned) index);
  return _mm512_mask_mov_epi16(v.v, mask, _mm512_set1_epi16(u.i));
}
CT_ALWAYS_FORCEINLINE auto set(Tag<float32_t, 16> t, VecOf(t) v, nint_t index, float32_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask16(1u << (unsigned) index);
  return _mm512_mask_mov_ps(v.v, mask, _mm512_set1_ps(x));
}
CT_ALWAYS_FORCEINLINE auto set(Tag<float64_t, 8> t, VecOf(t) v, nint_t index, float64_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask8(1u << (unsigned) index);
  return _mm512_mask_mov_pd(v.v, mask, _mm512_set1_pd(x));
}
CT_ALWAYS_FORCEINLINE auto set(Tag<int8_t, 64> t, VecOf(t) v, nint_t index, int8_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  auto mask = _cvtu64_mask64(1ull << (unsigned) index);
  return _mm512_mask_mov_epi8(v.v, mask, _mm512_set1_epi8(x));
}
CT_ALWAYS_FORCEINLINE auto set(Tag<uint8_t, 64> t, VecOf(t) v, nint_t index, uint8_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  auto mask = _cvtu64_mask64(1ull << (unsigned) index);
  return _mm512_mask_mov_epi8(v.v, mask, _mm512_set1_epi8((int8_t)x));
}
CT_ALWAYS_FORCEINLINE auto set(Tag<int16_t, 32> t, VecOf(t) v, nint_t index, int16_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask32(1u << (unsigned) index);
  return _mm512_mask_mov_epi16(v.v, mask, _mm512_set1_epi16(x));
}
CT_ALWAYS_FORCEINLINE auto set(Tag<uint16_t, 32> t, VecOf(t) v, nint_t index, uint16_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask32(1u << (unsigned) index);
  return _mm512_mask_mov_epi16(v.v, mask, _mm512_set1_epi16((int16_t)x));
}
CT_ALWAYS_FORCEINLINE auto set(Tag<int32_t, 16> t, VecOf(t) v, nint_t index, int32_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask16(1u << (unsigned) index);
  return _mm512_mask_mov_epi32(v.v, mask, _mm512_set1_epi32(x));
}
CT_ALWAYS_FORCEINLINE auto set(Tag<uint32_t, 16> t, VecOf(t) v, nint_t index, uint32_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask16(1u << (unsigned) index);
  return _mm512_mask_mov_epi32(v.v, mask, _mm512_set1_epi32((int32_t)x));
}
CT_ALWAYS_FORCEINLINE auto set(Tag<int64_t, 8> t, VecOf(t) v, nint_t index, int64_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask8(1u << (unsigned) index);
  return _mm512_mask_mov_epi64(v.v, mask, _mm512_set1_epi64(x));
}
CT_ALWAYS_FORCEINLINE auto set(Tag<uint64_t, 8> t, VecOf(t) v, nint_t index, uint64_t x) -> VecOf(t) {
  TL_CHECK_INDEX(index);
  auto mask = _cvtu32_mask8(1u << (unsigned) index);
  return _mm512_mask_mov_epi64(v.v, mask, _mm512_set1_epi64((int64_t)x));
}

// set(Mask<T> m, nint_t index, bool x)
CT_ALWAYS_FORCEINLINE auto set(Tag<bfloat16_t, 32> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_512(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<float16_t, 32> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_512(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<float32_t, 16> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_512(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<float64_t, 8> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_512(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<int8_t, 64> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_512(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<uint8_t, 64> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_512(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<int16_t, 32> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_512(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<uint16_t, 32> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_512(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<int32_t, 16> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_512(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<uint32_t, 16> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_512(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<int64_t, 8> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_512(v.v, (int) index, x); }
CT_ALWAYS_FORCEINLINE auto set(Tag<uint64_t, 8> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) { TL_CHECK_INDEX(index); return _set_mask_bit_512(v.v, (int) index, x); }



/* ************************************************************************** */
//                       Basic arithmetic operations                          //
/* ************************************************************************** */
// TODO arithmetics for float16_t and bfloat16_t


// add(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto add(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_add_ps(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_add_pd(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_add_epi8(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_add_epi8(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_add_epi16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_add_epi16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_add_epi32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_add_epi32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_add_epi64(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_add_epi64(a.v, b.v); }

// add(Vec<T> a, Vec<T> b, Mask<T> m)
CT_ALWAYS_FORCEINLINE auto add(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_add_ps(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_add_pd(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_add_epi8(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_add_epi8(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_add_epi16(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_add_epi16(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_add_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_add_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_add_epi64(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto add(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_add_epi64(a.v, m.v, a.v, b.v); }



// sub(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto sub(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_sub_ps(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_sub_pd(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_sub_epi8(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_sub_epi8(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_sub_epi16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_sub_epi16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_sub_epi32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_sub_epi32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_sub_epi64(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_sub_epi64(a.v, b.v); }

// sub(Vec<T> a, Vec<T> b, Mask<T> m)
CT_ALWAYS_FORCEINLINE auto sub(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_sub_ps(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_sub_pd(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_sub_epi8(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_sub_epi8(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_sub_epi16(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_sub_epi16(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_sub_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_sub_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_sub_epi64(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto sub(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_sub_epi64(a.v, m.v, a.v, b.v); }



// mul(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto mul(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_mul_ps(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_mul_pd(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) {
  // 8-bit multiplication using 16-bit intermediate
  auto zero = _mm512_setzero_si512();
  auto a_lo = _mm512_unpacklo_epi8(a.v, zero);
  auto a_hi = _mm512_unpackhi_epi8(a.v, zero);
  auto b_lo = _mm512_unpacklo_epi8(b.v, zero);
  auto b_hi = _mm512_unpackhi_epi8(b.v, zero);
  a_lo = _mm512_mullo_epi16(a_lo, b_lo);
  a_hi = _mm512_mullo_epi16(a_hi, b_hi);
  return _mm512_packus_epi16(a_lo, a_hi);
}
CT_ALWAYS_FORCEINLINE auto mul(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return mul(Tag<int8_t, 64>(), a.v, b.v).v; }
CT_ALWAYS_FORCEINLINE auto mul(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_mullo_epi16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_mullo_epi16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_mullo_epi32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_mullo_epi32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_mullo_epi64(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_mullo_epi64(a.v, b.v); }

// mul(Vec<T> a, Vec<T> b, Mask<T> m)
CT_ALWAYS_FORCEINLINE auto mul(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_mul_ps(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_mul_pd(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_blend_epi8(m.v, a.v, mul(t, a.v, b.v).v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_blend_epi8(m.v, a.v, mul(t, a.v, b.v).v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_mullo_epi16(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_mullo_epi16(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_mullo_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_mullo_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_mullo_epi64(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto mul(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_mullo_epi64(a.v, m.v, a.v, b.v); }



// div(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto div(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_div_ps(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto div(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_div_pd(a.v, b.v); }

// div(Vec<T> a, Vec<T> b, Mask<T> m)
CT_ALWAYS_FORCEINLINE auto div(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_div_ps(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto div(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_div_pd(a.v, m.v, a.v, b.v); }



// rcp(Vec<T> v)
CT_ALWAYS_FORCEINLINE auto rcp(Tag<float32_t, 16> t, VecOf(t) v) -> VecOf(t) { return _mm512_rcp14_ps(v.v); }
CT_ALWAYS_FORCEINLINE auto rcp(Tag<float64_t, 8> t, VecOf(t) v) -> VecOf(t) { return _mm512_rcp14_pd(v.v); }

// rcp(Vec<T> v, Mask<T> m, Vec<T> default_v)
CT_ALWAYS_FORCEINLINE auto rcp(Tag<float32_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_rcp14_ps(default_v.v, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto rcp(Tag<float64_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_rcp14_pd(default_v.v, m.v, v.v); }



// max(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto max(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_max_ps(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_max_pd(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_max_epi8(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_max_epu8(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_max_epi16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_max_epu16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_max_epi32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_max_epu32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_max_epi64(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_max_epu64(a.v, b.v); }

// max(Vec<T> a, Vec<T> b, Mask<T> m)
CT_ALWAYS_FORCEINLINE auto max(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_max_ps(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_max_pd(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_max_epi8(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_max_epu8(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_max_epi16(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_max_epu16(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_max_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_max_epu32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_max_epi64(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto max(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_max_epu64(a.v, m.v, a.v, b.v); }



// min(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto min(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_min_ps(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_min_pd(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_min_epi8(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_min_epu8(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_min_epi16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_min_epu16(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_min_epi32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_min_epu32(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_min_epi64(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_min_epu64(a.v, b.v); }

// min(Vec<T> a, Vec<T> b, Mask<T> m)
CT_ALWAYS_FORCEINLINE auto min(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_min_ps(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_min_pd(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_min_epi8(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_min_epu8(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_min_epi16(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_min_epu16(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_min_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_min_epu32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_min_epi64(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto min(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_min_epu64(a.v, m.v, a.v, b.v); }



// bit_and(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_and_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_and_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_and_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_and_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_and_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_and_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_and_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_and_si512(a.v, b.v); }

// bit_and(Vec<T> a, VecOf(t) b, Mask<T> m)
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_blend_epi8(m.v, a.v, _mm512_and_si512(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_blend_epi8(m.v, a.v, _mm512_and_si512(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_blend_epi16(m.v, a.v, _mm512_and_si512(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_blend_epi16(m.v, a.v, _mm512_and_si512(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_and_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_and_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_and_epi64(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_and(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_and_epi64(a.v, m.v, a.v, b.v); }



// bit_or(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_or_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_or_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_or_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_or_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_or_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_or_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_or_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_or_si512(a.v, b.v); }

// bit_or(Vec<T> a, Vec<T> b, Mask<T> m)
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_blend_epi8(m.v, a.v, _mm512_or_si512(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_blend_epi8(m.v, a.v, _mm512_or_si512(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_blend_epi16(m.v, a.v, _mm512_or_si512(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_blend_epi16(m.v, a.v, _mm512_or_si512(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_or_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_or_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_or_epi64(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_or(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_or_epi64(a.v, m.v, a.v, b.v); }



// bit_xor(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_xor_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_xor_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_xor_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_xor_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_xor_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_xor_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_xor_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_xor_si512(a.v, b.v); }

// bit_xor(Vec<T> a, Vec<T> b, Mask<T> m)
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_blend_epi8(m.v, a.v, _mm512_xor_si512(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_blend_epi8(m.v, a.v, _mm512_xor_si512(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_blend_epi16(m.v, a.v, _mm512_xor_si512(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_blend_epi16(m.v, a.v, _mm512_xor_si512(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_xor_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_xor_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_xor_epi64(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_xor(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_xor_epi64(a.v, m.v, a.v, b.v); }



// bit_andnot(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_andnot_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_andnot_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_andnot_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_andnot_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_andnot_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_andnot_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_andnot_si512(a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { return _mm512_andnot_si512(a.v, b.v); }

// bit_andnot(Vec<T> a, Vec<T> b, Mask<T> m)
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_blend_epi8(m.v, a.v, _mm512_andnot_si512(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_blend_epi8(m.v, a.v, _mm512_andnot_si512(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_blend_epi16(m.v, a.v, _mm512_andnot_si512(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_blend_epi16(m.v, a.v, _mm512_andnot_si512(a.v, b.v)); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_andnot_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_andnot_epi32(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_andnot_epi64(a.v, m.v, a.v, b.v); }
CT_ALWAYS_FORCEINLINE auto bit_andnot(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_andnot_epi64(a.v, m.v, a.v, b.v); }



// bit_not(Vec<T> v)
#define _tlmm_vec_not512(a) _mm512_xor_si512((a), _mm512_set1_epi32(-1))
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<int8_t, 64> t, VecOf(t) v) -> VecOf(t) { return _tlmm_vec_not512(v.v); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<uint8_t, 64> t, VecOf(t) v) -> VecOf(t) { return _tlmm_vec_not512(v.v); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<int16_t, 32> t, VecOf(t) v) -> VecOf(t) { return _tlmm_vec_not512(v.v); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<uint16_t, 32> t, VecOf(t) v) -> VecOf(t) { return _tlmm_vec_not512(v.v); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<int32_t, 16> t, VecOf(t) v) -> VecOf(t) { return _tlmm_vec_not512(v.v); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<uint32_t, 16> t, VecOf(t) v) -> VecOf(t) { return _tlmm_vec_not512(v.v); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<int64_t, 8> t, VecOf(t) v) -> VecOf(t) { return _tlmm_vec_not512(v.v); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<uint64_t, 8> t, VecOf(t) v) -> VecOf(t) { return _tlmm_vec_not512(v.v); }
#undef _tlmm_vec_not512

// bit_not(Vec<T> v, Mask<T> m, Vec<T> default_v)
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<int8_t, 64> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_blend_epi8(m.v, default_v.v, _mm512_xor_si512(v.v, _mm512_set1_epi32(-1))); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<uint8_t, 64> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_blend_epi8(m.v, default_v.v, _mm512_xor_si512(v.v, _mm512_set1_epi32(-1))); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<int16_t, 32> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_blend_epi16(m.v, default_v.v, _mm512_xor_si512(v.v, _mm512_set1_epi32(-1))); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<uint16_t, 32> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_blend_epi16(m.v, default_v.v, _mm512_xor_si512(v.v, _mm512_set1_epi32(-1))); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<int32_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_xor_epi32(default_v.v, m.v, v.v, _mm512_set1_epi32(-1)); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<uint32_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_xor_epi32(default_v.v, m.v, v.v, _mm512_set1_epi32(-1)); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<int64_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_xor_epi64(default_v.v, m.v, v.v, _mm512_set1_epi64(-1)); }
CT_ALWAYS_FORCEINLINE auto bit_not(Tag<uint64_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_xor_epi64(default_v.v, m.v, v.v, _mm512_set1_epi64(-1)); }



// neg(Vec<T> v)
CT_ALWAYS_FORCEINLINE auto neg(Tag<float32_t, 16> t, VecOf(t) v) -> VecOf(t) { return _mm512_sub_ps(_mm512_setzero_ps(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<float64_t, 8> t, VecOf(t) v) -> VecOf(t) { return _mm512_sub_pd(_mm512_setzero_pd(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<int8_t, 64> t, VecOf(t) v) -> VecOf(t) { return _mm512_sub_epi8(_mm512_setzero_si512(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<uint8_t, 64> t, VecOf(t) v) -> VecOf(t) { return _mm512_sub_epi8(_mm512_setzero_si512(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<int16_t, 32> t, VecOf(t) v) -> VecOf(t) { return _mm512_sub_epi16(_mm512_setzero_si512(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<uint16_t, 32> t, VecOf(t) v) -> VecOf(t) { return _mm512_sub_epi16(_mm512_setzero_si512(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<int32_t, 16> t, VecOf(t) v) -> VecOf(t) { return _mm512_sub_epi32(_mm512_setzero_si512(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<uint32_t, 16> t, VecOf(t) v) -> VecOf(t) { return _mm512_sub_epi32(_mm512_setzero_si512(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<int64_t, 8> t, VecOf(t) v) -> VecOf(t) { return _mm512_sub_epi64(_mm512_setzero_si512(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<uint64_t, 8> t, VecOf(t) v) -> VecOf(t) { return _mm512_sub_epi64(_mm512_setzero_si512(), v.v); }

// neg(Vec<T> v, Mask<T> m, Vec<T> default_v)
CT_ALWAYS_FORCEINLINE auto neg(Tag<float32_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_sub_ps(default_v.v, m.v, _mm512_setzero_ps(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<float64_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_sub_pd(default_v.v, m.v, _mm512_setzero_pd(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<int8_t, 64> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_sub_epi8(default_v.v, m.v, _mm512_setzero_si512(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<uint8_t, 64> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_sub_epi8(default_v.v, m.v, _mm512_setzero_si512(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<int16_t, 32> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_sub_epi16(default_v.v, m.v, _mm512_setzero_si512(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<uint16_t, 32> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_sub_epi16(default_v.v, m.v, _mm512_setzero_si512(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<int32_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_sub_epi32(default_v.v, m.v, _mm512_setzero_si512(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<uint32_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_sub_epi32(default_v.v, m.v, _mm512_setzero_si512(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<int64_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_sub_epi64(default_v.v, m.v, _mm512_setzero_si512(), v.v); }
CT_ALWAYS_FORCEINLINE auto neg(Tag<uint64_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_sub_epi64(default_v.v, m.v, _mm512_setzero_si512(), v.v); }



// abs(Vec<T> v)
CT_ALWAYS_FORCEINLINE auto abs(Tag<float32_t, 16> t, VecOf(t) v) -> VecOf(t) { return _mm512_and_ps(_mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)), v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<float64_t, 8> t, VecOf(t) v) -> VecOf(t) { return _mm512_and_pd(_mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFFLL)), v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<int8_t, 64> t, VecOf(t) v) -> VecOf(t) { return _mm512_abs_epi8(v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<uint8_t, 64> t, VecOf(t) v) -> VecOf(t) { return v.v; }
CT_ALWAYS_FORCEINLINE auto abs(Tag<int16_t, 32> t, VecOf(t) v) -> VecOf(t) { return _mm512_abs_epi16(v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<uint16_t, 32> t, VecOf(t) v) -> VecOf(t) { return v.v; }
CT_ALWAYS_FORCEINLINE auto abs(Tag<int32_t, 16> t, VecOf(t) v) -> VecOf(t) { return _mm512_abs_epi32(v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<uint32_t, 16> t, VecOf(t) v) -> VecOf(t) { return v.v; }
CT_ALWAYS_FORCEINLINE auto abs(Tag<int64_t, 8> t, VecOf(t) v) -> VecOf(t) { return _mm512_abs_epi64(v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<uint64_t, 8> t, VecOf(t) v) -> VecOf(t) { return v.v; }

// abs(Vec<T> v, Mask<T> m, Vec<T> default_v)
CT_ALWAYS_FORCEINLINE auto abs(Tag<float32_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_and_ps(default_v.v, m.v, _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF)), v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<float64_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_and_pd(default_v.v, m.v, _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFFLL)), v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<int8_t, 64> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_abs_epi8(default_v.v, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<uint8_t, 64> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_blend_epi8(m.v, default_v.v, v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<int16_t, 32> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_abs_epi16(default_v.v, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<uint16_t, 32> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_blend_epi16(m.v, default_v.v, v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<int32_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_abs_epi32(default_v.v, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<uint32_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_blend_epi32(m.v, default_v.v, v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<int64_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_abs_epi64(default_v.v, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto abs(Tag<uint64_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_blend_epi64(m.v, default_v.v, v.v); }



// sqrt(Vec<T> v)
CT_ALWAYS_FORCEINLINE auto sqrt(Tag<float32_t, 16> t, VecOf(t) v) -> VecOf(t) { return _mm512_sqrt_ps(v.v); }
CT_ALWAYS_FORCEINLINE auto sqrt(Tag<float64_t, 8> t, VecOf(t) v) -> VecOf(t) { return _mm512_sqrt_pd(v.v); }

// sqrt(Vec<T> v, Mask<T> m, Vec<T> default_v)
CT_ALWAYS_FORCEINLINE auto sqrt(Tag<float32_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_sqrt_ps(default_v.v, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto sqrt(Tag<float64_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_sqrt_pd(default_v.v, m.v, v.v); }



// rsqrt(Vec<T> v)
CT_ALWAYS_FORCEINLINE auto rsqrt(Tag<float32_t, 16> t, VecOf(t) v) -> VecOf(t) { return _mm512_rsqrt14_ps(v.v); }
CT_ALWAYS_FORCEINLINE auto rsqrt(Tag<float64_t, 8> t, VecOf(t) v) -> VecOf(t) { return _mm512_rsqrt14_pd(v.v); }

// rsqrt(Vec<T> v, Mask<T> m, Vec<T> default_v)
CT_ALWAYS_FORCEINLINE auto rsqrt(Tag<float32_t, 16> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_rsqrt14_ps(default_v.v, m.v, v.v); }
CT_ALWAYS_FORCEINLINE auto rsqrt(Tag<float64_t, 8> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { return _mm512_mask_rsqrt14_pd(default_v.v, m.v, v.v); }



// cmpeq(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_ps_mask(a.v, b.v, _CMP_EQ_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_pd_mask(a.v, b.v, _CMP_EQ_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_EQ); }

// cmpeq(Vec<T> a, Vec<T> b, Mask<T> m)
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_EQ_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_EQ_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }
CT_ALWAYS_FORCEINLINE auto cmpeq(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_EQ); }



// cmpne(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_ps_mask(a.v, b.v, _CMP_NEQ_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_pd_mask(a.v, b.v, _CMP_NEQ_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NE); }

// cmpne(Vec<T> a, Vec<T> b, Mask<T> m)
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_NEQ_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_NEQ_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }
CT_ALWAYS_FORCEINLINE auto cmpne(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NE); }



// cmplt(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_ps_mask(a.v, b.v, _CMP_LT_OQ); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_pd_mask(a.v, b.v, _CMP_LT_OQ); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_LT); }

// cmplt(Vec<T> a, Vec<T> b, Mask<T> m)
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_LT_OQ); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_LT_OQ); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }
CT_ALWAYS_FORCEINLINE auto cmplt(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_LT); }



// cmpgt(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_ps_mask(a.v, b.v, _CMP_GT_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_pd_mask(a.v, b.v, _CMP_GT_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NLE); }

// cmpgt(Vec<T> a, Vec<T> b, Mask<T> m)
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_GT_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_GT_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }
CT_ALWAYS_FORCEINLINE auto cmpgt(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NLE); }



// cmple(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto cmple(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_ps_mask(a.v, b.v, _CMP_LE_OQ); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_pd_mask(a.v, b.v, _CMP_LE_OQ); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_LE); }

// cmple(Vec<T> a, Vec<T> b, Mask<T> m)
CT_ALWAYS_FORCEINLINE auto cmple(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_LE_OQ); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_LE_OQ); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }
CT_ALWAYS_FORCEINLINE auto cmple(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_LE); }



// cmpge(Vec<T> a, Vec<T> b)
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_ps_mask(a.v, b.v, _CMP_GE_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_pd_mask(a.v, b.v, _CMP_GE_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi8_mask(a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epu8_mask(a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi16_mask(a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epu16_mask(a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi32_mask(a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epu32_mask(a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epi64_mask(a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { return _mm512_cmp_epu64_mask(a.v, b.v, _MM_CMPINT_NLT); }

// cmpge(Vec<T> a, Vec<T> b, Mask<T> m)
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<float32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_ps_mask(m.v, a.v, b.v, _CMP_GE_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<float64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_pd_mask(m.v, a.v, b.v, _CMP_GE_OQ); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi8_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint8_t, 64> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epu8_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi16_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint16_t, 32> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epu16_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi32_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint32_t, 16> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epu32_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<int64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epi64_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }
CT_ALWAYS_FORCEINLINE auto cmpge(Tag<uint64_t, 8> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_epu64_mask(m.v, a.v, b.v, _MM_CMPINT_NLT); }



// isnan(Vec<T> v)
CT_ALWAYS_FORCEINLINE auto isnan(Tag<float32_t, 16> t, VecOf(t) v) -> MaskOf(t) { return _mm512_cmp_ps_mask(v.v, v.v, _CMP_UNORD_Q); }
CT_ALWAYS_FORCEINLINE auto isnan(Tag<float64_t, 8> t, VecOf(t) v) -> MaskOf(t) { return _mm512_cmp_pd_mask(v.v, v.v, _CMP_UNORD_Q); }

// isnan(Vec<T> v, Mask<T> m)
CT_ALWAYS_FORCEINLINE auto isnan(Tag<float32_t, 16> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_ps_mask(m.v, v.v, v.v, _CMP_UNORD_Q); }
CT_ALWAYS_FORCEINLINE auto isnan(Tag<float64_t, 8> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_pd_mask(m.v, v.v, v.v, _CMP_UNORD_Q); }



// isposinf(Vec<T> v)
CT_ALWAYS_FORCEINLINE auto isposinf(Tag<float32_t, 16> t, VecOf(t) v) -> MaskOf(t) { return _mm512_cmp_ps_mask(v.v, _mm512_set1_ps(INFINITY), _CMP_EQ_OQ); }
CT_ALWAYS_FORCEINLINE auto isposinf(Tag<float64_t, 8> t, VecOf(t) v) -> MaskOf(t) { return _mm512_cmp_pd_mask(v.v, _mm512_set1_pd(INFINITY), _CMP_EQ_OQ); }

// isposinf(Vec<T> v, Mask<T> m)
CT_ALWAYS_FORCEINLINE auto isposinf(Tag<float32_t, 16> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_ps_mask(m.v, v.v, _mm512_set1_ps(INFINITY), _CMP_EQ_OQ); }
CT_ALWAYS_FORCEINLINE auto isposinf(Tag<float64_t, 8> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_pd_mask(m.v, v.v, _mm512_set1_pd(INFINITY), _CMP_EQ_OQ); }



// isneginf(Vec<T> v)
CT_ALWAYS_FORCEINLINE auto isneginf(Tag<float32_t, 16> t, VecOf(t) v) -> MaskOf(t) { return _mm512_cmp_ps_mask(v.v, _mm512_set1_ps(-INFINITY), _CMP_EQ_OQ); }
CT_ALWAYS_FORCEINLINE auto isneginf(Tag<float64_t, 8> t, VecOf(t) v) -> MaskOf(t) { return _mm512_cmp_pd_mask(v.v, _mm512_set1_pd(-INFINITY), _CMP_EQ_OQ); }

// isneginf(Vec<T> v, Mask<T> m)
CT_ALWAYS_FORCEINLINE auto isneginf(Tag<float32_t, 16> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_ps_mask(m.v, v.v, _mm512_set1_ps(-INFINITY), _CMP_EQ_OQ); }
CT_ALWAYS_FORCEINLINE auto isneginf(Tag<float64_t, 8> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return _mm512_mask_cmp_pd_mask(m.v, v.v, _mm512_set1_pd(-INFINITY), _CMP_EQ_OQ); }



// isinf(Vec<T> v)
CT_ALWAYS_FORCEINLINE auto isinf(Tag<float32_t, 16> t, VecOf(t) v) -> MaskOf(t) { return isposinf(t, abs(t, v)).v; }
CT_ALWAYS_FORCEINLINE auto isinf(Tag<float64_t, 8> t, VecOf(t) v) -> MaskOf(t) { return isposinf(t, abs(t, v)).v; }

// isinf(Vec<T> v, Mask<T> m)
CT_ALWAYS_FORCEINLINE auto isinf(Tag<float32_t, 16> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return isinf(t, v).v & m.v; }
CT_ALWAYS_FORCEINLINE auto isinf(Tag<float64_t, 8> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { return isinf(t, v).v & m.v; }



/* ************************************************************************** */
//                            Bit shift operations                            */
/* ************************************************************************** */

// Helper for 8-bit left shift (no native _mm512_sll_epi8)
static CT_ALWAYS_FORCEINLINE __m512i _bit_shl_epi8_512(__m512i v, int count) {
  auto zero = _mm512_setzero_si512();
  auto lo = _mm512_unpacklo_epi8(v, zero);
  auto hi = _mm512_unpackhi_epi8(v, zero);
  auto count_vec = _mm_cvtsi32_si128(count);
  lo = _mm512_sll_epi16(lo, count_vec);
  hi = _mm512_sll_epi16(hi, count_vec);
  auto mask = _mm512_set1_epi16(0xFF);
  lo = _mm512_and_si512(lo, mask);
  hi = _mm512_and_si512(hi, mask);
  return _mm512_packus_epi16(lo, hi);
}

// Helper for 8-bit logical right shift
static CT_ALWAYS_FORCEINLINE __m512i _bit_srl_epi8_512(__m512i v, int count) {
  auto zero = _mm512_setzero_si512();
  auto lo = _mm512_unpacklo_epi8(zero, v);
  auto hi = _mm512_unpackhi_epi8(zero, v);
  lo = _mm512_srli_epi16(lo, 8);
  hi = _mm512_srli_epi16(hi, 8);
  auto count_vec = _mm_cvtsi32_si128(count);
  lo = _mm512_srl_epi16(lo, count_vec);
  hi = _mm512_srl_epi16(hi, count_vec);
  return _mm512_packus_epi16(lo, hi);
}

// Helper for 8-bit arithmetic right shift
static CT_ALWAYS_FORCEINLINE __m512i _bit_sra_epi8_512(__m512i v, int count) {
  auto zero = _mm512_setzero_si512();
  auto signs = _mm512_movm_epi8(_mm512_cmpgt_epi8_mask(zero, v));
  auto lo = _mm512_unpacklo_epi8(v, signs);
  auto hi = _mm512_unpackhi_epi8(v, signs);
  auto count_vec = _mm_cvtsi32_si128(count);
  lo = _mm512_sra_epi16(lo, count_vec);
  hi = _mm512_sra_epi16(hi, count_vec);
  return _mm512_packs_epi16(lo, hi);
}

// bit_shl(Vec<T> v, int count)
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<int8_t, 64> t, VecOf(t) v, int count) -> VecOf(t) { return _bit_shl_epi8_512(v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<uint8_t, 64> t, VecOf(t) v, int count) -> VecOf(t) { return _bit_shl_epi8_512(v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<int16_t, 32> t, VecOf(t) v, int count) -> VecOf(t) { return _mm512_slli_epi16(v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<uint16_t, 32> t, VecOf(t) v, int count) -> VecOf(t) { return _mm512_slli_epi16(v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<int32_t, 16> t, VecOf(t) v, int count) -> VecOf(t) { return _mm512_slli_epi32(v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<uint32_t, 16> t, VecOf(t) v, int count) -> VecOf(t) { return _mm512_slli_epi32(v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<int64_t, 8> t, VecOf(t) v, int count) -> VecOf(t) { return _mm512_slli_epi64(v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<uint64_t, 8> t, VecOf(t) v, int count) -> VecOf(t) { return _mm512_slli_epi64(v.v, count); }

// bit_shl(Vec<T> v, int count, Mask<T> m)
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<int8_t, 64> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_blend_epi8(m.v, v.v, _bit_shl_epi8_512(v.v, count)); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<uint8_t, 64> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_blend_epi8(m.v, v.v, _bit_shl_epi8_512(v.v, count)); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<int16_t, 32> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_slli_epi16(v.v, m.v, v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<uint16_t, 32> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_slli_epi16(v.v, m.v, v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<int32_t, 16> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_slli_epi32(v.v, m.v, v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<uint32_t, 16> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_slli_epi32(v.v, m.v, v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<int64_t, 8> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_slli_epi64(v.v, m.v, v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shl(Tag<uint64_t, 8> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_slli_epi64(v.v, m.v, v.v, count); }



// bit_shr(Vec<T> v, int count) - Signed: arithmetic shift, Unsigned: logical shift
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<int8_t, 64> t, VecOf(t) v, int count) -> VecOf(t) { return _bit_sra_epi8_512(v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<int16_t, 32> t, VecOf(t) v, int count) -> VecOf(t) { return _mm512_srai_epi16(v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<int32_t, 16> t, VecOf(t) v, int count) -> VecOf(t) { return _mm512_srai_epi32(v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<int64_t, 8> t, VecOf(t) v, int count) -> VecOf(t) { return _mm512_srai_epi64(v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<uint8_t, 64> t, VecOf(t) v, int count) -> VecOf(t) { return _bit_srl_epi8_512(v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<uint16_t, 32> t, VecOf(t) v, int count) -> VecOf(t) { return _mm512_srli_epi16(v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<uint32_t, 16> t, VecOf(t) v, int count) -> VecOf(t) { return _mm512_srli_epi32(v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<uint64_t, 8> t, VecOf(t) v, int count) -> VecOf(t) { return _mm512_srli_epi64(v.v, count); }

// bit_shr(Vec<T> v, int count, Mask<T> m)
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<int8_t, 64> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_blend_epi8(m.v, v.v, _bit_sra_epi8_512(v.v, count)); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<int16_t, 32> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_srai_epi16(v.v, m.v, v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<int32_t, 16> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_srai_epi32(v.v, m.v, v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<int64_t, 8> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_srai_epi64(v.v, m.v, v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<uint8_t, 64> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_blend_epi8(m.v, v.v, _bit_srl_epi8_512(v.v, count)); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<uint16_t, 32> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_srli_epi16(v.v, m.v, v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<uint32_t, 16> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_srli_epi32(v.v, m.v, v.v, count); }
CT_ALWAYS_FORCEINLINE auto bit_shr(Tag<uint64_t, 8> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) { return _mm512_mask_srli_epi64(v.v, m.v, v.v, count); }



#undef TL_CHECK_COUNT
#undef TL_CHECK_INDEX
#undef TL_CHECK_ALIGN
} // namespace word
} // namespace ct::tl::vec
//@formatter:on

#endif //CTORCH_X86_512_H
