//
// Created by renyz on 2026/3/17.
//

#ifndef CTORCH_X86_128_H
#define CTORCH_X86_128_H

#include "CoreDefs.h"
#include "tl/cpu/impl/Scalar.h"
#include "tl/util/Math.h"

#ifndef ARCH_X86_FAMILY
#error "Not x86 platform"
#endif

#include <immintrin.h>

namespace ct::tl::vec {

template <>
struct VecDefs<float32_t, 4> : public ScalarVecDefs<float32_t, 4> {
  static constexpr nint_t num_words = 1;

  static constexpr nint_t word_size() { return 4; }

  static constexpr nint_t max_word_size = 4;

  static constexpr nint_t size() { return 4; };
  static constexpr nint_t max_size = 4;
  static constexpr bool is_scalable = false;
  static constexpr bool is_default_impl = false;
  static constexpr bool is_word_vec = true;
  using VecType = __m128;
  #ifdef HAS_AVX512DQ
  using MaskType = __mmask8;
  #else
  using MaskType = ScalarBitSet<sizeof(float32_t), 4>;
  #endif
  using WordDefs = VecDefs;
}; // struct VecDefs

template <>
struct VecDefs<float32_t, 2> : public VecDefs<float32_t, 4> {
  using TagType = Tag<float32_t, 2>;

  static constexpr nint_t word_size() { return 2; }

  static constexpr nint_t max_word_size = 2;

  static constexpr nint_t size() { return 2; };
  static constexpr nint_t max_size = 2;
  using VecType = __m128;
  #ifdef HAS_AVX512DQ
  using MaskType = __mmask8;
  #else
  using MaskType = ScalarBitSet<sizeof(float32_t), 4>; // compativble with Mask<Tag<float32_t, 4>>
  #endif
  using WordDefs = VecDefs;
}; // struct VecDefs

template <nint_t N>
struct VecDefs<float32_t, N, 0, std::enable_if_t<(N > 4)>> : public VecDefs<float32_t, 4, log2_floor(N / 4)> {};


namespace word {
/* ************************************************************************** */
//                               Constructors                                 //
/* ************************************************************************** */

auto fill(Tag<float32_t, 4> t, float32_t v) -> VecOf(t) {
  return _mm_set1_ps(v);
}

auto fill(Tag<float32_t, 2> t, float32_t v) -> VecOf(t) {
  // we do not guarantee that padded elements are zero
  return fill(Tag<float32_t, 4>(), v);
}

auto zeros(Tag<float32_t, 4> t) -> VecOf(t) {
  return _mm_setzero_ps();
}

auto zeros(Tag<float32_t, 2> t) -> VecOf(t) {
  return zeros(Tag<float32_t, 4>());
}

#ifdef HAS_AVX512DQ
auto mfill(Tag<float32_t, 4> t, bool value) -> MaskOf(t) {
  // we do not guarantee that padded elements are zero
  uint32_t x = value ? 0xff : 0x00;
  return _cvtu32_mask8(x);
}

auto mfill(Tag<float32_t, 2> t, bool value) -> MaskOf(t) {
  return mfill(Tag<float32_t, 4>(), value);
}

auto mwhilelt(Tag<float32_t, 4> t, nint_t a, nint_t b) -> MaskOf(t) {
  nint_t end = std::max(b - a, nint_t(0));
  return _cvtu32_mask8((1u << end) - 1);
}

auto mwhilelt(Tag<float32_t, 2> t, nint_t a, nint_t b) -> MaskOf(t) {
  return mwhilelt(Tag<float32_t, 4>(), a, b);
}

auto mwhilege(Tag<float32_t, 4> t, nint_t a, nint_t b) -> MaskOf(t) {
  nint_t end = std::max(b - a, nint_t(0));
  return _cvtu32_mask8(~((1u << end) - 1));
}

auto mwhilege(Tag<float32_t, 2> t, nint_t a, nint_t b) -> MaskOf(t) {
  return mwhilege(Tag<float32_t, 4>(), a, b);
}

#endif

/* ************************************************************************** */
//                          Consecutive load & store                          //
/* ************************************************************************** */

auto loadu(Tag<float32_t, 4> t, const float32_t* p) -> VecOf(t) {
  return _mm_loadu_ps(p);
}

auto loadu(Tag<float32_t, 2> t, const float32_t* p) -> VecOf(t) {
  return _mm_castpd_ps(_mm_load_sd((const float64_t*) p));
}

auto load(Tag<float32_t, 4> t, const float32_t* p) -> VecOf(t) {
  return _mm_load_ps(p);
}

auto load(Tag<float32_t, 2> t, const float32_t* p) -> VecOf(t) {
  return _mm_castpd_ps(_mm_load_sd((const float64_t*) p));
}

#ifndef HAS_AVX512DQ

CT_ALWAYS_FORCEINLINE
__m128i _cvt_mask(const ScalarBitSet<4, 4>& x) {
  alignas(16) static const uint32_t _MASKBIT_LUT[16][4] = {
      {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}, // 0000: no bit
      {0xffffffffu, 0x00000000u, 0x00000000u, 0x00000000u}, // 0001: bit 0
      {0x00000000u, 0xffffffffu, 0x00000000u, 0x00000000u}, // 0010: bit 1
      {0xffffffffu, 0xffffffffu, 0x00000000u, 0x00000000u}, // 0011: bit 0,1
      {0x00000000u, 0x00000000u, 0xffffffffu, 0x00000000u}, // 0100: bit 2
      {0xffffffffu, 0x00000000u, 0xffffffffu, 0x00000000u}, // 0101: bit 0,2
      {0x00000000u, 0xffffffffu, 0xffffffffu, 0x00000000u}, // 0110: bit 1,2
      {0xffffffffu, 0xffffffffu, 0xffffffffu, 0x00000000u}, // 0111: bit 0,1,2
      {0x00000000u, 0x00000000u, 0x00000000u, 0xffffffffu}, // 1000: bit 3
      {0xffffffffu, 0x00000000u, 0x00000000u, 0xffffffffu}, // 1001: bit 0,3
      {0x00000000u, 0xffffffffu, 0x00000000u, 0xffffffffu}, // 1010: bit 1,3
      {0xffffffffu, 0xffffffffu, 0x00000000u, 0xffffffffu}, // 1011: bit 0,1,3
      {0x00000000u, 0x00000000u, 0xffffffffu, 0xffffffffu}, // 1100: bit 2,3
      {0xffffffffu, 0x00000000u, 0xffffffffu, 0xffffffffu}, // 1101: bit 0,2,3
      {0x00000000u, 0xffffffffu, 0xffffffffu, 0xffffffffu}, // 1110: bit 1,2,3
      {0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}, // 1111: bit 0,1,2,3
  };

  nint_t shift = x.to_ulong();
  return _mm_load_si128((const __m128i*) _MASKBIT_LUT[shift]);
}

#endif //not HAS_AVX512DQ

auto loadu(Tag<float32_t, 4> t, const float32_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto mask = mwhilelt(t, 0, n);
  #ifdef HAS_AVX512DQ
  return _mm_mask_loadu_ps(default_v, mask, p);
  #else
  auto m = _cvt_mask(mask);
  return _mm_blendv_ps(default_v, _mm_maskload_ps(p, m), _mm_castsi128_ps(m));
  #endif
}

auto loadu(Tag<float32_t, 2> t, const float32_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  return loadu(Tag<float32_t, 4>(), p, n, default_v);
}

auto load(Tag<float32_t, 4> t, const float32_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto mask = mwhilelt(t, 0, n);
  #ifdef HAS_AVX512DQ
  return _mm_mask_load_ps(default_v, mask, p);
  #else
  auto m = _cvt_mask(mask);
  return _mm_blendv_ps(default_v, _mm_maskload_ps(p, m), _mm_castsi128_ps(m));
  #endif
}

auto load(Tag<float32_t, 2> t, const float32_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  return load(Tag<float32_t, 4>(), p, n, default_v);
}


auto loadu(Tag<float32_t, 4> t, const float32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) {
  #ifdef HAS_AVX512DQ
  return _mm_mask_loadu_ps(default_v, m, p);
  #else
  auto mask = _cvt_mask(m);
  return _mm_blendv_ps(default_v, _mm_maskload_ps(p, mask), _mm_castsi128_ps(mask));
  #endif
}

auto loadu(Tag<float32_t, 2> t, const float32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) {
  return loadu(Tag<float32_t, 4>(), p, m, default_v);
}

auto load(Tag<float32_t, 4> t, const float32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) {
  #ifdef HAS_AVX512DQ
  return _mm_mask_load_ps(default_v, m, p);
  #else
  auto mask = _cvt_mask(m);
  return _mm_blendv_ps(default_v, _mm_maskload_ps(p, mask), _mm_castsi128_ps(mask));
  #endif
}

void storeu(Tag<float32_t, 4> t, float32_t* p, Vec<Tag<float32_t, 4>> v) {
  _mm_storeu_ps(p, v);
}

void storeu(Tag<float32_t, 2> t, float32_t* p, Vec<Tag<float32_t, 2>> v) {
  _mm_store_sd((float64_t*) p, _mm_castps_pd(v));
}

void store(Tag<float32_t, 4> t, float32_t* p, Vec<Tag<float32_t, 4>> v) {
  _mm_store_ps(p, v);
}

void store(Tag<float32_t, 2> t, float32_t* p, Vec<Tag<float32_t, 2>> v) {
  _mm_store_sd((float64_t*) p, _mm_castps_pd(v));
}

void storeu(Tag<float32_t, 4> t, float32_t* p, nint_t n, VecOf(t) v) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto mask = mwhilelt(t, 0, n);
  #ifdef HAS_AVX512DQ
  _mm_mask_storeu_ps(p, mask, v);
  #else
  auto m = _cvt_mask(mask);
  _mm_maskstore_ps(p, m, v);
  #endif
}

void storeu(Tag<float32_t, 2> t, float32_t* p, nint_t n, VecOf(t) v) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  storeu(Tag<float32_t, 4>(), p, n, v);
}

void store(Tag<float32_t, 4> t, float32_t* p, nint_t n, VecOf(t) v) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  auto mask = mwhilelt(t, 0, n);
  #ifdef HAS_AVX512DQ
  _mm_mask_store_ps(p, mask, v);
  #else
  auto m = _cvt_mask(mask);
  _mm_maskstore_ps(p, m, v);
  #endif
}

void store(Tag<float32_t, 2> t, float32_t* p, nint_t n, VecOf(t) v) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  store(Tag<float32_t, 4>(), p, n, v);
}

void storeu(Tag<float32_t, 4> t, float32_t* p, MaskOf(t) m, VecOf(t) v) {
  #ifdef HAS_AVX512DQ
  _mm_mask_storeu_ps(p, m, v);
  #else
  auto mask = _cvt_mask(m);
  _mm_maskstore_ps(p, mask, v);
  #endif
}

void storeu(Tag<float32_t, 2> t, float32_t* p, MaskOf(t) m, VecOf(t) v) {
  storeu(Tag<float32_t, 4>(), p, m, v);
}

void store(Tag<float32_t, 4> t, float32_t* p, MaskOf(t) m, VecOf(t) v) {
  #ifdef HAS_AVX512DQ
  _mm_mask_store_ps(p, m, v);
  #else
  auto mask = _cvt_mask(m);
  _mm_maskstore_ps(p, mask, v);
  #endif
}

void store(Tag<float32_t, 2> t, float32_t* p, MaskOf(t) m, VecOf(t) v) {
  store(Tag<float32_t, 4>(), p, m, v);
}

/* ************************************************************************** */
//                             Get / set element                              //
/* ************************************************************************** */

float32_t get(Tag<float32_t, 4> t, VecOf(t) v, nint_t index) {
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  alignas(16) float tmp[4];
  _mm_store_ps(tmp, v);
  return tmp[index];
}

float32_t get(Tag<float32_t, 2> t, VecOf(t) v, nint_t index) {
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  return get(Tag<float32_t, 4>(), v, index);
}

bool get(Tag<float32_t, 4> t, MaskOf(t) v, nint_t index) {
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  #ifdef HAS_AVX512DQ
  return (_cvtmask8_u32(v) >> index) & 1;
  #else
  return v.test(index);
  #endif
}

bool get(Tag<float32_t, 2> t, MaskOf(t) v, nint_t index) {
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  return get(Tag<float32_t, 4>(), v, index);
}

auto set(Tag<float32_t, 4> t, VecOf(t) v, nint_t index, float32_t x) -> VecOf(t) {
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  alignas(16) float tmp[4];
  _mm_store_ps(tmp, v);
  tmp[index] = x;
  return _mm_load_ps(tmp);
}

auto set(Tag<float32_t, 2> t, VecOf(t) v, nint_t index, float32_t x) -> VecOf(t) {
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  return set(Tag<float32_t, 4>(), v, index, x);
}

auto set(Tag<float32_t, 4> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) {
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  #ifdef HAS_AVX512DQ
  auto m = _cvtmask8_u32(v);
  auto b = x ? 1 : 0;
  m = (m & ~(1 << index)) | (b << index);
  return _cvtu32_mask8(m);
  #else
  auto u = v;
  u.set(index, x);
  return u;
  #endif
}

auto set(Tag<float32_t, 2> t, MaskOf(t) v, nint_t index, float32_t x) -> MaskOf(t) {
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  return set(Tag<float32_t, 4>(), v, index, x);
}

} // namespace word
} // namespace ct::tl::vec

#endif //CTORCH_X86_128_H
