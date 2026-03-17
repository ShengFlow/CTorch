//
// Created by renyz on 2026/3/17.
//

#ifndef CTORCH_X86_128_H
#define CTORCH_X86_128_H

#include "CoreDefs.h"
#include "tl/cpu/impl/Scalar.h"
#include "tl/util/Math.h"

/**
 * @file x86_128.h
 * @brief SSE/AVX 128-bit SIMD implementation for x86/x86-64.
 * 
 * This file provides optimized SIMD implementations for 128-bit vectors
 * using Intel SSE/AVX intrinsics. It supports:
 * 
 * - 4 x float32 (single-precision)
 * - 2 x float32 (partial vector)
 * - Multi-word vectors (multiple 128-bit registers)
 * 
 * ## Implementation Details
 * 
 * ### Mask Representation
 * Masks are represented differently depending on CPU capabilities:
 * - **AVX-512DQ available**: Uses __mmask8 (hardware mask registers)
 * - **AVX-512DQ not available**: Uses ScalarBitSet with a lookup table
 *   to convert bits to a vector mask for blend operations
 * 
 * ### Partial Vectors (N=2)
 * When using Tag<float32_t, 2>, the implementation uses a full 128-bit
 * register but only operates on the lower 2 elements. This is useful for
 * operations that don't need 4-wide SIMD.
 * 
 * @warning Padded elements (indices >= 2 for N=2) are not guaranteed
 *          to be zero. Do not rely on their values.
 * 
 * ## Compatibility
 * 
 * This implementation requires at least SSE support. Some operations
 * benefit from AVX, AVX2, or AVX-512 when available.
 */

#ifndef ARCH_X86_FAMILY
#error "Not x86 platform"
#endif

#include <immintrin.h>

namespace ct::tl::vec {

/**
 * @brief VecDefs specialization for 4 x float32 using SSE.
 * 
 * Uses __m128 (128-bit XMM register) as the vector type.
 * Mask type depends on AVX-512DQ availability.
 */
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

/**
 * @brief VecDefs specialization for 2 x float32 (partial 128-bit vector).
 * 
 * Uses the same __m128 type as 4 x float32, but only operates on
 * the lower 2 elements. This allows partial vectors to share
 * implementation with full vectors where appropriate.
 * 
 * @note Mask type is the same as 4 x float32 for compatibility.
 */
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
  using MaskType = ScalarBitSet<sizeof(float32_t), 4>; // compativle with Mask<Tag<float32_t, 4>>
  #endif
  using WordDefs = VecDefs;
}; // struct VecDefs

/**
 * @brief VecDefs specialization for N > 4 x float32 (multi-word vectors).
 * 
 * Inherits from the appropriate multi-word base class, which handles
 * concatenation of multiple 128-bit registers.
 * 
 * @tparam N Number of elements (must be > 4 and power of 2)
 */
template <nint_t N>
struct VecDefs<float32_t, N, 0, std::enable_if_t<(N > 4)>> : public VecDefs<float32_t, 4, log2_floor(N / 4)> {};


namespace word {
/* ************************************************************************** */
//                               Constructors                                 //
/* ************************************************************************** */

/**
 * @brief Fill a 4 x float32 vector with a single value.
 * 
 * Uses _mm_set1_ps to broadcast the value to all lanes.
 * 
 * @param t The vector tag
 * @param v The value to broadcast
 * @return Vector with all elements set to v
 */
auto fill(Tag<float32_t, 4> t, float32_t v) -> VecOf(t) {
  return _mm_set1_ps(v);
}

/**
 * @brief Fill a 2 x float32 vector with a single value.
 * 
 * Uses the same implementation as 4 x float32. Padded elements
 * are not guaranteed to be zero.
 * 
 * @param t The vector tag
 * @param v The value to broadcast
 * @return Vector with lower 2 elements set to v
 */
auto fill(Tag<float32_t, 2> t, float32_t v) -> VecOf(t) {
  // we do not guarantee that padded elements are zero
  return fill(Tag<float32_t, 4>(), v);
}

/**
 * @brief Create a 4 x float32 vector of zeros.
 * 
 * Uses _mm_setzero_ps, which is typically faster than _mm_set1_ps(0.0f).
 * 
 * @param t The vector tag
 * @return Zero vector
 */
auto zeros(Tag<float32_t, 4> t) -> VecOf(t) {
  return _mm_setzero_ps();
}

/**
 * @brief Create a 2 x float32 vector of zeros.
 * 
 * @param t The vector tag
 * @return Zero vector (only lower 2 elements are meaningful)
 */
auto zeros(Tag<float32_t, 2> t) -> VecOf(t) {
  return zeros(Tag<float32_t, 4>());
}

#ifdef HAS_AVX512DQ
/**
 * @brief Fill a mask with a single boolean value (AVX-512DQ version).
 * 
 * Uses hardware mask registers when AVX-512DQ is available.
 * 
 * @param t The vector tag
 * @param value The boolean value
 * @return Mask with all lanes set to value
 * 
 * @note Padded bits are also set; for N=2, bits 2-7 are also set/unset.
 */
auto mfill(Tag<float32_t, 4> t, bool value) -> MaskOf(t) {
  // we do not guarantee that padded elements are zero
  uint32_t x = value ? 0xff : 0x00;
  return _cvtu32_mask8(x);
}

/**
 * @brief Fill a 2 x float32 mask with a single boolean value (AVX-512DQ).
 */
auto mfill(Tag<float32_t, 2> t, bool value) -> MaskOf(t) {
  return mfill(Tag<float32_t, 4>(), value);
}

/**
 * @brief Create a mask where lanes i are true if (a + i) < b (AVX-512DQ).
 * 
 * Uses efficient bit manipulation to create the mask.
 * 
 * @param t The vector tag
 * @param a Starting index
 * @param b Upper bound (exclusive)
 * @return Mask where lanes [a, b) are true
 */
auto mwhilelt(Tag<float32_t, 4> t, nint_t a, nint_t b) -> MaskOf(t) {
  nint_t end = std::clamp(b - a, nint_t(0), nint_t(sizeof(unsigned) * CHAR_BIT));
  return _cvtu32_mask8((1u << (unsigned)end) - 1);
}

/**
 * @brief Create a mask for 2 x float32 where lanes i are true if (a + i) < b.
 */
auto mwhilelt(Tag<float32_t, 2> t, nint_t a, nint_t b) -> MaskOf(t) {
  return mwhilelt(Tag<float32_t, 4>(), a, b);
}

/**
 * @brief Create a mask where lanes i are true if (a + i) >= b (AVX-512DQ).
 * 
 * @param t The vector tag
 * @param a Starting index
 * @param b Lower bound (inclusive)
 * @return Mask where lanes >= b are true
 */
auto mwhilege(Tag<float32_t, 4> t, nint_t a, nint_t b) -> MaskOf(t) {
  nint_t end = std::clamp(b - a, nint_t(0), nint_t(sizeof(unsigned) * CHAR_BIT));
  return _cvtu32_mask8(~((1u << (unsigned)end) - 1));
}

/**
 * @brief Create a mask for 2 x float32 where lanes i are true if (a + i) >= b.
 */
auto mwhilege(Tag<float32_t, 2> t, nint_t a, nint_t b) -> MaskOf(t) {
  return mwhilege(Tag<float32_t, 4>(), a, b);
}

#endif

/* ************************************************************************** */
//                          Consecutive load & store                          //
/* ************************************************************************** */

/**
 * @brief Load a 4 x float32 vector from unaligned memory.
 * 
 * Uses _mm_loadu_ps for unaligned loads.
 * 
 * @param t The vector tag
 * @param p Pointer to source memory (need not be aligned)
 * @return Loaded vector
 */
auto loadu(Tag<float32_t, 4> t, const float32_t* p) -> VecOf(t) {
  return _mm_loadu_ps(p);
}

/**
 * @brief Load a 2 x float32 vector from unaligned memory.
 * 
 * Loads 8 bytes (2 floats) using _mm_load_sd (double load) and
 * casts to float vector.
 * 
 * @param t The vector tag
 * @param p Pointer to source memory
 * @return Loaded vector (only lower 2 elements are valid)
 */
auto loadu(Tag<float32_t, 2> t, const float32_t* p) -> VecOf(t) {
  return _mm_castpd_ps(_mm_load_sd((const float64_t*) p));
}

/**
 * @brief Load a 4 x float32 vector from aligned memory.
 * 
 * Uses _mm_load_ps which requires 16-byte alignment.
 * 
 * @param t The vector tag
 * @param p Pointer to source memory (must be 16-byte aligned)
 * @return Loaded vector
 * 
 * @warning Using an unaligned pointer may cause #GP fault on older CPUs.
 */
auto load(Tag<float32_t, 4> t, const float32_t* p) -> VecOf(t) {
  return _mm_load_ps(p);
}

/**
 * @brief Load a 2 x float32 vector from aligned memory.
 * 
 * @param t The vector tag
 * @param p Pointer to source memory
 * @return Loaded vector
 */
auto load(Tag<float32_t, 2> t, const float32_t* p) -> VecOf(t) {
  return _mm_castpd_ps(_mm_load_sd((const float64_t*) p));
}

#ifndef HAS_AVX512DQ

/**
 * @brief Convert a ScalarBitSet mask to an __m128i for use with mask operations.
 * 
 * When AVX-512DQ is not available, we use a lookup table to convert
 * each possible 4-bit mask pattern to a vector of 32-bit masks
 * (0x00000000 for false, 0xFFFFFFFF for true).
 * 
 * This allows us to use _mm_blendv_ps for masked operations.
 * 
 * @param x The scalar bitset mask
 * @return __m128i where each 32-bit lane is 0 or 0xFFFFFFFF
 */
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

/**
 * @brief Load first n elements of 4 x float32 from unaligned memory.
 * 
 * Uses masked load when available, otherwise falls back to blend.
 * 
 * @param t The vector tag
 * @param p Pointer to source memory
 * @param n Number of elements to load (0-4)
 * @param default_v Default values for elements beyond n
 * @return Vector with n elements from memory, rest from default_v
 */
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

/**
 * @brief Load first n elements of 2 x float32 from unaligned memory.
 */
auto loadu(Tag<float32_t, 2> t, const float32_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  return loadu(Tag<float32_t, 4>(), p, n, default_v);
}

/**
 * @brief Load first n elements of 4 x float32 from aligned memory.
 */
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

/**
 * @brief Load first n elements of 2 x float32 from aligned memory.
 */
auto load(Tag<float32_t, 2> t, const float32_t* p, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  return load(Tag<float32_t, 4>(), p, n, default_v);
}


/**
 * @brief Masked load of 4 x float32 from unaligned memory.
 * 
 * @param t The vector tag
 * @param p Pointer to source memory
 * @param m Mask indicating which lanes to load
 * @param default_v Default values for masked-out lanes
 * @return Vector with masked-loaded elements
 */
auto loadu(Tag<float32_t, 4> t, const float32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) {
  #ifdef HAS_AVX512DQ
  return _mm_mask_loadu_ps(default_v, m, p);
  #else
  auto mask = _cvt_mask(m);
  return _mm_blendv_ps(default_v, _mm_maskload_ps(p, mask), _mm_castsi128_ps(mask));
  #endif
}

/**
 * @brief Masked load of 2 x float32 from unaligned memory.
 */
auto loadu(Tag<float32_t, 2> t, const float32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) {
  return loadu(Tag<float32_t, 4>(), p, m, default_v);
}

/**
 * @brief Masked load of 4 x float32 from aligned memory.
 */
auto load(Tag<float32_t, 4> t, const float32_t* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) {
  #ifdef HAS_AVX512DQ
  return _mm_mask_load_ps(default_v, m, p);
  #else
  auto mask = _cvt_mask(m);
  return _mm_blendv_ps(default_v, _mm_maskload_ps(p, mask), _mm_castsi128_ps(mask));
  #endif
}

/**
 * @brief Store a 4 x float32 vector to unaligned memory.
 */
void storeu(Tag<float32_t, 4> t, float32_t* p, Vec<Tag<float32_t, 4>> v) {
  _mm_storeu_ps(p, v);
}

/**
 * @brief Store a 2 x float32 vector to unaligned memory.
 * 
 * Uses _mm_store_sd to store 8 bytes (2 floats).
 */
void storeu(Tag<float32_t, 2> t, float32_t* p, Vec<Tag<float32_t, 2>> v) {
  _mm_store_sd((float64_t*) p, _mm_castps_pd(v));
}

/**
 * @brief Store a 4 x float32 vector to aligned memory.
 */
void store(Tag<float32_t, 4> t, float32_t* p, Vec<Tag<float32_t, 4>> v) {
  _mm_store_ps(p, v);
}

/**
 * @brief Store a 2 x float32 vector to aligned memory.
 */
void store(Tag<float32_t, 2> t, float32_t* p, Vec<Tag<float32_t, 2>> v) {
  _mm_store_sd((float64_t*) p, _mm_castps_pd(v));
}

/**
 * @brief Store first n elements of 4 x float32 to unaligned memory.
 */
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

/**
 * @brief Store first n elements of 2 x float32 to unaligned memory.
 */
void storeu(Tag<float32_t, 2> t, float32_t* p, nint_t n, VecOf(t) v) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  storeu(Tag<float32_t, 4>(), p, n, v);
}

/**
 * @brief Store first n elements of 4 x float32 to aligned memory.
 */
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

/**
 * @brief Store first n elements of 2 x float32 to aligned memory.
 */
void store(Tag<float32_t, 2> t, float32_t* p, nint_t n, VecOf(t) v) {
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  store(Tag<float32_t, 4>(), p, n, v);
}

/**
 * @brief Masked store of 4 x float32 to unaligned memory.
 */
void storeu(Tag<float32_t, 4> t, float32_t* p, MaskOf(t) m, VecOf(t) v) {
  #ifdef HAS_AVX512DQ
  _mm_mask_storeu_ps(p, m, v);
  #else
  auto mask = _cvt_mask(m);
  _mm_maskstore_ps(p, mask, v);
  #endif
}

/**
 * @brief Masked store of 2 x float32 to unaligned memory.
 */
void storeu(Tag<float32_t, 2> t, float32_t* p, MaskOf(t) m, VecOf(t) v) {
  storeu(Tag<float32_t, 4>(), p, m, v);
}

/**
 * @brief Masked store of 4 x float32 to aligned memory.
 */
void store(Tag<float32_t, 4> t, float32_t* p, MaskOf(t) m, VecOf(t) v) {
  #ifdef HAS_AVX512DQ
  _mm_mask_store_ps(p, m, v);
  #else
  auto mask = _cvt_mask(m);
  _mm_maskstore_ps(p, mask, v);
  #endif
}

/**
 * @brief Masked store of 2 x float32 to aligned memory.
 */
void store(Tag<float32_t, 2> t, float32_t* p, MaskOf(t) m, VecOf(t) v) {
  store(Tag<float32_t, 4>(), p, m, v);
}

/* ************************************************************************** */
//                             Get / set element                              //
/* ************************************************************************** */

/**
 * @brief Get a single element from a 4 x float32 vector.
 * 
 * @warning This is a slow operation as it stores the vector to memory
 *          and reads back the element. Avoid in hot loops.
 * 
 * @param t The vector tag
 * @param v The vector
 * @param index Element index (0-3)
 * @return The element at index
 */
float32_t get(Tag<float32_t, 4> t, VecOf(t) v, nint_t index) {
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  auto i = _mm_set_epi32(0, 0, 0, (int) index);
  return _mm_cvtss_f32(_mm_permutevar_ps(v, i));
}

/**
 * @brief Get a single element from a 2 x float32 vector.
 */
float32_t get(Tag<float32_t, 2> t, VecOf(t) v, nint_t index) {
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  return get(Tag<float32_t, 4>(), v, index);
}

/**
 * @brief Get a single element from a 4 x float32 mask.
 * 
 * @param t The vector tag
 * @param v The mask
 * @param index Element index (0-3)
 * @return The boolean value at index
 */
bool get(Tag<float32_t, 4> t, MaskOf(t) v, nint_t index) {
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  #ifdef HAS_AVX512DQ
  return (_cvtmask8_u32(v) >> index) & 1;
  #else
  return v.test(index);
  #endif
}

/**
 * @brief Get a single element from a 2 x float32 mask.
 */
bool get(Tag<float32_t, 2> t, MaskOf(t) v, nint_t index) {
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  return get(Tag<float32_t, 4>(), v, index);
}

/**
 * @brief Set a single element in a 4 x float32 vector.
 * 
 * @warning This is a slow operation. Avoid in hot loops.
 * 
 * @param t The vector tag
 * @param v The original vector
 * @param index Element index (0-3)
 * @param x The new value
 * @return Vector with the element at index set to x
 */
auto set(Tag<float32_t, 4> t, VecOf(t) v, nint_t index, float32_t x) -> VecOf(t) {
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  #if defined(HAS_AVX512DQ) && defined(HAS_AVX512VL)
  auto mask = _cvtu32_mask8(1u << (unsigned)index);
  return _mm_mask_mov_ps(v, mask, _mm_set1_ps(x));
  #else
  alignas(16) float tmp[4];
  _mm_store_ps(tmp, v);
  tmp[index] = x;
  return _mm_load_ps(tmp);
  #endif
}

/**
 * @brief Set a single element in a 2 x float32 vector.
 */
auto set(Tag<float32_t, 2> t, VecOf(t) v, nint_t index, float32_t x) -> VecOf(t) {
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  return set(Tag<float32_t, 4>(), v, index, x);
}

/**
 * @brief Set a single element in a 4 x float32 mask.
 * 
 * @param t The vector tag
 * @param v The original mask
 * @param index Element index (0-3)
 * @param x The new boolean value
 * @return Mask with the element at index set to x
 */
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

/**
 * @brief Set a single element in a 2 x float32 mask.
 */
auto set(Tag<float32_t, 2> t, MaskOf(t) v, nint_t index, float32_t x) -> MaskOf(t) {
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  return set(Tag<float32_t, 4>(), v, index, x);
}

} // namespace word
} // namespace ct::tl::vec

#endif //CTORCH_X86_128_H
