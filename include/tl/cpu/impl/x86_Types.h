//
// Created by renyz on 2026/3/18.
//

#ifndef CTORCH_X86_TYPES_H
#define CTORCH_X86_TYPES_H

#include "CoreDefs.h"
#include "tl/cpu/VecBase.h"

#ifndef ARCH_X86_FAMILY
#error "Not x86 platform"
#endif

#include <immintrin.h>
#define VEC_NUMEL(dtype) (VEC_WIDTH / 8 / sizeof(dtype))

namespace ct::tl::vec {
inline namespace x86 {
/**
 * Simple wrapper for intrinsic vector type that contains type for
 * overload dispatching.
 */
template <typename T, nint_t N>
struct RegType {
  static_assert(sizeof(T) == -1, "Unsupported type or size");
};

template <typename T>
struct WrapperType {
  T v;
  CT_ALWAYS_FORCEINLINE constexpr WrapperType() = default;
  CT_ALWAYS_FORCEINLINE constexpr WrapperType(const T& v) : v(v) {}
  CT_ALWAYS_FORCEINLINE constexpr WrapperType(T&& v) : v(std::move(v)) {}
  CT_ALWAYS_FORCEINLINE explicit constexpr operator T() { return this->v; }
};

#define TL_DEFINE_MMREG(name, N, raw_type) \
template <> struct RegType<name##_t, N> : public WrapperType<raw_type> { \
  using WrapperType<raw_type>::WrapperType; \
  CT_ALWAYS_FORCEINLINE constexpr name##_t operator[](nuint_t i) { return v[i]; } \
}; \
using v##name##x##N##_t = RegType<name##_t, N>

#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX)
TL_DEFINE_MMREG(bfloat16, 8, __m128i);
TL_DEFINE_MMREG(float16, 8, __m128i);
TL_DEFINE_MMREG(float32, 4, __m128);
TL_DEFINE_MMREG(float64, 2, __m128d);
TL_DEFINE_MMREG(int8, 16, __m128i);
TL_DEFINE_MMREG(uint8, 16, __m128i);
TL_DEFINE_MMREG(int16, 8, __m128i);
TL_DEFINE_MMREG(uint16, 8, __m128i);
TL_DEFINE_MMREG(int32, 4, __m128i);
TL_DEFINE_MMREG(uint32, 4, __m128i);
TL_DEFINE_MMREG(int64, 2, __m128i);
TL_DEFINE_MMREG(uint64, 2, __m128i);
#endif
#if defined(CPU_CAPABILITY_AVX512) || defined(CPU_CAPABILITY_AVX2)
TL_DEFINE_MMREG(bfloat16, 16, __m256i);
TL_DEFINE_MMREG(float16, 16, __m256i);
TL_DEFINE_MMREG(float32, 8, __m256);
TL_DEFINE_MMREG(float64, 4, __m256d);
TL_DEFINE_MMREG(int8, 32, __m256i);
TL_DEFINE_MMREG(uint8, 32, __m256i);
TL_DEFINE_MMREG(int16, 16, __m256i);
TL_DEFINE_MMREG(uint16, 16, __m256i);
TL_DEFINE_MMREG(int32, 8, __m256i);
TL_DEFINE_MMREG(uint32, 8, __m256i);
TL_DEFINE_MMREG(int64, 4, __m256i);
TL_DEFINE_MMREG(uint64, 4, __m256i);
#endif
#if defined(CPU_CAPABILITY_AVX512)
TL_DEFINE_MMREG(bfloat16, 32, __m512i);
TL_DEFINE_MMREG(float16, 32, __m512i);
TL_DEFINE_MMREG(float32, 16, __m512);
TL_DEFINE_MMREG(float64, 8, __m512d);
TL_DEFINE_MMREG(int8, 64, __m512i);
TL_DEFINE_MMREG(uint8, 64, __m512i);
TL_DEFINE_MMREG(int16, 32, __m512i);
TL_DEFINE_MMREG(uint16, 32, __m512i);
TL_DEFINE_MMREG(int32, 16, __m512i);
TL_DEFINE_MMREG(uint32, 16, __m512i);
TL_DEFINE_MMREG(int64, 8, __m512i);
TL_DEFINE_MMREG(uint64, 8, __m512i);
#endif
#undef TL_DEFINE_MMREG

template <nint_t ELSIZE, nint_t N>
struct MMMaskType {
  using Type = ScalarBitSet<ELSIZE, N>;
};

template <nint_t ELSIZE, typename T>
struct RegMask : public WrapperType<T> {
  using WrapperType<T>::WrapperType;
};

// Mask specialization for CPU feature AVX512DQ
#if defined(CPU_CAPABILITY_AVX512)
template <nint_t ELSIZE> struct MMMaskType<ELSIZE, 1> { using Type = RegMask<ELSIZE, __mmask8>; };
template <nint_t ELSIZE> struct MMMaskType<ELSIZE, 2> { using Type = RegMask<ELSIZE, __mmask8>; };
template <nint_t ELSIZE> struct MMMaskType<ELSIZE, 4> { using Type = RegMask<ELSIZE, __mmask8>; };
template <nint_t ELSIZE> struct MMMaskType<ELSIZE, 8> { using Type = RegMask<ELSIZE, __mmask8>; };
template <nint_t ELSIZE> struct MMMaskType<ELSIZE, 16> { using Type = RegMask<ELSIZE, __mmask16>; };
template <nint_t ELSIZE> struct MMMaskType<ELSIZE, 32> { using Type = RegMask<ELSIZE, __mmask32>; };
template <nint_t ELSIZE> struct MMMaskType<ELSIZE, 64> { using Type = RegMask<ELSIZE, __mmask64>; };
#else
#if defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX)
template <> struct MMMaskType<1, 1> { using Type = RegMask<1, __m128i>; };
template <> struct MMMaskType<2, 1> { using Type = RegMask<1, __m128i>; };
template <> struct MMMaskType<4, 1> { using Type = RegMask<1, __m128i>; };
template <> struct MMMaskType<8, 1> { using Type = RegMask<1, __m128i>; };
template <> struct MMMaskType<16, 1> { using Type = RegMask<1, __m128i>; };
template <> struct MMMaskType<1, 2> { using Type = RegMask<1, __m128i>; };
template <> struct MMMaskType<2, 2> { using Type = RegMask<1, __m128i>; };
template <> struct MMMaskType<4, 2> { using Type = RegMask<1, __m128i>; };
template <> struct MMMaskType<8, 2> { using Type = RegMask<1, __m128i>; };
template <> struct MMMaskType<1, 4> { using Type = RegMask<1, __m128i>; };
template <> struct MMMaskType<2, 4> { using Type = RegMask<1, __m128i>; };
template <> struct MMMaskType<4, 4> { using Type = RegMask<1, __m128i>; };
template <> struct MMMaskType<1, 8> { using Type = RegMask<1, __m128i>; };
template <> struct MMMaskType<2, 8> { using Type = RegMask<1, __m128i>; };
template <> struct MMMaskType<1, 16> { using Type = RegMask<1, __m128i>; };
#endif // defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX)

#if defined(CPU_CAPABILITY_AVX2)
template <> struct MMMaskType<32, 1> { using Type = RegMask<1, __m256i>; };
template <> struct MMMaskType<16, 2> { using Type = RegMask<1, __m256i>; };
template <> struct MMMaskType<8, 4> { using Type = RegMask<1, __m256i>; };
template <> struct MMMaskType<4, 8> { using Type = RegMask<1, __m256i>; };
template <> struct MMMaskType<2, 16> { using Type = RegMask<1, __m256i>; };
template <> struct MMMaskType<1, 32> { using Type = RegMask<1, __m256i>; };
#endif // CPU_CAPABILITY_AVX2
#endif // CPU_CAPABILITY_AVX512

template <typename T, nint_t N>
using MaskType = MMMaskType<sizeof(T), N>::Type;
} // namespace x86

/**
 * @brief VecDefs specialization for (S / sizeof(dtype)) x dtype using SIMD register.
 *
 * Uses __m128/__m256/__m512 as the vector type.
 */
#define TL_DEFINE_VEC(dtype, S) \
template <> struct VecDefs<dtype, (S) / sizeof(dtype)> : public ScalarVecDefs<dtype, (S) / sizeof(dtype)> { \
  static constexpr nint_t num_words = 1; \
  static constexpr nint_t word_size() { return (S) / sizeof(dtype); } \
  static constexpr nint_t max_word_size = (S) / sizeof(dtype); \
  static constexpr nint_t size() { return (S) / sizeof(dtype); }; \
  static constexpr nint_t max_size = (S) / sizeof(dtype); \
  static constexpr bool is_scalable = false; \
  static constexpr bool is_default_impl = false; \
  static constexpr bool is_word_vec = true; \
  using VecType = x86::RegType<dtype, (S) / sizeof(dtype)>; \
  using MaskType = x86::MaskType<dtype, (S) / sizeof(dtype)>; \
  using WordDefs = VecDefs; \
}; \
template <> struct Vec2Tag<x86::RegType<dtype, (S) / sizeof(dtype)>> { using Type = Tag<dtype, (S) / sizeof(dtype)>; } \

#if defined(CPU_CAPABILITY_AVX512)
  #define TL_DEFINE_VEC_ALL(dtype) \
    TL_DEFINE_VEC(dtype, 16); \
    TL_DEFINE_VEC(dtype, 32); \
    TL_DEFINE_VEC(dtype, 64);
#elif defined(CPU_CAPABILITY_AVX2)
  #define TL_DEFINE_VEC_ALL(dtype) \
    TL_DEFINE_VEC(dtype, 16); \
    TL_DEFINE_VEC(dtype, 32);
#elif defined(CPU_CAPABILITY_AVX)
  #define TL_DEFINE_VEC_ALL(dtype) \
    TL_DEFINE_VEC(dtype, 16);
#endif

TL_DEFINE_VEC_ALL(bfloat16_t);
TL_DEFINE_VEC_ALL(float16_t);
TL_DEFINE_VEC_ALL(float32_t);
TL_DEFINE_VEC_ALL(float64_t);
TL_DEFINE_VEC_ALL(int8_t);
TL_DEFINE_VEC_ALL(uint8_t);
TL_DEFINE_VEC_ALL(int16_t);
TL_DEFINE_VEC_ALL(uint16_t);
TL_DEFINE_VEC_ALL(int32_t);
TL_DEFINE_VEC_ALL(uint32_t);
TL_DEFINE_VEC_ALL(int64_t);
TL_DEFINE_VEC_ALL(uint64_t);
#undef TL_DEFINE_VEC_ALL


#define _TL_XMM_APPLY_2(FN, ...) FN(2, __VA_ARGS__);
#define _TL_XMM_APPLY_4(FN, ...) FN(4, __VA_ARGS__); _TL_XMM_APPLY_2(FN, __VA_ARGS__)
#define _TL_XMM_APPLY_8(FN, ...) FN(8, __VA_ARGS__); _TL_XMM_APPLY_4(FN, __VA_ARGS__)
#define _TL_XMM_APPLY_bfloat16_t _TL_XMM_APPLY_4
#define _TL_XMM_APPLY_float16_t _TL_XMM_APPLY_4
#define _TL_XMM_APPLY_float32_t _TL_XMM_APPLY_2
#define _TL_XMM_APPLY_float64_t(...)
#define _TL_XMM_APPLY_int8_t _TL_XMM_APPLY_8
#define _TL_XMM_APPLY_uint8_t _TL_XMM_APPLY_8
#define _TL_XMM_APPLY_int16_t _TL_XMM_APPLY_4
#define _TL_XMM_APPLY_uint16_t _TL_XMM_APPLY_4
#define _TL_XMM_APPLY_int32_t _TL_XMM_APPLY_2
#define _TL_XMM_APPLY_uint32_t _TL_XMM_APPLY_2
#define _TL_XMM_APPLY_int64_t(...)
#define _TL_XMM_APPLY_uint64_t(...)
#define TL_XMM_APPLY_TO_ALL_HALVES(dtype, FN, ...) _TL_XMM_APPLY_##dtype(FN, __VA_ARGS__)


/**
 * @brief VecDefs specialization for N x dtype (partial 128-bit vector).
 * 
 * Uses the same __m128 type as (16 / sizeof(dtype)) x dtype, but only operates on
 * the lower half / quarter elements. This allows partial vectors to share
 * implementation with full vectors where appropriate.
 */
#define TL_XMM_DEFINE_VEC_HALF(N, dtype) \
template <> struct VecDefs<dtype, (N)> : public VecDefs<dtype, 2 * (N)> { \
  using TagType = Tag<dtype, (N)>; \
  static constexpr nint_t word_size() { return (N); } \
  static constexpr nint_t max_word_size = (N); \
  static constexpr nint_t size() { return (N); }; \
  static constexpr nint_t max_size = (N); \
  using VecType = x86::RegType<dtype, 16 / sizeof(dtype)>; \
  /* compatible with Mask<Tag<dtype, 16 / sizeof(dtype)>> */ \
  using MaskType = x86::MaskType<dtype, 16 / sizeof(dtype)>;  \
  using WordDefs = VecDefs; \
} \

TL_XMM_APPLY_TO_ALL_HALVES(bfloat16_t, TL_XMM_DEFINE_VEC_HALF, bfloat16_t)
TL_XMM_APPLY_TO_ALL_HALVES(float16_t, TL_XMM_DEFINE_VEC_HALF, float16_t)
TL_XMM_APPLY_TO_ALL_HALVES(float32_t, TL_XMM_DEFINE_VEC_HALF, float32_t)
TL_XMM_APPLY_TO_ALL_HALVES(int8_t, TL_XMM_DEFINE_VEC_HALF, int8_t)
TL_XMM_APPLY_TO_ALL_HALVES(uint8_t, TL_XMM_DEFINE_VEC_HALF, uint8_t)
TL_XMM_APPLY_TO_ALL_HALVES(int16_t, TL_XMM_DEFINE_VEC_HALF, int16_t)
TL_XMM_APPLY_TO_ALL_HALVES(uint16_t, TL_XMM_DEFINE_VEC_HALF, uint16_t)
TL_XMM_APPLY_TO_ALL_HALVES(int32_t, TL_XMM_DEFINE_VEC_HALF, int32_t)
TL_XMM_APPLY_TO_ALL_HALVES(uint32_t, TL_XMM_DEFINE_VEC_HALF, uint32_t)
#undef TL_XMM_DEFINE_VEC_HALF

/**
 * @brief VecDefs specialization for N > VEC_NUMEL(dtype) x dtype (multi-word vectors).
 * 
 * Inherits from the appropriate multi-word base class, which handles
 * concatenation of multiple VEC_WIDTH-bit registers.
 */
#define TL_DEFINE_BATCH_VEC(dtype) \
template <nint_t N> struct VecDefs<dtype, N, 0, std::enable_if_t<(N > VEC_NUMEL(dtype))>>  \
    : public VecDefs<dtype, VEC_NUMEL(dtype), log2_floor(N / VEC_NUMEL(dtype))> {} \

TL_DEFINE_BATCH_VEC(bfloat16_t);
TL_DEFINE_BATCH_VEC(float16_t);
TL_DEFINE_BATCH_VEC(float32_t);
TL_DEFINE_BATCH_VEC(float64_t);
TL_DEFINE_BATCH_VEC(int8_t);
TL_DEFINE_BATCH_VEC(uint8_t);
TL_DEFINE_BATCH_VEC(int16_t);
TL_DEFINE_BATCH_VEC(uint16_t);
TL_DEFINE_BATCH_VEC(int32_t);
TL_DEFINE_BATCH_VEC(uint32_t);
TL_DEFINE_BATCH_VEC(int64_t);
TL_DEFINE_BATCH_VEC(uint64_t);
#undef TL_DEFINE_BATCH_VEC

} // namespace ct::tl::vec

#endif //CTORCH_X86_TYPES_H
