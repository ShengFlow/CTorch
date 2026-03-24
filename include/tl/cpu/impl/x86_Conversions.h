//
// Created by renyz on 2026/3/23.
//

#ifndef CTORCH_X86_CONVERSIONS_H
#define CTORCH_X86_CONVERSIONS_H

#include "CoreDefs.h"
#include "tl/util/TypeTraits.h"
#include "tl/cpu/Vec.h"

#ifndef HAS_CPU_CAPABILITY_AVX
#error "AVX instruction set required"
#endif

//@formatter:off
namespace ct::tl::vec {

namespace word {
namespace details {

template <typename T> struct BitCastFromInt128 {
  CT_ALWAYS_FORCEINLINE __m128i operator()(__m128i v) { return v; }
};
template <> struct BitCastFromInt128<float32_t> {
  CT_ALWAYS_FORCEINLINE __m128 operator()(__m128i v) { return _mm_castsi128_ps(v); }
};
template <> struct BitCastFromInt128<float64_t> {
  CT_ALWAYS_FORCEINLINE __m128d operator()(__m128i v) { return _mm_castsi128_pd(v); }
};
#ifdef HAS_AVX512_FP16
template <> struct BitCastFromInt128<float16_t> {
  CT_ALWAYS_FORCEINLINE __m128h operator()(__m128i v) { return _mm_castsi128_ph(v); }
};
#endif
template <typename T> struct BitCastToInt128 {
  CT_ALWAYS_FORCEINLINE __m128i operator()(__m128i v) { return v; }
};
template <> struct BitCastToInt128<float32_t> {
  CT_ALWAYS_FORCEINLINE __m128i operator()(__m128 v) { return _mm_castps_si128(v); }
};
template <> struct BitCastToInt128<float64_t> {
  CT_ALWAYS_FORCEINLINE __m128i operator()(__m128d v) { return _mm_castpd_si128(v); }
};
#ifdef HAS_AVX512_FP16
template <> struct BitCastToInt128<float16_t> {
  CT_ALWAYS_FORCEINLINE __m128i operator()(__m128h v) { return _mm_castph_si128(v); }
};
#endif

template <typename T> CT_ALWAYS_FORCEINLINE
RegType<uint8_t, 16> bitcast_to_int(RegType<T, 16 / sizeof(T)> v) { return BitCastToInt128<T>()(v.v); }
template <typename T> CT_ALWAYS_FORCEINLINE
RegType<T, 16 / sizeof(T)> bitcast_from_int(RegType<uint8_t, 16> v) { return BitCastFromInt128<T>()(v.v); }

#if VEC_WIDTH >= 256
template <typename T> struct BitCastFromInt256 {
  CT_ALWAYS_FORCEINLINE __m256i operator()(__m256i v) { return v; }
};
template <> struct BitCastFromInt256<float32_t> {
  CT_ALWAYS_FORCEINLINE __m256 operator()(__m256i v) { return _mm256_castsi256_ps(v); }
};
template <> struct BitCastFromInt256<float64_t> {
  CT_ALWAYS_FORCEINLINE __m256d operator()(__m256i v) { return _mm256_castsi256_pd(v); }
};
#ifdef HAS_AVX512_FP16
template <> struct BitCastFromInt256<float16_t> {
  CT_ALWAYS_FORCEINLINE __m256h operator()(__m256i v) { return _mm256_castsi256_ph(v); }
};
#endif
template <typename T> struct BitCastToInt256 {
  CT_ALWAYS_FORCEINLINE __m256i operator()(__m256i v) { return v; }
};
template <> struct BitCastToInt256<float32_t> {
  CT_ALWAYS_FORCEINLINE __m256i operator()(__m256 v) { return _mm256_castps_si256(v); }
};
template <> struct BitCastToInt256<float64_t> {
  CT_ALWAYS_FORCEINLINE __m256i operator()(__m256d v) { return _mm256_castpd_si256(v); }
};
#ifdef HAS_AVX512_FP16
template <> struct BitCastToInt256<float16_t> {
  CT_ALWAYS_FORCEINLINE __m256i operator()(__m256h v) { return _mm256_castph_si256(v); }
};
#endif

template <typename T> CT_ALWAYS_FORCEINLINE
RegType<uint8_t, 32> bitcast_to_int(RegType<T, 32 / sizeof(T)> v) { return BitCastToInt256<T>()(v.v); }
template <typename T> CT_ALWAYS_FORCEINLINE
RegType<T, 32 / sizeof(T)> bitcast_from_int(RegType<uint8_t, 32> v) { return BitCastFromInt256<T>()(v.v); }
#endif // VEC_WIDTH >= 256


#if VEC_WIDTH >= 512
template <typename T> struct BitCastFromInt512 {
  CT_ALWAYS_FORCEINLINE __m512i operator()(__m512i v) { return v; }
};
template <> struct BitCastFromInt512<float32_t> {
  CT_ALWAYS_FORCEINLINE __m512 operator()(__m512i v) { return _mm512_castsi512_ps(v); }
};
template <> struct BitCastFromInt512<float64_t> {
  CT_ALWAYS_FORCEINLINE __m512d operator()(__m512i v) { return _mm512_castsi512_pd(v); }
};
#ifdef HAS_AVX512_FP16
template <> struct BitCastFromInt512<float16_t> {
  CT_ALWAYS_FORCEINLINE __m512h operator()(__m512i v) { return _mm512_castsi512_ph(v); }
};
#endif
template <typename T> struct BitCastToInt512 {
  CT_ALWAYS_FORCEINLINE __m512i operator()(__m512i v) { return v; }
};
template <> struct BitCastToInt512<float32_t> {
  CT_ALWAYS_FORCEINLINE __m512i operator()(__m512 v) { return _mm512_castps_si512(v); }
};
template <> struct BitCastToInt512<float64_t> {
  CT_ALWAYS_FORCEINLINE __m512i operator()(__m512d v) { return _mm512_castpd_si512(v); }
};
#ifdef HAS_AVX512_FP16
template <> struct BitCastToInt512<float16_t> {
  CT_ALWAYS_FORCEINLINE __m512i operator()(__m512h v) { return _mm512_castph_si512(v); }
};
#endif

template <typename T> CT_ALWAYS_FORCEINLINE
RegType<uint8_t, 64> bitcast_to_int(RegType<T, 64 / sizeof(T)> v) { return BitCastToInt512<T>()(v.v); }
template <typename T> CT_ALWAYS_FORCEINLINE
RegType<T, 64 / sizeof(T)> bitcast_from_int(RegType<uint8_t, 64> v) { return BitCastFromInt512<T>()(v.v); }
#endif // VEC_WIDTH >= 512

} // namespace details

/* ************************************************************************** */
//                                Bit Casts                                   //
/* ************************************************************************** */

template <typename T, typename V, TL_IF(sizeof(V) == sizeof(Vec<T>)), TL_IF(sizeof(V) <= VEC_WIDTH / 8)>
CT_ALWAYS_FORCEINLINE Vec<T> bitcast(T t, V v) {
  return details::bitcast_from_int<TypeOf<T>>(details::bitcast_to_int(v));
}

template <typename T, typename V, TL_IF(sizeof(V) == sizeof(Vec<T>)), TL_IF(sizeof(V) == 2 * VEC_WIDTH / 8)>
CT_ALWAYS_FORCEINLINE Vec<T> bitcast(T t, V v) {
  Tag<TypeOf<T>, T::N / 2> t1;
  return Vec<T>{ word::bitcast(t1, v[0]), word::bitcast(t1, v[1]) };
}

#if VEC_WIDTH >= 256
template <typename T, typename V, TL_IF(sizeof(V) == 16 && sizeof(Vec<T>) == 32)>
CT_ALWAYS_FORCEINLINE Vec<T> bitcast(T t, V v) {
  return details::bitcast_from_int<TypeOf<T>>(
      RegType<uint8_t, 32>{_mm256_castsi128_si256(details::bitcast_to_int(v).v)});
}
template <typename T, typename V, TL_IF(sizeof(V) == 32 && sizeof(Vec<T>) == 16)>
CT_ALWAYS_FORCEINLINE Vec<T> bitcast(T t, V v) {
  return details::bitcast_from_int<TypeOf<T>>(
      RegType<uint8_t, 16>{_mm256_castsi256_si128(details::bitcast_to_int(v).v)});
}
#endif // VEC_WIDTH >= 256
template <typename T, typename V, TL_IF(sizeof(V) == 16 && sizeof(Vec<T>) == 64)>
CT_ALWAYS_FORCEINLINE Vec<T> bitcast(T t, V v) {
  return details::bitcast_from_int<TypeOf<T>>(
      RegType<uint8_t, 64>{_mm512_castsi128_si512(details::bitcast_to_int(v).v)});
}
template <typename T, typename V, TL_IF(sizeof(V) == 64 && sizeof(Vec<T>) == 16)>
CT_ALWAYS_FORCEINLINE Vec<T> bitcast(T t, V v) {
  return details::bitcast_from_int<TypeOf<T>>(
      RegType<uint8_t, 16>{_mm512_castsi512_si128(details::bitcast_to_int(v).v)});
}
template <typename T, typename V, TL_IF(sizeof(V) == 32 && sizeof(Vec<T>) == 64)>
CT_ALWAYS_FORCEINLINE Vec<T> bitcast(T t, V v) {
  return details::bitcast_from_int<TypeOf<T>>(
      RegType<uint8_t, 64>{_mm512_castsi256_si512(details::bitcast_to_int(v).v)});
}
template <typename T, typename V, TL_IF(sizeof(V) == 64 && sizeof(Vec<T>) == 32)>
CT_ALWAYS_FORCEINLINE Vec<T> bitcast(T t, V v) {
  return details::bitcast_from_int<TypeOf<T>>(
      RegType<uint8_t, 32>{_mm512_castsi512_si256(details::bitcast_to_int(v).v)});
}

/* ************************************************************************** */
//                           Intra-block Shuffle                              //
/* ************************************************************************** */

namespace details {
template <nint_t N, int... Is>
struct StaticIndexChecker;

template <nint_t N>
struct StaticIndexChecker<N> {
  CT_ALWAYS_FORCEINLINE void operator()() {}
};

template <nint_t N, int I0, int... Is>
struct StaticIndexChecker<N, I0, Is...> {
  CT_ALWAYS_FORCEINLINE void operator()() {
    static_assert(0 <= I0 && I0 < N, "Index out of range");
    StaticIndexChecker<N, Is...>()();
  }
};

template <nint_t N, nint_t J, typename... Is>
struct RuntimeIndexChecker;

template <nint_t N, nint_t J>
struct RuntimeIndexChecker<N, J> {
  CT_ALWAYS_FORCEINLINE void operator()() {}
};

template <nint_t N, nint_t J, typename I0, typename... Is>
struct RuntimeIndexChecker<N, J, I0, Is...> {
  CT_ALWAYS_FORCEINLINE void operator()(I0 i0, Is... is) {
    CT_ASSERT(0 <= i0 && i0 < N, "Index #%zd out of range: %d !in 0:%zd", N - J - 1, int(i0), N);
    RuntimeIndexChecker<N, J + 1, Is...>()(is...);
  }
};

template <int... Is>
CT_ALWAYS_FORCEINLINE void assert_index() { StaticIndexChecker<sizeof...(Is), Is...>()(); }

template <typename... Is>
CT_ALWAYS_FORCEINLINE void assert_index(Is... is) { RuntimeIndexChecker<sizeof...(is), 0, Is...>()(is...); }
} // namespace details

/* ******************************** float32_t ******************************* */

template <int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<float32_t, 4>> local_shuf(Vec<Tag<float32_t, 4>> v) {
  details::assert_index<I3, I2, I1, I0>();
  return _mm_shuffle_ps(v.v, v.v, _MM_SHUFFLE(I3, I2, I1, I0));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<float32_t, 4>> local_shuf(Vec<Tag<float32_t, 4>> v, Vec<Tag<int32_t, 4>> i) {
  return _mm_permutevar_ps(v.v, i.v);
}

CT_ALWAYS_FORCEINLINE Vec<Tag<float32_t, 4>> local_shuf(Vec<Tag<float32_t, 4>> v, int i3, int i2, int i1, int i0) {
  details::assert_index(i3, i2, i1, i0);
  return word::local_shuf(v, _mm_set_epi32(i3, i2, i1, i0));
}

#if VEC_WIDTH >= 256
template <int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<float32_t, 8>> local_shuf(Vec<Tag<float32_t, 8>> v) {
  details::assert_index<I3, I2, I1, I0>();
  return _mm256_shuffle_ps(v.v, v.v, _MM_SHUFFLE(I3, I2, I1, I0));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<float32_t, 8>> local_shuf(Vec<Tag<float32_t, 8>> v, Vec<Tag<int32_t, 8>> i) {
  return _mm256_permutevar_ps(v.v, i.v);
}

CT_ALWAYS_FORCEINLINE Vec<Tag<float32_t, 8>> local_shuf(Vec<Tag<float32_t, 8>> v, int i3, int i2, int i1, int i0) {
  details::assert_index(i3, i2, i1, i0);
  return word::local_shuf(v, _mm256_set_epi32(i3, i2, i1, i0, i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<float32_t, 16>> local_shuf(Vec<Tag<float32_t, 16>> v) {
  details::assert_index<I3, I2, I1, I0>();
  return _mm512_shuffle_ps(v.v, v.v, _MM_SHUFFLE(I3, I2, I1, I0));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<float32_t, 16>> local_shuf(Vec<Tag<float32_t, 16>> v, Vec<Tag<int32_t, 16>> i) {
  return _mm512_permutevar_ps(v.v, i.v);
}

CT_ALWAYS_FORCEINLINE Vec<Tag<float32_t, 16>> local_shuf(Vec<Tag<float32_t, 16>> v, int i3, int i2, int i1, int i0) {
  details::assert_index(i3, i2, i1, i0);
  return word::local_shuf(v, _mm512_set_epi32(i3, i2, i1, i0, i3, i2, i1, i0, i3, i2, i1, i0, i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 512

/* ********************************* int32_t ******************************** */

template <int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<int32_t, 4>> local_shuf(Vec<Tag<int32_t, 4>> v) {
  details::assert_index<I3, I2, I1, I0>();
  return _mm_shuffle_epi32(v.v, _MM_SHUFFLE(I3, I2, I1, I0));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int32_t, 4>> local_shuf(Vec<Tag<int32_t, 4>> v, Vec<Tag<int32_t, 4>> i) {
  Tag<float32_t, 4> t1; Tag<int32_t, 4> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int32_t, 4>> local_shuf(Vec<Tag<int32_t, 4>> v, int i3, int i2, int i1, int i0) {
  Tag<float32_t, 4> t1; Tag<int32_t, 4> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i3, i2, i1, i0));
}

#if VEC_WIDTH >= 256
template <int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<int32_t, 8>> local_shuf(Vec<Tag<int32_t, 8>> v) {
  details::assert_index<I3, I2, I1, I0>();
  return _mm256_shuffle_epi32(v.v, _MM_SHUFFLE(I3, I2, I1, I0));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int32_t, 8>> local_shuf(Vec<Tag<int32_t, 8>> v, Vec<Tag<int32_t, 8>> i) {
  Tag<float32_t, 8> t1; Tag<int32_t, 8> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int32_t, 8>> local_shuf(Vec<Tag<int32_t, 8>> v, int i3, int i2, int i1, int i0) {
  details::assert_index(i3, i2, i1, i0);
  return word::local_shuf(v, _mm256_set_epi32(i3, i2, i1, i0, i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<int32_t, 16>> local_shuf(Vec<Tag<int32_t, 16>> v) {
  details::assert_index<I3, I2, I1, I0>();
  return _mm512_shuffle_epi32(v.v, _MM_SHUFFLE(I3, I2, I1, I0));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int32_t, 16>> local_shuf(Vec<Tag<int32_t, 16>> v, Vec<Tag<int32_t, 16>> i) {
  Tag<float32_t, 16> t1; Tag<int32_t, 16> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int32_t, 16>> local_shuf(Vec<Tag<int32_t, 16>> v, int i3, int i2, int i1, int i0) {
  details::assert_index(i3, i2, i1, i0);
  return word::local_shuf(v, _mm512_set_epi32(i3, i2, i1, i0, i3, i2, i1, i0, i3, i2, i1, i0, i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 512

/* ********************************* uint32_t ******************************* */

template <int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<uint32_t, 4>> local_shuf(Vec<Tag<uint32_t, 4>> v) {
  Tag<int32_t, 4> t1; Tag<uint32_t, 4> t2;
  return word::bitcast(t2, word::local_shuf<I3, I2, I1, I0>(word::bitcast(t1, v)));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint32_t, 4>> local_shuf(Vec<Tag<uint32_t, 4>> v, Vec<Tag<int32_t, 4>> i) {
  Tag<int32_t, 4> t1; Tag<uint32_t, 4> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint32_t, 4>> local_shuf(Vec<Tag<uint32_t, 4>> v, int i3, int i2, int i1, int i0) {
  Tag<int32_t, 4> t1; Tag<uint32_t, 4> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i3, i2, i1, i0));
}

#if VEC_WIDTH >= 256
template <int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<uint32_t, 8>> local_shuf(Vec<Tag<uint32_t, 8>> v) {
  Tag<int32_t, 8> t1; Tag<uint32_t, 8> t2;
  return word::bitcast(t2, word::local_shuf<I3, I2, I1, I0>(bitcast(t1, v)));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint32_t, 8>> local_shuf(Vec<Tag<uint32_t, 8>> v, Vec<Tag<int32_t, 8>> i) {
  Tag<int32_t, 8> t1; Tag<uint32_t, 8> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint32_t, 8>> local_shuf(Vec<Tag<uint32_t, 8>> v, int i3, int i2, int i1, int i0) {
  Tag<int32_t, 8> t1; Tag<uint32_t, 8> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<uint32_t, 16>> local_shuf(Vec<Tag<uint32_t, 16>> v) {
  Tag<int32_t, 16> t1; Tag<uint32_t, 16> t2;
  return word::bitcast(t2, word::local_shuf<I3, I2, I1, I0>(bitcast(t1, v)));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint32_t, 16>> local_shuf(Vec<Tag<uint32_t, 16>> v, Vec<Tag<int32_t, 16>> i) {
  Tag<int32_t, 16> t1; Tag<uint32_t, 16> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint32_t, 16>> local_shuf(Vec<Tag<uint32_t, 16>> v, int i3, int i2, int i1, int i0) {
  Tag<int32_t, 16> t1; Tag<uint32_t, 16> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 512

/* ********************************* float64_t ****************************** */

template <int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<float64_t, 2>> local_shuf(Vec<Tag<float64_t, 2>> v) {
  details::assert_index<I1, I0>();
  return _mm_shuffle_pd(v.v, v.v, _MM_SHUFFLE2(I1, I0));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<float64_t, 2>> local_shuf(Vec<Tag<float64_t, 2>> v, Vec<Tag<int64_t, 2>> i) {
  return _mm_permutevar_pd(v.v, _mm_slli_epi64(i.v, 1));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<float64_t, 2>> local_shuf(Vec<Tag<float64_t, 2>> v, int i1, int i0) {
  details::assert_index(i1, i0);
  return local_shuf(v, _mm_set_epi64x(i1, i0));
}

#if VEC_WIDTH >= 256
template <int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<float64_t, 4>> local_shuf(Vec<Tag<float64_t, 4>> v) {
  details::assert_index<I1, I0>();
  return _mm256_shuffle_pd(v.v, v.v, ((I1 << 3) | (I0 << 2) | (I1 << 1) | (I0 << 0)));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<float64_t, 4>> local_shuf(Vec<Tag<float64_t, 4>> v, Vec<Tag<int64_t, 4>> i) {
  return _mm256_permutevar_pd(v.v, _mm256_slli_epi64(i.v, 1));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<float64_t, 4>> local_shuf(Vec<Tag<float64_t, 4>> v, int i1, int i0) {
  details::assert_index(i1, i0);
  return local_shuf(v, _mm256_set_epi64x(i1, i0, i1, i0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<float64_t, 8>> local_shuf(Vec<Tag<float64_t, 8>> v) {
  details::assert_index<I1, I0>();
  return _mm512_shuffle_pd(v.v, v.v, ((I1 << 7) | (I0 << 6) | (I1 << 5) | (I0 << 4) | (I1 << 3) | (I0 << 2) | (I1 << 1) | (I0 << 0)));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<float64_t, 8>> local_shuf(Vec<Tag<float64_t, 8>> v, Vec<Tag<int64_t, 8>> i) {
  return _mm512_permutevar_pd(v.v, _mm512_slli_epi64(i.v, 1));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<float64_t, 8>> local_shuf(Vec<Tag<float64_t, 8>> v, int i1, int i0) {
  details::assert_index(i1, i0);
  return local_shuf(v, _mm512_set_epi64(i1, i0, i1, i0, i1, i0, i1, i0));
}
#endif // VEC_WIDTH >= 512

/* ********************************** int64_t ******************************* */

template <int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<int64_t, 2>> local_shuf(Vec<Tag<int64_t, 2>> v) {
  Tag<float64_t, 2> t1; Tag<int64_t, 2> t2;
  return word::bitcast(t2, word::local_shuf<I1, I0>(bitcast(t1, v)));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int64_t, 2>> local_shuf(Vec<Tag<int64_t, 2>> v, Vec<Tag<int64_t, 2>> i) {
  Tag<float64_t, 2> t1; Tag<int64_t, 2> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int64_t, 2>> local_shuf(Vec<Tag<int64_t, 2>> v, int i1, int i0) {
  Tag<float64_t, 2> t1; Tag<int64_t, 2> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i1, i0));
}

#if VEC_WIDTH >= 256
template <int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<int64_t, 4>> local_shuf(Vec<Tag<int64_t, 4>> v) {
  Tag<float64_t, 4> t1; Tag<int64_t, 4> t2;
  return word::bitcast(t2, word::local_shuf<I1, I0>(bitcast(t1, v)));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int64_t, 4>> local_shuf(Vec<Tag<int64_t, 4>> v, Vec<Tag<int64_t, 4>> i) {
  Tag<float64_t, 4> t1; Tag<int64_t, 4> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int64_t, 4>> local_shuf(Vec<Tag<int64_t, 4>> v, int i1, int i0) {
  Tag<float64_t, 4> t1; Tag<int64_t, 4> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i1, i0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<int64_t, 8>> local_shuf(Vec<Tag<int64_t, 8>> v) {
  Tag<float64_t, 8> t1; Tag<int64_t, 8> t2;
  return word::bitcast(t2, word::local_shuf<I1, I0>(bitcast(t1, v)));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int64_t, 8>> local_shuf(Vec<Tag<int64_t, 8>> v, Vec<Tag<int64_t, 8>> i) {
  Tag<float64_t, 8> t1; Tag<int64_t, 8> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int64_t, 8>> local_shuf(Vec<Tag<int64_t, 8>> v, int i1, int i0) {
  Tag<float64_t, 8> t1; Tag<int64_t, 8> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i1, i0));
}
#endif // VEC_WIDTH >= 512

/* ********************************* uint64_t ******************************* */

template <int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<uint64_t, 2>> local_shuf(Vec<Tag<uint64_t, 2>> v) {
  Tag<int64_t, 2> t1; Tag<uint64_t, 2> t2;
  return word::bitcast(t2, word::local_shuf<I1, I0>(bitcast(t1, v)));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint64_t, 2>> local_shuf(Vec<Tag<uint64_t, 2>> v, Vec<Tag<int64_t, 2>> i) {
  Tag<int64_t, 2> t1; Tag<uint64_t, 2> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint64_t, 2>> local_shuf(Vec<Tag<uint64_t, 2>> v, int i1, int i0) {
  Tag<int64_t, 2> t1; Tag<uint64_t, 2> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i1, i0));
}


#if VEC_WIDTH >= 256
template <int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<uint64_t, 4>> local_shuf(Vec<Tag<uint64_t, 4>> v) {
  Tag<int64_t, 4> t1; Tag<uint64_t, 4> t2;
  return word::bitcast(t2, word::local_shuf<I1, I0>(bitcast(t1, v)));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint64_t, 4>> local_shuf(Vec<Tag<uint64_t, 4>> v, Vec<Tag<int64_t, 4>> i) {
  Tag<int64_t, 4> t1; Tag<uint64_t, 4> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint64_t, 4>> local_shuf(Vec<Tag<uint64_t, 4>> v, int i1, int i0) {
  Tag<int64_t, 4> t1; Tag<uint64_t, 4> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i1, i0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<uint64_t, 8>> local_shuf(Vec<Tag<uint64_t, 8>> v) {
  Tag<int64_t, 8> t1; Tag<uint64_t, 8> t2;
  return word::bitcast(t2, word::local_shuf<I1, I0>(bitcast(t1, v)));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint64_t, 8>> local_shuf(Vec<Tag<uint64_t, 8>> v, Vec<Tag<int64_t, 8>> i) {
  Tag<int64_t, 8> t1; Tag<uint64_t, 8> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint64_t, 8>> local_shuf(Vec<Tag<uint64_t, 8>> v, int i1, int i0) {
  Tag<int64_t, 8> t1; Tag<uint64_t, 8> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i1, i0));
}
#endif // VEC_WIDTH >= 512

/* ********************************** int8_t ******************************** */

CT_ALWAYS_FORCEINLINE Vec<Tag<int8_t, 16>> local_shuf(Vec<Tag<int8_t, 16>> v, Vec<Tag<int8_t, 16>> i) {
  return _mm_shuffle_epi8(v.v, i.v);
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int8_t, 16>> local_shuf(Vec<Tag<int8_t, 16>> v, int i15, int i14, int i13, int i12, int i11, int i10, int i9, int i8, int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  details::assert_index(i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0);
  return local_shuf(v, _mm_set_epi8(i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0));
}

template <int I15, int I14, int I13, int I12, int I11, int I10, int I9, int I8, int I7, int I6, int I5, int I4, int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<int8_t, 16>> local_shuf(Vec<Tag<int8_t, 16>> v) {
  details::assert_index<I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0>();
  return local_shuf(v, _mm_set_epi8(I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0));
}


#if VEC_WIDTH >= 256
CT_ALWAYS_FORCEINLINE Vec<Tag<int8_t, 32>> local_shuf(Vec<Tag<int8_t, 32>> v, Vec<Tag<int8_t, 32>> i) {
  return _mm256_shuffle_epi8(v.v, i.v);
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int8_t, 32>> local_shuf(Vec<Tag<int8_t, 32>> v, int i15, int i14, int i13, int i12, int i11, int i10, int i9, int i8, int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  details::assert_index(i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0);
  return local_shuf(v, _mm256_set_epi8(i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0, i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0));
}

template <int I15, int I14, int I13, int I12, int I11, int I10, int I9, int I8, int I7, int I6, int I5, int I4, int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<int8_t, 32>> local_shuf(Vec<Tag<int8_t, 32>> v) {
  details::assert_index<I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0>();
  return local_shuf(v, _mm256_set_epi8(I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0, I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
CT_ALWAYS_FORCEINLINE Vec<Tag<int8_t, 64>> local_shuf(Vec<Tag<int8_t, 64>> v, Vec<Tag<int8_t, 64>> i) {
  return _mm512_shuffle_epi8(v.v, i.v);
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int8_t, 64>> local_shuf(Vec<Tag<int8_t, 64>> v, int i15, int i14, int i13, int i12, int i11, int i10, int i9, int i8, int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  details::assert_index(i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0);
  return local_shuf(v, _mm512_set_epi8(i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0, i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0, i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0, i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0));
}

template <int I15, int I14, int I13, int I12, int I11, int I10, int I9, int I8, int I7, int I6, int I5, int I4, int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<int8_t, 64>> local_shuf(Vec<Tag<int8_t, 64>> v) {
  details::assert_index<I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0>();
  return local_shuf(v, _mm512_set_epi8(I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0, I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0, I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0, I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0));
}

#endif // VEC_WIDTH >= 512


/* ********************************* uint8_t ******************************** */

template <int I15, int I14, int I13, int I12, int I11, int I10, int I9, int I8, int I7, int I6, int I5, int I4, int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<uint8_t, 16>> local_shuf(Vec<Tag<uint8_t, 16>> v) {
  Tag<int8_t, 16> t1; Tag<uint8_t, 16> t2;
  return word::bitcast(t2, word::local_shuf<I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0>(bitcast(t1, v)));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint8_t, 16>> local_shuf(Vec<Tag<uint8_t, 16>> v, Vec<Tag<int8_t, 16>> i) {
  Tag<int8_t, 16> t1; Tag<uint8_t, 16> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint8_t, 16>> local_shuf(Vec<Tag<uint8_t, 16>> v, int i15, int i14, int i13, int i12, int i11, int i10, int i9, int i8, int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  Tag<int8_t, 16> t1; Tag<uint8_t, 16> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0));
}


#if VEC_WIDTH >= 256
template <int I15, int I14, int I13, int I12, int I11, int I10, int I9, int I8, int I7, int I6, int I5, int I4, int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<uint8_t, 32>> local_shuf(Vec<Tag<uint8_t, 32>> v) {
  Tag<int8_t, 32> t1; Tag<uint8_t, 32> t2;
  return word::bitcast(t2, word::local_shuf<I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0>(bitcast(t1, v)));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint8_t, 32>> local_shuf(Vec<Tag<uint8_t, 32>> v, Vec<Tag<int8_t, 32>> i) {
  Tag<int8_t, 32> t1; Tag<uint8_t, 32> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint8_t, 32>> local_shuf(Vec<Tag<uint8_t, 32>> v, int i15, int i14, int i13, int i12, int i11, int i10, int i9, int i8, int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  Tag<int8_t, 32> t1; Tag<uint8_t, 32> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <int I15, int I14, int I13, int I12, int I11, int I10, int I9, int I8, int I7, int I6, int I5, int I4, int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<uint8_t, 64>> local_shuf(Vec<Tag<uint8_t, 64>> v) {
  Tag<int8_t, 64> t1; Tag<uint8_t, 64> t2;
  return word::bitcast(t2, word::local_shuf<I15, I14, I13, I12, I11, I10, I9, I8, I7, I6, I5, I4, I3, I2, I1, I0>(bitcast(t1, v)));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint8_t, 64>> local_shuf(Vec<Tag<uint8_t, 64>> v, Vec<Tag<int8_t, 64>> i) {
  Tag<int8_t, 64> t1; Tag<uint8_t, 64> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint8_t, 64>> local_shuf(Vec<Tag<uint8_t, 64>> v, int i15, int i14, int i13, int i12, int i11, int i10, int i9, int i8, int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  Tag<int8_t, 64> t1; Tag<uint8_t, 64> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 512

/* ********************************* int16_t ******************************** */

template <int I7, int I6, int I5, int I4, int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<int16_t, 8>> local_shuf(Vec<Tag<int16_t, 8>> v) {
  if constexpr (I3 < 4 && I2 < 4 && I1 < 4 && I0 < 4 &&
                I7 >= 4 && I6 >= 4 && I5 >= 4 && I4 >= 4) {
    // Fast path: low 4 elements all from [0,3], high 4 all from [4,7]
    auto u = _mm_shufflelo_epi16(v.v, _MM_SHUFFLE(I3, I2, I1, I0));
    return _mm_shufflehi_epi16(u, _MM_SHUFFLE(I7 - 4, I6 - 4, I5 - 4, I4 - 4));
  } else {
    Tag<int8_t, 16> t1; Tag<int16_t, 8> t2;
    return word::bitcast(t2, word::local_shuf<
        2 * I7 + 1, 2 * I7, 2 * I6 + 1, 2 * I6, 2 * I5 + 1, 2 * I5, 2 * I4 + 1, 2 * I4,
        2 * I3 + 1, 2 * I3, 2 * I2 + 1, 2 * I2, 2 * I1 + 1, 2 * I1, 2 * I0 + 1, 2 * I0
    >(word::bitcast(t1, v)));
  }
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int16_t, 8>> local_shuf(Vec<Tag<int16_t, 8>> v, Vec<Tag<int16_t, 8>> i) {
  Tag<int8_t, 16> t1; Tag<int16_t, 8> t2;
  // we assume that high byte of each index in i are zero
  auto i2 = _mm_slli_epi32(i.v, 1); // * 2
  auto i3 = _mm_slli_epi32(i.v, 9); // move to hi byte
  // idx = i2 | i3 | 0x0100
  #ifdef HAS_AVX512F
  auto idx = _mm_ternarylogic_epi32(i2, i3, _mm_set1_epi16(0x0100), _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C);
  #else
  auto idx = _mm_or_si128(_mm_or_si128(i2, i3), _mm_set1_epi16(0x0100));
  #endif
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), Vec<decltype(t1)>{idx}));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int16_t, 8>> local_shuf(Vec<Tag<int16_t, 8>> v, int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  return word::local_shuf(v, _mm_set_epi16(i7, i6, i5, i4, i3, i2, i1, i0));
}


#if VEC_WIDTH >= 256
template <int I7, int I6, int I5, int I4, int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<int16_t, 16>> local_shuf(Vec<Tag<int16_t, 16>> v) {
  if constexpr (I3 < 4 && I2 < 4 && I1 < 4 && I0 < 4 &&
                I7 >= 4 && I6 >= 4 && I5 >= 4 && I4 >= 4) {
    // Fast path: low 4 elements all from [0,3], high 4 all from [4,7]
    auto u = _mm256_shufflelo_epi16(v.v, _MM_SHUFFLE(I3, I2, I1, I0));
    return _mm256_shufflehi_epi16(u, _MM_SHUFFLE(I7 - 4, I6 - 4, I5 - 4, I4 - 4));
  } else {
    Tag<int8_t, 32> t1; Tag<int16_t, 16> t2;
    return word::bitcast(t2, word::local_shuf<
        2 * I7 + 1, 2 * I7, 2 * I6 + 1, 2 * I6, 2 * I5 + 1, 2 * I5, 2 * I4 + 1, 2 * I4,
        2 * I3 + 1, 2 * I3, 2 * I2 + 1, 2 * I2, 2 * I1 + 1, 2 * I1, 2 * I0 + 1, 2 * I0
    >( word::bitcast(t1, v)));
  }
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int16_t, 16>> local_shuf(Vec<Tag<int16_t, 16>> v, Vec<Tag<int16_t, 16>> i) {
  Tag<int8_t, 32> t1; Tag<int16_t, 16> t2;
  // we assume that high byte of each index in i are zero
  auto i2 = _mm256_slli_epi32(i.v, 1); // * 2
  auto i3 = _mm256_slli_epi32(i.v, 9); // move to hi byte
  // idx = i2 | i3 | 0x0100
  #ifdef HAS_AVX512F
  auto idx = _mm256_ternarylogic_epi32(i2, i3, _mm256_set1_epi16(0x0100), _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C);
  #else
  auto idx = _mm256_or_si256(_mm256_or_si256(i2, i3), _mm256_set1_epi16(0x0100));
  #endif
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), Vec<decltype(t1)>{idx}));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int16_t, 16>> local_shuf(Vec<Tag<int16_t, 16>> v, int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  return word::local_shuf(v, _mm256_set_epi16(i7, i6, i5, i4, i3, i2, i1, i0, i7, i6, i5, i4, i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <int I7, int I6, int I5, int I4, int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<int16_t, 32>> local_shuf(Vec<Tag<int16_t, 32>> v) {
  if constexpr (I3 < 4 && I2 < 4 && I1 < 4 && I0 < 4 &&
                I7 >= 4 && I6 >= 4 && I5 >= 4 && I4 >= 4) {
    // Fast path: low 4 elements all from [0,3], high 4 all from [4,7]
    auto u = _mm512_shufflelo_epi16(v.v, _MM_SHUFFLE(I3, I2, I1, I0));
    return _mm512_shufflehi_epi16(u, _MM_SHUFFLE(I7 - 4, I6 - 4, I5 - 4, I4 - 4));
  } else {
    Tag<int8_t, 64> t1; Tag<int16_t, 32> t2;
    return word::bitcast(t2, word::local_shuf<
        2 * I7 + 1, 2 * I7, 2 * I6 + 1, 2 * I6, 2 * I5 + 1, 2 * I5, 2 * I4 + 1, 2 * I4,
        2 * I3 + 1, 2 * I3, 2 * I2 + 1, 2 * I2, 2 * I1 + 1, 2 * I1, 2 * I0 + 1, 2 * I0
    >( word::bitcast(t1, v)));
  }
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int16_t, 32>> local_shuf(Vec<Tag<int16_t, 32>> v, Vec<Tag<int16_t, 32>> i) {
  Tag<int8_t, 64> t1; Tag<int16_t, 32> t2;
  // we assume that high byte of each index in i are zero
  auto i2 = _mm512_slli_epi32(i.v, 1); // * 2
  auto i3 = _mm512_slli_epi32(i.v, 9); // move to hi byte
  // idx = i2 | i3 | 0x0100
  auto idx = _mm512_ternarylogic_epi32(i2, i3, _mm512_set1_epi16(0x0100), _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C);
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), Vec<decltype(t1)>{idx}));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int16_t, 32>> local_shuf(Vec<Tag<int16_t, 32>> v, int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  return word::local_shuf(v, _mm512_set_epi16(i7, i6, i5, i4, i3, i2, i1, i0, i7, i6, i5, i4, i3, i2, i1, i0, i7, i6, i5, i4, i3, i2, i1, i0, i7, i6, i5, i4, i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 512


/* ********************************* uint16_t ******************************* */

template <int I7, int I6, int I5, int I4, int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<uint16_t, 8>> local_shuf(Vec<Tag<uint16_t, 8>> v) {
  Tag<int16_t, 8> t1; Tag<uint16_t, 8> t2;
  return word::bitcast(t2, word::local_shuf<I7, I6, I5, I4, I3, I2, I1, I0>(word::bitcast(t1, v)));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint16_t, 8>> local_shuf(Vec<Tag<uint16_t, 8>> v, Vec<Tag<int16_t, 8>> i) {
  Tag<int16_t, 8> t1; Tag<uint16_t, 8> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint16_t, 8>> local_shuf(Vec<Tag<uint16_t, 8>> v, int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  Tag<int16_t, 8> t1; Tag<uint16_t, 8> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i7, i6, i5, i4, i3, i2, i1, i0));
}


#if VEC_WIDTH >= 256
template <int I7, int I6, int I5, int I4, int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<uint16_t, 16>> local_shuf(Vec<Tag<uint16_t, 16>> v) {
  Tag<int16_t, 16> t1; Tag<uint16_t, 16> t2;
  return word::bitcast(t2, word::local_shuf<I7, I6, I5, I4, I3, I2, I1, I0>(word::bitcast(t1, v)));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint16_t, 16>> local_shuf(Vec<Tag<uint16_t, 16>> v, Vec<Tag<int16_t, 16>> i) {
  Tag<int16_t, 16> t1; Tag<uint16_t, 16> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint16_t, 16>> local_shuf(Vec<Tag<uint16_t, 16>> v, int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  Tag<int16_t, 16> t1; Tag<uint16_t, 16> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i7, i6, i5, i4, i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <int I7, int I6, int I5, int I4, int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<uint16_t, 32>> local_shuf(Vec<Tag<uint16_t, 32>> v) {
  Tag<int16_t, 32> t1; Tag<uint16_t, 32> t2;
  return word::bitcast(t2, word::local_shuf<I7, I6, I5, I4, I3, I2, I1, I0>(word::bitcast(t1, v)));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint16_t, 32>> local_shuf(Vec<Tag<uint16_t, 32>> v, Vec<Tag<int16_t, 32>> i) {
  Tag<int16_t, 32> t1; Tag<uint16_t, 32> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint16_t, 32>> local_shuf(Vec<Tag<uint16_t, 32>> v, int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0) {
  Tag<int16_t, 32> t1; Tag<uint16_t, 32> t2;
  return word::bitcast(t2, word::local_shuf(word::bitcast(t1, v), i7, i6, i5, i4, i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 512



/* ************************************************************************** */
//                             Block-level Shuffle                            //
/* ************************************************************************** */
// xmm only have 1 block/lane
template <int I0, typename V, TL_IF(sizeof(V) == 16)>
CT_ALWAYS_FORCEINLINE V block_shuf(V v) {
  details::assert_index<I0>();
  return v;
}

template <typename V, TL_IF(sizeof(V) == 16)>
CT_ALWAYS_FORCEINLINE V block_shuf(V v, int i0) {
  details::assert_index(i0);
  return v;
}

#if VEC_WIDTH >= 256
template <int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<float32_t, 8>> block_shuf(Vec<Tag<float32_t, 8>> v) {
  details::assert_index<I1, I0>();
  return _mm256_permute2f128_ps(v.v, v.v, ((I1 << 4) | (I0)));
}

template <int I1, int I0, typename V, TL_IF(sizeof(V) == 32), TL_IF(is_none<TypeOf<Vec2Tag<V>>, float32_t>)>
CT_ALWAYS_FORCEINLINE V block_shuf(V v) {
  Tag<float32_t, 8> t1; Vec2Tag<V> t2;
  return word::bitcast(t2, word::block_shuf<I1, I0>(word::bitcast(t1, v)));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<float32_t, 8>> block_shuf(Vec<Tag<float32_t, 8>> v, int i1, int i0) {
  details::assert_index(i1, i0);
  return _mm256_permutevar8x32_ps(v.v, _mm256_set_epi32(i1 + 3, i1 + 2, i1 + 1, i1 + 0, i0 + 3, i0 + 2, i0 + 1, i0 + 0));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<float64_t, 4>> block_shuf(Vec<Tag<float64_t, 4>> v, int i1, int i0) {
  Tag<float32_t, 8> t1; Tag<float64_t, 4> t2;
  return word::bitcast(t2, word::block_shuf(word::bitcast(t1, v), i1, i0));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int32_t, 8>> block_shuf(Vec<Tag<int32_t, 8>> v, int i1, int i0) {
  details::assert_index(i1, i0);
  return _mm256_permutevar8x32_epi32(v.v, _mm256_set_epi32(i1 + 3, i1 + 2, i1 + 1, i1 + 0, i0 + 3, i0 + 2, i0 + 1, i0 + 0));
}

template <typename V, TL_IF(sizeof(V) == 32), TL_IF(is_none<TypeOf<Vec2Tag<V>>, float32_t, float64_t, int32_t>)>
CT_ALWAYS_FORCEINLINE V block_shuf(V v, int i1, int i0) {
  Tag<int32_t, 8> t1; Vec2Tag<V> t2;
  return word::bitcast(t2, word::block_shuf(word::bitcast(t1, v), i1, i0));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<float32_t, 16>> block_shuf(Vec<Tag<float32_t, 16>> v) {
  details::assert_index<I3, I2, I1, I0>();
  return _mm512_shuffle_f32x4(v.v, v.v, _MM_SHUFFLE(I3, I2, I1, I0));
}

template <int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<float64_t, 8>> block_shuf(Vec<Tag<float64_t, 8>> v) {
  details::assert_index<I3, I2, I1, I0>();
  return _mm512_shuffle_f64x2(v.v, v.v, _MM_SHUFFLE(I3, I2, I1, I0));
}

template <int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<int32_t, 16>> block_shuf(Vec<Tag<int32_t, 16>> v) {
  details::assert_index<I3, I2, I1, I0>();
  return _mm512_shuffle_i32x4(v.v, v.v, _MM_SHUFFLE(I3, I2, I1, I0));
}

template <int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<uint32_t, 16>> block_shuf(Vec<Tag<uint32_t, 16>> v) {
  Tag<int32_t, 16> t1; Tag<uint32_t, 16> t2;
  return word::bitcast(t2, word::block_shuf<I3, I2, I1, I0>(word::bitcast(t1, v)));
}

template <int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<int64_t, 8>> block_shuf(Vec<Tag<int64_t, 8>> v) {
  details::assert_index<I3, I2, I1, I0>();
  return _mm512_shuffle_i64x2(v.v, v.v, _MM_SHUFFLE(I3, I2, I1, I0));
}

template <int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<uint64_t, 8>> block_shuf(Vec<Tag<uint64_t, 8>> v) {
  Tag<int64_t, 8> t1; Tag<uint64_t, 8> t2;
  return word::bitcast(t2, word::block_shuf<I3, I2, I1, I0>(word::bitcast(t1, v)));
}

template <int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<int16_t, 32>> block_shuf(Vec<Tag<int16_t, 32>> v) {
  Tag<int32_t, 16> t1; Tag<int16_t, 32> t2;
  return word::bitcast(t2, word::block_shuf<I3, I2, I1, I0>(word::bitcast(t1, v)));
}

template <int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<uint16_t, 32>> block_shuf(Vec<Tag<uint16_t, 32>> v) {
  Tag<int16_t, 32> t1; Tag<uint16_t, 32> t2;
  return word::bitcast(t2, word::block_shuf<I3, I2, I1, I0>(word::bitcast(t1, v)));
}

template <int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<int8_t, 64>> block_shuf(Vec<Tag<int8_t, 64>> v) {
  Tag<int32_t, 16> t1; Tag<int8_t, 64> t2;
  return word::bitcast(t2, word::block_shuf<I3, I2, I1, I0>(word::bitcast(t1, v)));
}

template <int I3, int I2, int I1, int I0>
CT_ALWAYS_FORCEINLINE Vec<Tag<uint8_t, 64>> block_shuf(Vec<Tag<uint8_t, 64>> v) {
  Tag<int8_t, 64> t1; Tag<uint8_t, 64> t2;
  return word::bitcast(t2, word::block_shuf<I3, I2, I1, I0>(word::bitcast(t1, v)));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<float64_t, 8>> block_shuf(Vec<Tag<float64_t, 8>> v, int i3, int i2, int i1, int i0) {
  details::assert_index(i3, i2, i1, i0);
  return _mm512_permutexvar_pd(_mm512_set_epi64(i3 + 1, i3, i2 + 1, i2, i1 + 1, i1, i0 + 1, i0), v.v);
}

CT_ALWAYS_FORCEINLINE Vec<Tag<float32_t, 16>> block_shuf(Vec<Tag<float32_t, 16>> v, int i3, int i2, int i1, int i0) {
  Tag<float64_t, 8> t1; Tag<float32_t, 16> t2;
  return word::bitcast(t2, word::block_shuf(word::bitcast(t1, v), i3, i2, i1, i0));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int64_t, 8>> block_shuf(Vec<Tag<int64_t, 8>> v, int i3, int i2, int i1, int i0) {
  details::assert_index(i3, i2, i1, i0);
  return _mm512_permutexvar_epi64(_mm512_set_epi64(i3 + 1, i3, i2 + 1, i2, i1 + 1, i1, i0 + 1, i0), v.v);
}

template <typename V, TL_IF(sizeof(V) == 64), TL_IF(is_none<TypeOf<Vec2Tag<V>>, float32_t, float64_t, int64_t>)>
CT_ALWAYS_FORCEINLINE V block_shuf(V v, int i3, int i2, int i1, int i0) {
  Tag<int64_t, 8> t1; Vec2Tag<V> t2;
  return word::bitcast(t2, word::block_shuf(word::bitcast(t1, v), i3, i2, i1, i0));
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                       Global Shuffle / Table Lookup                        //
/* ************************************************************************** */
template <typename V, TL_IF(sizeof(TypeOf<Vec2Tag<V>>) == 1), TL_IF(sizeof(V) == 16)>
CT_ALWAYS_FORCEINLINE V shuf(V v, Vec<Tag<int8_t, 16>> i) {
  return word::local_shuf(v, i);
}

template <typename V, TL_IF(sizeof(TypeOf<Vec2Tag<V>>) == 2), TL_IF(sizeof(V) == 16)>
CT_ALWAYS_FORCEINLINE V shuf(V v, Vec<Tag<int16_t, 8>> i) {
  return word::local_shuf(v, i);
}

template <typename V, TL_IF(sizeof(TypeOf<Vec2Tag<V>>) == 4), TL_IF(sizeof(V) == 16)>
CT_ALWAYS_FORCEINLINE V shuf(V v, Vec<Tag<int32_t, 4>> i) {
  return word::local_shuf(v, i);
}

template <typename V, TL_IF(sizeof(TypeOf<Vec2Tag<V>>) == 8), TL_IF(sizeof(V) == 16)>
CT_ALWAYS_FORCEINLINE V shuf(V v, Vec<Tag<int64_t, 2>> i) {
  return word::local_shuf(v, i);
}

#if VEC_WIDTH >= 256
CT_ALWAYS_FORCEINLINE Vec<Tag<float32_t, 8>> shuf(Vec<Tag<float32_t, 8>> v, Vec<Tag<int32_t, 8>> i) {
  return _mm256_permutevar8x32_ps(v.v, i.v);
}

CT_ALWAYS_FORCEINLINE Vec<Tag<float64_t, 4>> shuf(Vec<Tag<float64_t, 4>> v, Vec<Tag<int64_t, 4>> i) {
  #ifdef HAS_AVX512F
  return _mm256_permutexvar_pd(i.v, v.v);
  #else
  auto i1 = _mm256_slli_epi64(i.v, 1); // i * 2
  auto i2 = _mm256_slli_epi64(i.v, 33); // move to hi 4 bytes
  auto idx = _mm256_or_si256(_mm256_or_si256(i1, i2), _mm256_set1_epi64x(0x00000001'00000000));
  Tag<float32_t, 8> t1; Tag<float64_t, 4> t2; Tag<int32_t, 8> ti;
  return word::bitcast(t2, word::shuf(word::bitcast(t1, v), Vec<decltype(ti)>{idx}));
  #endif
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int64_t, 4>> shuf(Vec<Tag<int64_t, 4>> v, Vec<Tag<int64_t, 4>> i) {
  #ifdef HAS_AVX512F
  return _mm256_permutexvar_epi64(i.v, v.v);
  #else
  Tag<float64_t, 4> t1; Tag<int64_t, 4> t2;
  return word::bitcast(t2, word::shuf(word::bitcast(t1, v), i));
  #endif
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint64_t, 4>> shuf(Vec<Tag<uint64_t, 4>> v, Vec<Tag<int64_t, 4>> i) {
  Tag<int64_t, 4> t1; Tag<uint64_t, 4> t2;
  return word::bitcast(t2, word::shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int32_t, 8>> shuf(Vec<Tag<int32_t, 8>> v, Vec<Tag<int32_t, 8>> i) {
  return _mm256_permutevar8x32_epi32(v.v, i.v);
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint32_t, 8>> shuf(Vec<Tag<uint32_t, 8>> v, Vec<Tag<int32_t, 8>> i) {
  Tag<int32_t, 8> t1; Tag<uint32_t, 8> t2;
  return word::bitcast(t2, word::shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int8_t, 32>> shuf(Vec<Tag<int8_t, 32>> v, Vec<Tag<int8_t, 32>> i) {
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

CT_ALWAYS_FORCEINLINE Vec<Tag<uint8_t, 32>> shuf(Vec<Tag<uint8_t, 32>> v, Vec<Tag<int8_t, 32>> i) {
  Tag<int8_t, 32> t1; Tag<uint8_t, 32> t2;
  return word::bitcast(t2, word::shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int16_t, 16>> shuf(Vec<Tag<int16_t, 16>> v, Vec<Tag<int16_t, 16>> i) {
  #ifdef HAS_AVX512BW
  return _mm256_permutexvar_epi16(i.v, v.v);
  #else
  Tag<int8_t, 32> t1; Tag<int16_t, 16> t2;
  auto i2 = _mm256_slli_epi32(i.v, 1); // * 2
  auto i3 = _mm256_slli_epi32(i.v, 9); // move to hi byte
  // idx = i2 | i3 | 0x0100
  #ifdef HAS_AVX512F
  auto idx = _mm256_ternarylogic_epi32(i2, i3, _mm256_set1_epi16(0x0100), _MM_TERNLOG_A | _MM_TERNLOG_B | _MM_TERNLOG_C);
  #else
  auto idx = _mm256_or_si256(_mm256_or_si256(i2, i3), _mm256_set1_epi16(0x0100));
  #endif
  return bitcast(t2, shuf(bitcast(t1, v), idx));
  #endif
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint16_t, 16>> shuf(Vec<Tag<uint16_t, 16>> v, Vec<Tag<int16_t, 16>> i) {
  Tag<int16_t, 16> t1; Tag<uint16_t, 16> t2;
  return word::bitcast(t2, word::shuf(word::bitcast(t1, v), i));
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
CT_ALWAYS_FORCEINLINE Vec<Tag<float32_t, 16>> shuf(Vec<Tag<float32_t, 16>> v, Vec<Tag<int32_t, 16>> i) {
  return _mm512_permutexvar_ps(i.v, v.v);
}

CT_ALWAYS_FORCEINLINE Vec<Tag<float64_t, 8>> shuf(Vec<Tag<float64_t, 8>> v, Vec<Tag<int64_t, 8>> i) {
  return _mm512_permutexvar_pd(i.v, v.v);
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int64_t, 8>> shuf(Vec<Tag<int64_t, 8>> v, Vec<Tag<int64_t, 8>> i) {
  return _mm512_permutexvar_epi64(i.v, v.v);
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint64_t, 8>> shuf(Vec<Tag<uint64_t, 8>> v, Vec<Tag<int64_t, 8>> i) {
  Tag<int64_t, 8> t1; Tag<uint64_t, 8> t2;
  return word::bitcast(t2, word::shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int32_t, 16>> shuf(Vec<Tag<int32_t, 16>> v, Vec<Tag<int32_t, 16>> i) {
  return _mm512_permutexvar_epi32(i.v, v.v);
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint32_t, 16>> shuf(Vec<Tag<uint32_t, 16>> v, Vec<Tag<int32_t, 16>> i) {
  Tag<int32_t, 16> t1; Tag<uint32_t, 16> t2;
  return word::bitcast(t2, word::shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int8_t, 64>> shuf(Vec<Tag<int8_t, 64>> v, Vec<Tag<int8_t, 64>> i) {
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

CT_ALWAYS_FORCEINLINE Vec<Tag<uint8_t, 64>> shuf(Vec<Tag<uint8_t, 64>> v, Vec<Tag<int8_t, 64>> i) {
  Tag<int8_t, 64> t1; Tag<uint8_t, 64> t2;
  return word::bitcast(t2, word::shuf(word::bitcast(t1, v), i));
}

CT_ALWAYS_FORCEINLINE Vec<Tag<int16_t, 32>> shuf(Vec<Tag<int16_t, 32>> v, Vec<Tag<int16_t, 32>> i) {
  return _mm512_permutexvar_epi16(i.v, v.v);
}

CT_ALWAYS_FORCEINLINE Vec<Tag<uint16_t, 32>> shuf(Vec<Tag<uint16_t, 32>> v, Vec<Tag<int16_t, 32>> i) {
  Tag<int16_t, 32> t1; Tag<uint16_t, 32> t2;
  return word::bitcast(t2, word::shuf(word::bitcast(t1, v), i));
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                              Pack & Unconcat                                 //
/* ************************************************************************** */

template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 2), TL_IF(sizeof(TypeOf<T>) == 1)>
CT_ALWAYS_FORCEINLINE Vec<T> concat(T t, V v1, V v2) {
  Tag<uint8_t, 16> t1;
  auto u1 = word::bitcast(t1, v1);
  auto u2 = word::bitcast(t1, v2);
  auto w1 = _mm_cvtsi128_si32(v1.v);
  auto w2 = _mm_cvtsi128_si32(v2.v);
  auto w3 = (w1 & 0xff) | ((w2 && 0xff) << 8);
  auto r = _mm_cvtsi32_si128(w3);
  return word::bitcast(t, Vec<decltype(t1)>{r});
}

template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 4), TL_IF(sizeof(TypeOf<T>) <= 2)>
CT_ALWAYS_FORCEINLINE Vec<T> concat(T t, V v1, V v2) {
  Tag<uint8_t, 16> t1;
  auto u1 = word::bitcast(t1, v1);
  auto u2 = word::bitcast(t1, v2);
  auto shuf = _mm_shufflelo_epi16(u2.v, _MM_SHUFFLE(0, 0, 0, 0));
  auto r = _mm_blend_epi16(u1.v, shuf, 0b10);
  return word::bitcast(t, Vec<decltype(t1)>{r});
}

template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 8), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> concat(T t, V v1, V v2) {
  auto hi = word::local_shuf<0, 0, 0, 0>(v1);
  return _mm_move_ss(hi.v, v2.v);
}

template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 8), TL_IF(is_none<TypeOf<T>, float32_t> && sizeof(TypeOf<T>) <= 4)>
CT_ALWAYS_FORCEINLINE Vec<T> concat(T t, V v1, V v2) {
  Tag<uint8_t, 16> t1;
  Tag<float32_t, 4> t2;
  auto hi = word::bitcast(t2, word::local_shuf<0, 0, 0, 0>(word::bitcast(t1, v1)));
  return word::bitcast(t, Vec<decltype(t2)>{_mm_move_ss(hi.v, v2.v)});
}

template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 8)>
CT_ALWAYS_FORCEINLINE Vec<T> concat(T t, V v1, V v2) {
  Tag<float32_t, 4> t1;
  auto u1 = word::bitcast(t1, v1);
  auto u2 = word::bitcast(t1, v2);
  return word::bitcast(t, Vec<decltype(t1)>{_mm_movelh_ps(u1.v, u2.v)});
}

template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 16)>
CT_ALWAYS_FORCEINLINE Vec<T> concat(T t, V v1, V v2) {
  Tag<float32_t, 4> t1;
  auto u1 = word::bitcast(t1, v1);
  auto u2 = word::bitcast(t1, v2);
  return word::bitcast(t, Vec<decltype(t1)>{_mm_movelh_ps(u1.v, u2.v)});
}

#if VEC_WIDTH >= 256
template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 32)>
CT_ALWAYS_FORCEINLINE Vec<T> concat(T t, V v1, V v2) {
  Tag<uint8_t, 32> t1;
  Tag<uint8_t, 16> t2;
  auto u1 = word::bitcast(t1, v1);
  auto u2 = word::bitcast(t2, v2);
  return word::bitcast(t, Vec<decltype(t1)>{_mm256_insertf128_si256(u1.v, u2.v, 1)});
}
#else
template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 32)>
CT_ALWAYS_FORCEINLINE Vec<T> concat(T t, V v1, V v2) {
  return Vec<T>{ v1, v2 };
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 64)>
CT_ALWAYS_FORCEINLINE Vec<T> concat(T t, V v1, V v2) {
  Tag<uint8_t, 64> t1;
  Tag<uint8_t, 32> t2;
  auto u1 = word::bitcast(t1, v1);
  auto u2 = word::bitcast(t2, v2);
  return word::bitcast(t, Vec<decltype(t1)>{_mm512_inserti64x4(u1.v, u2.v, 1)});
}
template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 128)>
CT_ALWAYS_FORCEINLINE Vec<T> concat(T t, V v1, V v2) {
  return Vec<T>{ v1, v2 };
}
#elif VEC_WIDTH >= 256
template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 64)>
CT_ALWAYS_FORCEINLINE Vec<T> concat(T t, V v1, V v2) {
  return Vec<T>{ v1, v2 };
}
#else
template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 64)>
CT_ALWAYS_FORCEINLINE Vec<T> concat(T t, V v1, V v2) {
  return Vec<T>{ v1[0], v1[1], v2[0], v2[1] };
}
#endif // VEC_WIDTH >= 512

template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes <= VEC_WIDTH / 8)>
CT_ALWAYS_FORCEINLINE V lower(T t, Vec<T> v) {
  Tag<TypeOf<T>, T::N / 2> t1;
  return word::bitcast(t1, v);
}

template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 2 * VEC_WIDTH / 8)>
CT_ALWAYS_FORCEINLINE V lower(T t, Vec<T> v) {
  return v[0];
}

template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 2), TL_IF(sizeof(TypeOf<T>) == 1)>
CT_ALWAYS_FORCEINLINE V upper(T t, Vec<T> v) {
  Tag<uint8_t, 16> t1;
  Tag<TypeOf<T>, T::N / 2> tr;
  auto u = word::bitcast(t1, v);
  auto r = _mm_srli_epi16(u.v, 8);
  return word::bitcast(tr, Vec<decltype(t1)>{r});
}

template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 4), TL_IF(sizeof(TypeOf<T>) <= 2)>
CT_ALWAYS_FORCEINLINE V upper(T t, Vec<T> v) {
  Tag<uint8_t, 16> t1;
  Tag<TypeOf<T>, T::N / 2> tr;
  auto u = word::bitcast(t1, v);
  auto r = _mm_srli_epi32(u.v, 16);
  return word::bitcast(tr, Vec<decltype(t1)>{r});
}

template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 8), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V upper(T t, Vec<T> v) {
  return word::local_shuf<0, 0, 0, 1>(v);
}

template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 8), TL_IF(is_none<TypeOf<T>, float32_t> && sizeof(TypeOf<T>) <= 4)>
CT_ALWAYS_FORCEINLINE V upper(T t, Vec<T> v) {
  Tag<uint8_t, 16> t1;
  Tag<TypeOf<T>, T::N / 2> tr;
  auto u = word::bitcast(t1, v);
  auto r = _mm_srli_epi64(u.v, 32);
  return word::bitcast(tr, Vec<decltype(t1)>{r});
}

template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V upper(T t, Vec<T> v) {
  return word::local_shuf<0, 0, 3, 2>(v);
}

template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V upper(T t, Vec<T> v) {
  return word::local_shuf<0, 1>(v);
}

template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 16), TL_IF(is_none<TypeOf<T>, float32_t, float64_t>)>
CT_ALWAYS_FORCEINLINE V upper(T t, Vec<T> v) {
  Tag<uint32_t, 4> t1;
  Tag<TypeOf<T>, T::N / 2> tr;
  auto u = word::bitcast(t1, v);
  auto r = word::local_shuf<0, 0, 3, 2>(u);
  return word::bitcast(tr, r);
}

#if VEC_WIDTH >= 256
template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 32), TL_IF(is_any<TypeOf<T>, float32_t, float64_t>)>
CT_ALWAYS_FORCEINLINE V upper(T t, Vec<T> v) {
  Tag<uint8_t, 32> t1;
  Tag<uint8_t, 16> t2;
  Tag<TypeOf<T>, T::N / 2> tr;
  auto u = word::bitcast(t1, v);
  auto r = _mm256_extractf128_si256(u.v, 1);
  return word::bitcast(tr, Vec<decltype(t2)>{r});
}
template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 32), TL_IF(is_none<TypeOf<T>, float32_t, float64_t>)>
CT_ALWAYS_FORCEINLINE V upper(T t, Vec<T> v) {
  Tag<uint8_t, 32> t1;
  Tag<uint8_t, 16> t2;
  Tag<TypeOf<T>, T::N / 2> tr;
  auto u = word::bitcast(t1, v);
  auto r = _mm256_extracti128_si256(u.v, 1);
  return word::bitcast(tr, Vec<decltype(t2)>{r});
}
#else
template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 32)>
CT_ALWAYS_FORCEINLINE V upper(T t, Vec<T> v) {
  return v[1];
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE V upper(T t, Vec<T> v) {
  return _mm512_extractf32x8_ps(v.v, 1);
}
template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 64), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE V upper(T t, Vec<T> v) {
  return _mm512_extractf64x4_ps(v.v, 1);
}
template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 64)>
CT_ALWAYS_FORCEINLINE V upper(T t, Vec<T> v) {
  return _mm512_extracti32x8_epi32(v.v, 1);
}
template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 128)>
CT_ALWAYS_FORCEINLINE V upper(T t, Vec<T> v) {
  return v[1];
}
#elif VEC_WIDTH >= 256
template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 64)>
CT_ALWAYS_FORCEINLINE V upper(T t, Vec<T> v) {
  return v[1];
}
#else
template <typename T, typename V = Vec<Tag<TypeOf<T>, T::N / 2>>, TL_IF(T::Bytes == 64)>
CT_ALWAYS_FORCEINLINE V upper(T t, Vec<T> v) {
  return V{ v[2], v[3] };
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                           Generic Conversions                              //
/* ************************************************************************** */
template <typename T>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<T> v) { return v; }

template <typename T>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<T> v) { return v; }

template <typename T>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<T> v) { return v; }


/* ************************************************************************** */
//                          int64_t <=> float64_t                             //
/* ************************************************************************** */
template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<float64_t, T>> v) {
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

template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<int64_t, T>> v) {
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
template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<float64_t, T>> v) {
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

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<int64_t, T>> v) {
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
template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<float64_t, T>> v) {
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

template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<int64_t, T>> v) {
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
template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<float64_t, T>> v) {
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

template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<uint64_t, T>> v) {
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
template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<float64_t, T>> v) {
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

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<uint64_t, T>> v) {
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
template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<float64_t, T>> v) {
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

template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<uint64_t, T>> v) {
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
template <typename T, TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<uint64_t, T>> v) {
  return v.v;
}

template <typename T, TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<int64_t, T>> v) {
  return v.v;
}


/* ************************************************************************** */
//                         float32_t <=> float64_t                            //
/* ************************************************************************** */
template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  return _mm_cvtpd_ps(v.v);
}

template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
  return _mm_cvtps_pd(v.v);
}

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
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

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
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
template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
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

template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
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
template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  Tag<float32_t, 8> t1;
  Rebind<float64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}

template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
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
template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
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

template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
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

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
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

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
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
template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
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

template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
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
template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  Tag<float32_t, 8> t1;
  Rebind<int64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}

template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
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
template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
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

template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
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

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
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

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
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
template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
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

template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
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
template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
  Tag<float32_t, 8> t1;
  Rebind<uint64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}

template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<float32_t, T>> v) {
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
template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  return _mm_cvttpd_epi32(v.v);
}

template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
  return _mm_cvtepi32_pd(v.v);
}

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
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

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
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
template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
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

template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
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
template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  Tag<int32_t, 8> t1;
  Rebind<float64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}

template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
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
template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
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

template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
  return _mm_cvtepi32_epi64(v.v);
}

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
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

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
  return _mm256_cvtepi32_epi64(v.v);
}

#if VEC_WIDTH >= 256
template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
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

template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
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
template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  Tag<int32_t, 8> t1;
  Rebind<int64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}

template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
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
template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
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

template <typename T, TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int32_t, T>> v) {
  Rebind<int64_t, T> t1;
  return word::bitcast(t, word::promote(t1, v));
}

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
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
template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
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
template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
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
template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
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

template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
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

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
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

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
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
template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
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

template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
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
template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  Tag<uint32_t, 8> t1;
  Rebind<float64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}

template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
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
template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
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

template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
  return _mm_cvtepu32_epi64(v.v);
}

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
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

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
  return _mm256_cvtepu32_epi64(v.v);
}

#if VEC_WIDTH >= 256
template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
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

template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
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
template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  Tag<uint32_t, 8> t1;
  Rebind<int64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}

template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
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
template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
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

template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
  return _mm_cvtepu32_epi64(v.v);
}

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
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

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
  return _mm256_cvtepu32_epi64(v.v);
}

#if VEC_WIDTH >= 256
template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
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

template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
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
template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
  Tag<uint32_t, 8> t1;
  Rebind<uint64_t, T> t2;
  auto lo = word::demote(t1, word::lower(t2, v));
  auto hi = word::demote(t1, word::upper(t2, v));
  return word::concat(t, lo, hi);
}

template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint32_t, T>> v) {
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
template <typename T, TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<int32_t, T>> v) {
  return _mm_cvtepi32_ps(v.v);
}

template <typename T, TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<float32_t, T>> v) {
  return _mm_cvttps_epi32(v.v);
}

#if VEC_WIDTH >= 256
template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<int32_t, T>> v) {
  return _mm256_cvtepi32_ps(v.v);
}

template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<float32_t, T>> v) {
  return _mm256_cvttps_epi32(v.v);
}
#endif // VEC_WIDTH >= 256

#if VEC_WIDTH >= 512
template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<int32_t, T>> v) {
  return _mm512_cvtepi32_ps(v.v);
}

template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<float32_t, T>> v) {
  return _mm512_cvttps_epi32(v.v);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                          float32_t <=> uint32_t                             //
/* ************************************************************************** */
template <typename T, TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<uint32_t, T>> v) {
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

template <typename T, TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<float32_t, T>> v) {
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
template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<uint32_t, T>> v) {
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

template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<float32_t, T>> v) {
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
template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<uint32_t, T>> v) {
  return _mm512_cvtepu32_ps(v.v);
}

template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<float32_t, T>> v) {
  return _mm512_cvttps_epu32(v.v);
}
#endif // VEC_WIDTH >= 512


/* ************************************************************************** */
//                          int32_t <=> uint32_t                             //
/* ************************************************************************** */

template <typename T, TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<uint32_t, T>> v) {
  return v.v;
}

template <typename T, TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<int32_t, T>> v) {
  return v.v;
}


/* ************************************************************************** */
//                           int16_t <=> int32_t                              //
/* ************************************************************************** */
template <typename T, TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int32_t, T>> v) {
  return _mm_packs_epi32(v.v, v.v);
}

template <typename T, TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  return _mm_cvtepi16_epi32(v.v);
}

template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int32_t, T>> v) {
  Rebind<int32_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  return _mm_packs_epi32(lo.v, hi.v);
}

template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
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
template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int32_t, T>> v) {
  Rebind<int32_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  auto u = _mm256_packs_epi32(lo.v, hi.v);
  return _mm256_permute4x64_epi64(u, _MM_SHUFFLE(3, 1, 2, 0));
}

template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
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
template <typename T, TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int32_t, T>> v) {
  Rebind<int32_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  auto u = _mm512_packs_epi32(lo.v, hi.v);
  static const __m512i idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_epi64(u, idx);
}

template <typename T, TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
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
template <typename T, TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint32_t, T>> v) {
  static constexpr uint32_t max_val = INT32_MAX;
  auto u = _mm_min_epu32(v.v, _mm_set1_epi32(max_val));
  return _mm_packs_epi32(u, u);
}

template <typename T, TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::bitcast(t, word::promote(t1, v));
}

template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint32_t, T>> v) {
  Rebind<uint32_t, T> t1;
  static constexpr uint32_t max_val = INT32_MAX;
  Vec<decltype(t1)> u = _mm256_min_epu32(v.v, _mm256_set1_epi32(max_val));
  auto lo = word::lower(t1, u);
  auto hi = word::upper(t1, u);
  return _mm_packs_epi32(lo.v, hi.v);
}

#if VEC_WIDTH >= 256
template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint32_t, T>> v) {
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
template <typename T, TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint32_t, T>> v) {
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
template <typename T, TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<float32_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

template <typename T, TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                           uint16_t <=> int32_t                             //
/* ************************************************************************** */
template <typename T, TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int32_t, T>> v) {
  return _mm_packus_epi32(v.v, v.v);
}

template <typename T, TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  return _mm_cvtepu16_epi32(v.v);
}

template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int32_t, T>> v) {
  Rebind<int32_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  return _mm_packus_epi32(lo.v, hi.v);
}

template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
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
template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int32_t, T>> v) {
  Rebind<int32_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  auto u = _mm256_packus_epi32(lo.v, hi.v);
  return _mm256_permute4x64_epi64(u, _MM_SHUFFLE(3, 1, 2, 0));
}

template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
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
template <typename T, TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int32_t, T>> v) {
  Rebind<int32_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  auto u = _mm512_packus_epi32(lo.v, hi.v);
  static const __m512i idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_epi64(u, idx);
}

template <typename T, TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
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
template <typename T, TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint32_t, T>> v) {
  static constexpr uint32_t max_val = INT32_MAX;
  auto u = _mm_min_epu32(v.v, _mm_set1_epi32(max_val));
  return _mm_packus_epi32(u, u);
}

template <typename T, TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::bitcast(t, word::promote(t1, v));
}

template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint32_t, T>> v) {
  Rebind<uint32_t, T> t1;
  static constexpr uint32_t max_val = INT32_MAX;
  Vec<decltype(t1)> u = _mm256_min_epu32(v.v, _mm256_set1_epi32(max_val));
  auto lo = word::lower(t1, u);
  auto hi = word::upper(t1, u);
  return _mm_packus_epi32(lo.v, hi.v);
}

#if VEC_WIDTH >= 256
template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint32_t, T>> v) {
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
template <typename T, TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint32_t, T>> v) {
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
template <typename T, TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<float32_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

template <typename T, TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                           int16_t <=> float64_t                            //
/* ************************************************************************** */
template <typename T, TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

template <typename T, TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                           int16_t <=> int64_t                              //
/* ************************************************************************** */
template <typename T, TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  return _mm_cvtepi16_epi64(v.v);
}

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
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
template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
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
template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
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
template <typename T, TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <typename T, TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int64_t, T> t1;
  return word::bitcast(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                          uint16_t <=> float64_t                            //
/* ************************************************************************** */
template <typename T, TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return demote(t, convert(t1, v));
}

template <typename T, TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<int32_t, T> t1;
  return convert(t, promote(t1, v));
}


/* ************************************************************************** */
//                           uint16_t <=> int64_t                             //
/* ************************************************************************** */
template <typename T, TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  return _mm_cvtepu16_epi64(v.v);
}

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
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
template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
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
template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
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
template <typename T, TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <typename T, TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<int64_t, T> t1;
  return word::bitcast(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                            int8_t <=> int16_t                              //
/* ************************************************************************** */
template <typename T, TL_IF(T::N <= 8), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int16_t, T>> v) {
  return _mm_packs_epi16(v.v, v.v);
}

template <typename T, TL_IF(T::N <= 8), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  return _mm_cvtepi8_epi16(v.v);
}

template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int16_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  return _mm_packs_epi16(lo.v, hi.v);
}

template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
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
template <typename T, TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int16_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  auto u = _mm256_packs_epi16(lo.v, hi.v);
  return _mm256_permute4x64_epi64(u, _MM_SHUFFLE(3, 1, 2, 0));
}

template <typename T, TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
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
template <typename T, TL_IF(T::N == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int16_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  auto u = _mm512_packs_epi16(lo.v, hi.v);
  static const __m512i idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_epi64(u, idx);
}

template <typename T, TL_IF(T::N == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
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

template <typename T, TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<uint16_t, T>> v) {
  return v.v;
}

template <typename T, TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<int16_t, T>> v) {
  return v.v;
}

/* ************************************************************************** */
//                           int8_t <=> uint16_t                              //
/* ************************************************************************** */
template <typename T, TL_IF(T::N <= 8), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint16_t, T>> v) {
  static constexpr uint16_t max_val = INT16_MAX;
  auto u = _mm_min_epu16(v.v, _mm_set1_epi16(max_val));
  return _mm_packs_epi16(u, u);
}

template <typename T, TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int16_t, T> t1;
  return word::bitcast(t, word::promote(t1, v));
}

template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<uint16_t, T> t1;
  static constexpr uint16_t max_val = INT16_MAX;
  Vec<decltype(t1)> u = _mm256_min_epu16(v.v, _mm256_set1_epi16(max_val));
  auto lo = word::lower(t1, u);
  auto hi = word::upper(t1, u);
  return _mm_packs_epi16(lo.v, hi.v);
}

#if VEC_WIDTH >= 256
template <typename T, TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint16_t, T>> v) {
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
template <typename T, TL_IF(T::N == 64), TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint16_t, T>> v) {
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
template <typename T, TL_IF(T::N <= 8), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int16_t, T>> v) {
  return _mm_packus_epi16(v.v, v.v);
}

template <typename T, TL_IF(T::N <= 8), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  return _mm_cvtepu8_epi16(v.v);
}

template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int16_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  return _mm_packus_epi16(lo.v, hi.v);
}

template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
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
template <typename T, TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int16_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  auto u = _mm256_packus_epi16(lo.v, hi.v);
  return _mm256_permute4x64_epi64(u, _MM_SHUFFLE(3, 1, 2, 0));
}

template <typename T, TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
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
template <typename T, TL_IF(T::N == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int16_t, T>> v) {
  Rebind<int16_t, T> t1;
  auto lo = word::lower(t1, v);
  auto hi = word::upper(t1, v);
  auto u = _mm512_packus_epi16(lo.v, hi.v);
  static const __m512i idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
  return _mm512_permutexvar_epi64(u, idx);
}

template <typename T, TL_IF(T::N == 64), TL_IF(is_any<TypeOf<T>, int16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
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
template <typename T, TL_IF(T::N <= 8), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint16_t, T>> v) {
  static constexpr uint16_t max_val = INT16_MAX;
  auto u = _mm_min_epu16(v.v, _mm_set1_epi16(max_val));
  return _mm_packus_epi16(u, u);
}

template <typename T, TL_IF(is_any<TypeOf<T>, uint16_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<int16_t, T> t1;
  return word::bitcast(t, word::promote(t1, v));
}

template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint16_t, T>> v) {
  Rebind<uint16_t, T> t1;
  static constexpr uint16_t max_val = INT16_MAX;
  Vec<decltype(t1)> u = _mm256_min_epu16(v.v, _mm256_set1_epi16(max_val));
  auto lo = word::lower(t1, u);
  auto hi = word::upper(t1, u);
  return _mm_packus_epi16(lo.v, hi.v);
}

#if VEC_WIDTH >= 256
template <typename T, TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint16_t, T>> v) {
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
template <typename T, TL_IF(T::N == 64), TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint16_t, T>> v) {
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
template <typename T, TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<float32_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

template <typename T, TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                            int8_t <=> int32_t                              //
/* ************************************************************************** */
template <typename T, TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int32_t, T>> v) {
  Rebind<int16_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <typename T, TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  return _mm_cvtepi8_epi32(v.v);
}

template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
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
template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
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
template <typename T, TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
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
template <typename T, TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint32_t, T>> v) {
  Rebind<int16_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <typename T, TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::bitcast(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                           uint8_t <=> float32_t                            //
/* ************************************************************************** */
template <typename T, TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<float32_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

template <typename T, TL_IF(is_any<TypeOf<T>, float32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                           uint8_t <=> int32_t                              //
/* ************************************************************************** */
template <typename T, TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int32_t, T>> v) {
  Rebind<int16_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <typename T, TL_IF(T::N <= 4), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  return _mm_cvtepu8_epi32(v.v);
}

template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
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
template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
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
template <typename T, TL_IF(T::N == 32), TL_IF(is_any<TypeOf<T>, int32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
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
template <typename T, TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint32_t, T>> v) {
  Rebind<uint16_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <typename T, TL_IF(is_any<TypeOf<T>, uint32_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::bitcast(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                           int8_t <=> float64_t                             //
/* ************************************************************************** */
template <typename T, TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

template <typename T, TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                            int8_t <=> int64_t                              //
/* ************************************************************************** */
template <typename T, TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  return _mm_cvtepi8_epi64(v.v);
}

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
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
template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
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
template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
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
template <typename T, TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <typename T, TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<int8_t, T>> v) {
  Rebind<int64_t, T> t1;
  return word::bitcast(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                           uint8_t <=> float64_t                            //
/* ************************************************************************** */
template <typename T, TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<float64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::convert(t1, v));
}

template <typename T, TL_IF(is_any<TypeOf<T>, float64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::convert(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                           uint8_t <=> int64_t                              //
/* ************************************************************************** */
template <typename T, TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<int64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <typename T, TL_IF(T::N <= 2), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  return _mm_cvtepu8_epi64(v.v);
}

template <typename T, TL_IF(T::N == 4), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
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
template <typename T, TL_IF(T::N == 8), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
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
template <typename T, TL_IF(T::N == 16), TL_IF(is_any<TypeOf<T>, int64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
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
template <typename T, TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, Vec<Rebind<uint64_t, T>> v) {
  Rebind<int32_t, T> t1;
  return word::demote(t, word::demote(t1, v));
}

template <typename T, TL_IF(is_any<TypeOf<T>, uint64_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, Vec<Rebind<uint8_t, T>> v) {
  Rebind<int64_t, T> t1;
  return word::bitcast(t, word::promote(t1, v));
}


/* ************************************************************************** */
//                            int8_t <=> uint8_t                              //
/* ************************************************************************** */

template <typename T, TL_IF(is_any<TypeOf<T>, int8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<uint8_t, T>> v) {
  return v.v;
}

template <typename T, TL_IF(is_any<TypeOf<T>, uint8_t>)>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, Vec<Rebind<int8_t, T>> v) {
  return v.v;
}

} // namespace word
} // namespace ct::tl::vec
//@formatter:on

#endif //CTORCH_X86_CONVERSIONS_H
