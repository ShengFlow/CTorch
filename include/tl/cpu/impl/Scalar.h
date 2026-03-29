//
// Created by renyz on 2026/3/17.
//

#ifndef CTORCH_SCALAR_H
#define CTORCH_SCALAR_H

#include <cmath>

#include "CoreDefs.h"
#include "tl/cpu/VecBase.h"
#include "tl/util/ScalarConvert.h"

/**
 * @file Scalar.h
 * @brief Scalar (fallback) implementation of vector operations.
 * 
 * This file provides the default implementation of all vector operations
 * using scalar loops. It is used when:
 * - No SIMD implementation is available for the target architecture
 * - The architecture is unrecognized
 * - The SIMD width is not supported
 * 
 * The scalar implementation serves as:
 * 1. A reference implementation for correctness verification
 * 2. A portable fallback for unsupported platforms
 * 3. A baseline for performance comparison
 * 
 * All functions in this file operate on ScalarArray and ScalarBitSet,
 * which are defined in VecBase.h.
 */
//@formatter:off
namespace ct::tl::vec::CPU_CAPABILITY {
namespace word {
namespace details {
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> vectorized_v(auto&& fn) {
  constexpr T t;
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  
  Vec<T> v;
  for (nint_t i = 0; i < size(t); ++i) {
    v[i] = TypeOf<T>(fn(i));
  }
  return v;
}

template <TLV_DECL_TAG(T)>
TLV_INLINE Mask<T> vectorized_m(auto&& fn) {
  constexpr T t;
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));

  Mask<T> v;
  for (nint_t i = 0; i < size(t); ++i) {
    v[i] = bool(fn(i));
  }
  return v;
}
} // namespace details

/* ************************************************************************** */
//                               Constructors                                 //
/* ************************************************************************** */

template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> value) {
  return details::vectorized_v<T>([&](nint_t i){ return value; });
}

template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> value, nint_t n, Vec<T> default_v) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  Vec<T> v;
  nint_t i;
  for (i = 0; i < n; ++i) {
    v[i] = value;
  }
  for (; i < size(t); ++i) {
    v[i] = default_v[i];
  }
  return v;
}

template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> value, Mask<T> m, Vec<T> default_v) {
  return details::vectorized_v<T>([&](nint_t i){ return m[i] ? value : default_v[i]; });
}

template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> zeros(T t) {
  return word::fill(t, TypeOf<T>());
}

/**
 * @brief Fill a mask with a single boolean value (scalar implementation).
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Mask<T> mfill(T t, bool value) {
  static_assert(is_word_vec(t));
  Mask<T> m; // default inits to zero
  if (value) m.set();
  return m;
}

/**
 * @brief Create a mask where lanes i are true if (a + i) < b (scalar implementation).
 * 
 * This implementation uses either:
 * - Bit manipulation (fast path) for masks up to 64 bits
 * - Loop-based setting for larger masks
 * 
 * @example
 *   Tag<float32_t, 8> t;
 *   auto m = mwhilelt(t, 0, 5);  // m = [T, T, T, T, T, F, F, F]
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Mask<T> mwhilelt(T t, nint_t a, nint_t b) {
  static_assert(is_word_vec(t));
  // Fast path for masks that fit in a machine word
  if constexpr (size(t) <= sizeof(unsigned long long) * CHAR_BIT) {
    nint_t end = std::clamp(b - a, nint_t(0), nint_t(sizeof(unsigned long long) * CHAR_BIT));
    // Create a bitmask with 'end' bits set
    auto bits = tailing_mask(int64_t(end));
    return { bits };
  } else {
    // Slow path for large masks: set bits individually
    // TODO should can be made optimal by custom bitset type that exposes underlying nuint_t storage
    nint_t end = std::min(b - a, size(t));
    Mask<T> m;
    for (nint_t i = 0; i < end; ++i) {
      m.set(i);
    }
    return m;
  }
}

/**
 * @brief Create a mask where lanes i are true if (a + i) >= b (scalar implementation).
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Mask<T> mwhilege(T t, nint_t a, nint_t b) {
  static_assert(is_word_vec(t));
  // Fast path for masks that fit in a machine word
  if constexpr (size(t) <= sizeof(unsigned long long) * CHAR_BIT) {
    nint_t end = std::clamp(b - a, nint_t(0), nint_t(sizeof(unsigned long long) * CHAR_BIT));
    // Create a bitmask with 'end' bits clear (upper bits set)
    auto bits = tailing_mask(int64_t(end));
    return { ~bits };
  } else {
    // Slow path for large masks
    Mask<T> m;
    nint_t start = std::max(b - a, nint_t(0));
    for (nint_t i = start; i < size(t); ++i) {
      m.set(i);
    }
    return m;
  }
}


/* ************************************************************************** */
//                             Shuffle & Permute                              //
/* ************************************************************************** */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
CT_ALWAYS_FORCEINLINE V local_shuf(V v, Vec<Rebind<Index<TypeOf<T>>, T>> vi) {
  constexpr T t;
  constexpr Rebind<TypeOf<T>, T> ti;
  constexpr nint_t group_el = 16 / sizeof(TypeOf<T>);
  static_assert(is_default_impl(t) && is_default_impl(ti));
  static_assert(is_word_vec(t) && is_word_vec(ti));
  static_assert(size(t) >= group_el && size(t) % group_el == 0);
  V u;
  for (nint_t i = 0; i < size(t); i += group_el) {
    CT_UNROLL for (nint_t j = 0; j < group_el; ++j) {
      u[i + j] = v[nint_t(vi[i + j]) + i];
    }
  }
  return u;
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, typename... Is>
CT_ALWAYS_FORCEINLINE V local_shuf(V v, Is... is) {
  constexpr T t;
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  using Ei = Index<TypeOf<T>>;
  using Ti = Rebind<Ei, T>;
  constexpr nint_t group_el = 16 / sizeof(Ei);
  std::array<Ei, group_el> gi = {Ei(is)...}; // accepting inverse order
  for (nint_t j = 0; j < group_el  / 2; ++j)
    std::swap(gi[j], gi[group_el - j - 1]);

  Vec<Ti> i; // copy and propogate
  for (nint_t j = 0; j < size(t); j += group_el) {
    for (nint_t k = 0; k < group_el; ++k) {
      i[j + k] = gi[k];
    }
  }
  return word::local_shuf(v, i);
}

template <int... Is, TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
CT_ALWAYS_FORCEINLINE V local_shuf(V v) {
  return word::local_shuf(v, Is...);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
CT_ALWAYS_FORCEINLINE V shuf(V v, Vec<Rebind<Index<TypeOf<T>>, T>> vi) {
  constexpr T t;
  constexpr Rebind<TypeOf<T>, T> ti;
  static_assert(is_default_impl(t) && is_default_impl(ti));
  static_assert(is_word_vec(t) && is_word_vec(ti));
  V u;
  for (nint_t i = 0; i < size(t); ++i) {
    u[i] = v[nint_t(vi[i])];
  }
  return u;
}

/* ************************************************************************** */
//                          Consecutive load & store                          //
/* ************************************************************************** */

/**
 * @brief Load a vector from unaligned memory (scalar implementation).
 * 
 * Copies elements one by one from memory to the vector.
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T>* p) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  Vec<T> v;
  for (nint_t i = 0; i < size(t); ++i) {
    v[i] = p[i];
  }
  return v;
}

/**
 * @brief Load a vector from aligned memory (scalar implementation).
 * 
 * Asserts that the pointer is properly aligned, then calls loadu.
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> load(T t, const TypeOf<T>* p) {
  CT_ASSERT(((nuint_t)(p) & (DEFAULT_ALIGNMENT - 1)) == 0, "Not aligned");
  return word::loadu(t, p);
}

/**
 * @brief Load first n elements, with default for rest (scalar implementation).
 */
template <TLV_DECL_TAG(T)>
auto loadu(T t, const TypeOf<T>* p, nint_t n, Vec<T> default_v) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  Vec<T> v;
  nint_t i;
  // Load n elements from memory
  for (i = 0; i < n; ++i) {
    v[i] = p[i];
  }
  // Fill remaining elements from default
  for (; i < size(t); ++i) {
    v[i] = default_v[i];
  }
  return v;
}

/**
 * @brief Aligned load of first n elements (scalar implementation).
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> load(T t, const TypeOf<T>* p, nint_t n, Vec<T> default_v) {
  CT_ASSERT(((nuint_t)(p) & (DEFAULT_ALIGNMENT - 1)) == 0, "Not aligned");
  return word::loadu(t, p, n, default_v);
}

/**
 * @brief Masked load (scalar implementation).
 * 
 * For each lane i where mask[i] is true, loads from p[i].
 * For masked-out lanes, takes the value from default_v[i].
 */
template <TLV_DECL_TAG(T)>
auto loadu(T t, const TypeOf<T>* p, Mask<T> m, Vec<T> default_v) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  Vec<T> v;
  for (nint_t i = 0; i < size(t); ++i) {
    v[i] = m[i] ? p[i] : default_v[i];
  }
  return v;
}

/**
 * @brief Aligned masked load (scalar implementation).
 */
template <TLV_DECL_TAG(T)>
auto load(T t, const TypeOf<T>* p, Mask<T> m, Vec<T> default_v) {
  CT_ASSERT(((nuint_t)(p) & (DEFAULT_ALIGNMENT - 1)) == 0, "Not aligned");
  return word::loadu(t, p, m, default_v);
}

/**
 * @brief Store a vector to unaligned memory (scalar implementation).
 */
template <TLV_DECL_TAG(T)>
void storeu(T t, TypeOf<T>* p, Vec<T> v) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  for (nint_t i = 0; i < size(t); ++i) {
    p[i] = v[i];
  }
}

/**
 * @brief Store a vector to aligned memory (scalar implementation).
 */
template <TLV_DECL_TAG(T)>
void store(T t, TypeOf<T>* p, Vec<T> v) {
  CT_ASSERT(((nuint_t)(p) & (DEFAULT_ALIGNMENT - 1)) == 0, "Not aligned");
  return word::storeu(t, p, v);
}

/**
 * @brief Store first n elements (scalar implementation).
 */
template <TLV_DECL_TAG(T)>
void storeu(T t, TypeOf<T>* p, nint_t n, Vec<T> v) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  for (nint_t i = 0; i < n; ++i) {
    p[i] = v[i];
  }
}

/**
 * @brief Aligned store of first n elements (scalar implementation).
 */
template <TLV_DECL_TAG(T)>
void store(T t, TypeOf<T>* p, nint_t n, Vec<T> v) {
  CT_ASSERT(((nuint_t)(p) & (DEFAULT_ALIGNMENT - 1)) == 0, "Not aligned");
  return word::storeu(t, p, n, v);
}

/**
 * @brief Masked store (scalar implementation).
 */
template <TLV_DECL_TAG(T)>
void storeu(T t, TypeOf<T>* p, Mask<T> m, Vec<T> v) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  for (nint_t i = 0; i < size(t); ++i) {
    if (m[i]) p[i] = v[i];
  }
}

/**
 * @brief Aligned masked store (scalar implementation).
 */
template <TLV_DECL_TAG(T)>
void store(T t, TypeOf<T>* p, Mask<T> m, Vec<T> v) {
  CT_ASSERT(((nuint_t)(p) & (DEFAULT_ALIGNMENT - 1)) == 0, "Not aligned");
  return word::storeu(t, p, m, v);
}

/* ************************************************************************** */
//                         Indexed gather & scatter                           //
/* ************************************************************************** */

/**
 * @brief Gather elements using an index vector (scalar implementation).
 * 
 * For each lane j, loads from p[i[j]].
 *
 * @return Gathered vector
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  static_assert(is_default_impl(t));
  static_assert(is_default_impl(Rebind<Index<TypeOf<T>>, T>()));
  static_assert(is_word_vec(t));
  static_assert(is_word_vec(Rebind<Index<TypeOf<T>>, T>()));
  Vec<T> v;
  for (nint_t j = 0; j < size(t); ++j) {
    v[j] = p[(nint_t) i[j]];
  }
  return v;
}

/**
 * @brief Gather first n elements (scalar implementation).
 *
 * @return Gathered vector
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, nint_t n, Vec<T> default_v) {
  static_assert(is_default_impl(t));
  static_assert(is_default_impl(Rebind<Index<TypeOf<T>>, T>()));
  static_assert(is_word_vec(t));
  static_assert(is_word_vec(Rebind<Index<TypeOf<T>>, T>()));
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  Vec<T> v;
  nint_t j;
  // Gather n elements
  for (j = 0; j < n; ++j) {
    v[j] = p[(nint_t) i[j]];
  }
  // Fill remaining from default
  for(; j < size(t); ++j) {
    v[j] = default_v[j];
  }
  return v;
}

/**
 * @brief Masked gather (scalar implementation).
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  static_assert(is_default_impl(t));
  static_assert(is_default_impl(Rebind<Index<TypeOf<T>>, T>()));
  static_assert(is_word_vec(t));
  static_assert(is_word_vec(Rebind<Index<TypeOf<T>>, T>()));
  Vec<T> v;
  for (nint_t j = 0; j < size(t); ++j) {
    v[j] = m[j] ? p[(nint_t) i[j]] : default_v[j];
  }
  return v;
}

/**
 * @brief Scatter elements using an index vector (scalar implementation).
 * 
 * For each lane j, stores v[j] to p[i[j]].
 */
template <TLV_DECL_TAG(T)>
void scatter(T t, TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v) {
  static_assert(is_default_impl(t));
  static_assert(is_default_impl(Rebind<Index<TypeOf<T>>, T>()));
  static_assert(is_word_vec(t));
  static_assert(is_word_vec(Rebind<Index<TypeOf<T>>, T>()));
  for (nint_t j = 0; j < size(t); ++j) {
    p[i[j]] = v[j];
  }
}

/**
 * @brief Scatter first n elements (scalar implementation).
 */
template <TLV_DECL_TAG(T)>
void scatter(T t, TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v, nint_t n) {
  static_assert(is_default_impl(t));
  static_assert(is_default_impl(Rebind<Index<TypeOf<T>>, T>()));
  static_assert(is_word_vec(t));
  static_assert(is_word_vec(Rebind<Index<TypeOf<T>>, T>()));
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  for (nint_t j = 0; j < n; ++j) {
    p[i[j]] = v[j];
  }
}

/**
 * @brief Masked scatter (scalar implementation).
 */
template <TLV_DECL_TAG(T)>
void scatter(T t, TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v, Mask<T> m) {
  static_assert(is_default_impl(t));
  static_assert(is_default_impl(Rebind<Index<TypeOf<T>>, T>()));
  static_assert(is_word_vec(t));
  static_assert(is_word_vec(Rebind<Index<TypeOf<T>>, T>()));
  for (nint_t j = 0; j < size(t); ++j) {
    if (m[j]) p[i[j]] = v[j];
  }
}

/* ************************************************************************** */
//                             Get / set element                              //
/* ************************************************************************** */

/**
 * @brief Get a single element from a vector (scalar implementation).
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TypeOf<T> get(V v, nint_t index) {
  constexpr T t;
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  return v[index];
}

/**
 * @brief Get a single element from a mask (scalar implementation).
 */
template <TLV_DECL_MASK(M)>
bool get(M v, nint_t index) {
//  static_assert(is_word_vec(t));
//  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  return v[index];
}

/**
 * @brief Set a single element in a vector (scalar implementation).
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V set(V v, nint_t index, TypeOf<T> x) {
  constexpr T t;
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  Vec<T> u = v;
  u[index] = x;
  return u;
}

/**
 * @brief Set a single element in a mask (scalar implementation).
 */
template <TLV_DECL_MASK(M)>
TLV_INLINE M set(M v, nint_t index, bool x) {
//  static_assert(is_default_impl(t));
//  static_assert(is_word_vec(t));
//  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  M u = v;
  u[index] = x;
  return u;
}

/* ************************************************************************** */
//                       Basic arithmetic operations                          //
/* ************************************************************************** */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V add(V a, V b) {
  return details::vectorized_v<T>([&](nint_t i){ return a[i] + b[i]; });
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V add(V a, V b, Mask<T> m) {
  return details::vectorized_v<T>([&](nint_t i){ return m[i] ? a[i] + b[i] : a[i]; });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V sub(V a, V b) {
  return details::vectorized_v<T>([&](nint_t i){ return a[i] - b[i]; });
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V sub(V a, V b, Mask<T> m) {
  return details::vectorized_v<T>([&](nint_t i){ return m[i] ? a[i] - b[i] : a[i]; });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V mul(V a, V b) {
  return details::vectorized_v<T>([&](nint_t i){ return a[i] * b[i]; });
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V mul(V a, V b, Mask<T> m) {
  return details::vectorized_v<T>([&](nint_t i){ return m[i] ? a[i] * b[i] : a[i]; });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
TLV_INLINE V div(V a, V b) {
  return details::vectorized_v<T>([&](nint_t i){ return a[i] / b[i]; });
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
TLV_INLINE V div(V a, V b, Mask<T> m) {
  return details::vectorized_v<T>([&](nint_t i){ return m[i] ? a[i] / b[i] : a[i]; });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V max(V a, V b) {
  return details::vectorized_v<T>([&](nint_t i){ return (TypeOf<T>)std::max(a[i], b[i]); });
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V max(V a, V b, Mask<T> m) {
  return details::vectorized_v<T>([&](nint_t i){ return m[i] ? (TypeOf<T>)std::max(a[i], b[i]) : a[i]; });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V min(V a, V b) {
  return details::vectorized_v<T>([&](nint_t i){ return (TypeOf<T>)std::min(a[i], b[i]); });
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V min(V a, V b, Mask<T> m) {
  return details::vectorized_v<T>([&](nint_t i){ return m[i] ? (TypeOf<T>)std::min(a[i], b[i]) : a[i]; });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
TLV_INLINE V bit_and(V a, V b) {
  return details::vectorized_v<T>([&](nint_t i){ return a[i] & b[i]; });
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
TLV_INLINE V bit_and(V a, V b, Mask<T> m) {
  return details::vectorized_v<T>([&](nint_t i){ return m[i] ? (a[i] & b[i]) : a[i]; });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
TLV_INLINE V bit_or(V a, V b) {
  return details::vectorized_v<T>([&](nint_t i){ return a[i] | b[i]; });
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
TLV_INLINE V bit_or(V a, V b, Mask<T> m) {
  return details::vectorized_v<T>([&](nint_t i){ return m[i] ? (a[i] | b[i]) : a[i]; });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
TLV_INLINE V bit_xor(V a, V b) {
  return details::vectorized_v<T>([&](nint_t i){ return a[i] ^ b[i]; });
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
TLV_INLINE V bit_xor(V a, V b, Mask<T> m) {
  return details::vectorized_v<T>([&](nint_t i){ return m[i] ? (a[i] ^ b[i]) : a[i]; });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
TLV_INLINE V bit_andnot(V a, V b) {
  return details::vectorized_v<T>([&](nint_t i){ return ~a[i] & b[i]; });
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
TLV_INLINE V bit_andnot(V a, V b, Mask<T> m) {
  return details::vectorized_v<T>([&](nint_t i){ return m[i] ? (~a[i] & b[i]) : a[i]; });
}

template <typename T>
static TLV_INLINE T _safe_shl(T v, int count) {
  if (count >= int(sizeof(T) * 8)) return T();
  if (count < 0) return v;
  return v << count;
}
template <typename T>
static TLV_INLINE T _safe_shr(T v, int count) {
  if (count >= int(sizeof(T) * 8)) {
    if constexpr (std::is_signed_v<T>) {
      return (v < 0) ? T(-1) : T();
    } else {
      return T();
    }
  }
  if (count < 0) return v;
  return v >> count;
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
TLV_INLINE V bit_shl(V v, int count) {
  return details::vectorized_v<T>([&](nint_t i){ return _safe_shl<TypeOf<T>>(v[i], count); });
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
TLV_INLINE V bit_shl(V v, int count, Mask<T> m) {
  return details::vectorized_v<T>([&](nint_t i){ return m[i] ? _safe_shl<TypeOf<T>>(v[i], count) : v[i]; });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
TLV_INLINE V bit_shr(V v, int count) {
  return details::vectorized_v<T>([&](nint_t i){ return _safe_shr<TypeOf<T>>(v[i], count); });
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
TLV_INLINE V bit_shr(V v, int count, Mask<T> m) {
  return details::vectorized_v<T>([&](nint_t i){ return m[i] ? _safe_shr<TypeOf<T>>(v[i], count) : v[i]; });
}

template <typename T>
static TLV_INLINE T _safe_abs(T v) {
  if constexpr (std::is_unsigned_v<T>) {
    return v;
  } else {
    return std::abs(v);
  }
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
TLV_INLINE V bit_not(V v) {
  return details::vectorized_v<T>([&](nint_t i){ return ~v[i]; });
  }
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_int<TypeOf<T>>)>
TLV_INLINE V bit_not(V v, Mask<T> m, V default_v) {
  return details::vectorized_v<T>([&](nint_t i){ return m[i] ? (~v[i]) : default_v[i]; });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V neg(V v) {
  return details::vectorized_v<T>([&](nint_t i){ return -v[i]; });
  }
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V neg(V v, Mask<T> m, V default_v) {
  return details::vectorized_v<T>([&](nint_t i){ return m[i] ? (-v[i]) : default_v[i]; });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V abs(V v) {
  return details::vectorized_v<T>([&](nint_t i){ return _safe_abs<TypeOf<T>>(v[i]); });
  }
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V abs(V v, Mask<T> m, V default_v) {
  return details::vectorized_v<T>([&](nint_t i){ return m[i] ? _safe_abs<TypeOf<T>>(v[i]) : default_v[i]; });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
TLV_INLINE V sqrt(V v) {
  return details::vectorized_v<T>([&](nint_t i){ return (TypeOf<T>)std::sqrt(v[i]); });
  }
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
TLV_INLINE V sqrt(V v, Mask<T> m, V default_v) {
  return details::vectorized_v<T>([&](nint_t i){ return m[i] ? (TypeOf<T>)std::sqrt(v[i]) : default_v[i]; });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
TLV_INLINE V rsqrt(V v) {
  return details::vectorized_v<T>([&](nint_t i){ return 1 / (TypeOf<T>)std::sqrt(v[i]); });
  }
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
TLV_INLINE V rsqrt(V v, Mask<T> m, V default_v) {
  return details::vectorized_v<T>([&](nint_t i){ return m[i] ? 1 / (TypeOf<T>)std::sqrt(v[i]) : default_v[i]; });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
TLV_INLINE V rcp(V v) {
  return details::vectorized_v<T>([&](nint_t i){ return 1 / v[i]; });
  }
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
TLV_INLINE V rcp(V v, Mask<T> m, V default_v) {
  return details::vectorized_v<T>([&](nint_t i){ return m[i] ? 1 / v[i] : default_v[i]; });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpeq(V a, V b) {
  return details::vectorized_m<T>([&](nint_t i){ return a[i] == b[i]; });
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  return details::vectorized_m<T>([&](nint_t i){ return m[i] && (a[i] == b[i]); });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpne(V a, V b) {
  return details::vectorized_m<T>([&](nint_t i){ return a[i] != b[i]; });
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  return details::vectorized_m<T>([&](nint_t i){ return m[i] && (a[i] != b[i]); });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmplt(V a, V b) {
  return details::vectorized_m<T>([&](nint_t i){ return a[i] < b[i]; });
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  return details::vectorized_m<T>([&](nint_t i){ return m[i] && (a[i] < b[i]); });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmple(V a, V b) {
  return details::vectorized_m<T>([&](nint_t i){ return a[i] <= b[i]; });
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  return details::vectorized_m<T>([&](nint_t i){ return m[i] && (a[i] <= b[i]); });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpgt(V a, V b) {
  return details::vectorized_m<T>([&](nint_t i){ return a[i] > b[i]; });
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  return details::vectorized_m<T>([&](nint_t i){ return m[i] && (a[i] > b[i]); });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpge(V a, V b) {
  return details::vectorized_m<T>([&](nint_t i){ return a[i] >= b[i]; });
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  return details::vectorized_m<T>([&](nint_t i){ return m[i] && (a[i] >= b[i]); });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
TLV_INLINE Mask<T> isnan(V v) {
  return details::vectorized_m<T>([&](nint_t i){ return std::isnan(v[i]); });
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
TLV_INLINE Mask<T> isnan(V v, Mask<T> m) {
  return details::vectorized_m<T>([&](nint_t i){ return m[i] && std::isnan(v[i]); });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
TLV_INLINE Mask<T> isposinf(V v) {
  return details::vectorized_m<T>([&](nint_t i){ return v[i] > 0 && std::isinf(v[i]); });
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
TLV_INLINE Mask<T> isposinf(V v, Mask<T> m) {
  return details::vectorized_m<T>([&](nint_t i){ return m[i] && v[i] > 0 && std::isinf(v[i]); });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
TLV_INLINE Mask<T> isneginf(V v) {
  return details::vectorized_m<T>([&](nint_t i){ return v[i] < 0 && std::isinf(v[i]); });
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
TLV_INLINE Mask<T> isneginf(V v, Mask<T> m) {
  return details::vectorized_m<T>([&](nint_t i){ return m[i] && v[i] < 0 && std::isinf(v[i]); });
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
TLV_INLINE Mask<T> isinf(V v) {
  return details::vectorized_m<T>([&](nint_t i){ return std::isinf(v[i]); });
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>, TL_IF(is_float<TypeOf<T>>)>
TLV_INLINE Mask<T> isinf(V v, Mask<T> m) {
  return details::vectorized_m<T>([&](nint_t i){ return m[i] && std::isinf(v[i]); });
}


/* ************************************************************************** */
//                          Data type conversions                             //
/* ************************************************************************** */
namespace details {
template <typename T, typename V, TL_IF(is_word_vec(T())), TL_IF(is_default_impl(T())), TL_IF(is_default_impl(Vec2Tag<V>()))>
TLV_INLINE Vec<T> convert_impl(T t, V v) {
  using TOut = TypeOf<T>;
  using TIn = TypeOf<Vec2Tag<V>>;
  Vec2Tag<V> t_in;
  Vec<T> u;
  auto n_out = size(t);
  auto n_in = size(t_in);
  CT_ASSERT(n_out <= n_in, "Insufficient element (expected %zd, got %zd)", n_out, n_in);

  if constexpr (is_word_vec(t_in)) {
    for (nint_t i = 0; i < n_out; ++i) {
      u[i] = tl::convert<TOut, TIn>(v[i]);
    }
  } else {
    TOut* p = u.data();
    for (nint_t i = 0; i < num_words(t_in); ++i, p += word_size(t_in)) {
      for (nint_t j = 0; j < word_size(t_in); ++j) {
        p[j] = tl::convert<TOut, TIn>(v[i][j]);
      }
    }
  }
  return u;
}
} // namespace details

template <typename T, typename V, TL_IF(sizeof(TypeOf<T>) > sizeof(TypeOf<Vec2Tag<V>>))>
TLV_INLINE Vec<T> promote(T t, V v) {
  static_assert(is_default_impl(T()) && is_default_impl(Vec2Tag<V>()));
  static_assert(is_word_vec(T()));
  return details::convert_impl(t, v);
}

template <typename T, typename V, TL_IF(sizeof(TypeOf<T>) < sizeof(TypeOf<Vec2Tag<V>>))>
TLV_INLINE Vec<T> demote(T t, V v) {
  static_assert(is_default_impl(T()) && is_default_impl(Vec2Tag<V>()));
  static_assert(is_word_vec(T()));
  return details::convert_impl(t, v);
}

template <typename T, typename V, TL_IF(sizeof(TypeOf<T>) == sizeof(TypeOf<Vec2Tag<V>>))>
TLV_INLINE Vec<T> convert(T t, V v) {
  static_assert(is_default_impl(T()) && is_default_impl(Vec2Tag<V>()));
  static_assert(is_word_vec(T()));
  return details::convert_impl(t, v);
}

template <typename To, typename Vi, typename Ti = Vec2Tag<Vi>>
TLV_INLINE Vec<To> reshape(To t_out, Vi v_in) {
  static_assert(is_default_impl(Ti()) && is_default_impl(t_out));
  static_assert(std::is_same_v<TypeOf<To>, TypeOf<Ti>>, "Not same type");
  if constexpr (std::is_same_v<Vec<To>, Vi>) {
    return v_in;
  }
  constexpr Ti t_in;
  constexpr nint_t NIn = Ti::N;
  constexpr nint_t NOut = To::N;
  using T = TypeOf<Ti>;
  Vec<To> v_out;
  if constexpr (is_word_vec(t_in)) {
    if constexpr (is_word_vec(t_out)) {
      static_assert(NIn == NOut);
      return v_in;
    } else {
      const T* p = v_in.data();
      for (nint_t i = 0; i < num_words(t_out); ++i, p += word_size(t_out)) {
        std::copy(p, p + word_size(t_out), v_out[i].data());
      }
    }
  } else if constexpr (is_word_vec(t_out)) {
    T* p = v_out.data();
    for (nint_t i = 0; i < num_words(t_in); ++i, p += word_size(t_in)) {
      std::copy(t_in[i].data(), t_in[i].data() + word_size(t_in), p);
    }
  } else {
    for (nint_t i = 0; i < size(t_in); ++i) {
      t_out[i / word_size(t_out)][i % word_size(t_out)] = t_in[i / word_size(t_in)][i % word_size(t_in)];
    }
  }
  return v_out;
}

} // namespace word
} // namespace ct::tl::vec::CPU_CAPABILITY
//@formatter:on

#endif //CTORCH_SCALAR_H
