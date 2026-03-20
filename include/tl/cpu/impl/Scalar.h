//
// Created by renyz on 2026/3/17.
//

#ifndef CTORCH_SCALAR_H
#define CTORCH_SCALAR_H

#include <cmath>

#include "tl/cpu/VecBase.h"

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

namespace ct::tl::vec {
namespace word {
/* ************************************************************************** */
//                               Constructors                                 //
/* ************************************************************************** */

/**
 * @brief Fill a vector with a single value (scalar implementation).
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param value The value to fill
 * @return Vector with all elements set to value
 */
template <typename T, nint_t N, int P>
auto fill(Tag<T, N, P> t, T value) -> VecOf(t) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  VecOf(t) v;
  v.fill(value);
  return v;
}

template <typename T, nint_t N, int P>
auto fill(Tag<T, N, P> t, T value, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  VecOf(t) v;
  nint_t i;
  for (i = 0; i < n; ++i) {
    v[i] = value;
  }
  for (; i < size(t); ++i) {
    v[i] = default_v[i];
  }
  return v;
}

template <typename T, nint_t N, int P>
auto fill(Tag<T, N, P> t, T value, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  VecOf(t) v;
  for (nint_t i = 0; i < size(t); ++i) {
    v[i] = m[i] ? value : default_v[i];
  }
  return v;
}

/**
 * @brief Fill a mask with a single boolean value (scalar implementation).
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param value The boolean value
 * @return Mask with all elements set to value
 */
template <typename T, nint_t N, int P>
auto mfill(Tag<T, N, P> t, bool value) -> MaskOf(t) {
  static_assert(is_word_vec(t));
  MaskOf(t) m; // default inits to zero
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
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param a Starting index
 * @param b Upper bound (exclusive)
 * @return Mask where lanes [a, b-a) are true
 * 
 * @example
 *   Tag<float32_t, 8> t;
 *   auto m = mwhilelt(t, 0, 5);  // m = [T, T, T, T, T, F, F, F]
 */
template <typename T, nint_t N, int P>
auto mwhilelt(Tag<T, N, P> t, nint_t a, nint_t b) -> MaskOf(t) {
  static_assert(is_word_vec(t));
  // Fast path for masks that fit in a machine word
  if constexpr (size(t) <= sizeof(unsigned long long) * CHAR_BIT) {
    nint_t end = std::clamp(b - a, nint_t(0), nint_t(sizeof(unsigned long long) * CHAR_BIT));
    // Create a bitmask with 'end' bits set
    // Note: (1 << 64) - 1 correctly produces all ones when end = 64
    auto bits = ((1uLL) << end) - 1;
    return { bits };
  } else {
    // Slow path for large masks: set bits individually
    // TODO should can be made optimal by custom bitset type that exposes underlying nuint_t storage
    nint_t end = std::min(b - a, size(t));
    MaskOf(t) m;
    for (nint_t i = 0; i < end; ++i) {
      m.set(i);
    }
    return m;
  }
}

/**
 * @brief Create a mask where lanes i are true if (a + i) >= b (scalar implementation).
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param a Starting index
 * @param b Lower bound (inclusive)
 * @return Mask where lanes [b-a, N) are true
 */
template <typename T, nint_t N, int P>
auto mwhilege(Tag<T, N, P> t, nint_t a, nint_t b) -> MaskOf(t) {
  static_assert(is_word_vec(t));
  // Fast path for masks that fit in a machine word
  if constexpr (size(t) <= sizeof(unsigned long long) * CHAR_BIT) {
    nint_t end = std::clamp(b - a, nint_t(0), nint_t(sizeof(unsigned long long) * CHAR_BIT));
    // Create a bitmask with 'end' bits clear (upper bits set)
    auto bits = (1uLL << end) - 1;
    return { ~bits };
  } else {
    // Slow path for large masks
    MaskOf(t) m;
    nint_t start = std::max(b - a, nint_t(0));
    for (nint_t i = start; i < size(t); ++i) {
      m.set(i);
    }
    return m;
  }
}

/* ************************************************************************** */
//                          Consecutive load & store                          //
/* ************************************************************************** */

/**
 * @brief Load a vector from unaligned memory (scalar implementation).
 * 
 * Copies elements one by one from memory to the vector.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Pointer to source memory
 * @return Loaded vector
 */
template <typename T, nint_t N, int P>
auto loadu(Tag<T, N, P> t, const T* p) -> VecOf(t) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  VecOf(t) v;
  for (nint_t i = 0; i < size(t); ++i) {
    v[i] = p[i];
  }
  return v;
}

/**
 * @brief Load a vector from aligned memory (scalar implementation).
 * 
 * Asserts that the pointer is properly aligned, then calls loadu.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Pointer to source memory (must be aligned)
 * @return Loaded vector
 */
template <typename T, nint_t N, int P>
auto load(Tag<T, N, P> t, const T* p) -> VecOf(t) {
  CT_ASSERT(((nuint_t)(p) & (DEFAULT_ALIGNMENT - 1)) == 0, "Not aligned");
  return word::loadu(t, p);
}

/**
 * @brief Load first n elements, with default for rest (scalar implementation).
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Pointer to source memory
 * @param n Number of elements to load
 * @param default_v Default values for elements beyond n
 * @return Vector with loaded and default elements
 */
template <typename T, nint_t N, int P>
auto loadu(Tag<T, N, P> t, const T* p, nint_t n, VecOf(t) default_v) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  VecOf(t) v;
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
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Pointer to source memory (must be aligned)
 * @param n Number of elements to load
 * @param default_v Default values for elements beyond n
 * @return Vector with loaded and default elements
 */
template <typename T, nint_t N, int P>
auto load(Tag<T, N, P> t, const T* p, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  CT_ASSERT(((nuint_t)(p) & (DEFAULT_ALIGNMENT - 1)) == 0, "Not aligned");
  return word::loadu(t, p, n, default_v);
}

/**
 * @brief Masked load (scalar implementation).
 * 
 * For each lane i where mask[i] is true, loads from p[i].
 * For masked-out lanes, takes the value from default_v[i].
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Pointer to source memory
 * @param m Mask indicating which lanes to load
 * @param default_v Default values for masked-out lanes
 * @return Vector with masked-loaded elements
 */
template <typename T, nint_t N, int P>
auto loadu(Tag<T, N, P> t, const T* p, MaskOf(t) m, VecOf(t) default_v) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  VecOf(t) v;
  for (nint_t i = 0; i < size(t); ++i) {
    v[i] = m[i] ? p[i] : default_v[i];
  }
  return v;
}

/**
 * @brief Aligned masked load (scalar implementation).
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Pointer to source memory (must be aligned)
 * @param m Mask indicating which lanes to load
 * @param default_v Default values for masked-out lanes
 * @return Vector with masked-loaded elements
 */
template <typename T, nint_t N, int P>
auto load(Tag<T, N, P> t, const T* p, MaskOf(t) m, VecOf(t) default_v) {
  CT_ASSERT(((nuint_t)(p) & (DEFAULT_ALIGNMENT - 1)) == 0, "Not aligned");
  return word::loadu(t, p, m, default_v);
}

/**
 * @brief Store a vector to unaligned memory (scalar implementation).
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Pointer to destination memory
 * @param v The vector to store
 */
template <typename T, nint_t N, int P>
void storeu(Tag<T, N, P> t, T* p, VecOf(t) v) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  for (nint_t i = 0; i < size(t); ++i) {
    p[i] = v[i];
  }
}

/**
 * @brief Store a vector to aligned memory (scalar implementation).
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Pointer to destination memory (must be aligned)
 * @param v The vector to store
 */
template <typename T, nint_t N, int P>
void store(Tag<T, N, P> t, T* p, VecOf(t) v) {
  CT_ASSERT(((nuint_t)(p) & (DEFAULT_ALIGNMENT - 1)) == 0, "Not aligned");
  return word::storeu(t, p, v);
}

/**
 * @brief Store first n elements (scalar implementation).
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Pointer to destination memory
 * @param n Number of elements to store
 * @param v The vector to store
 */
template <typename T, nint_t N, int P>
void storeu(Tag<T, N, P> t, T* p, nint_t n, VecOf(t) v) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  for (nint_t i = 0; i < n; ++i) {
    p[i] = v[i];
  }
}

/**
 * @brief Aligned store of first n elements (scalar implementation).
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Pointer to destination memory (must be aligned)
 * @param n Number of elements to store
 * @param v The vector to store
 */
template <typename T, nint_t N, int P>
void store(Tag<T, N, P> t, T* p, nint_t n, VecOf(t) v) {
  CT_ASSERT(((nuint_t)(p) & (DEFAULT_ALIGNMENT - 1)) == 0, "Not aligned");
  return word::storeu(t, p, n, v);
}

/**
 * @brief Masked store (scalar implementation).
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Pointer to destination memory
 * @param m Mask indicating which lanes to store
 * @param v The vector to store
 */
template <typename T, nint_t N, int P>
void storeu(Tag<T, N, P> t, T* p, MaskOf(t) m, VecOf(t) v) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  for (nint_t i = 0; i < size(t); ++i) {
    if (m[i]) p[i] = v[i];
  }
}

/**
 * @brief Aligned masked store (scalar implementation).
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Pointer to destination memory (must be aligned)
 * @param m Mask indicating which lanes to store
 * @param v The vector to store
 */
template <typename T, nint_t N, int P>
void store(Tag<T, N, P> t, T* p, MaskOf(t) m, VecOf(t) v) {
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
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Base pointer for gather
 * @param i Index vector
 * @return Gathered vector
 */
template <typename T, nint_t N, int P>
auto gather(Tag<T, N, P> t, const T* p, Vec<Tag<Index<T>, N, P>> i) -> VecOf(t) {
  static_assert(is_default_impl(t));
  static_assert(is_default_impl(Tag<Index<T>, N, P>()));
  static_assert(is_word_vec(t));
  static_assert(is_word_vec(Tag<Index<T>, N, P>()));
  VecOf(t) v;
  for (nint_t j = 0; j < size(t); ++j) {
    v[j] = p[(nint_t) i[j]];
  }
  return v;
}

/**
 * @brief Gather first n elements (scalar implementation).
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Base pointer for gather
 * @param i Index vector
 * @param n Number of elements to gather
 * @param default_v Default values for elements beyond n
 * @return Gathered vector
 */
template <typename T, nint_t N, int P>
auto gather(Tag<T, N, P> t, const T* p, Vec<Tag<Index<T>, N, P>> i, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  static_assert(is_default_impl(t));
  static_assert(is_default_impl(Tag<Index<T>, N, P>()));
  static_assert(is_word_vec(t));
  static_assert(is_word_vec(Tag<Index<T>, N, P>()));
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  VecOf(t) v;
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
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Base pointer for gather
 * @param i Index vector
 * @param m Mask indicating which lanes to gather
 * @param default_v Default values for masked-out lanes
 * @return Gathered vector
 */
template <typename T, nint_t N, int P>
auto gather(Tag<T, N, P> t, const T* p, Vec<Tag<Index<T>, N, P>> i, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) {
  static_assert(is_default_impl(t));
  static_assert(is_default_impl(Tag<Index<T>, N, P>()));
  static_assert(is_word_vec(t));
  static_assert(is_word_vec(Tag<Index<T>, N, P>()));
  VecOf(t) v;
  for (nint_t j = 0; j < size(t); ++j) {
    v[j] = m[j] ? p[(nint_t) i[j]] : default_v[j];
  }
  return v;
}

/**
 * @brief Scatter elements using an index vector (scalar implementation).
 * 
 * For each lane j, stores v[j] to p[i[j]].
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Base pointer for scatter
 * @param i Index vector
 * @param v The vector to scatter
 */
template <typename T, nint_t N, int P>
void scatter(Tag<T, N, P> t, T* p, Vec<Tag<Index<T>, N, P>> i, VecOf(t) v) {
  static_assert(is_default_impl(t));
  static_assert(is_default_impl(Tag<Index<T>, N, P>()));
  static_assert(is_word_vec(t));
  static_assert(is_word_vec(Tag<Index<T>, N, P>()));
  for (nint_t j = 0; j < size(t); ++j) {
    p[i[j]] = v[j];
  }
}

/**
 * @brief Scatter first n elements (scalar implementation).
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Base pointer for scatter
 * @param i Index vector
 * @param v The vector to scatter
 * @param n Number of elements to scatter
 */
template <typename T, nint_t N, int P>
void scatter(Tag<T, N, P> t, T* p, Vec<Tag<Index<T>, N, P>> i, VecOf(t) v, nint_t n) {
  static_assert(is_default_impl(t));
  static_assert(is_default_impl(Tag<Index<T>, N, P>()));
  static_assert(is_word_vec(t));
  static_assert(is_word_vec(Tag<Index<T>, N, P>()));
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  for (nint_t j = 0; j < n; ++j) {
    p[i[j]] = v[j];
  }
}

/**
 * @brief Masked scatter (scalar implementation).
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Base pointer for scatter
 * @param i Index vector
 * @param v The vector to scatter
 * @param m Mask indicating which lanes to scatter
 */
template <typename T, nint_t N, int P>
void scatter(Tag<T, N, P> t, T* p, Vec<Tag<Index<T>, N, P>> i, VecOf(t) v, MaskOf(t) m) {
  static_assert(is_default_impl(t));
  static_assert(is_default_impl(Tag<Index<T>, N, P>()));
  static_assert(is_word_vec(t));
  static_assert(is_word_vec(Tag<Index<T>, N, P>()));
  for (nint_t j = 0; j < size(t); ++j) {
    if (m[j]) p[i[j]] = v[j];
  }
}

/* ************************************************************************** */
//                             Get / set element                              //
/* ************************************************************************** */

/**
 * @brief Get a single element from a vector (scalar implementation).
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param v The vector
 * @param index Element index
 * @return The element at the specified index
 */
template <typename T, nint_t N, int P>
T get(Tag<T, N, P> t, VecOf(t) v, nint_t index) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  return v[index];
}

/**
 * @brief Get a single element from a mask (scalar implementation).
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param v The mask
 * @param index Element index
 * @return The boolean value at the specified index
 */
template <typename T, nint_t N, int P>
bool get(Tag<T, N, P> t, MaskOf(t) v, nint_t index) {
  static_assert(is_word_vec(t));
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  return v[index];
}

/**
 * @brief Set a single element in a vector (scalar implementation).
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param v The original vector
 * @param index Element index
 * @param x The new value
 * @return Vector with the element at index set to x
 */
template <typename T, nint_t N, int P>
auto set(Tag<T, N, P> t, VecOf(t) v, nint_t index, T x) -> VecOf(t) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  VecOf(t) u = v;
  u[index] = x;
  return u;
}

/**
 * @brief Set a single element in a mask (scalar implementation).
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param v The original mask
 * @param index Element index
 * @param x The new boolean value
 * @return Mask with the element at index set to x
 */
template <typename T, nint_t N, int P>
auto set(Tag<T, N, P> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  MaskOf(t) u = v;
  u[index] = x;
  return u;
}

/* ************************************************************************** */
//                       Basic arithmetic operations                          //
/* ************************************************************************** */
#define _CT_SCALAR_VECTORIZED_BINARY_V(name, ...) \
template <typename T, nint_t N, int P> \
auto name(Tag<T, N, P> t, VecOf(t) a, VecOf(t) b) -> VecOf(t) { \
  static_assert(is_default_impl(t)); \
  static_assert(is_word_vec(t)); \
  VecOf(t) r; \
  for (nint_t i = 0; i < size(t); ++i) { \
    r[i] = (__VA_ARGS__); \
  } \
  return r; \
} \
template <typename T, nint_t N, int P> \
auto name(Tag<T, N, P> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> VecOf(t) { \
  static_assert(is_default_impl(t)); \
  static_assert(is_word_vec(t)); \
  VecOf(t) r; \
  for (nint_t i = 0; i < size(t); ++i) { \
    r[i] = m[i] ? (__VA_ARGS__) : a[i]; \
  } \
  return r; \
}

_CT_SCALAR_VECTORIZED_BINARY_V(add, a[i] + b[i])
_CT_SCALAR_VECTORIZED_BINARY_V(sub, a[i] - b[i])
_CT_SCALAR_VECTORIZED_BINARY_V(mul, a[i] * b[i])
_CT_SCALAR_VECTORIZED_BINARY_V(div, a[i] / b[i])
_CT_SCALAR_VECTORIZED_BINARY_V(rem, a[i] % b[i])
_CT_SCALAR_VECTORIZED_BINARY_V(max, (T)std::max(a[i], b[i]))
_CT_SCALAR_VECTORIZED_BINARY_V(min, (T)std::min(a[i], b[i]))
_CT_SCALAR_VECTORIZED_BINARY_V(bit_and, a[i] & b[i])
_CT_SCALAR_VECTORIZED_BINARY_V(bit_or, a[i] | b[i])
_CT_SCALAR_VECTORIZED_BINARY_V(bit_xor, a[i] ^ b[i])
_CT_SCALAR_VECTORIZED_BINARY_V(bit_andnot, ~a[i] & b[i])
#undef _CT_SCALAR_VECTORIZED_BINARY_V

template <typename T>
static CT_ALWAYS_FORCEINLINE T _safe_shl(T v, int count) {
  if (count >= int(sizeof(T) * 8)) return T();
  if (count < 0) return v;
  return v << count;
}

template <typename T>
static CT_ALWAYS_FORCEINLINE T _safe_shr(T v, int count) {
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

template <typename T, nint_t N, int P>
auto bit_shl(Tag<T, N, P> t, VecOf(t) v, int count) -> VecOf(t) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  VecOf(t) r;
  for (nint_t i = 0; i < size(t); ++i) {
    r[i] = _safe_shl(v[i], count);
  }
  return r;
}

template <typename T, nint_t N, int P>
auto bit_shl(Tag<T, N, P> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  VecOf(t) r;
  for (nint_t i = 0; i < size(t); ++i) {
    r[i] = m[i] ? _safe_shl(v[i], count) : v[i];
  }
  return r;
}

template <typename T, nint_t N, int P>
auto bit_shr(Tag<T, N, P> t, VecOf(t) v, int count) -> VecOf(t) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  VecOf(t) r;
  for (nint_t i = 0; i < size(t); ++i) {
    r[i] = _safe_shr(v[i], count);
  }
  return r;
}

template <typename T, nint_t N, int P>
auto bit_shr(Tag<T, N, P> t, VecOf(t) v, int count, MaskOf(t) m) -> VecOf(t) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  VecOf(t) r;
  for (nint_t i = 0; i < size(t); ++i) {
    r[i] = m[i] ? _safe_shr(v[i], count) : v[i];
  }
  return r;
}

#define _CT_SCALAR_VECTORIZED_UNARY_V(name, ...) \
template <typename T, nint_t N, int P> \
auto name(Tag<T, N, P> t, VecOf(t) v) -> VecOf(t) { \
  static_assert(is_default_impl(t)); \
  static_assert(is_word_vec(t)); \
  VecOf(t) r; \
  for (nint_t i = 0; i < size(t); ++i) { \
    r[i] = (__VA_ARGS__); \
  } \
  return r; \
} \
template <typename T, nint_t N, int P> \
auto name(Tag<T, N, P> t, VecOf(t) v, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) { \
  static_assert(is_default_impl(t)); \
  static_assert(is_word_vec(t)); \
  VecOf(t) r; \
  for (nint_t i = 0; i < size(t); ++i) { \
    r[i] = m[i] ? (__VA_ARGS__) : default_v[i]; \
  } \
  return r; \
}

template <typename T>
static CT_ALWAYS_FORCEINLINE T _safe_abs(T v) {
  if constexpr (std::is_unsigned_v<T>) {
    return v;
  } else {
    return std::abs(v);
  }
}

_CT_SCALAR_VECTORIZED_UNARY_V(bit_not, ~v[i])
_CT_SCALAR_VECTORIZED_UNARY_V(neg, -v[i])
_CT_SCALAR_VECTORIZED_UNARY_V(abs, _safe_abs<T>(v[i]))
_CT_SCALAR_VECTORIZED_UNARY_V(sqrt, (T) std::sqrt(v[i]))
_CT_SCALAR_VECTORIZED_UNARY_V(rcp, (T) T(1) / v[i])
_CT_SCALAR_VECTORIZED_UNARY_V(rsqrt, T(1) / (T) std::sqrt(v[i]))

#undef _CT_SCALAR_VECTORIZED_UNARY_V

#define _CT_SCALAR_VECTORIZED_BINARY_M(name, ...) \
template <typename T, nint_t N, int P> \
auto name(Tag<T, N, P> t, VecOf(t) a, VecOf(t) b) -> MaskOf(t) { \
  static_assert(is_default_impl(t)); \
  static_assert(is_word_vec(t)); \
  MaskOf(t) r; \
  for (nint_t i = 0; i < size(t); ++i) { \
    r[i] = (__VA_ARGS__); \
  } \
  return r; \
} \
template <typename T, nint_t N, int P> \
auto name(Tag<T, N, P> t, VecOf(t) a, VecOf(t) b, MaskOf(t) m) -> MaskOf(t) { \
  static_assert(is_default_impl(t)); \
  static_assert(is_word_vec(t)); \
  MaskOf(t) r; \
  for (nint_t i = 0; i < size(t); ++i) { \
    r[i] = m[i] ? (__VA_ARGS__) : false; \
  } \
  return r; \
}

_CT_SCALAR_VECTORIZED_BINARY_M(cmpeq, a[i] == b[i])
_CT_SCALAR_VECTORIZED_BINARY_M(cmpne, a[i] != b[i])
_CT_SCALAR_VECTORIZED_BINARY_M(cmplt, a[i] < b[i])
_CT_SCALAR_VECTORIZED_BINARY_M(cmple, a[i] <= b[i])
_CT_SCALAR_VECTORIZED_BINARY_M(cmpgt, a[i] > b[i])
_CT_SCALAR_VECTORIZED_BINARY_M(cmpge, a[i] >= b[i])
#undef _CT_SCALAR_VECTORIZED_BINARY_M

#define _CT_SCALAR_VECTORIZED_UNARY_M(name, ...) \
template <typename T, nint_t N, int P> \
auto name(Tag<T, N, P> t, VecOf(t) v) -> MaskOf(t) { \
  static_assert(is_default_impl(t)); \
  static_assert(is_word_vec(t)); \
  MaskOf(t) r; \
  for (nint_t i = 0; i < size(t); ++i) { \
    r[i] = (__VA_ARGS__); \
  } \
  return r; \
} \
template <typename T, nint_t N, int P> \
auto name(Tag<T, N, P> t, VecOf(t) v, MaskOf(t) m) -> MaskOf(t) { \
  static_assert(is_default_impl(t)); \
  static_assert(is_word_vec(t)); \
  MaskOf(t) r; \
  for (nint_t i = 0; i < size(t); ++i) { \
    r[i] = m[i] ? (__VA_ARGS__) : false; \
  } \
  return r; \
}

_CT_SCALAR_VECTORIZED_UNARY_M(isnan, std::isnan(v[i]))
_CT_SCALAR_VECTORIZED_UNARY_M(isposinf, v[i] > 0 && std::isinf(v[i]))
_CT_SCALAR_VECTORIZED_UNARY_M(isneginf, v[i] < 0 && std::isinf(v[i]))
_CT_SCALAR_VECTORIZED_UNARY_M(isinf, std::isinf(v[i]))
#undef _CT_SCALAR_VECTORIZED_UNARY_M
} // namespace word
} // namespace ct::tl::vec

#endif //CTORCH_SCALAR_H
