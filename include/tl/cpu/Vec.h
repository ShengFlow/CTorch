//
// Created by renyz on 2026/3/16.
//

#ifndef CTORCH_VEC_H
#define CTORCH_VEC_H

#include "tl/cpu/VecBase.h"
#include "tl/cpu/impl/VectorizedUtil.h"
#include "tl/cpu/Capabilities.h"

/**
 * @file Vec.h
 * @brief High-level SIMD vector operations API.
 * 
 * This file provides the user-facing API for SIMD vector operations, designed
 * with a Highway-like interface. The API abstracts over different SIMD
 * architectures and automatically handles:
 * 
 * - Single-word vectors (one hardware register)
 * - Multi-word vectors (multiple registers concatenated)
 * - Scalable vectors (runtime-determined size like SVE)
 * - Scalar fallback (when no SIMD is available)
 * 
 * ## Usage Example
 * @code
 *   using namespace ct::tl::vec;
 *   Tag<float32_t, 4> t;  // 4-element float vector
 *   
 *   // Fill with a value
 *   auto v = fill(t, 1.0f);
 *   
 *   // Load from memory
 *   float data[4] = {1, 2, 3, 4};
 *   auto v2 = loadu(t, data);
 *   
 *   // Store to memory
 *   storeu(t, data, v);
 *   
 *   // Process partial data (n elements)
 *   auto v3 = loadu(t, data, 3, v);  // Load 3 elements, 4th from default_v
 * @endcode
 * 
 * ## Design Notes
 * 
 * The API uses a tag-based dispatch system. All operations take a Tag parameter
 * that specifies the vector type. This design:
 * 
 * 1. Enables compile-time type checking and optimization
 * 2. Supports both fixed-size and scalable vectors uniformly
 * 3. Allows the same API to work across different architectures
 * 
 * For multi-word vectors, operations are automatically unrolled across all words.
 * The internal `vmap` functions handle this transparently.
 */

#if defined(ARCH_X86_FAMILY) && defined(HAS_AVX)
  #include "tl/cpu/impl/x86_Basic.h"
  #include "tl/cpu/impl/x86_Bit.h"
  #include "tl/cpu/impl/x86_Conversions.h"
  #include "tl/cpu/impl/x86_LoadStore.h"
  #include "tl/cpu/impl/x86_Arithmetic.h"
#elif defined(ARCH_ARM_FAMILY)
#else
  #include "tl/cpu/impl/Scalar.h"
#endif

#include "tl/cpu/impl/Common.h"

namespace ct::tl::vec {
using namespace CPU_CAPABILITY;

/* ************************************************************************** */
//                               Constructors                                 //
/* ************************************************************************** */

/**
 * @brief Create a vector filled with a single value.
 * 
 * All elements of the result vector are set to `value`.
 * 
 * @example
 *   Tag<float32_t, 4> t;
 *   auto v = fill(t, 3.14f);  // v = [3.14, 3.14, 3.14, 3.14]
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> value) {
  using namespace details;
  return vmap(
      t, [=](auto tt) { return word::fill(tt, value); }
  );
}

/**
 * @brief Create a vector with first n elements filled with a value.
 *
 * Elements [0, n) are set to `value`; elements [n, size(t)) are taken from `default_v`.
 *
 * @return Vector with first n elements from value, rest from default_v
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> value, nint_t n, Vec<T> default_v) {
  using namespace details;
  return vmap(
      t, n, [=](auto tt, auto&& dd) { return word::fill(tt, value); },
      [=](auto tt, nint_t rem, auto&& dd) { return word::fill(tt, value, rem, dd); },
      ShardVec(t, default_v)
  );
}

template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> value, nint_t n, TypeOf<T> default_v = T()) {
  return vec::fill(t, value, n, vec::fill(t, default_v));
}

/**
 * @brief Create a vector where masked lanes are set to a value.
 *
 * For each lane i where mask[i] is true, result[i] = value.
 * Masked-out lanes take values from default_v.
 *
 * @return Vector with masked lanes from value, rest from default_v
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> value, Mask<T> m, Vec<T> default_v) {
  using namespace details;
  return vmap(
      t, [=](auto tt, auto&& mm, auto&& dd) { return word::fill(tt, value, mm, dd); },
      ShardMask(t, m), ShardVec(t, default_v)
  );
}

template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> fill(T t, TypeOf<T> value, Mask<T> m, TypeOf<T> default_v = TypeOf<T>()) {
  return vec::fill(t, value, m, vec::fill(t, default_v));
}


/**
 * @brief Create a vector filled with zeros (default-constructed values).
 *
 * Equivalent to fill(t, T()) where T is default-constructible.
 *
 * @return Vector with all elements zero-initialized
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> zeros(T t) {
  using namespace details;
  return vmap(
      t, [=](auto tt) { return word::zeros(tt); }
  );
}

/**
 * @brief Create a mask filled with a single boolean value.
 *
 * @return Mask with all elements set to `value`
 *
 * @example
 *   Tag<float32_t, 4> t;
 *   auto m = mfill(t, true);  // All lanes active
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Mask<T> mfill(T t, bool value) {
  using namespace details;
  return vmap(
      t, [=](auto tt) { return word::mfill(tt, value); }
  );
}

/**
 * @brief Create a mask with all lanes active (all true).
 *
 * @return Mask with all elements true
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Mask<T> mtrue(T t) {
  return vec::mfill(t, true);
}

/**
 * @brief Create a mask with all lanes inactive (all false).
 *
 * @return Mask with all elements false
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Mask<T> mfalse(T t) {
  return vec::mfill(t, false);
}

/**
 * @brief Create a mask where lanes i are true if (a + i) < b.
 *
 * This is useful for creating masks for processing a partial number of
 * elements. For a vector of size N:
 *   result[i] = (a + i) < b
 *
 * For multi-word vectors, each word's mask is computed independently
 * with appropriate offsets.
 *
 * @return Mask where lanes [a, b) are true
 *
 * @example
 *   Tag<float32_t, 4> t;
 *   auto m = mwhilelt(t, 0, 3);  // m = [true, true, true, false]
 *
 *   // Useful for processing partial data:
 *   auto m = mwhilelt(t, 0, n);  // Process n elements
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Mask<T> mwhilelt(T t, nint_t a, nint_t b) {
  using namespace details;
  nint_t ws = word_size(t);
  return vmap(
      t, [=] <nint_t I>(auto tt) { return word::mwhilelt(tt, a + I * ws, b); }
  );
}

/**
 * @brief Create a mask where lanes i are true if (a + i) <= b.
 *
 * Equivalent to mwhilelt(t, a, b + 1).
 *
 * @return Mask where lanes [a, b] are true
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Mask<T> mwhilele(T t, nint_t a, nint_t b) {
  return vec::mwhilelt(t, a, b + 1);
}

/**
 * @brief Create a mask where lanes i are true if (a + i) >= b.
 *
 * For a vector of size N:
 *   result[i] = (a + i) >= b
 *
 * @return Mask where lanes >= b are true
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Mask<T> mwhilege(T t, nint_t a, nint_t b) {
  using namespace details;
  nint_t ws = word_size(t);
  return vmap(
      t, [=] <nint_t I>(auto tt) { return word::mwhilege(tt, a + I * ws, b); }
  );
}

/**
 * @brief Create a mask where lanes i are true if (a + i) > b.
 *
 * Equivalent to mwhilege(t, a, b + 1).
 *
 * @return Mask where lanes > b are true
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Mask<T> mwhilegt(T t, nint_t a, nint_t b) {
  return vec::mwhilege(t, a, b + 1);
}


/* ************************************************************************** */
//                          Consecutive load & store                          //
/* ************************************************************************** */

/**
 * @brief Load a vector from unaligned memory.
 *
 * Loads size(t) consecutive elements from memory starting at address p.
 * The pointer p does not need to be aligned.
 *
 * @return Loaded vector
 *
 * @note For multi-word vectors, each word is loaded from consecutive addresses
 *       (p, p + word_size, p + 2*word_size, etc.)
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T>* p) {
  using namespace details;
  return vmap(
      t, [=](auto tt, const TypeOf<T>* pp) { return word::loadu(tt, pp); },
      StepPointer(t, p)
  );
}

/**
 * @brief Load a vector from an initializer list.
 *
 * Convenience overload for creating vectors from brace-enclosed values.
 *
 * @return Loaded vector
 *
 * @example
 *   Tag<float32_t, 4> t;
 *   auto v = loadu(t, {1.0f, 2.0f, 3.0f, 4.0f});
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> loadu(T t, std::initializer_list<TypeOf<T>> list) {
  CT_ASSERT(list.size() >= size(t), "insufficient elements: %zd v.s. %zd", (nint_t) list.size(), size(t));
  return vec::loadu(t, (const TypeOf<T>*) list.begin());
}

/**
 * @brief Load a vector from aligned memory.
 *
 * Loads size(t) consecutive elements from memory starting at address p.
 * The pointer p must be aligned to DEFAULT_ALIGNMENT bytes.
 *
 * @return Loaded vector
 *
 * @warning Passing an unaligned pointer may cause crashes or performance issues.
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> load(T t, const TypeOf<T>* p) {
  using namespace details;
  return vmap(
      t, [=](auto tt, const TypeOf<T>* pp) { return word::load(tt, pp); },
      StepPointer(t, p)
  );
}

/**
 * @brief Load a vector from an aligned initializer list.
 *
 * @return Loaded vector
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> load(T t, std::initializer_list<T> list) {
  return vec::load(t, (const TypeOf<T>*) list.begin());
}

/**
 * @brief Load first n elements from unaligned memory, with default for rest.
 *
 * Loads n consecutive elements from memory. Elements beyond n are filled
 * from default_v. This is useful for processing partial data at the end
 * of an array.
 *
 * @return Vector with first n elements from memory, rest from default_v
 *
 * @example
 *   Tag<float32_t, 4> t;
 *   float data[4] = {1, 2, 3, 4};
 *   auto v = loadu(t, data, 2, zeros(t));  // v = [1, 2, 0, 0]
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T>* p, nint_t n, Vec<T> default_v) {
  using namespace details;
  return vmap(
      t, n, [=](auto tt, const TypeOf<T>* p, auto v_d) { return word::loadu(tt, p); },
      [=](auto tt, nint_t rem, const TypeOf<T>* p, auto v_d) { return word::loadu(tt, p, rem, v_d); },
      StepPointer(t, p), ShardVec(t, default_v)
  );
}

template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T>* p, nint_t n, T default_v = T()) {
  return vec::loadu(t, p, n, vec::fill(t, default_v));
}

/**
 * @brief Load first n elements from aligned memory, with default for rest.
 *
 * @return Vector with loaded elements
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> load(T t, const TypeOf<T>* p, nint_t n, Vec<T> default_v) {
  using namespace details;
  return vmap(
      t, n, [=](auto tt, const TypeOf<T>* p, auto v_d) { return word::load(tt, p); },
      [=](auto tt, nint_t rem, const TypeOf<T>* p, auto v_d) { return word::load(tt, p, rem, v_d); },
      StepPointer(t, p), ShardVec(t, default_v)
  );
}

template <TLV_DECL_TAG(T)>
TLV_INLINE auto load(T t, const TypeOf<T>* p, nint_t n, T default_v = T()) {
  return vec::load(t, p, n, vec::fill(t, default_v));
}

/**
 * @brief Masked load from unaligned memory.
 *
 * For each lane i where mask[i] is true, loads from p[i]. For lanes
 * where mask is false, takes the value from default_v[i].
 *
 * @return Vector with masked-loaded elements
 *
 * @note This operation may load from masked-out addresses; ensure
 *       those addresses are valid even if the values won't be used.
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T>* p, Mask<T> m, Vec<T> default_v) {
  using namespace details;
  return vmap(
      t, [=](auto tt, const TypeOf<T>* p, auto mm, auto v_d) { return word::loadu(tt, p, mm, v_d); },
      StepPointer(t, p), ShardMask(t, m), ShardVec(t, default_v)
  );
}

template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> loadu(T t, const TypeOf<T>* p, Mask<T> m, T default_v = T()) {
  return vec::loadu(t, p, m, vec::fill(t, default_v));
}

/**
 * @brief Masked load from aligned memory.
 *
 * @return Vector with masked-loaded elements
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> load(T t, const TypeOf<T>* p, Mask<T> m, Vec<T> default_v) {
  using namespace details;
  return vmap(
      t, [=](auto tt, const TypeOf<T>* p, auto mm, auto v_d) { return word::load(tt, p, mm, v_d); },
      StepPointer(t, p), ShardMask(t, m), ShardVec(t, default_v)
  );
}

template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> load(T t, const TypeOf<T>* p, Mask<T> m, T default_v = T()) {
  return vec::load(t, p, m, vec::fill(t, default_v));
}

/**
 * @brief Store a vector to unaligned memory.
 *
 * Stores size(t) consecutive elements to memory starting at address p.
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE void storeu(T t, TypeOf<T>* p, Vec<T> v) {
  using namespace details;
  return vmap(
      t, [=](auto tt, TypeOf<T>* pp, auto&& vv) { word::storeu(tt, pp, vv); },
      StepPointer(t, p), ShardVec(t, v)
  );
}

/**
 * @brief Store a vector to aligned memory.
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE void store(T t, TypeOf<T>* p, Vec<T> v) {
  using namespace details;
  return vmap(
      t, [=](auto tt, TypeOf<T>* pp, auto&& vv) { word::store(tt, pp, vv); },
      StepPointer(t, p), ShardVec(t, v)
  );
}

/**
 * @brief Store first n elements of a vector to unaligned memory.
 *
 * Only the first n elements are written to memory.
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE void storeu(T t, TypeOf<T>* p, nint_t n, Vec<T> v) {
  using namespace details;
  return vmap(
      t, n, [=](auto tt, TypeOf<T>* pp, auto&& vv) { word::storeu(tt, pp, vv); },
      [=](auto tt, nint_t rem, TypeOf<T>* pp, auto&& vv) { word::storeu(tt, pp, rem, vv); },
      StepPointer(t, p), ShardVec(t, v)
  );
}

/**
 * @brief Store first n elements of a vector to aligned memory.
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE void store(T t, TypeOf<T>* p, nint_t n, Vec<T> v) {
  using namespace details;
  return vmap(
      t, n, [=](auto tt, TypeOf<T>* pp, auto&& vv) { word::store(tt, pp, vv); },
      [=](auto tt, nint_t rem, TypeOf<T>* pp, auto&& vv) { word::store(tt, pp, rem, vv); },
      StepPointer(t, p), ShardVec(t, v)
  );
}

/**
 * @brief Masked store to unaligned memory.
 *
 * For each lane i where mask[i] is true, stores v[i] to p[i].
 * Masked-out lanes are not written.
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE void storeu(T t, TypeOf<T>* p, Mask<T> m, Vec<T> v) {
  using namespace details;
  return vmap(
      t, [=](auto tt, TypeOf<T>* pp, auto&& mm, auto&& vv) { word::storeu(tt, pp, mm, vv); },
      StepPointer(t, p), ShardMask(t, m), ShardVec(t, v)
  );
}

/**
 * @brief Masked store to aligned memory.
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE void store(T t, TypeOf<T>* p, Mask<T> m, Vec<T> v) {
  using namespace details;
  return vmap(
      t, [=](auto tt, TypeOf<T>* pp, auto&& mm, auto&& vv) { word::store(tt, pp, mm, vv); },
      StepPointer(t, p), ShardMask(t, m), ShardVec(t, v)
  );
}


/* ************************************************************************** */
//                         Indexed gather & scatter                           //
/* ************************************************************************** */

/**
 * @brief Gather elements from memory using an index vector.
 *
 * For each lane i, loads from p[i[i]] and returns the result.
 * This is the vectorized equivalent of:
 *   result[i] = p[index[i]]
 *
 * @return Gathered vector
 *
 * @note Gather operations can be significantly slower than consecutive loads
 *       due to memory access patterns. Use consecutive loads when possible.
 *
 * @example
 *   Tag<float32_t, 4> t;
 *   float data[100] = {...};
 *   int32_t indices[4] = {10, 20, 5, 15};
 *   auto idx = loadu(Tag<int32_t, 4>(), indices);
 *   auto v = gather(t, data, idx);  // v[i] = data[indices[i]]
 */
template <TLV_DECL_TAG(T), TL_IF(sizeof(TypeOf<T>) >= 4)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i) {
  using namespace details;
  constexpr Rebind<Index<TypeOf<T>>, T> it;
  return vmap(
      t, [=](auto tt, TypeOf<T>* pp, auto&& ii) { return word::gather(tt, pp, ii); },
      StepPointer(t, p), ShardVec(it, i)
  );
}

/**
 * @brief Gather first n elements using an index vector, with default for rest.
 *
 * @return Gathered vector
 */
template <TLV_DECL_TAG(T), TL_IF(sizeof(TypeOf<T>) >= 4)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, nint_t n, Vec<T> default_v) {
  using namespace details;
  constexpr Rebind<Index<TypeOf<T>>, T> it;
  return vmap(
      t, n, [=](auto tt, TypeOf<T>* pp, auto&& ii, auto&& vv) { return word::gather(tt, pp, ii); },
      [=](auto tt, nint_t rem, TypeOf<T>* pp, auto&& ii, auto&& vv) { return word::gather(tt, pp, ii, rem, vv); },
      StepPointer(t, p), ShardVec(it, i), ShardVec(t, default_v)
  );
}

/**
 * @brief Gather first n elements using an index vector, with scalar default.
 *
 * Convenience overload that broadcasts a scalar default value.
 *
 * @return Gathered vector
 */
template <TLV_DECL_TAG(T), TL_IF(sizeof(TypeOf<T>) >= 4)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, nint_t n, TypeOf<T> default_v) {
  return vec::gather(t, p, i, n, vec::fill(t, default_v));
}

/**
 * @brief Masked gather from memory using an index vector.
 *
 * For each lane i where mask[i] is true, loads from p[index[i]].
 * For masked-out lanes, takes the value from default_v[i].
 *
 * @return Gathered vector
 *
 * @note May access p[index[i]] for masked-out lanes; ensure indices are valid.
 */
template <TLV_DECL_TAG(T), TL_IF(sizeof(TypeOf<T>) >= 4)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> default_v) {
  using namespace details;
  constexpr Rebind<Index<TypeOf<T>>, T> it;
  return vmap(
      t, [=](auto tt, TypeOf<T>* pp, auto&& ii, auto&& mm, auto&& vv) { return word::gather(tt, pp, ii, mm, vv); },
      StepPointer(t, p), ShardVec(it, i), ShardMask(t, m), ShardVec(t, default_v)
  );
}

/**
 * @brief Masked gather with scalar default.
 *
 * @return Gathered vector
 */
template <TLV_DECL_TAG(T), TL_IF(sizeof(TypeOf<T>) >= 4)>
TLV_INLINE Vec<T> gather(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, TypeOf<T> default_v) {
  return vec::gather(t, p, i, m, vec::fill(t, default_v));
}

/**
 * @brief Scatter elements to memory using an index vector.
 *
 * For each lane i, stores v[i] to p[index[i]].
 * This is the vectorized equivalent of:
 *   p[index[i]] = v[i]
 *
 * @warning If indices are not unique, the result depends on execution order.
 *          Multiple writes to the same location may race.
 *
 * @note Scatter operations can be significantly slower than consecutive stores.
 */
template <TLV_DECL_TAG(T), TL_IF(sizeof(TypeOf<T>) >= 4)>
TLV_INLINE void scatter(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Vec<T> v) {
  using namespace details;
  constexpr Rebind<Index<TypeOf<T>>, T> it;
  return vmap(
      t, [=](auto tt, TypeOf<T>* pp, auto&& ii, auto&& vv) { word::scatter(tt, pp, ii, vv); },
      StepPointer(t, p), ShardVec(it, i), ShardVec(t, v)
  );
}

/**
 * @brief Scatter first n elements to memory using an index vector.
 *
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Base pointer for scatter
 * @param i Index vector
 * @param v The vector to scatter
 * @param n Number of elements to scatter (0 <= n <= size(t))
 */
template <TLV_DECL_TAG(T), TL_IF(sizeof(TypeOf<T>) >= 4)>
TLV_INLINE void scatter(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, nint_t n, Vec<T> v) {
  using namespace details;
  constexpr Rebind<Index<TypeOf<T>>, T> it;
  return vmap(
      t, n, [=](auto tt, TypeOf<T>* pp, auto&& ii, auto&& vv) { word::scatter(tt, pp, ii, vv); },
      [=](auto tt, nint_t rem, TypeOf<T>* pp, auto&& ii, auto&& vv) { word::scatter(tt, pp, ii, rem, vv); },
      StepPointer(t, p), ShardVec(it, i), ShardVec(t, v)
  );
}

/**
 * @brief Masked scatter to memory using an index vector.
 *
 * For each lane i where mask[i] is true, stores v[i] to p[index[i]].
 * Masked-out lanes are not written.
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
template <TLV_DECL_TAG(T), TL_IF(sizeof(TypeOf<T>) >= 4)>
TLV_INLINE void scatter(T t, const TypeOf<T>* p, Vec<Rebind<Index<TypeOf<T>>, T>> i, Mask<T> m, Vec<T> v) {
  using namespace details;
  constexpr Rebind<Index<TypeOf<T>>, T> it;
  return vmap(
      t, [=](auto tt, TypeOf<T>* pp, auto&& ii, auto&& mm, auto&& vv) { word::scatter(tt, pp, ii, mm, vv); },
      StepPointer(t, p), ShardVec(it, i), ShardMask(t, m), ShardVec(t, v)
  );
}


/* ************************************************************************** */
//                             Get / set element                              //
/* ************************************************************************** */

/**
 * @brief Get a single element from a vector by index.
 *
 * @warning This operation is relatively slow as it requires extracting
 *          the element from a SIMD register. Avoid using in performance-
 *          critical inner loops.
 * @return The element at the specified index
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE TypeOf<T> get(V v, nint_t index) {
  constexpr T t;
  nint_t ws = word_size(t);
  nint_t ord = index / ws, off = index % ws;
  auto word = get_word(t, v, ord);
  return word::get(word, off);
}

template <TLV_DECL_TAG(T)>
TLV_INLINE TypeOf<T> get(T t, Vec<T> v, nint_t index) {
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0:%zd", index, size(t));
  return vec::get<Vec<T>, T>(v, index);
}


/**
 * @brief Get a single element from a mask by index.
 *
 * @warning This operation is relatively slow. Avoid using in hot loops.
 * @return The boolean value at the specified index
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE bool get(T t, Mask<T> m, nint_t index) {
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0:%zd", index, size(t));
  nint_t ws = word_size(t);
  nint_t ord = index / ws, off = index % ws;
  auto word = get_word_mask(t, m, ord);
  return word::get(word, off);
}

/**
 * @brief Set a single element in a vector by index.
 *
 * @warning This operation is relatively slow as it requires modifying
 *          the SIMD register. Avoid using in performance-critical loops.
 *
 * @return Vector with the element at index set to x
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V set(V v, nint_t index, TypeOf<T> x) {
  constexpr T t;
  nint_t ws = word_size(t);
  nint_t ord = index / ws, off = index % ws;
  auto word = get_word(t, v, ord);
  return set_word(t, v, ord, word::set(word, off, x));
}
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> set(T t, Vec<T> v, nint_t index, TypeOf<T> x) {
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0:%zd", index, size(t));
  return vec::set<Vec<T>, T>(v, index, x);
}

/**
 * @brief Set a single element in a mask by index.
 *
 * @warning This operation is relatively slow. Avoid using in hot loops.
 *
 * @return Mask with the element at index set to x
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE auto set(T t, Mask<T> m, nint_t index, bool x) -> Mask<T> {
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0:%zd", index, size(t));
  nint_t ws = word_size(t);
  nint_t ord = index / ws, off = index % ws;
  auto word = get_word_mask(t, m, ord);
  return set_word_mask(t, m, ord, word::set(word, off, x));
}



/* ************************************************************************** */
//                       Basic arithmetic operations                          //
/* ************************************************************************** */

/**
 * @brief Element-wise addition: result[i] = a[i] + b[i].
 * @return Sum of the two vectors
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V add(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::add(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked element-wise addition: result[i] = a[i] + b[i] for masked lanes.
 *
 * For masked-out lanes, the value is undefined (typically a[i]).
 *
 * @return Sum of the two vectors for masked lanes
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V add(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::add(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise subtraction: result[i] = a[i] - b[i].
 * @return Difference of the two vectors
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V sub(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::sub(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked element-wise subtraction: result[i] = a[i] - b[i] for masked lanes.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V sub(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::sub(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise multiplication: result[i] = a[i] * b[i].
 * @return Product of the two vectors
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V mul(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::mul(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked element-wise multiplication: result[i] = a[i] * b[i] for masked lanes.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V mul(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::mul(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise division: result[i] = a[i] / b[i].
 * @return Quotient of the two vectors
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V div(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::div(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked element-wise division: result[i] = a[i] / b[i] for masked lanes.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V div(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::div(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise maximum: result[i] = max(a[i], b[i]).
 * @return Vector with maximum elements
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V max(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::max(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked element-wise maximum: result[i] = max(a[i], b[i]) for masked lanes.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V max(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::max(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise minimum: result[i] = min(a[i], b[i]).
 * @return Vector with minimum elements
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V min(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::min(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked element-wise minimum: result[i] = min(a[i], b[i]) for masked lanes.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V min(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::min(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise bitwise AND: result[i] = a[i] & b[i].
 * @return Bitwise AND of the two vectors
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_and(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::bit_and(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked element-wise bitwise AND for masked lanes.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_and(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::bit_and(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise bitwise OR: result[i] = a[i] | b[i].
 * @return Bitwise OR of the two vectors
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_or(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::bit_or(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked element-wise bitwise OR for masked lanes.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_or(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::bit_or(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise bitwise XOR: result[i] = a[i] ^ b[i].
 * @return Bitwise XOR of the two vectors
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_xor(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::bit_xor(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked element-wise bitwise XOR for masked lanes.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_xor(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::bit_xor(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise bitwise AND-NOT: result[i] = (~a[i]) & b[i].
 * @return Bitwise AND-NOT of the two vectors
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_andnot(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::bit_andnot(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked element-wise bitwise AND-NOT for masked lanes.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_andnot(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::bit_andnot(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise left shift: result[i] = v[i] << count.
 * @param count Shift count (same for all lanes)
 * @return Shifted vector
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_shl(V v, int count) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::bit_shl(vv, count); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked element-wise left shift for masked lanes.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_shl(V v, int count, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm) { return word::bit_shl(vv, count, mm); },
      ShardVec(t, v), ShardMask(t, m)
  );
}

/**
 * @brief Element-wise right shift: result[i] = v[i] >> count.
 * @param count Shift count (same for all lanes)
 * @return Shifted vector
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_shr(V v, int count) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::bit_shr(vv, count); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked element-wise right shift for masked lanes.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_shr(V v, int count, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm) { return word::bit_shr(vv, count, mm); },
      ShardVec(t, v), ShardMask(t, m)
  );
}


/**
 * @brief Element-wise bitwise NOT: result[i] = ~v[i].
 * @return Bitwise complement of the vector
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_not(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::bit_not(vv); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked bitwise NOT: result[i] = ~v[i] for masked lanes, default_v[i] otherwise.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_not(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm, auto&& dd) { return word::bit_not(vv, mm, dd); },
      ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v)
  );
}

/**
 * @brief Masked bitwise NOT with original value as default for masked-out lanes.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_not(V v, Mask<T> m) {
  return vec::bit_not(v, m, v);
}

/**
 * @brief Element-wise negation: result[i] = -v[i].
 * @return Negated vector
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V neg(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::neg(vv); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked negation: result[i] = -v[i] for masked lanes, default_v[i] otherwise.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V neg(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm, auto&& dd) { return word::neg(vv, mm, dd); },
      ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v)
  );
}

/**
 * @brief Masked negation with original value as default for masked-out lanes.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V neg(V v, Mask<T> m) {
  return vec::neg(v, m, v);
}


/**
 * @brief Element-wise absolute value: result[i] = |v[i]|.
 * @return Absolute value of the vector
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V abs(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::abs(vv); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked absolute value: result[i] = |v[i]| for masked lanes, default_v[i] otherwise.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V abs(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm, auto&& dd) { return word::abs(vv, mm, dd); },
      ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v)
  );
}

/**
 * @brief Masked absolute value with original value as default for masked-out lanes.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V abs(V v, Mask<T> m) {
  return vec::abs(v, m, v);
}


/**
 * @brief Element-wise square root: result[i] = sqrt(v[i]).
 * @return Square root of the vector (floating-point only)
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V sqrt(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::sqrt(vv); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked square root: result[i] = sqrt(v[i]) for masked lanes, default_v[i] otherwise.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V sqrt(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm, auto&& dd) { return word::sqrt(vv, mm, dd); },
      ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v)
  );
}

/**
 * @brief Masked square root with original value as default for masked-out lanes.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V sqrt(V v, Mask<T> m) {
  return vec::sqrt(v, m, v);
}


/**
 * @brief Element-wise reciprocal square root: result[i] = 1/sqrt(v[i]).
 * @return Reciprocal square root of the vector (floating-point only)
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V rsqrt(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::rsqrt(vv); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked reciprocal square root: result[i] = 1/sqrt(v[i]) for masked lanes.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V rsqrt(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm, auto&& dd) { return word::rsqrt(vv, mm, dd); },
      ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v)
  );
}

/**
 * @brief Masked reciprocal square root with original value as default.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V rsqrt(V v, Mask<T> m) {
  return vec::rsqrt(v, m, v);
}


/**
 * @brief Element-wise reciprocal: result[i] = 1/v[i].
 * @return Reciprocal of the vector (floating-point only)
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V rcp(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::rcp(vv); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked reciprocal: result[i] = 1/v[i] for masked lanes.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V rcp(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm, auto&& dd) { return word::rcp(vv, mm, dd); },
      ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v)
  );
}

/**
 * @brief Masked reciprocal with original value as default.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V rcp(V v, Mask<T> m) {
  return vec::rcp(v, m, v);
}


/**
 * @brief Element-wise equality comparison: result[i] = (a[i] == b[i]).
 * @return Mask where lanes are true if elements are equal
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpeq(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::cmpeq(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked equality comparison: only compare lanes where mask is true.
 * @return Mask where lanes are true if elements are equal and mask is true
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::cmpeq(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}


/**
 * @brief Element-wise inequality comparison: result[i] = (a[i] != b[i]).
 * @return Mask where lanes are true if elements are not equal
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpne(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::cmpne(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked inequality comparison: only compare lanes where mask is true.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::cmpne(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}


/**
 * @brief Element-wise less-than comparison: result[i] = (a[i] < b[i]).
 * @return Mask where lanes are true if a[i] < b[i]
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmplt(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::cmplt(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked less-than comparison: only compare lanes where mask is true.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::cmplt(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}


/**
 * @brief Element-wise greater-than comparison: result[i] = (a[i] > b[i]).
 * @return Mask where lanes are true if a[i] > b[i]
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpgt(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::cmpgt(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked greater-than comparison: only compare lanes where mask is true.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::cmpgt(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}


/**
 * @brief Element-wise less-than-or-equal comparison: result[i] = (a[i] <= b[i]).
 * @return Mask where lanes are true if a[i] <= b[i]
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmple(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::cmple(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked less-than-or-equal comparison: only compare lanes where mask is true.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::cmple(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}


/**
 * @brief Element-wise greater-than-or-equal comparison: result[i] = (a[i] >= b[i]).
 * @return Mask where lanes are true if a[i] >= b[i]
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpge(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::cmpge(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Masked greater-than-or-equal comparison: only compare lanes where mask is true.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::cmpge(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}


/**
 * @brief Check if elements are NaN: result[i] = isnan(v[i]).
 * @return Mask where lanes are true if elements are NaN (floating-point only)
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> isnan(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::isnan(vv); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked NaN check: only check lanes where mask is true.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> isnan(V v, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm) { return word::isnan(vv, mm); },
      ShardVec(t, v), ShardMask(t, m)
  );
}


/**
 * @brief Check if elements are positive infinity: result[i] = (v[i] == +inf).
 * @return Mask where lanes are true if elements are positive infinity
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> isposinf(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::isposinf(vv); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked positive infinity check: only check lanes where mask is true.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> isposinf(V v, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm) { return word::isposinf(vv, mm); },
      ShardVec(t, v), ShardMask(t, m)
  );
}


/**
 * @brief Check if elements are negative infinity: result[i] = (v[i] == -inf).
 * @return Mask where lanes are true if elements are negative infinity
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> isneginf(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::isneginf(vv); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked negative infinity check: only check lanes where mask is true.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> isneginf(V v, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm) { return word::isneginf(vv, mm); },
      ShardVec(t, v), ShardMask(t, m)
  );
}


/**
 * @brief Check if elements are infinity (positive or negative): result[i] = isinf(v[i]).
 * @return Mask where lanes are true if elements are infinity
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> isinf(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::isinf(vv); },
      ShardVec(t, v)
  );
}

/**
 * @brief Masked infinity check: only check lanes where mask is true.
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> isinf(V v, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm) { return word::isinf(vv, mm); },
      ShardVec(t, v), ShardMask(t, m)
  );
}


/* ************************************************************************** */
//                           Shuffle & Permutation                            //
/* ************************************************************************** */

/**
 * @brief Shuffle elements within each 16-byte block using compile-time indices.
 *
 * Permutes elements within each 16-byte lane independently. All lanes apply
 * the same shuffle pattern. For float32_t (4 elements per block), indices
 * must be in [0, 3].
 *
 * Result: result[j] = v[Is[j % M] + floor(j / M)]
 * where M = 16 / sizeof(element_type), j = 0...N-1.
 *
 * @tparam Is Compile-time shuffle indices
 * @return Shuffled vector
 *
 * @note Vectors smaller than word_size are treated as word_size vectors.
 */
template <int... Is, TLV_DECL_VEC(V)>
TLV_INLINE V local_shuf(V v) {
  using namespace details;
  constexpr Vec2Tag<V> t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::local_shuf<Is...>(vv); },
      ShardVec(t, v)
  );
}

/**
 * @brief Shuffle elements within each 16-byte block using runtime indices.
 *
 * Permutes elements within each 16-byte lane independently. Each lane can
 * use different indices from the index vector. Index vector element type
 * must be a signed integer with same width as the data element type.
 *
 * Result: result[j] = v[i[j] + floor(j / M)]
 * where M = 16 / sizeof(element_type), j = 0...N-1.
 * Undefined if i[j] not in [0, M).
 *
 * @param v Input vector
 * @param i Index vector (signed integer, same bit-width as v)
 * @return Shuffled vector
 */
template <TLV_DECL_VEC(V), TLV_DECL_VEC(Vi)>
TLV_INLINE V local_shuf(V v, Vi i) {
  using namespace details;
  constexpr Vec2Tag<V> t;
  constexpr Vec2Tag<Vi> ti;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& ii) { return word::local_shuf(vv, ii); },
      ShardVec(t, v), ShardVec(ti, i)
  );
}

/**
 * @brief Shuffle elements within each 16-byte block using runtime integer indices.
 *
 * Same as the compile-time index version, but accepts runtime values.
 * All lanes apply the same shuffle pattern.
 *
 * @param v Input vector
 * @param is Runtime shuffle indices (one per element in block)
 * @return Shuffled vector
 */
template <TLV_DECL_VEC(V), typename... Is, TL_IF(is_any<Is, int> && ...)>
TLV_INLINE V local_shuf(V v, Is... is) {
  using namespace details;
  constexpr Vec2Tag<V> t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::local_shuf(vv, is...); },
      ShardVec(t, v)
  );
}

//template <int... Is, TLV_DECL_VEC(V)>
//V block_shuf(V v) {
//  using namespace details;
//  constexpr Vec2Tag<V> t;
//  return vmap(
//      t, [=](auto tt, auto&& vv) { return word::block_shuf<Is...>(vv); },
//      ShardVec(t, v)
//  );
//}
//
//template <TLV_DECL_VEC(V), typename... Is, TL_IF(is_any<Is, int> && ...)>
//V block_shuf(V v, Is... is) {
//  using namespace details;
//  constexpr Vec2Tag<V> t;
//  return vmap(
//      t, [=](auto tt, auto&& vv) { return word::block_shuf(vv, is...); },
//      ShardVec(t, v)
//  );
//}

/**
 * @brief Shuffle elements across the entire vector using an index vector.
 *
 * Permutes elements across the whole vector (word). Each element can be
 * selected from any position in the input. Index vector element type must
 * be a signed integer with same width as the data element type.
 *
 * Result: result[j] = v[i[j]], where j = 0...N-1.
 * Undefined if i[j] not in [0, N).
 *
 * @param v Input vector
 * @param i Index vector (signed integer, same bit-width as v)
 * @return Shuffled vector
 *
 * @note On x86 AVX2+, this involves cross-lane data movement and is slower
 *       than local_shuf. Without AVX512, performance is further reduced.
 *       For int8_t/uint8_t without AVX512_VBMI, this is slower than wider types.
 */
template <TLV_DECL_VEC(V), TLV_DECL_VEC(Vi)>
TLV_INLINE V shuf(V v, Vi i) {
  using namespace details;
  constexpr Vec2Tag<V> t;
  constexpr Vec2Tag<Vi> ti;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& ii) { return word::shuf(vv, ii); },
      ShardVec(t, v), ShardVec(ti, i)
  );
}

/**
 * @brief Extract the upper half of a vector.
 *
 * Returns a vector of half the length containing the upper (high-indexed)
 * elements of the input vector.
 *
 * For a vector v of size N: result[i] = v[i + N/2]
 *
 * @tparam T Input vector tag type
 * @tparam V Output vector type (Vec<Half<T>>)
 * @param t Input vector tag
 * @param v Input vector of size N
 * @return Vector of size N/2 containing upper half elements
 *
 * @note For multi-word vectors, this extracts words from the upper half
 *       without additional processing. For single-word vectors, delegates
 *       to the architecture-specific word::upper implementation.
 *
 * @example
 *   Tag<float32_t, 8> t;
 *   auto v = loadu(t, {0, 1, 2, 3, 4, 5, 6, 7});
 *   auto u = upper(t, v);  // u = [4, 5, 6, 7]
 */
template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>>
TLV_INLINE V upper(T t, Vec<T> v) {
  using namespace details;
  constexpr nint_t NWi = num_words(t);
  if constexpr (NWi > 1) {
    static_assert(NWi == 2 * num_words(Vec2Tag<V>{}));
    return vmap(Half<T>{}, [&]<nint_t I>(auto tt) { return get_word<I + NWi / 2>(t, v); });
  } else {
    return word::upper(t, v);
  }
}

/**
 * @brief Extract the lower half of a vector.
 *
 * Returns a vector of half the length containing the lower (low-indexed)
 * elements of the input vector.
 *
 * For a vector v of size N: result[i] = v[i]
 *
 * @tparam T Input vector tag type
 * @tparam V Output vector type (Vec<Half<T>>)
 * @param t Input vector tag
 * @param v Input vector of size N
 * @return Vector of size N/2 containing lower half elements
 *
 * @note For multi-word vectors, this extracts words from the lower half
 *       without additional processing. For single-word vectors, delegates
 *       to the architecture-specific word::lower implementation.
 *
 * @example
 *   Tag<float32_t, 8> t;
 *   auto v = loadu(t, {0, 1, 2, 3, 4, 5, 6, 7});
 *   auto u = lower(t, v);  // u = [0, 1, 2, 3]
 */
template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>>
TLV_INLINE V lower(T t, Vec<T> v) {
  using namespace details;
  constexpr nint_t NWi = num_words(t);
  if constexpr (NWi > 1) {
    static_assert(NWi == 2 * num_words(Vec2Tag<V>{}));
    return vmap(Half<T>{}, [&]<nint_t I>(auto tt) { return get_word<I>(t, v); });
  } else {
    return word::lower(t, v);
  }
}

/**
 * @brief Extract elements at even indices from a vector.
 *
 * Returns a vector of half the length containing elements at even positions
 * (indices 0, 2, 4, ...) from the input vector.
 *
 * For a vector v of size N: result[i] = v[2*i]
 *
 * @tparam T Input vector tag type
 * @tparam V Output vector type (Vec<Half<T>>)
 * @param t Input vector tag
 * @param v Input vector of size N
 * @return Vector of size N/2 containing even-indexed elements
 *
 * @pre Input vector must have at least 2 elements.
 *
 * @note For multi-word vectors, this processes pairs of adjacent words
 *       together to extract even elements. For single-word vectors,
 *       delegates to the architecture-specific word::even implementation.
 *
 * @example
 *   Tag<float32_t, 8> t;
 *   auto v = loadu(t, {0, 1, 2, 3, 4, 5, 6, 7});
 *   auto u = even(t, v);  // u = [0, 2, 4, 6]
 */
template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>>
TLV_INLINE V even(T t, Vec<T> v) {
  using namespace details;
  static_assert(size(t) >= 2, "Insufficient elements");
  constexpr nint_t NWi = num_words(t);
  if constexpr (NWi > 1) {
    static_assert(NWi == 2 * num_words(Vec2Tag<V>{}));
    constexpr Twice<WordOf<T>> t2;
    V v_o;
    foreach<NWi / 2>([&]<nint_t I>{
      auto u = word::even(t2, Vec<decltype(t2)>{get_word<2 * I>(t, v), get_word<2 * I + 1>(t, v)});
      v_o = set_word<I>(Half<T>{}, v_o, u);
    });
    return v_o;
  } else {
    return word::even(t, v);
  }
}

/**
 * @brief Extract elements at odd indices from a vector.
 *
 * Returns a vector of half the length containing elements at odd positions
 * (indices 1, 3, 5, ...) from the input vector.
 *
 * For a vector v of size N: result[i] = v[2*i + 1]
 *
 * @tparam T Input vector tag type
 * @tparam V Output vector type (Vec<Half<T>>)
 * @param t Input vector tag
 * @param v Input vector of size N
 * @return Vector of size N/2 containing odd-indexed elements
 *
 * @pre Input vector must have at least 2 elements.
 *
 * @note For multi-word vectors, this processes pairs of adjacent words
 *       together to extract odd elements. For single-word vectors,
 *       delegates to the architecture-specific word::odd implementation.
 *
 * @example
 *   Tag<float32_t, 8> t;
 *   auto v = loadu(t, {0, 1, 2, 3, 4, 5, 6, 7});
 *   auto u = odd(t, v);  // u = [1, 3, 5, 7]
 */
template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>>
TLV_INLINE V odd(T t, Vec<T> v) {
  using namespace details;
  static_assert(size(t) >= 2, "Insufficient elements");
  constexpr nint_t NWi = num_words(t);
  if constexpr (NWi > 1) {
    static_assert(NWi == 2 * num_words(Vec2Tag<V>{}));
    constexpr Twice<WordOf<T>> t2;
    V v_o;
    foreach<NWi / 2>([&]<nint_t I>{
      auto u = word::odd(t2, Vec<decltype(t2)>{get_word<2 * I>(t, v), get_word<2 * I + 1>(t, v)});
      v_o = set_word<I>(Half<T>{}, v_o, u);
    });
    return v_o;
  } else {
    return word::odd(t, v);
  }
}

/**
 * @brief Concatenate two half-length vectors into a full-length vector.
 *
 * Combines two vectors of half the target size into a single vector of
 * the target size. The lower vector's elements occupy the lower indices,
 * and the higher vector's elements occupy the upper indices.
 *
 * Result: result[i] = v_lo[i] for i < N/2, result[i] = v_hi[i - N/2] for i >= N/2
 *
 * @tparam T Output vector tag type
 * @tparam V Input vector type (Vec<Half<T>>)
 * @param t Output vector tag
 * @param v_lo Lower half vector (placed at lower indices)
 * @param v_hi Upper half vector (placed at upper indices)
 * @return Concatenated vector of size N
 *
 * @note This is the inverse operation of upper() and lower() combined:
 *       concat(t, lower(t, v), upper(t, v)) == v
 *
 * @example
 *   Tag<float32_t, 8> t;
 *   Tag<float32_t, 4> th;
 *   auto lo = loadu(th, {0, 1, 2, 3});
 *   auto hi = loadu(th, {4, 5, 6, 7});
 *   auto v = concat(t, lo, hi);  // v = [0, 1, 2, 3, 4, 5, 6, 7]
 */
template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>>
TLV_INLINE Vec<T> concat(T t, V v_lo, V v_hi) {
  using namespace details;
  using Ti = Vec2Tag<V>;
  constexpr nint_t NWo = num_words(t);
  if constexpr (NWo > 1) {
    static_assert(NWo == 2 * num_words(Ti{}));
    Vec<T> v_o;
    foreach<NWo / 2>([&]<nint_t I>{
      v_o = set_word<I>(t, v_o, get_word<I>(Ti(), v_lo));
    });
    foreach<NWo / 2>([&]<nint_t I>{
      v_o = set_word<I + NWo / 2>(t, v_o, get_word<I>(Ti(), v_hi));
    });
    return v_o;
  } else {
    return word::concat(t, v_lo, v_hi);
  }
}

/**
 * @brief Extract even-indexed elements from two vectors and concatenate.
 *
 * Extracts elements at even indices from both input vectors and concatenates
 * them into a single output vector. The result has the same total element
 * count as each input vector.
 *
 * For input vectors v_lo and v_hi of size N:
 *   result[i] = v_lo[2*i]     for i < N/2
 *   result[i] = v_hi[2*(i-N/2)] for i >= N/2
 *
 * @tparam T Vector tag type
 * @param t Vector tag
 * @param v_lo First input vector (lower half of result comes from its even elements)
 * @param v_hi Second input vector (upper half of result comes from its even elements)
 * @return Vector of size N containing even elements from both inputs
 *
 * @note This operation is commonly used after local_interleave_lower to undo
 *       the interleaving. Combined with concat_odd, these two operations can
 *       deinterleave a pair of interleaved vectors.
 *
 * @example
 *   Tag<float32_t, 8> t;
 *   auto lo = loadu(t, {0, 1, 2, 3, 4, 5, 6, 7});
 *   auto hi = loadu(t, {8, 9, 10, 11, 12, 13, 14, 15});
 *   auto v = concat_even(t, lo, hi);  // v = [0, 2, 4, 6, 8, 10, 12, 14]
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> concat_even(T t, Vec<T> v_lo, Vec<T> v_hi) {
  using namespace details;
  constexpr nint_t NWo = num_words(t);
  if constexpr (NWo > 1) {
    Half<T> th;
    auto u_lo = vec::even(t, v_lo);
    auto u_hi = vec::even(t, v_hi);
    Vec<T> v_o;
    foreach<NWo / 2>([&]<nint_t I>{
      v_o = set_word<I>(t, v_o, get_word<I>(th, u_lo));
    });
    foreach<NWo / 2>([&]<nint_t I>{
      v_o = set_word<NWo / 2 + I>(t, v_o, get_word<I>(th, u_hi));
    });
    return v_o;
  } else {
    return word::concat_even(t, v_lo, v_hi);
  }
}

/**
 * @brief Extract odd-indexed elements from two vectors and concatenate.
 *
 * Extracts elements at odd indices from both input vectors and concatenates
 * them into a single output vector. The result has the same total element
 * count as each input vector.
 *
 * For input vectors v_lo and v_hi of size N:
 *   result[i] = v_lo[2*i + 1]     for i < N/2
 *   result[i] = v_hi[2*(i-N/2) + 1] for i >= N/2
 *
 * @tparam T Vector tag type
 * @param t Vector tag
 * @param v_lo First input vector (lower half of result comes from its odd elements)
 * @param v_hi Second input vector (upper half of result comes from its odd elements)
 * @return Vector of size N containing odd elements from both inputs
 *
 * @note This operation is commonly used after local_interleave_upper to undo
 *       the interleaving. Combined with concat_even, these two operations can
 *       deinterleave a pair of interleaved vectors.
 *
 * @example
 *   Tag<float32_t, 8> t;
 *   auto lo = loadu(t, {0, 1, 2, 3, 4, 5, 6, 7});
 *   auto hi = loadu(t, {8, 9, 10, 11, 12, 13, 14, 15});
 *   auto v = concat_odd(t, lo, hi);  // v = [1, 3, 5, 7, 9, 11, 13, 15]
 */
template <TLV_DECL_TAG(T)>
TLV_INLINE Vec<T> concat_odd(T t, Vec<T> v_lo, Vec<T> v_hi) {
  using namespace details;
  constexpr nint_t NWo = num_words(t);
  if constexpr (NWo > 1) {
    Half<T> th;
    auto u_lo = vec::odd(t, v_lo);
    auto u_hi = vec::odd(t, v_hi);
    Vec<T> v_o;
    foreach<NWo / 2>([&]<nint_t I>{
      v_o = set_word<I>(t, v_o, get_word<I>(th, u_lo));
    });
    foreach<NWo / 2>([&]<nint_t I>{
      v_o = set_word<NWo / 2 + I>(t, v_o, get_word<I>(th, u_hi));
    });
    return v_o;
  } else {
    return word::concat_odd(t, v_lo, v_hi);
  }
}

/**
 * @brief Interleave elements from the lower half of each 16-byte block.
 *
 * Performs element-wise interleaving within each 16-byte block, using only
 * elements from the lower 8 bytes of each input vector. The operation is
 * local to each 16-byte block (no cross-block data movement).
 *
 * For float32 x 8 (two 16-byte blocks):
 *   Input:  a = [a7, a6, a5, a4, a3, a2, a1, a0], b = [b7, b6, b5, b4, b3, b2, b1, b0]
 *   Result: [b5, a5, b4, a4, b1, a1, b0, a0]
 *
 * More generally, for each 16-byte block (M = 16 / sizeof(element)):
 *   result[2*i]   = a[i]     for i < M/2
 *   result[2*i+1] = b[i]     for i < M/2
 *
 * @tparam V Input/output vector type
 * @tparam T Vector tag type
 * @param a First input vector
 * @param b Second input vector
 * @return Interleaved vector containing lower-half elements from both inputs
 *
 * @note This operation stays within each 16-byte block, making it efficient
 *       on x86 architectures. Use concat_even(a, b) to deinterleave the result.
 *
 * @see local_interleave_upper for interleaving upper-half elements
 * @see interleave for full-vector interleaving (may be slower due to cross-block movement)
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V local_interleave_lower(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [&](auto tt, auto&& aa, auto&& bb) { return word::local_interleave_lower(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Interleave elements from the upper half of each 16-byte block.
 *
 * Performs element-wise interleaving within each 16-byte block, using only
 * elements from the upper 8 bytes of each input vector. The operation is
 * local to each 16-byte block (no cross-block data movement).
 *
 * For float32 x 8 (two 16-byte blocks):
 *   Input:  a = [a7, a6, a5, a4, a3, a2, a1, a0], b = [b7, b6, b5, b4, b3, b2, b1, b0]
 *   Result: [b7, a7, b6, a6, b3, a3, b2, a2]
 *
 * More generally, for each 16-byte block (M = 16 / sizeof(element)):
 *   result[2*i]   = a[M/2 + i]     for i < M/2
 *   result[2*i+1] = b[M/2 + i]     for i < M/2
 *
 * @tparam V Input/output vector type
 * @tparam T Vector tag type
 * @param a First input vector
 * @param b Second input vector
 * @return Interleaved vector containing upper-half elements from both inputs
 *
 * @note This operation stays within each 16-byte block, making it efficient
 *       on x86 architectures. Use concat_odd(a, b) to deinterleave the result.
 *
 * @see local_interleave_lower for interleaving lower-half elements
 * @see interleave for full-vector interleaving (may be slower due to cross-block movement)
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V local_interleave_upper(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [&](auto tt, auto&& aa, auto&& bb) { return word::local_interleave_upper(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Interleave elements from two half-length vectors into a full vector.
 *
 * Performs full interleaving of two half-length vectors, producing a vector
 * of twice the length where elements alternate between the two inputs.
 *
 * Result: [v_hi[N/2-1], v_lo[N/2-1], ..., v_hi[1], v_lo[1], v_hi[0], v_lo[0]]
 *
 * More formally: result[2*i] = v_lo[i], result[2*i+1] = v_hi[i]
 *
 * @tparam T Output vector tag type
 * @tparam V Input vector type (Vec<Half<T>>)
 * @param t Output vector tag
 * @param v_lo First input vector (placed at even indices in result)
 * @param v_hi Second input vector (placed at odd indices in result)
 * @return Interleaved vector of size N
 *
 * @warning This operation may involve cross-block data movement on x86 AVX/AVX2,
 *          which can result in performance penalties compared to local_interleave_*
 *          operations. For better performance on these architectures, consider
 *          using local_interleave_lower/local_interleave_upper instead.
 *
 * @note This is the inverse operation of even() and odd() combined:
 *       interleave(t, even(t, v), odd(t, v)) == v
 *
 * @example
 *   Tag<float32_t, 8> t;
 *   Tag<float32_t, 4> th;
 *   auto lo = loadu(th, {0, 1, 2, 3});
 *   auto hi = loadu(th, {4, 5, 6, 7});
 *   auto v = interleave(t, lo, hi);  // v = [4, 0, 5, 1, 6, 2, 7, 3] (MSB to LSB)
 */
template <TLV_DECL_TAG(T), typename V = Vec<Half<T>>>
TLV_INLINE Vec<T> interleave(T t, V v_lo, V v_hi) {
  using namespace details;
  using Ti = Vec2Tag<V>;
  using TWo = WordOf<T>;
  constexpr nint_t NWo = num_words(t);
  if constexpr (num_words(t) > 1) {
    static_assert(NWo == 2 * num_words(Ti{}));
    Vec<T> v_o;
    foreach<NWo / 2>([&]<nint_t I>{
      auto vi_lo = get_word<I>(Ti(), v_lo);
      auto vi_hi = get_word<I>(Ti(), v_hi);
      v_o = set_word<2 * I>(t, v_o, word::interleave(
          TWo(), word::lower(TWo(), vi_lo), word::lower(TWo(), vi_hi)
      ));
      v_o = set_word<2 * I + 1>(t, v_o, word::interleave(
          TWo(), word::upper(TWo(), vi_lo), word::upper(TWo(), vi_hi)
      ));
    });
    return v_o;
  } else {
    return word::interleave(t, v_lo, v_hi);
  }
}

/**
 * @brief Interleave even-indexed elements from two vectors.
 *
 * Takes elements at even indices from both input vectors and interleaves them.
 * The result has half the length of each input.
 *
 * For vectors a and b of size N:
 *   result[2*i]   = a[2*i]
 *   result[2*i+1] = b[2*i]
 *   for i in [0, N/2)
 *
 * @tparam V Input/output vector type
 * @tparam T Vector tag type
 * @param a First input vector
 * @param b Second input vector
 * @return Vector of size N/2 containing interleaved even elements
 *
 * @note This is the inverse operation of concat_even for deinterleaving.
 *       Combined with interleave_odd, these can separate interleaved data.
 *
 * @example
 *   Tag<float32_t, 8> t;
 *   auto a = loadu(t, {0, 1, 2, 3, 4, 5, 6, 7});
 *   auto b = loadu(t, {8, 9, 10, 11, 12, 13, 14, 15});
 *   auto v = interleave_even(a, b);  // v = [8, 0, 10, 2, 12, 4, 14, 6]
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V interleave_even(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [&](auto tt, auto&& aa, auto&& bb) { return word::interleave_even(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}

/**
 * @brief Interleave odd-indexed elements from two vectors.
 *
 * Takes elements at odd indices from both input vectors and interleaves them.
 * The result has half the length of each input.
 *
 * For vectors a and b of size N:
 *   result[2*i]   = a[2*i + 1]
 *   result[2*i+1] = b[2*i + 1]
 *   for i in [0, N/2)
 *
 * @tparam V Input/output vector type
 * @tparam T Vector tag type
 * @param a First input vector
 * @param b Second input vector
 * @return Vector of size N/2 containing interleaved odd elements
 *
 * @note This is the inverse operation of concat_odd for deinterleaving.
 *       Combined with interleave_even, these can separate interleaved data.
 *
 * @example
 *   Tag<float32_t, 8> t;
 *   auto a = loadu(t, {0, 1, 2, 3, 4, 5, 6, 7});
 *   auto b = loadu(t, {8, 9, 10, 11, 12, 13, 14, 15});
 *   auto v = interleave_odd(a, b);  // v = [9, 1, 11, 3, 13, 5, 15, 7]
 */
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V interleave_odd(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [&](auto tt, auto&& aa, auto&& bb) { return word::interleave_odd(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}


/* ************************************************************************** */
//                       Data type & size conversions                         //
/* ************************************************************************** */

/**
 * @brief Promote elements to a larger type.
 *
 * Converts elements to a larger type (e.g., int16_t -> int32_t, float32_t -> float64_t).
 * Output element size must be larger than input element size. Input vector must
 * have sufficient elements to fill the output vector.
 *
 * Conversion follows C++ standard promotion rules.
 *
 * @tparam To Target type tag
 * @param t Target vector tag
 * @param v Input vector
 * @return Promoted vector
 *
 * @note Assumes byte size of a word in input and output vectors is consistent.
 */
template <TLV_DECL_TAG(To), TLV_DECL_VEC(Vi)>
TLV_INLINE Vec<To> promote(To t, Vi v) {
  using namespace details;
  using     Ti       = Vec2Tag<Vi>;
  constexpr Ti   t_i;                  constexpr To   t_o;
  constexpr auto NWi = num_words(t_i); constexpr auto NWo = num_words(t_o);
  using     TWi      = WordOf<Ti>;     using     TWo      = WordOf<To>;
  using     Ei       = TypeOf<Ti>;     using     Eo       = TypeOf<To>;

  static_assert(sizeof(Ei) < sizeof(Eo));
  static_assert(!(is_scalable(t_i) ^ is_scalable(t_o)));

  constexpr nint_t factor = sizeof(Eo) / sizeof(Ei);
  // required number of words from input vector
  constexpr nint_t nw_i_r = is_scalable(t_i)
      ? (NWo + factor - 1) / factor
      : (To::AdjustedN + TWi::N - 1) / TWi::N;
  static_assert(nw_i_r <= NWi, "Insufficient elements");

  Vec<To> v_o;
  if constexpr (NWo > 1) {
    foreach<nw_i_r>([&]<nint_t I>{
      auto v_i = get_word<I>(t_i, v);
      auto v_bo_raw = word::promote(Rebind<Eo, TWi>(), v_i);
      // batch output
      using TBo = Tag<Eo, TWo::N, log2_floor(factor)>;
      static_assert(num_words(TBo()) == factor, "Output element count mismatch");
      auto v_bo = word::reshape(TBo(), v_bo_raw);

      foreach<factor>([&]<nint_t J>{
        v_o = set_word<I * factor + J>(t_o, v_o, get_word<J>(TBo(), v_bo));
      });
    });
  } else {
    static_assert(nw_i_r == 1);
    auto v_i = get_word<0>(t_i, v);
    auto v_bo = word::promote(TWo(), v_i);
    v_o = set_word<0>(t_o, v_o, v_bo);
  }
  return v_o;
}

/**
 * @brief Demote elements to a smaller type.
 *
 * Converts elements to a smaller type (e.g., int32_t -> int16_t, float64_t -> float32_t).
 * Output element size must be smaller than input element size. Input vector must
 * have sufficient elements to fill the output vector.
 *
 * Demotion rules:
 * - larger int -> smaller int: value clamped to target range
 * - int -> float: standard conversion
 * - float -> signed int: standard conversion
 * - float -> unsigned int: standard conversion (negative input undefined)
 *
 * @tparam To Target type tag
 * @param t Target vector tag
 * @param v Input vector
 * @return Demoted vector
 */
template <TLV_DECL_TAG(To), TLV_DECL_VEC(Vi)>
TLV_INLINE Vec<To> demote(To t, Vi v) {
  using namespace details;
  using     Ti       = Vec2Tag<Vi>;
  constexpr Ti   t_i;                  constexpr To   t_o;
  constexpr auto NWi = num_words(t_i); constexpr auto NWo = num_words(t_o);
  using     TWi      = WordOf<Ti>;     using     TWo      = WordOf<To>;
  using     Ei       = TypeOf<Ti>;     using     Eo       = TypeOf<To>;

  static_assert(sizeof(Ei) > sizeof(Eo));
  static_assert(!(is_scalable(t_i) ^ is_scalable(t_o)));

  constexpr nint_t factor = sizeof(Ei) / sizeof(Eo);
  constexpr nint_t nw_i_r = is_scalable(t_i)
      ? (NWo + factor - 1) / factor
      : (To::AdjustedN + TWi::N - 1) / TWi::N;
  static_assert(nw_i_r <= NWi, "Insufficient elements");

  Vec<To> v_o;
  if constexpr (NWo > 1) {
    static_assert (nw_i_r == NWo * factor);
    foreach<NWo>([&]<nint_t I>{
      // batch input
      using TBi = Tag<Ei, TWi::N, log2_floor(factor)>;
      auto v_bi = vmap(TBi(), [&]<nuint_t J>(auto tt){
        return get_word<I * factor + J>(t_i, v);
      });
      auto v_bo = word::demote(TWo(), v_bi);
      v_o = set_word<I>(t_o, v_o, v_bo);
    });
  } else {
    static_assert(nw_i_r <= factor);
    // minibatch input
    using TBmi = Tag<Ei, TWi::N, log2_floor(nw_i_r)>;
    auto v_bi = vmap(TBmi(), [&]<nuint_t J>(auto tt){
      return get_word<J>(t_i, v);
    });
    auto v_bo = word::demote(TWo(), v_bi);
    v_o = set_word<0>(t_o, v_o, v_bo);
  }
  return v_o;
}

/**
 * @brief Convert elements to a type of the same size.
 *
 * Converts between types of equal size (e.g., int32_t <-> float32_t).
 * Input vector size must not be smaller than output vector size.
 * Conversion follows C++ standard rules.
 *
 * @tparam To Target type tag
 * @param t Target vector tag
 * @param v Input vector
 * @return Converted vector
 */
template <TLV_DECL_TAG(To), TLV_DECL_VEC(Vi)>
TLV_INLINE Vec<To> convert(To t, Vi v) {
  using namespace details;
  constexpr Vec2Tag<Vi> ti;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::convert(tt, vv); },
      ShardVec(ti, v)
  );
}

/**
 * @brief Reinterpret cast vector bits to a different type.
 *
 * Reinterprets the bit pattern of the input vector as a different type
 * without any conversion. If output is smaller, high elements are discarded.
 * If output is larger, high elements are filled with undefined values.
 *
 * @tparam To Target type tag
 * @param t Target vector tag
 * @param v Input vector
 * @return Reinterpreted vector
 */
template <TLV_DECL_TAG(To), TLV_DECL_VEC(Vi)>
TLV_INLINE Vec<To> bitcast(To t, Vi v) {
  using namespace details;
  using     Ti       = Vec2Tag<Vi>;
  constexpr Ti   t_i;                  constexpr To   t_o;
  constexpr auto NWi = num_words(t_i); constexpr auto NWo = num_words(t_o);
  using     TWi      = WordOf<Ti>;     using     TWo      = WordOf<To>;

  static_assert(!(is_scalable(t_i) ^ is_scalable(t_o)));
  constexpr nint_t n_cpy_iter = std::min(NWi, NWo);

  Vec<To> v_o;
  foreach<n_cpy_iter>([&]<nint_t I>{
    v_o = set_word<I>(t_o, v_o, word::bitcast(TWo(), get_word<I>(t_i, v)));
  });
  return v_o;
}

} // namespace ct::tl::vec

#endif //CTORCH_VEC_H
