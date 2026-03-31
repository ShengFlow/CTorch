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
 * Create a vector with first n elements filled with a single value. other sets to default_v
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
 * Create a vector where masked lanes set to single value, other sets to default_v
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

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V add(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::add(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V add(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::add(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V sub(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::sub(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V sub(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::sub(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V mul(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::mul(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V mul(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::mul(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V div(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::div(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V div(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::div(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V max(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::max(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V max(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::max(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V min(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::min(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V min(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::min(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_and(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::bit_and(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_and(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::bit_and(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_or(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::bit_or(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_or(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::bit_or(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_xor(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::bit_xor(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_xor(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::bit_xor(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

// not(a) and (b)
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_andnot(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::bit_andnot(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_andnot(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::bit_andnot(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_shl(V v, int count) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::bit_shl(vv, count); },
      ShardVec(t, v)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_shl(V v, int count, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm) { return word::bit_shl(vv, count, mm); },
      ShardVec(t, v), ShardMask(t, m)
  );
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_shr(V v, int count) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::bit_shr(vv, count); },
      ShardVec(t, v)
  );
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_shr(V v, int count, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm) { return word::bit_shr(vv, count, mm); },
      ShardVec(t, v), ShardMask(t, m)
  );
}


template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_not(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::bit_not(vv); },
      ShardVec(t, v)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_not(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm, auto&& dd) { return word::bit_not(vv, mm, dd); },
      ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V bit_not(V v, Mask<T> m) {
  return vec::bit_not(v, m, v);
}

template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V neg(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::neg(vv); },
      ShardVec(t, v)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V neg(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm, auto&& dd) { return word::neg(vv, mm, dd); },
      ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V neg(V v, Mask<T> m) {
  return vec::neg(v, m, v);
}


template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V abs(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::abs(vv); },
      ShardVec(t, v)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V abs(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm, auto&& dd) { return word::abs(vv, mm, dd); },
      ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V abs(V v, Mask<T> m) {
  return vec::abs(v, m, v);
}


template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V sqrt(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::sqrt(vv); },
      ShardVec(t, v)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V sqrt(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm, auto&& dd) { return word::sqrt(vv, mm, dd); },
      ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V sqrt(V v, Mask<T> m) {
  return vec::sqrt(v, m, v);
}


template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V rsqrt(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::rsqrt(vv); },
      ShardVec(t, v)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V rsqrt(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm, auto&& dd) { return word::rsqrt(vv, mm, dd); },
      ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V rsqrt(V v, Mask<T> m) {
  return vec::rsqrt(v, m, v);
}


template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V rcp(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::rcp(vv); },
      ShardVec(t, v)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V rcp(V v, Mask<T> m, V default_v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm, auto&& dd) { return word::rcp(vv, mm, dd); },
      ShardVec(t, v), ShardMask(t, m), ShardVec(t, default_v)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE V rcp(V v, Mask<T> m) {
  return vec::rcp(v, m, v);
}


template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpeq(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::cmpeq(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpeq(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::cmpeq(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}


template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpne(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::cmpne(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpne(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::cmpne(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}


template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmplt(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::cmplt(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmplt(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::cmplt(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}


template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpgt(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::cmpgt(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpgt(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::cmpgt(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}


template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmple(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::cmple(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmple(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::cmple(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}


template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpge(V a, V b) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb) { return word::cmpge(aa, bb); },
      ShardVec(t, a), ShardVec(t, b)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> cmpge(V a, V b, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& aa, auto&& bb, auto&& mm) { return word::cmpge(aa, bb, mm); },
      ShardVec(t, a), ShardVec(t, b), ShardMask(t, m)
  );
}


template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> isnan(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::isnan(vv); },
      ShardVec(t, v)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> isnan(V v, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm) { return word::isnan(vv, mm); },
      ShardVec(t, v), ShardMask(t, m)
  );
}


template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> isposinf(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::isposinf(vv); },
      ShardVec(t, v)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> isposinf(V v, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm) { return word::isposinf(vv, mm); },
      ShardVec(t, v), ShardMask(t, m)
  );
}


template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> isneginf(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::isneginf(vv); },
      ShardVec(t, v)
  );
}
template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> isneginf(V v, Mask<T> m) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv, auto&& mm) { return word::isneginf(vv, mm); },
      ShardVec(t, v), ShardMask(t, m)
  );
}


template <TLV_DECL_VEC(V), typename T = Vec2Tag<V>>
TLV_INLINE Mask<T> isinf(V v) {
  using namespace details;
  constexpr T t;
  return vmap(
      t, [=](auto tt, auto&& vv) { return word::isinf(vv); },
      ShardVec(t, v)
  );
}
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
 * 使用固定下标Is对向量v中每一个元素在（16字节）块（即一个x86的lane）内进行重排。下标数量与块内元素数量对应。
 * 即float32_t下块内4个元素，则下标长度应为4，下标应属于范围[0, 3].
 * 在此函数下所有的块均会应用相同的下标进行重排。
 * 小于一个字长(word_size)的向量会按照一个字长的向量进行处理，因此如果字长为16，Vec<Tag<float32_t, 2>>也会要求有4个下标。
 * 重排结果:
 *      result[j] = v[Is[j % M] + floor(j / M)]
 *      where j = 0...N;
 *            M = 16 / sizeof(element type of V).
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
 * 使用i中对应位置的下标对向量v中每一个元素在（16字节）块（即一个x86的lane）内进行重排。下标数量与块内元素数量对应。
 * 即下标向量i元素类型应为与v元素类型宽度相同的有符号整数，且i的位宽与v完全相同。
 * 在此函数下不同的块内应用的下标可以不同。
 * 重排结果：
 *      result[j] = v[i[j] + floor(j / M)]
 *      where j = 0...N;
 *            M = 16 / sizeof(element type of V);
 *            i[j] in [0, M), 否则result[j]未定义.
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
 * 使用运行时下标is对向量v中每一个元素在（16字节）块（即一个x86的lane）内进行重排。要求和个规则与使用固定下标的版本完全相同。
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
 * 使用i中对应位置的下标对v中每一个元素在字(word)内进行重排。下标数量与块内元素数量对应。
 * 即下标向量i元素类型应为与v元素类型宽度相同的有符号整数，且i的位宽与v完全相同。
 * 在此函数下不同的字内应用的下标可以不同（如果向量是多字的）。
 * 重排结果：
 *      result[j] = v[i[j]]
 *      where j = 0...N;
 *      i[j] in [0, N), 否则result[j]未定义.
 * 注：x86 AVX2往上（存在多块向量）下此操作涉及块间数据传递，比块内重排慢。同时，如果没有
 * AVX512，则此操作会更慢。而如果元素类型为int8_t/uint8_t，且没有AVX512_VBMI特性，
 * 则即是有AVX512，此操作会慢于其他更宽的数据类型。
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

/* ************************************************************************** */
//                       Data type & size conversions                         //
/* ************************************************************************** */

/**
 * We assume that byte size of a word in input vector v and output vector is consistent.
 */
/**
 * Size of input vector must not smaller than requested output
 * Size of output dtype must be larger than size of input dtype.
 * Promotion rules:
 *   - all type conversions conform C++ standard.
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
 * Size of input vector must not smaller than requested output
 * Size of output dtype must be smaller than size of input dtype.
 * Demotion rules:
 *   - larger int -> smaller int (whether signed or unsigned): value will be clamped to target range.
 *   - int -> float: standard int/float conversion
 *   - float -> signed int: standard float/int conversion
 *   - float -> unsigned int: standard float/int conversion (result for negative input undefined)
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
 * Size of input vector must not smaller than requested output
 * Size of output dtype must be equals to size of input dtype.
 * Conversion rules:
 *   - all type conversions conform C++ standard.
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
 * 将输入v按位重新解释成标签t的形式。如果t长度小于v的长度，则v中高位元素会被丢弃。
 * 如果t长度大于v的长度，则v高位会被填充未定义值的元素。
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
