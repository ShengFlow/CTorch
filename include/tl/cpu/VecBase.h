//
// Created by renyz on 2026/3/15.
//

#ifndef CTORCH_VECBASE_H
#define CTORCH_VECBASE_H

#include <array>
#include <bitset>

#include "Assertion.h"
#include "CoreDefs.h"
#include "tl/util/Math.h"
#include "tl/cpu/Capabilities.h"

/**
 * @file VecBase.h
 * @brief Foundation types and definitions for SIMD vector operations.
 *
 * This module provides a Highway-like vector abstraction layer that supports:
 * - Multiple SIMD architectures (x86 SSE/AVX, ARM NEON/SVE)
 * - Scalable vectors (runtime-determined size, e.g., SVE)
 * - Fixed-size vectors
 * - Multi-word vectors (concatenation of multiple hardware vectors)
 * - Scalar fallback implementation
 *
 * The design uses a Tag-based type system to specify vector properties at
 * compile time, enabling efficient code generation while maintaining portability.
 */

#if defined(HAS_SVE)
  #define _VEC_SIZE(type) VEC_WIDTH
#else
  #define _VEC_SIZE(type) (VEC_WIDTH / 8 / sizeof(type))
#endif

namespace ct::tl::vec {
namespace details {
/**
 * @brief Computes the size of a vector.
 * 
 * For fixed-size vectors (N >= 0), returns N directly.
 * For scalable vectors (N < 0), returns the runtime-determined size.
 * 
 * @tparam N The nominal size (negative for scalable vectors)
 * @tparam POW2 Power-of-2 multiplier (not used in this function)
 * @param scalable_size The actual size for scalable vectors (default -1)
 * @return The vector size
 */
template <nint_t N, int POW2>
static constexpr nint_t size(nint_t scalable_size = -1) {
  if constexpr (N < 0) {
    return scalable_size;
  } else {
    return N;
  }
}

/**
 * @brief Computes the adjusted size of a vector with POW2 scaling.
 * 
 * The actual vector size is N * 2^POW2.
 * - POW2 > 0: Creates a larger vector (multi-word)
 * - POW2 = 0: No adjustment
 * - POW2 < 0: Creates a smaller vector (partial word)
 * 
 * @tparam N The nominal size
 * @tparam POW2 Power-of-2 multiplier
 * @param scalable_size The actual size for scalable vectors
 * @return The adjusted vector size
 */
template <nint_t N, int POW2>
static constexpr nint_t adjusted_size(nint_t scalable_size = -1) {
  if constexpr (N < 0) {
    return scalable_size;
  } else {
    if (POW2 > 0)
      return N << POW2;
    else
      return N >> (-POW2);
  }
}

/**
 * Check if type T is supported element type for vector
 */
template <typename T>
struct IsElementType {
  static constexpr bool value = false;
};

template <> struct IsElementType<bfloat16_t> { static constexpr bool value = true; };
template <> struct IsElementType<float16_t> { static constexpr bool value = true; };
template <> struct IsElementType<float32_t> { static constexpr bool value = true; };
template <> struct IsElementType<float64_t> { static constexpr bool value = true; };
template <> struct IsElementType<int8_t> { static constexpr bool value = true; };
template <> struct IsElementType<uint8_t> { static constexpr bool value = true; };
template <> struct IsElementType<int16_t> { static constexpr bool value = true; };
template <> struct IsElementType<uint16_t> { static constexpr bool value = true; };
template <> struct IsElementType<int32_t> { static constexpr bool value = true; };
template <> struct IsElementType<uint32_t> { static constexpr bool value = true; };
template <> struct IsElementType<int64_t> { static constexpr bool value = true; };
template <> struct IsElementType<uint64_t> { static constexpr bool value = true; };
} // namespace details

template <typename T>
static constexpr bool is_element_type = details::IsElementType<T>::value;

/**
 * @brief Type tag that describes vector properties.
 * 
 * A Tag encodes the element type, nominal size, and size multiplier (POW2)
 * for a vector. It is used throughout the vector API to specify vector types
 * at compile time.
 * 
 * @tparam T Element type (e.g., float32_t, int32_t)
 * @tparam N_ Nominal number of elements. Must be a power of 2, or -1 for scalable vectors.
 * @tparam POW2_ Size multiplier exponent. Actual size = N_ * 2^POW2_.
 *               - POW2 > 0: Multi-word vector (concatenation of 2^POW2 hardware vectors)
 *               - POW2 = 0: Single hardware vector
 *               - POW2 < 0: Partial vector (only supported for fixed-size N)
 * 
 * @example
 *   Tag<float32_t, 4>           // 4 floats (single 128-bit SSE register)
 *   Tag<float32_t, 4, 1>        // 8 floats (two 128-bit SSE registers)
 *   Tag<float32_t, 4, 2>        // 16 floats (four 128-bit SSE registers)
 *   Tag<float32_t, -1>          // Scalable vector (SVE-style)
 */
template <typename T, nint_t N_, int POW2_ = 0>
struct Tag {
  using Type = T;
  static constexpr nint_t N = details::size<N_, POW2_>(N_);
  static constexpr nint_t AdjustedN = details::adjusted_size<N_, POW2_>();
  static constexpr int POW2 = POW2_;
  static constexpr bool is_runtime_size = N_ < 0;
  static constexpr nint_t Bytes = is_runtime_size ? (-1) : AdjustedN * sizeof(T);

  static_assert(N_ < 0 || ((N_ & (N_ - 1)) == 0 && N_ != 0), "Not 2^x");
  static_assert(is_element_type<T>, "Unsupported element type");
};

/**
 * @brief Tag for a scalable vector with hardware-determined size.
 * 
 * The vector size is determined at runtime based on the target SIMD width
 * and element type. Useful for writing portable code that adapts to different
 * hardware capabilities.
 * 
 * @tparam T Element type
 * @tparam POW2 Optional size multiplier (default 0)
 */
template <typename T, int POW2 = 0>
using ScalableTag = Tag<T, _VEC_SIZE(T), POW2>;

/**
 * @brief Tag for a fixed-size vector.
 * 
 * @tparam T Element type
 * @tparam N Number of elements (must be power of 2)
 */
template <typename T, nint_t N>
using FixedTag = Tag<T, N, 0>;

namespace details {
template <typename T>
struct IsTag {
  static constexpr bool value = false;
};

template <typename T, nint_t N, int POW2>
struct IsTag<Tag<T, N, POW2>> {
  static constexpr bool value = true;
};
} // namespace details

template <typename T>
static constexpr bool is_tag = details::IsTag<T>::value;

/**
 * @brief Default memory alignment for vector types (16 bytes if is scalable).
 * Note: VEC_WIDTH always smaller than SIMD_WIDTH, to ensure best compatibility
 *   we intentionally use SIMD_WIDTH here.
 */
#if SIMD_WIDTH < 0
static constexpr nint_t DEFAULT_ALIGNMENT = 16;
#else
static constexpr nint_t DEFAULT_ALIGNMENT = std::max<nint_t>(8, SIMD_WIDTH / 8);
#endif

/**
 * @brief Default vector length in bytes for scalar fallback implementation.
 */
static constexpr nint_t DEFAULT_LENGTH = VEC_WIDTH / 8;

/**
 * @brief Default number of elements for scalar vectors of type T.
 */
template <typename T>
static constexpr nint_t DEFAULT_SIZE = DEFAULT_LENGTH / sizeof(T);

/**
 * @brief Aligned array type used as the underlying storage for scalar vectors.
 * 
 * Provides std::array functionality with guaranteed memory alignment.
 * 
 * @tparam T Element type
 * @tparam N Number of elements
 */
template <typename T, nint_t N>
class alignas(DEFAULT_ALIGNMENT) ScalarArray : public std::array<T, N> {};

/**
 * @brief Bitset-based mask type for scalar implementation.
 * 
 * This is a special std::bitset wrapper that can be distinguished from
 * other types via the ELSIZE template parameter.
 * 
 * @tparam ELSIZE Size of the element type in bytes (for type distinction)
 * @tparam N Number of elements (bits)
 */
template <nint_t ELSIZE, nint_t N>
class ScalarBitSet : public std::bitset<N> {};

/**
 * @brief Base class providing vector type definitions and operations.
 * 
 * This template defines the interface that all vector implementations must follow.
 * Architecture-specific specializations inherit from this class and provide
 * actual implementations for SIMD operations.
 * 
 * @tparam T Element type
 * @tparam N Nominal vector size (negative for scalable vectors)
 * @tparam POW2 Size multiplier exponent
 * 
 * Key concepts:
 * - Word: A single hardware vector register (e.g., one XMM register = 128 bits)
 * - Multi-word vector: A vector composed of multiple words (POW2 > 0)
 * - Scalable vector: A vector whose size is determined at runtime (SVE)
 */
template <
    typename T, nint_t N, int POW2 = 0
>
struct BaseVecDefs {
  using TagType = Tag<T, N, POW2>;
  
  /**
   * @brief Number of hardware vector words in this vector.
   * 
   * For single-word vectors, this is 1.
   * For multi-word vectors (POW2 > 0), this is 2^POW2.
   */
  static constexpr nint_t num_words = -1; // placeholder
  
  /**
   * @brief Number of elements in one hardware word.
   * 
   * May be a runtime value for scalable vectors.
   * For multi-word vectors, this represents the word size, not the total size.
   * 
   * @return Number of elements per word
   */
  static constexpr nint_t word_size() { return -1; }

  /**
   * @brief Maximum (upper bound) number of elements per word.
   * 
   * For fixed-size vectors, equal to word_size().
   * For scalable vectors, provides a compile-time upper bound.
   */
  static constexpr nint_t max_word_size = -1; // placeholder
  
  /**
   * @brief Total number of elements in this vector.
   * 
   * May be a runtime value for scalable vectors.
   * For multi-word vectors, this equals word_size() * num_words.
   * 
   * @return Total number of elements
   */
  static constexpr nint_t size() { return -1; };
  
  /**
   * @brief Maximum (upper bound) total size of this vector.
   */
  static constexpr nint_t max_size = -1;
  
  /**
   * @brief Whether this vector has runtime-determined size.
   */
  static constexpr bool is_scalable = TagType::is_runtime_size;
  
  /**
   * @brief Whether this is a scalar fallback implementation.
   * 
   * If true, operations use scalar loops rather than SIMD instructions.
   */
  static constexpr bool is_default_impl = false;  // default sets to false
  
  /**
   * @brief Whether this vector is exactly one hardware word.
   */
  static constexpr bool is_word_vec = true;
  
  /**
   * @brief Underlying vector type.
   * 
   * For SIMD implementations, this is typically an intrinsic type (e.g., __m128).
   * For scalar implementation, this is ScalarArray<T, N>.
   * 
   * @warning For SVE, this type may be sizeless and cannot be stored in memory
   *          or have its address taken. It can only be used for local variables
   *          and function parameters.
   */
  using VecType = int; // placeholder
  
  /**
   * @brief Underlying mask type for conditional operations.
   * 
   * For AVX-512, this is a bitmask type (__mmask8, __mmask16, etc.).
   * For other architectures, this is ScalarBitSet.
   * For SVE, this is a predicate register type.
   * 
   * @warning For SVE, this type may be sizeless (see VecType warning).
   */
  using MaskType = int; // placeholder
  
  /**
   * @brief VecDefs for the underlying word type.
   * 
   * For single-word vectors, this refers to itself.
   * For multi-word vectors, this refers to the definition of one word.
   */
  using WordDefs = std::conditional_t<is_scalable,
      BaseVecDefs<T, N, 0>, BaseVecDefs<T, word_size(), 0>
  >;

  /**
   * @brief Get the Index-th word from a vector.
   * 
   * @tparam Index Word index (0 to num_words-1)
   * @param v The vector
   * @return The specified word
   */
  template <nint_t Index>
  static typename WordDefs::VecType get(VecType v) { return v; };

  /**
   * @brief Get a word from a vector by runtime index.
   * 
   * @param v The vector
   * @param index Word index (0 to num_words-1)
   * @return The specified word
   */
  static typename WordDefs::VecType get(VecType v, nint_t index) { return v; };

  /**
   * @brief Set the Index-th word of a vector.
   * 
   * @tparam Index Word index (0 to num_words-1)
   * @param v The original vector
   * @param u The new word value
   * @return Vector with the word updated
   */
  template <nint_t Index>
  static VecType set(VecType v, typename WordDefs::VecType u) { return v; };

  /**
   * @brief Set a word of a vector by runtime index.
   * 
   * @param v The original vector
   * @param index Word index (0 to num_words-1)
   * @param u The new word value
   * @return Vector with the word updated
   */
  static VecType set(VecType v, nint_t index, typename WordDefs::VecType u) { return v; };


  template <nint_t Index>
  static typename WordDefs::MaskType get_mask(MaskType m) { return m; }

  static typename WordDefs::MaskType get_mask(MaskType m, nint_t index) { return m; };

  template <nint_t Index>
  static MaskType set_mask(MaskType m, typename WordDefs::MaskType u) { return m; }

  static MaskType set_mask(MaskType m, nint_t index, typename WordDefs::MaskType u) { return m; };
}; // struct BaseVecDefs

/**
 * @brief Scalar (fallback) vector implementation definition.
 * 
 * This is the default implementation used when no SIMD implementation
 * is available for the target architecture/type combination. It uses
 * ScalarArray and ScalarBitSet as the underlying storage.
 */
template <typename T, nint_t N, int P = 0, typename = void /*  SFINAE */>
struct ScalarVecDefs {};

/**
 * @brief Main vector definition template.
 * 
 * Specialized for each architecture and type combination to provide
 * SIMD-optimized implementations. Falls back to ScalarVecDefs if no
 * specialization exists.
 */
template <typename T, nint_t N, int POW2 = 0, typename = void /* SFINAE */>
struct VecDefs : public ScalarVecDefs<T, N, POW2> {};

/**
 * @brief Scalar implementation for single-word vectors (POW2 = 0).
 * 
 * Uses DEFAULT_SIZE elements by default if N is not specified by
 * architecture-specific specialization.
 */
template <typename T, nint_t N>
struct ScalarVecDefs<T, N, 0> : public BaseVecDefs<T, N, 0> {
private:
  static constexpr nint_t ADJUSTED_SIZE = details::adjusted_size<N, 0>(DEFAULT_SIZE<T>);

public:
  using TagType = Tag<T, ADJUSTED_SIZE>;
  static constexpr nint_t num_words = 1;
  static constexpr nint_t word_size() { return ADJUSTED_SIZE; }
  static constexpr nint_t max_word_size = ADJUSTED_SIZE;
  static constexpr nint_t size() { return ADJUSTED_SIZE; }
  static constexpr nint_t max_size = ADJUSTED_SIZE;
  static constexpr bool is_scalable = false;
  static constexpr bool is_default_impl = true;
  static constexpr bool is_word_vec = true;
  using VecType = ScalarArray<T, ADJUSTED_SIZE>;
  using MaskType = ScalarBitSet<sizeof(T), ADJUSTED_SIZE>;
  using WordDefs = ScalarVecDefs<T, ADJUSTED_SIZE>;

  static_assert(std::is_same_v<VecType, typename WordDefs::VecType>);

  template <nint_t Index>
  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto get(auto v) {
    static_assert(Index == 0, "Static index out of range");
    return v;
  }

  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto get(auto v, nint_t index) {
    CT_ASSERT(index == 0, "%lld !in 0..1", index);
    return v;
  };

  template <nint_t Index>
  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto set(auto v, auto u) {
    static_assert(Index == 0, "Static index out of range");
    return u;
  }

  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto set(auto v, nint_t index, auto u) {
    CT_ASSERT(index == 0, "%lld !in 0..1", index);
    return u;
  };

  template <nint_t Index>
  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto get_mask(auto m) {
    static_assert(Index == 0, "Static index out of range");
    return m;
  }

  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto get_mask(auto m, nint_t index) {
    CT_ASSERT(index == 0, "%lld !in 0..1", index);
    return m;
  };

  template <nint_t Index>
  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto set_mask(auto m, auto u) {
    static_assert(Index == 0, "Static index out of range");
    return u;
  }

  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto set_mask(auto m, nint_t index, auto u) {
    CT_ASSERT(index == 0, "%lld !in 0..1", index);
    return u;
  };
}; // class ScalarVecDefs

/**
 * @brief Scalar implementation for multi-word vectors (POW2 > 0).
 * 
 * A multi-word vector stores 2^POW2 hardware words in an array.
 * Operations on multi-word vectors iterate over each word.
 */
template <typename T, nint_t N, int POW2>
struct ScalarVecDefs<T, N, POW2, std::enable_if_t<(POW2 > 0)>> : BaseVecDefs<T, N, POW2> {
  using TagType = Tag<T, N, POW2>;
  using WordDefs = typename VecDefs<T, N, 0>::WordDefs;
  static_assert(!WordDefs::is_scalable);
  static constexpr nint_t word_size() { return WordDefs::word_size(); }
  static constexpr nint_t max_word_size = WordDefs::max_word_size;
  static constexpr nint_t size() { return details::adjusted_size<N, POW2>(); }
  static constexpr nint_t num_words = size() / WordDefs::size();
  static constexpr bool is_scalable = WordDefs::is_scalable;
  static constexpr bool is_default_impl = WordDefs::is_default_impl;
  static constexpr bool is_word_vec = false;
  using VecType = ScalarArray<typename WordDefs::VecType, num_words>;
  using MaskType = ScalarArray<typename WordDefs::MaskType, num_words>;

  template <nint_t Index>
  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto get(auto v) {
    static_assert(0 <= Index && Index < num_words, "Static index out of range");
    return v[Index];
  }

  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto get(auto v, nint_t index) {
    CT_ASSERT(0 <= index && index < num_words, "%lld !in 0..%lld", index, num_words);
    return v[index];
  };

  template <nint_t Index>
  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto set(auto v, auto u) {
    static_assert(0 <= Index && Index < num_words, "Static index out of range");
    auto r = v;
    r[Index] = u;
    return r;
  };

  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto set(auto v, nint_t index, auto u) {
    CT_ASSERT(0 <= index && index < num_words, "%lld !in 0..%lld", index, num_words);
    auto r = v;
    r[index] = u;
    return r;
  };

  template <nint_t Index>
  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto get_mask(auto m) {
    static_assert(0 <= Index && Index < num_words, "Static index out of range");
    return m[Index];
  }

  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto get_mask(auto m, nint_t index) {
    CT_ASSERT(0 <= index && index < num_words, "%lld !in 0..%lld", index, num_words);
    return m[index];
  };

  template <nint_t Index>
  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto set_mask(auto m, auto u) {
    static_assert(0 <= Index && Index < num_words, "Static index out of range");
    auto r = m;
    r[Index] = u;
    return r;
  }

  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto set_mask(auto m, nint_t index, auto u) {
    CT_ASSERT(0 <= index && index < num_words, "%lld !in 0..%lld", index, num_words);
    auto r = m;
    r[index] = u;
    return r;
  };
}; // class ScalarVecDefs

/**
 * @brief Scalar implementation for partial vectors (POW2 < 0).
 * 
 * A partial vector uses only a subset of the elements in a hardware word.
 * This is useful when you need a vector with fewer elements than the
 * hardware natively supports.
 * 
 * @warning This is only supported for fixed-size vectors (N >= 0).
 *          Scalable vectors with POW2 < 0 require architecture-specific support.
 */
template <typename T, nint_t N, int POW2>
struct ScalarVecDefs<T, N, POW2, std::enable_if_t<(POW2 < 0)>> : BaseVecDefs<T, N, POW2>
{
  static_assert(N >= 0, "Scalable implementation needs specialization!");

  using TagType = Tag<T, N, POW2>;
  using WordDefs = typename VecDefs<T, (N >> (-POW2))>::WordDefs;
  static_assert(!WordDefs::is_scalable);
  static constexpr nint_t num_words = 1;
  static constexpr nint_t word_size() { return WordDefs::word_size(); }
  static constexpr nint_t max_word_size = WordDefs::max_word_size;
  static constexpr nint_t size() { return WordDefs::size(); }
  static constexpr bool is_scalable = false; // WordDefs::is_Scalable
  static constexpr bool is_default_impl = WordDefs::is_default_impl;
  static constexpr bool is_word_vec = true;
  using VecType = typename WordDefs::VecType;
  using MaskType = typename WordDefs::MaskType;

  template <nint_t Index>
  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto get(auto v) {
    static_assert(Index == 0, "Static index out of range");
    return v;
  }

  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto get(auto v, nint_t index) {
    CT_ASSERT(index == 0, "%lld !in 0..1", index);
    return v;
  };

  template <nint_t Index>
  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto set(auto v, auto u) {
    static_assert(Index == 0, "Static index out of range");
    return u;
  }

  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto set(auto v, nint_t index, auto u) {
    CT_ASSERT(index == 0, "%lld !in 0..1", index);
    return u;
  };

  template <nint_t Index>
  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto get_mask(auto m) {
    static_assert(Index == 0, "Static index out of range");
    return m;
  }

  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto get_mask(auto m, nint_t index) {
    CT_ASSERT(index == 0, "%lld !in 0..1", index);
    return m;
  };

  template <nint_t Index>
  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto set_mask(auto m, auto u) {
    static_assert(Index == 0, "Static index out of range");
    return u;
  }

  CT_ALWAYS_FORCEINLINE CT_PURE
  static auto set_mask(auto m, nint_t index, auto u) {
    CT_ASSERT(index == 0, "%lld !in 0..1", index);
    return u;
  };
}; // class VecDefs

/**
 * @brief Returns the number of hardware words in a vector.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam POW2 Size multiplier
 * @param t Optional tag parameter (for type deduction)
 * @return Number of words
 */
template <typename T, nint_t N, int POW2>
CT_ALWAYS_FORCEINLINE CT_PURE
static constexpr nint_t num_words(Tag<T, N, POW2> t = {}) {
  return VecDefs<T, N, POW2>::num_words;
}

/**
 * @brief Returns the number of elements per hardware word.
 * 
 * @note For scalable vectors, this may be a runtime constant.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam POW2 Size multiplier
 * @param t Optional tag parameter
 * @return Number of elements per word
 */
template <typename T, nint_t N, int POW2>
CT_ALWAYS_FORCEINLINE CT_PURE
static constexpr nint_t word_size(Tag<T, N, POW2> t = {}) {
  return VecDefs<T, N, POW2>::word_size();
}

/**
 * @brief Returns the maximum number of elements per word.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam POW2 Size multiplier
 * @param t Optional tag parameter
 * @return Maximum elements per word
 */
template <typename T, nint_t N, int POW2>
CT_ALWAYS_FORCEINLINE CT_PURE
static constexpr nint_t max_word_size(Tag<T, N, POW2> t = {}) {
  return VecDefs<T, N, POW2>::max_word_size;
}

/**
 * @brief Returns the total number of elements in a vector.
 * 
 * @note For scalable vectors, this may be a runtime constant.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam POW2 Size multiplier
 * @param t Optional tag parameter
 * @return Total number of elements
 */
template <typename T, nint_t N, int POW2>
CT_ALWAYS_FORCEINLINE CT_PURE
static constexpr nint_t size(Tag<T, N, POW2> t = {}) {
  return VecDefs<T, N, POW2>::size();
}

/**
 * @brief Checks if a vector has runtime-determined size.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam POW2 Size multiplier
 * @param t Optional tag parameter
 * @return true if scalable, false if fixed-size
 */
template <typename T, nint_t N, int POW2>
CT_ALWAYS_FORCEINLINE CT_PURE
static constexpr bool is_scalable(Tag<T, N, POW2> t = {}) {
  return VecDefs<T, N, POW2>::is_scalable;
}

/**
 * @brief Checks if a vector uses scalar fallback implementation.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam POW2 Size multiplier
 * @param t Optional tag parameter
 * @return true if using scalar implementation, false if using SIMD
 */
template <typename T, nint_t N, int POW2>
CT_ALWAYS_FORCEINLINE CT_PURE
static constexpr bool is_default_impl(Tag<T, N, POW2> t = {}) {
  return VecDefs<T, N, POW2>::is_default_impl;
}

/**
 * @brief Checks if a vector is exactly one hardware word.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam POW2 Size multiplier
 * @param t Optional tag parameter
 * @return true if single-word, false if multi-word
 */
template <typename T, nint_t N, int POW2>
CT_ALWAYS_FORCEINLINE CT_PURE
static constexpr bool is_word_vec(Tag<T, N, POW2> t = {}) {
  return VecDefs<T, N, POW2>::is_word_vec;
}

namespace details {
/**
 * @brief Helper to determine the index type for a given element type.
 * 
 * The index type has the same size as the element type but is always signed.
 * Used for gather/scatter operations where indices must be signed integers.
 */
template <typename T, typename = void/*SFINAE*/>
struct IndexType { static_assert(sizeof(T) == -1, "Unsupported type"); };
template <typename T>
struct IndexType<T, std::enable_if_t<sizeof(T) == sizeof(int8_t)>> { using Type = int8_t; };
template <typename T>
struct IndexType<T, std::enable_if_t<sizeof(T) == sizeof(int16_t)>> { using Type = int16_t; };
template <typename T>
struct IndexType<T, std::enable_if_t<sizeof(T) == sizeof(int32_t)>> { using Type = int32_t; };
template <typename T>
struct IndexType<T, std::enable_if_t<sizeof(T) == sizeof(int64_t)>> { using Type = int64_t; };
} // namespace details

/**
 * @brief Type alias for the vector storage type.
 * 
 * @warning This is a forwarded type and cannot be used for template argument
 *          deduction. Use the tag directly instead.
 * 
 * @tparam Tag The tag type describing the vector
 */
template <typename Tag>
using Vec = typename VecDefs<typename Tag::Type, Tag::N, Tag::POW2>::VecType;

/**
 * @brief Macro to extract the vector type from a tag expression.
 * 
 * Useful when the tag is a complex expression, e.g., VecOf(t) where t is a Tag.
 */
#define VecOf(x) Vec<decltype(x)>

/**
 * @brief Macro to extract the mask type from a tag expression.
 */
#define MaskOf(x) Mask<decltype(x)>

/**
 * @brief Type alias for the mask storage type.
 * 
 * @warning This is a forwarded type and cannot be used for template argument
 *          deduction.
 * 
 * @tparam Tag The tag type describing the vector
 */
template <typename Tag>
using Mask = typename VecDefs<typename Tag::Type, Tag::N, Tag::POW2>::MaskType;

/**
 * @brief Type alias for the signed index type corresponding to element type T.
 * 
 * The index type is used in gather/scatter operations.
 * 
 * @warning This is a forwarded type and cannot be used for template argument
 *          deduction.
 */
template <typename T>
using Index = typename details::IndexType<T>::Type;

/**
 * @brief Get a specific word from a vector by compile-time index.
 * 
 * @tparam Index The word index (0 to num_words-1)
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The tag
 * @param v The vector
 * @return The specified word
 */
template <nint_t Index, typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE CT_PURE
static typename VecDefs<T, N, P>::WordDefs::VecType get_word(Tag<T, N, P> t, Vec<Tag<T, N, P>> v) {
  return VecDefs<T, N, P>::template get<Index>(v);
}

/**
 * @brief Get a specific word from a vector by runtime index.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The tag
 * @param v The vector
 * @param index The word index (0 to num_words-1)
 * @return The specified word
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE CT_PURE
static typename VecDefs<T, N, P>::WordDefs::VecType get_word(Tag<T, N, P> t, Vec<Tag<T, N, P>> v, nint_t index) {
  return VecDefs<T, N, P>::get(v, index);
}

/**
 * @brief Set a specific word in a vector by compile-time index.
 * 
 * @tparam Index The word index (0 to num_words-1)
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The tag
 * @param v The original vector
 * @param u The new word value
 * @return Vector with the word updated
 */
template <nint_t Index, typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE CT_PURE
static Vec<Tag<T, N, P>> set_word(Tag<T, N, P> t, Vec<Tag<T, N, P>> v, typename VecDefs<T, N, P>::WordDefs::VecType u) {
  return VecDefs<T, N, P>::template set<Index>(v, u);
}

/**
 * @brief Set a specific word in a vector by runtime index.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The tag
 * @param v The original vector
 * @param index The word index
 * @param u The new word value
 * @return Vector with the word updated
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE CT_PURE
static Vec<Tag<T, N, P>> set_word(Tag<T, N, P> t, Vec<Tag<T, N, P>> v, nint_t index, typename VecDefs<T, N, P>::WordDefs::VecType u) {
  return VecDefs<T, N, P>::set(v, index, u);
}

/**
 * @brief Get a specific mask word from a mask by compile-time index.
 * 
 * @tparam Index The word index (0 to num_words-1)
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The tag
 * @param v The mask
 * @return The specified mask word
 */
template <nint_t Index, typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE CT_PURE
static typename VecDefs<T, N, P>::WordDefs::MaskType get_word_mask(Tag<T, N, P> t, Mask<Tag<T, N, P>> v) {
  return VecDefs<T, N, P>::template get_mask<Index>(v);
}

/**
 * @brief Get a specific mask word from a mask by runtime index.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The tag
 * @param v The mask
 * @param index The word index
 * @return The specified mask word
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE CT_PURE
static typename VecDefs<T, N, P>::WordDefs::MaskType get_word_mask(Tag<T, N, P> t, Mask<Tag<T, N, P>> v, nint_t index) {
  return VecDefs<T, N, P>::get_mask(v, index);
}

/**
 * @brief Set a specific mask word in a mask by compile-time index.
 * 
 * @tparam Index The word index (0 to num_words-1)
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The tag
 * @param v The original mask
 * @param u The new mask word value
 * @return Mask with the word updated
 */
template <nint_t Index, typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE CT_PURE
static Mask<Tag<T, N, P>> set_word_mask(Tag<T, N, P> t, Mask<Tag<T, N, P>> v, typename VecDefs<T, N, P>::WordDefs::MaskType u) {
  return VecDefs<T, N, P>::template set_mask<Index>(v, u);
}

/**
 * @brief Set a specific mask word in a mask by runtime index.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The tag
 * @param v The original mask
 * @param index The word index
 * @param u The new mask word value
 * @return Mask with the word updated
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE CT_PURE
static Mask<Tag<T, N, P>> set_word_mask(Tag<T, N, P> t, Mask<Tag<T, N, P>> v, nint_t index, typename VecDefs<T, N, P>::WordDefs::MaskType u) {
  return VecDefs<T, N, P>::set_mask(v, index, u);
}

/**
 * @brief Returns the tag for a single word of this vector type.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @return Tag for a single hardware word
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE CT_PURE
static constexpr auto word_tag(Tag<T, N, P> t) {
  return typename VecDefs<T, N, P>::WordDefs::TagType();
}

/**
 * Helper struct getting (word or vector of word) Tag type from a vec type
 * The value returned may not be same to original Tag, which may not be used for
 * scenario where a precise number of elements is required (load & store).
 */
template <typename TVec, typename = void/* SFINAE*/>
struct Vec2TagDefs {
  using Type = void;
};

template <typename T, nint_t N>
struct Vec2TagDefs<ScalarArray<T, N>, std::enable_if_t<!is_element_type<T> && sizeof(typename Vec2TagDefs<T>::Type) != -1>> {
  static_assert((N & (N - 1)) == 0, "N not power of 2");
  using Type = Tag<typename Vec2TagDefs<T>::Type::Type, Vec2TagDefs<T>::Type::N, (Vec2TagDefs<T>::Type::POW2 + log2_floor(N))>;
};

template <typename T, nint_t N>
struct Vec2TagDefs<ScalarArray<T, N>, std::enable_if_t<is_element_type<T>>> {
  using Type = Tag<T, N, 0>;
};

template <typename TVec>
using Vec2Tag = typename Vec2TagDefs<TVec>::Type;

template <typename TTag>
using TypeOf = typename TTag::Type;

namespace details {
constexpr nint_t size_shift(nuint_t from_size, nuint_t to_size) {
  if (from_size > to_size) {
    return -log2_floor(from_size / to_size);
  } else {
    return log2_floor(to_size / from_size);
  }
}
} // namespace details

template <typename TFrom, typename TTo>
static constexpr nint_t SizeShift = details::size_shift(sizeof(TFrom), sizeof(TTo));

/**
 * Keep number of elements unchanged but with a new dtype.
 * The power factor might change in scalable vector.
 */
template <typename TNew, typename TTag>
using Rebind = std::conditional_t<TTag::is_runtime_size, Tag<TNew, TTag::N, TTag::POW2 + SizeShift<TypeOf<TTag>, TNew>>, Tag<TNew, TTag::N, TTag::POW2>>;

/**
 * Keep number of bytes (vector width) unchanged but with a new dtype.
 * The power factor might change in non-scalable vector.
 */
template <typename TNew, typename TTag>
using ViewAs = std::conditional_t<TTag::is_runtime_size, Tag<TNew, TTag::N, TTag::POW2>, Tag<TNew, TTag::N, TTag::POW2 + SizeShift<TypeOf<TTag>, TNew>>>;

template <typename TVec>
CT_ALWAYS_FORCEINLINE CT_PURE
static constexpr auto vec_to_tag(TVec v) {
  return Vec2Tag<TVec>();
}

#define TL_IF(...) std::enable_if_t<(__VA_ARGS__), bool> = true
#define TL_IF_TAG_N_EQ(tag_type, size) TL_IF(!tag_type::is_runtime_size && tag_type::N == (size))
#define TL_IF_TAG_N_LE(tag_type, size) TL_IF(!tag_type::is_runtime_size && tag_type::N <= (size))
#define TL_IF_TAG_DTYPE(tag_type, dtype) TL_IF(std::is_same_v<typename tag_type::Type, dtype>)
#define TL_IF_TAG_IS_INT(tag_type) TL_IF(std::is_integral_v<typename tag_type::Type>)
#define TL_IF_TAG_IS_FLOAT(tag_type) TL_IF(std::is_floating_point_v<typename tag_type::Type>)


// Forward declaration for vector conversion API
template <typename TOut, typename TIn, nint_t NOut>
struct VecConvert {
  template <typename TTag>
  Vec<Tag < TOut, NOut>> convert(TTag t, Vec<TTag> v) const;
};

template <typename T, nint_t NOut, int POut, nint_t NIn, int PIn>
struct VecReshape {
  static_assert(details::adjusted_size<NOut, POut>() == details::adjusted_size<NOut, POut>());
  Vec<Tag<T, NOut, POut>> reshape(Tag<T, NIn, PIn> t, Vec<Tag<T, NIn, PIn>> v) const;
};

//template <typename TOut, typename TIn, nint_t NOut>
//struct VecBitCast {
//  template <typename TVec>
//  Vec<Tag < TOut, NOut>> cast(TVec v) const;
//};
//
//template <typename TOut, typename TIn, nint_t NOut>
//struct VecZeroExtendBitCast {
//  template <typename TVec>
//  Vec<Tag < TOut, NOut>> cast(TVec v) const;
//};

} // namespace ct::tl::vec

#endif //CTORCH_VECBASE_H
