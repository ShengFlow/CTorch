//
// Created by renyz on 2026/3/15.
//

#ifndef CTORCH_VECBASE_H
#define CTORCH_VECBASE_H

#include <array>
#include <bitset>

#include "Assertion.h"
#include "CoreDefs.h"

#if defined(HAS_SVE)
#define _VEC_SIZE(type) SIMD_WIDTH
#else
#define _VEC_SIZE(type) (SIMD_WIDTH / 8 / sizeof(type))
#endif

namespace ct::tl::vec {
namespace details {
template <nint_t N, int POW2>
static constexpr nint_t size(nint_t scalable_size = -1) {
  if constexpr (N < 0) {
    return scalable_size;
  } else {
    return N;
  }
}

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
} // namespace details

template <typename T, nint_t N_, int POW2_ = 0>
struct Tag {
  using Type = T;
  static constexpr nint_t N = details::size<N_, POW2_>(N_);
  static constexpr nint_t AdjustedN = details::adjusted_size<N_, POW2_>();
  static constexpr int POW2 = POW2_;
  static constexpr bool is_runtime_size = N_ < 0;

  static_assert(N_ < 0 || ((N_ & (N_ - 1)) == 0 && N_ != 0), "Not 2^x");
};

template <typename T, int POW2 = 0>
using ScalableTag = Tag<T, _VEC_SIZE(T), POW2>;

template <typename T, nint_t N>
using FixedTag = Tag<T, N, 0>;

static constexpr nint_t DEFAULT_ALIGNMENT = 16;

/**
* Default bytes length of a vector, only used for default vec impl.
*/
static constexpr nint_t DEFAULT_LENGTH = 64;

template <typename T>
static constexpr nint_t DEFAULT_SIZE = DEFAULT_LENGTH / sizeof(T);

/**
 *
 */
template <typename T, nint_t N>
class alignas(DEFAULT_ALIGNMENT) ScalarArray : public std::array<T, N> {};

/**
 * Special bitset for default mask repr with marker ELSIZE to disguise with other types
 */
template <nint_t ELSIZE, nint_t N>
class ScalarBitSet : public std::bitset<N> {};

/**
 * Helper struct providing Vec & Mask type for given tag.
 * @tparam T
 * @tparam N vector size, must be power of 2, or -1 if it's scalable
 * @tparam POW2 size power of N, actual size is vector size * 2 ^ POW2
 * @see VecDefs
 */
template <
    typename T, nint_t N, int POW2 = 0
>
struct BaseVecDefs {
  /**
   * Tag type
   */
  using TagType = Tag<T, N, POW2>;
  /**
   * Number of machine vector word in this vector (i.e. a xmm register, SVE register)
   */
  static constexpr nint_t num_words = -1; // placeholder
  /**
   * Number of elements in a machine vector word, may be runtime value (if is scalar).
   * The value may be greater than size().
   */
  static constexpr nint_t word_size() { return -1; }

  /**
   * Upperbound size (in elements) of machine vector word. Not smaller than word_size()
   */
  static constexpr nint_t max_word_size = -1; // placeholder
  /**
   * Number of elements in this vector, may be runtime value (if is scalar).
   * Which means that value returned may not be constepxr.
   */
  static constexpr nint_t size() { return -1; };
  /**
   * Upperbound size of this vector, not smaller than size()
   */
  static constexpr nint_t max_size = -1;
  /**
   * If this vector has runtime size
   */
  static constexpr bool is_scalable = TagType::is_runtime_size;
  /**
   * If this vector is implemented by a default vector. (No native acceleration)
   */
  static constexpr bool is_default_impl = false;  // default sets to false
  /**
   * If VecType represents directly a machine vector word.
   */
  static constexpr bool is_word_vec = true;
  /**
   * Underlying vector type, can only be used to declare local variables and parameters, may not be stored in memory or addressed.
   * i.e. SVE vectors is sizeless.
   */
  using VecType = int; // placeholder
  /**
   * Underlying mask type for VecType, can only be used to declare local variables and parameters, may not be stored in memory or addressed.
   * i.e. SVE predicate vectors.
   */
  using MaskType = int; // placeholder
  /**
   * VecDefs for word vector.
   */
  using WordDefs = std::conditional_t<is_scalable,
      BaseVecDefs<T, N, 0>, BaseVecDefs<T, word_size(), 0>
  >;

  /**
   * Get Index-th word of this vector
   */
  template <nint_t Index>
  static typename WordDefs::VecType get(VecType v) { return v; };

  static typename WordDefs::VecType get(VecType v, nint_t index) { return v; };

  /**
   * Set Index-th word of this vector to u
   */
  template <nint_t Index>
  static VecType set(VecType v, typename WordDefs::VecType u) { return v; };

  static VecType set(VecType v, nint_t index, typename WordDefs::VecType u) { return v; };


  template <nint_t Index>
  static typename WordDefs::MaskType get_mask(MaskType m) { return m; }

  static typename WordDefs::MaskType get_mask(MaskType m, nint_t index) { return m; };

  template <nint_t Index>
  static MaskType set_mask(MaskType m, typename WordDefs::MaskType u) { return m; }

  static MaskType set_mask(MaskType m, nint_t index, typename WordDefs::MaskType u) { return m; };
}; // struct BaseVecDefs

/**
 * Definition for default vector impl in a scalar sematic.
 */
template <typename T, nint_t N, int P = 0, typename = void /*  SFINAE */>
struct ScalarVecDefs {};

template <typename T, nint_t N, int POW2 = 0, typename = void /* SFINAE */>
struct VecDefs : public ScalarVecDefs<T, N, POW2> {};

/**
 * Default impl for size 0
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
 * Vectorization when POW2 > 0
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
 * Delegates to smaller vector when POW2 < 0
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

template <typename T, nint_t N, int POW2>
CT_ALWAYS_FORCEINLINE CT_PURE
static constexpr nint_t num_words(Tag<T, N, POW2> t = {}) {
  return VecDefs<T, N, POW2>::num_words;
}

/**
 * Note: may be runtime constant
 */
template <typename T, nint_t N, int POW2>
CT_ALWAYS_FORCEINLINE CT_PURE
static constexpr nint_t word_size(Tag<T, N, POW2> t = {}) {
  return VecDefs<T, N, POW2>::word_size();
}

template <typename T, nint_t N, int POW2>
CT_ALWAYS_FORCEINLINE CT_PURE
static constexpr nint_t max_word_size(Tag<T, N, POW2> t = {}) {
  return VecDefs<T, N, POW2>::max_word_size;
}

/**
 * Note: may be runtime constant
 */
template <typename T, nint_t N, int POW2>
CT_ALWAYS_FORCEINLINE CT_PURE
static constexpr nint_t size(Tag<T, N, POW2> t = {}) {
  return VecDefs<T, N, POW2>::size();
}

template <typename T, nint_t N, int POW2>
CT_ALWAYS_FORCEINLINE CT_PURE
static constexpr bool is_scalable(Tag<T, N, POW2> t = {}) {
  return VecDefs<T, N, POW2>::is_scalable;
}

template <typename T, nint_t N, int POW2>
CT_ALWAYS_FORCEINLINE CT_PURE
static constexpr bool is_default_impl(Tag<T, N, POW2> t = {}) {
  return VecDefs<T, N, POW2>::is_default_impl;
}

template <typename T, nint_t N, int POW2>
CT_ALWAYS_FORCEINLINE CT_PURE
static constexpr bool is_word_vec(Tag<T, N, POW2> t = {}) {
  return VecDefs<T, N, POW2>::is_word_vec;
}

namespace details {
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
 * Note: forwarded type, cannot be used for template deduction
 */
template <typename Tag>
using Vec = typename VecDefs<typename Tag::Type, Tag::N, Tag::POW2>::VecType;

#define VecOf(x) Vec<decltype(x)>
#define MaskOf(x) Mask<decltype(x)>

/**
 * Note: forwarded type, cannot be used for template deduction
 */
template <typename Tag>
using Mask = typename VecDefs<typename Tag::Type, Tag::N, Tag::POW2>::MaskType;

/**
 * Note: forwarded type, cannot be used for template deduction
 */
template <typename T>
using Index = typename details::IndexType<T>::Type;

template <nint_t Index, typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE CT_PURE
static typename VecDefs<T, N, P>::WordDefs::VecType get_word(Tag<T, N, P> t, Vec<Tag<T, N, P>> v) {
  return VecDefs<T, N, P>::template get<Index>(v);
}

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE CT_PURE
static typename VecDefs<T, N, P>::WordDefs::VecType get_word(Tag<T, N, P> t, Vec<Tag<T, N, P>> v, nint_t index) {
  return VecDefs<T, N, P>::get(v, index);
}

template <nint_t Index, typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE CT_PURE
static Vec<Tag<T, N, P>> set_word(Tag<T, N, P> t, Vec<Tag<T, N, P>> v, typename VecDefs<T, N, P>::WordDefs::VecType u) {
  return VecDefs<T, N, P>::template set<Index>(v, u);
}

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE CT_PURE
static Vec<Tag<T, N, P>> set_word(Tag<T, N, P> t, Vec<Tag<T, N, P>> v, nint_t index, typename VecDefs<T, N, P>::WordDefs::VecType u) {
  return VecDefs<T, N, P>::set(v, index, u);
}

template <nint_t Index, typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE CT_PURE
static typename VecDefs<T, N, P>::WordDefs::MaskType get_word_mask(Tag<T, N, P> t, Mask<Tag<T, N, P>> v) {
  return VecDefs<T, N, P>::template get_mask<Index>(v);
}

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE CT_PURE
static typename VecDefs<T, N, P>::WordDefs::MaskType get_word_mask(Tag<T, N, P> t, Mask<Tag<T, N, P>> v, nint_t index) {
  return VecDefs<T, N, P>::get_mask(v, index);
}

template <nint_t Index, typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE CT_PURE
static Mask<Tag<T, N, P>> set_word_mask(Tag<T, N, P> t, Mask<Tag<T, N, P>> v, typename VecDefs<T, N, P>::WordDefs::MaskType u) {
  return VecDefs<T, N, P>::template set_mask<Index>(v, u);
}

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE CT_PURE
static Mask<Tag<T, N, P>> set_word_mask(Tag<T, N, P> t, Mask<Tag<T, N, P>> v, nint_t index, typename VecDefs<T, N, P>::WordDefs::MaskType u) {
  return VecDefs<T, N, P>::set_mask(v, index, u);
}

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE CT_PURE
static constexpr auto word_tag(Tag<T, N, P> t) {
  return typename VecDefs<T, N, P>::WordDefs::TagType();
}

} // namespace ct::tl::vec

#endif //CTORCH_VECBASE_H
