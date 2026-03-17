//
// Created by renyz on 2026/3/16.
//

#ifndef CTORCH_VEC_H
#define CTORCH_VEC_H

#include "VecBase.h"

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
 * The internal `vectorized_map_*` functions handle this transparently.
 */

#if defined(ARCH_X86_FAMILY)
#if SIMD_WIDTH == 128
#include "tl/cpu/impl/x86_128.h"
#endif
#if SIMD_WIDTH == 256
#include "tl/cpu/impl/x86_128.h"
//    #include "tl/cpu/impl/x86_256.h"  // TODO implement it
#endif
#if SIMD_WIDTH == 512
#include "tl/cpu/impl/x86_128.h"
//    #include "tl/cpu/impl/x86_256.h"
//    #include "tl/cpu/impl/x86_512.h"
#endif
#if SIMD_WIDTH != 128 && SIMD_WIDTH != 256 && SIMD_WIDTH != 512
#warning "Unsupported SIMD width, falling back to scalar implementation."
#endif
#elif defined(ARCH_ARM_FAMILY)
// TODO untested
#if SIMD_WIDTH == 128
#include "tl/cpu/impl/arm_neon.h"
#endif
#if SSIMD_WIDTH == (-1)
#include "tl/cpu/impl/arm_neon.h"
#include "tl/cpu/impl/arm_sve.h"
#endif
#else
#warning "Unrecognized architecture, falling back to scalar implementation."

#include "tl/cpu/impl/Scalar.h"

#endif

namespace ct::tl::vec {
namespace details {

/**
 * @brief Wrapper for a vector that should be split into words.
 * 
 * Used internally to pass vectors to vectorized_map functions where
 * each word should be processed separately.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 */
template <typename T, nint_t N, int P>
struct ShardVec {
  Tag<T, N, P> tag;
  Vec<Tag<T, N, P>>& v;

  CT_ALWAYS_FORCEINLINE
  constexpr ShardVec(Tag<T, N, P> t, Vec<Tag<T, N, P>>& v) : tag(t), v(v) {}
};

/**
 * @brief Wrapper for a mask that should be split into words.
 * 
 * Used internally to pass masks to vectorized_map functions where
 * each word should be processed separately.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 */
template <typename T, nint_t N, int P>
struct ShardMask {
  Tag<T, N, P> tag;
  Mask<Tag<T, N, P>>& m;

  CT_ALWAYS_FORCEINLINE
  constexpr ShardMask(Tag<T, N, P> t, Mask<Tag<T, N, P>>& m) : tag(t), m(m) {}
};

/**
 * @brief A pointer with a non-contiguous stride.
 * 
 * Used internally for operations that need to access data at fixed intervals,
 * such as when processing multiple words of a multi-word vector.
 * 
 * @tparam T Element type
 */
template <typename T>
struct StepPointer {
  T* p;
  nint_t step;

  CT_ALWAYS_FORCEINLINE
  constexpr StepPointer(const T* p, nint_t step) : p(const_cast<T*>(p)), step(step) {}

  template <nint_t N, int P>
  CT_ALWAYS_FORCEINLINE
  constexpr StepPointer(Tag<T, N, P> t, const T* p) : StepPointer(p, word_size(t)) {}
};

/**
 * @brief Default argument transformer that passes arguments through unchanged.
 * 
 * Used by the vectorized_map infrastructure to transform arguments based on
 * their type. This base case does no transformation.
 * 
 * @tparam Index The iteration index (unused in this case)
 * @tparam T The argument type
 */
template <nint_t Index, typename T>
struct ArgTransform {
  CT_ALWAYS_FORCEINLINE
  constexpr decltype(auto) operator()(T&& a) {
    return std::forward<T>(a);
  }

  CT_ALWAYS_FORCEINLINE
  constexpr decltype(auto) operator()(T&& a, nint_t index) {
    return this->operator()(std::forward<T>(a));
  }
};

/**
 * @brief Argument transformer that extracts a word from a ShardVec.
 * 
 * When a ShardVec is passed to a vectorized operation, this transformer
 * extracts the Index-th word from the wrapped vector.
 * 
 * @tparam Index The word index to extract
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 */
template <nint_t Index, typename T, nint_t N, int P>
struct ArgTransform<Index, ShardVec<T, N, P>> {
  CT_ALWAYS_FORCEINLINE
  constexpr decltype(auto) operator()(ShardVec<T, N, P>&& a) {
    return get_word<Index>(a.tag, a.v);
  }

  CT_ALWAYS_FORCEINLINE
  constexpr decltype(auto) operator()(ShardVec<T, N, P>&& a, nint_t index) {
    return get_word(a.tag, a.v, index);
  }
};

/**
 * @brief Argument transformer that extracts a word from a ShardMask.
 * 
 * Similar to the ShardVec transformer, but for masks.
 * 
 * @tparam Index The word index to extract
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 */
template <nint_t Index, typename T, nint_t N, int P>
struct ArgTransform<Index, ShardMask<T, N, P>> {
  CT_ALWAYS_FORCEINLINE
  constexpr decltype(auto) operator()(ShardMask<T, N, P>&& a) {
    return get_word_mask<Index>(a.tag, a.m);
  }

  CT_ALWAYS_FORCEINLINE
  constexpr decltype(auto) operator()(ShardMask<T, N, P>&& a, nint_t index) {
    return get_word_mask(a.tag, a.m, index);
  }
};

/**
 * @brief Argument transformer for StepPointer that computes the actual address.
 * 
 * Transforms a StepPointer by computing: p + Index * step
 * This is used when processing multi-word vectors where each word's data
 * is at a different offset.
 * 
 * @tparam Index The iteration index
 * @tparam T Element type
 */
template <nint_t Index, typename T>
struct ArgTransform<Index, StepPointer<T>> {
  CT_ALWAYS_FORCEINLINE
  constexpr decltype(auto) operator()(StepPointer<T>&& a) {
    return this->operator()(std::forward<StepPointer<T>>(a), Index);
  }

  CT_ALWAYS_FORCEINLINE
  constexpr decltype(auto) operator()(StepPointer<T>&& a, nint_t index) {
    return a.p + index * a.step;
  }
};

/**
 * @brief Compile-time loop for applying operations to transformed arguments.
 * 
 * This template implements a loop unrolling mechanism that iterates from I
 * to NLoop with the given Step, transforming arguments at each iteration
 * and applying the user's function.
 * 
 * @tparam NLoop Loop bound (exclusive)
 * @tparam Step Loop step (must be positive)
 * @tparam I Current iteration (starts at 0)
 */
template <nint_t NLoop, nint_t Step = 1, nint_t I = 0, typename = void /* SFINAE*/>
struct ForEachTransformed {
  static_assert((Step > 0 && I < NLoop) || (Step < 0 && I > NLoop));

  /**
   * @brief Iterate from I to NLoop, calling f with transformed arguments.
   * 
   * @tparam F Function type (must have template operator()<Index>)
   * @tparam TArgs Argument types
   * @param f The function to call at each iteration
   * @param args Arguments to transform and pass to f
   */
  template <typename F, typename... TArgs>
  CT_ALWAYS_FORCEINLINE
  constexpr void operator()(F&& f, TArgs&& ... args) {
    f.template operator()<I>(ArgTransform<I, std::remove_cvref_t<TArgs>>()(std::forward<TArgs>(args))...);
    ForEachTransformed<NLoop, Step, I + Step>()(std::forward<F>(f), std::forward<TArgs>(args)...);
  }

  /**
   * @brief Iterate from I to n (runtime bound), calling f with transformed arguments.
   * 
   * @tparam F Function type
   * @tparam TArgs Argument types
   * @param n The runtime loop bound
   * @param f The function to call
   * @param args Arguments to transform and pass to f
   */
  template <typename F, typename... TArgs>
  CT_ALWAYS_FORCEINLINE
  constexpr void operator()(nint_t n, F&& f, TArgs&& ... args) {
    if (I >= n - (Step - 1)) return;
    f.template operator()<I>(ArgTransform<I, std::remove_cvref_t<TArgs>>()(std::forward<TArgs>(args))...);
    ForEachTransformed<NLoop, Step, I + Step>()(n, std::forward<F>(f), std::forward<TArgs>(args)...);
  }
};

/**
 * @brief Base case for ForEachTransformed when iteration is complete.
 */
template <nint_t NLoop, nint_t Step, nint_t I>
struct ForEachTransformed<NLoop, Step, I, std::enable_if_t<(I >= NLoop)>> {
  template <typename F, typename... TArgs>
  CT_ALWAYS_FORCEINLINE
  constexpr void operator()(F&& f, TArgs&& ... args) {}

  template <typename F, typename... TArgs>
  CT_ALWAYS_FORCEINLINE
  constexpr void operator()(nint_t n, F&& f, TArgs&& ... args) {}
};

/**
 * @brief Apply a word-level operation to produce a vector result.
 * 
 * This is the core infrastructure for vector operations. It:
 * 1. Checks if the vector is a single word (fast path)
 * 2. If multi-word, iterates over all words, applying f to each
 * 3. Collects results into the output vector
 * 
 * @tparam TTag The vector tag type
 * @tparam Fn Function type that takes (word_tag, word_args...) and returns a word
 * @tparam TArgs Argument types (may include ShardVec, ShardMask, StepPointer)
 * @param t The vector tag
 * @param f The operation to apply to each word
 * @param args Arguments to pass (will be transformed per-word)
 * @return The result vector
 */
template <typename TTag, typename Fn, typename... TArgs>
CT_ALWAYS_FORCEINLINE
Vec<TTag> vectorized_map_v(TTag t, Fn&& f, TArgs&& ... args) {
  constexpr auto wt = word_tag(t);
  if constexpr (is_word_vec(t)) {
    return std::forward<Fn>(f)(
        wt, ArgTransform<nint_t(0), std::remove_cvref_t<TArgs>>()(std::forward<TArgs>(args))...
    );
  } else {
    constexpr nint_t nloop = num_words(t);
    Vec<TTag> r;
    ForEachTransformed<nloop>()(
        [&r, &f, t, wt] <nint_t I>(auto&& ... args) {
          r = set_word<I>(t, r, f(wt, std::forward<std::remove_cvref_t<decltype(args)>>(args)...));
        }, std::forward<TArgs>(args)...
    );
    return r;
  }
} // vectorized_map_v

/**
 * @brief Apply a word-level operation to produce a mask result.
 * 
 * Similar to vectorized_map_v, but for operations that return masks.
 * 
 * @tparam TTag The vector tag type
 * @tparam Fn Function type
 * @tparam TArgs Argument types
 * @param t The vector tag
 * @param f The operation to apply to each word
 * @param args Arguments to pass
 * @return The result mask
 */
template <typename TTag, typename Fn, typename... TArgs>
CT_ALWAYS_FORCEINLINE
Mask<TTag> vectorized_map_m(TTag t, Fn&& f, TArgs&& ... args) {
  constexpr auto wt = word_tag(t);
  if constexpr (is_word_vec(t)) {
    return std::forward<Fn>(f)(
        wt, ArgTransform<nint_t(0), std::remove_cvref_t<TArgs>>()(std::forward<TArgs>(args))...
    );
  } else {
    constexpr nint_t nloop = num_words(t);
    Mask<TTag> r;
    ForEachTransformed<nloop>()(
        [&r, &f, t, wt] <nint_t I>(auto&& ... args) {
          r = set_word_mask<I>(t, r, f(wt, std::forward<std::remove_cvref_t<decltype(args)>>(args)...));
        }, std::forward<TArgs>(args)...
    );
    return r;
  }
} // vectorized_map_m


/**
 * @brief Apply a word-level operation with index awareness to produce a vector result.
 * 
 * Similar to vectorized_map_v, but the function f receives the word index
 * as a template parameter. Useful for operations that need to know which
 * word they're processing (e.g., mwhilelt for generating sequential masks).
 * 
 * @tparam TTag The vector tag type
 * @tparam Fn Function type with template operator()<Index>(word_tag, args...)
 * @tparam TArgs Argument types
 * @param t The vector tag
 * @param f The operation to apply (receives word index as template param)
 * @param args Arguments to pass
 * @return The result vector
 */
template <typename TTag, typename Fn, typename... TArgs>
CT_ALWAYS_FORCEINLINE
Vec<TTag> vectorized_map_v_indexed(TTag t, Fn&& f, TArgs&& ... args) {
  constexpr auto wt = word_tag(t);
  if constexpr (is_word_vec(t)) {
    return std::forward<Fn>(f).template operator()<nint_t(0)>(
        wt, ArgTransform<nint_t(0), std::remove_cvref_t<TArgs>>()(std::forward<TArgs>(args))...
    );
  } else {
    constexpr nint_t nloop = num_words(t);
    Vec<TTag> r;
    ForEachTransformed<nloop>()(
        [&r, &f, t, wt] <nint_t I>(auto&& ... args) {
          r = set_word<I>(t, r, f.template operator()<I>(wt, std::forward<std::remove_cvref_t<decltype(args)>>(args)...));
        }, std::forward<TArgs>(args)...
    );
    return r;
  }
} // vectorized_map_v_indexed

/**
 * @brief Apply a word-level operation with index awareness to produce a mask result.
 * 
 * @tparam TTag The vector tag type
 * @tparam Fn Function type with template operator()<Index>
 * @tparam TArgs Argument types
 * @param t The vector tag
 * @param f The operation to apply
 * @param args Arguments to pass
 * @return The result mask
 */
template <typename TTag, typename Fn, typename... TArgs>
CT_ALWAYS_FORCEINLINE
Mask<TTag> vectorized_map_m_indexed(TTag t, Fn&& f, TArgs&& ... args) {
  constexpr auto wt = word_tag(t);
  if constexpr (is_word_vec(t)) {
    return std::forward<Fn>(f).template operator()<nint_t(0)>(
        wt, ArgTransform<nint_t(0), std::remove_cvref_t<TArgs>>()(std::forward<TArgs>(args))...
    );
  } else {
    constexpr nint_t nloop = num_words(t);
    Mask<TTag> r;
    ForEachTransformed<nloop>()(
        [&r, &f, t, wt] <nint_t I>(auto&& ... args) {
          r = set_word_mask<I>(t, r, f.template operator()<I>(wt, std::forward<std::remove_cvref_t<decltype(args)>>(args)...));
        }, std::forward<TArgs>(args)...
    );
    return r;
  }
} // vectorized_map_m_indexed


/**
 * @brief Apply word-level operations with a partial (tail) handling.
 * 
 * This variant handles operations that may only process n elements,
 * where n <= size(t). It uses f_complete for full words and f_tail
 * for the partial last word.
 * 
 * @tparam TTag The vector tag type
 * @tparam FnC Function type for complete words
 * @tparam FnT Function type for tail (partial) word
 * @tparam TArgs Argument types
 * @param t The vector tag
 * @param n Number of elements to process (0 <= n <= size(t))
 * @param f_complete Function for full words: f(word_tag, args...)
 * @param f_tail Function for tail: f(word_tag, remaining, args...)
 * @param args Arguments to pass
 * @return The result vector
 */
template <typename TTag, typename FnC, typename FnT, typename... TArgs>
CT_ALWAYS_FORCEINLINE
Vec<TTag> vectorized_map_v(TTag t, nint_t n, FnC&& f_complete, FnT&& f_tail, TArgs&& ... args) {
  constexpr auto wt = word_tag(t);
  nint_t L = size(t);
  CT_ASSERT(0 <= n && n <= L, "%zd !in 0..%zd", n, L);

  if constexpr (is_word_vec(t)) {
    return std::forward<FnT>(f_tail)(
        wt, n, ArgTransform<nint_t(0), std::remove_cvref_t<TArgs>>()(std::forward<TArgs>(args))...
    );
  } else {
    constexpr nint_t nloop = num_words(t);
    nint_t ws = word_size(t);
    nint_t full_nloop = n / ws;
    nint_t rem = n % ws;
    Vec<TTag> r;
    ForEachTransformed<nloop>()(
        full_nloop, [&r, &f_complete, t, wt] <nint_t I>(auto&& ... args) {
          r = set_word<I>(t, r, f_complete(wt, std::forward<std::remove_cvref_t<decltype(args)>>(args)...));
        }, std::forward<TArgs>(args)...
    );
    if (rem > 0) {
      r = set_word(
          t, r, full_nloop,
          f_tail(wt, rem, ArgTransform<-1, std::remove_cvref_t<TArgs>>()(std::forward<std::remove_cvref_t<TArgs>>(args), full_nloop)...)
      );
    }
    return r;
  }
} // vectorized_map_v

/**
 * @brief Apply word-level operations with partial handling for mask results.
 * 
 * @tparam TTag The vector tag type
 * @tparam FnC Function type for complete words
 * @tparam FnT Function type for tail word
 * @tparam TArgs Argument types
 * @param t The vector tag
 * @param n Number of elements to process
 * @param f_complete Function for full words
 * @param f_tail Function for tail word
 * @param args Arguments to pass
 * @return The result mask
 */
template <typename TTag, typename FnC, typename FnT, typename... TArgs>
CT_ALWAYS_FORCEINLINE
Mask<TTag> vectorized_map_m(TTag t, nint_t n, FnC&& f_complete, FnT&& f_tail, TArgs&& ... args) {
  constexpr auto wt = word_tag(t);
  nint_t L = size(t);
  CT_ASSERT(0 <= n && n <= L, "%zd !in 0..%zd", n, L);

  if constexpr (is_word_vec(t)) {
    return std::forward<FnT>(f_tail)(
        wt, n, ArgTransform<nint_t(0), std::remove_cvref_t<TArgs>>()(std::forward<TArgs>(args))...
    );
  } else {
    constexpr nint_t nloop = num_words(t);
    nint_t ws = word_size(t);
    nint_t full_nloop = n / ws;
    nint_t rem = n % ws;
    Mask<TTag> r;
    ForEachTransformed<nloop>()(
        full_nloop, [&r, &f_complete, t, wt] <nint_t I>(auto&& ... args) {
          r = set_word_mask<I>(t, r, f_complete(wt, std::forward<std::remove_cvref_t<decltype(args)>>(args)...));
        }, std::forward<TArgs>(args)...
    );
    if (rem > 0) {
      r = set_word_mask(
          t, r, full_nloop,
          f_tail(wt, rem, ArgTransform<-1, std::remove_cvref_t<TArgs>>()(std::forward<std::remove_cvref_t<TArgs>>(args), full_nloop)...)
      );
    }
    return r;
  }
} // vectorized_map_m

/**
 * @brief Apply a word-level operation without returning a result.
 * 
 * Used for operations like store that don't produce a value.
 * 
 * @tparam TTag The vector tag type
 * @tparam Fn Function type
 * @tparam TArgs Argument types
 * @param t The vector tag
 * @param f The operation to apply
 * @param args Arguments to pass
 */
template <typename TTag, typename Fn, typename... TArgs>
CT_ALWAYS_FORCEINLINE
void vectorized_foreach(TTag t, Fn&& f, TArgs&& ... args) {
  constexpr auto wt = word_tag(t);
  if constexpr (is_word_vec(t)) {
    std::forward<Fn>(f)(
        wt, ArgTransform<nint_t(0), std::remove_cvref_t<TArgs>>()(std::forward<TArgs>(args))...
    );
    return;
  } else {
    constexpr nint_t nloop = num_words(t);
    ForEachTransformed<nloop>()(
        [&f, wt] <nint_t I>(auto&& ... args) {
          f(wt, std::forward<std::remove_cvref_t<decltype(args)>>(args)...);
        }, std::forward<TArgs>(args)...
    );
    return;
  }
} // vectorized_foreach


/**
 * @brief Apply a word-level operation with index awareness, without returning a result.
 * 
 * @tparam TTag The vector tag type
 * @tparam Fn Function type with template operator()<Index>
 * @tparam TArgs Argument types
 * @param t The vector tag
 * @param f The operation to apply
 * @param args Arguments to pass
 */
template <typename TTag, typename Fn, typename... TArgs>
CT_ALWAYS_FORCEINLINE
void vectorized_foreach_indexed(TTag t, Fn&& f, TArgs&& ... args) {
  constexpr auto wt = word_tag(t);
  if constexpr (is_word_vec(t)) {
    std::forward<Fn>(f).template operator()<nint_t(0)>(
        wt, ArgTransform<nint_t(0), std::remove_cvref_t<TArgs>>()(std::forward<TArgs>(args))...
    );
    return;
  } else {
    constexpr nint_t nloop = num_words(t);
    ForEachTransformed<nloop>()(
        [&f, wt] <nint_t I>(auto&& ... args) {
          f.template operator()<I>(wt, std::forward<std::remove_cvref_t<decltype(args)>>(args)...);
        }, std::forward<TArgs>(args)...
    );
    return;
  }
} // vectorized_foreach


/**
 * @brief Apply word-level operations with partial handling, without returning a result.
 * 
 * Used for store operations with partial element counts.
 * 
 * @tparam TTag The vector tag type
 * @tparam FnC Function type for complete words
 * @tparam FnT Function type for tail word
 * @tparam TArgs Argument types
 * @param t The vector tag
 * @param n Number of elements to process
 * @param f_complete Function for full words
 * @param f_tail Function for tail word
 * @param args Arguments to pass
 */
template <typename TTag, typename FnC, typename FnT, typename... TArgs>
CT_ALWAYS_FORCEINLINE
void vectorized_foreach(TTag t, nint_t n, FnC&& f_complete, FnT&& f_tail, TArgs&& ... args) {
  constexpr auto wt = word_tag(t);
  nint_t L = size(t);
  CT_ASSERT(0 <= n && n <= L, "%zd !in 0..%zd", n, L);

  if constexpr (is_word_vec(t)) {
    std::forward<FnT>(f_tail)(
        wt, n, ArgTransform<nint_t(0), std::remove_cvref_t<TArgs>>()(std::forward<TArgs>(args))...
    );
    return;
  } else {
    constexpr nint_t nloop = num_words(t);
    nint_t ws = word_size(t);
    nint_t full_nloop = n / ws;
    nint_t rem = n % ws;
    ForEachTransformed<nloop>()(
        full_nloop, [&f_complete, t, wt] <nint_t I>(auto&& ... args) {
          f_complete(wt, std::forward<std::remove_cvref_t<decltype(args)>>(args)...);
        }, std::forward<TArgs>(args)...
    );
    if (rem > 0) {
      f_tail(
          wt, rem,
          ArgTransform<-1, std::remove_cvref_t<TArgs>>()(std::forward<std::remove_cvref_t<TArgs>>(args), full_nloop)...
      );
    }
    return;
  }
} // vectorized_foreach
} // namespace details

/* ************************************************************************** */
//                               Constructors                                 //
/* ************************************************************************** */

/**
 * @brief Create a vector filled with a single value.
 * 
 * All elements of the result vector are set to `value`.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param value The value to fill with
 * @return Vector with all elements set to `value`
 * 
 * @example
 *   Tag<float32_t, 4> t;
 *   auto v = fill(t, 3.14f);  // v = [3.14, 3.14, 3.14, 3.14]
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto fill(Tag<T, N, P> t, T value) -> VecOf(t) {
  using namespace details;
  return vectorized_map_v(
      t, [=](auto tt) { return word::fill(tt, value); }
  );
}

/**
 * @brief Create a vector filled with zeros (default-constructed values).
 * 
 * Equivalent to fill(t, T()) where T is default-constructible.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @return Vector with all elements zero-initialized
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto zeros(Tag<T, N, P> t) -> VecOf(t) {
  return fill(t, T());
}

/**
 * @brief Create a mask filled with a single boolean value.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param value The boolean value (true or false)
 * @return Mask with all elements set to `value`
 * 
 * @example
 *   Tag<float32_t, 4> t;
 *   auto m = mfill(t, true);  // All lanes active
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto mfill(Tag<T, N, P> t, bool value) -> MaskOf(t) {
  using namespace details;
  return vectorized_map_m(
      t, [=](auto tt) { return word::mfill(tt, value); }
  );
}

/**
 * @brief Create a mask with all lanes active (all true).
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @return Mask with all elements true
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto mtrue(Tag<T, N, P> t) -> MaskOf(t) {
  return mfill(t, true);
}

/**
 * @brief Create a mask with all lanes inactive (all false).
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @return Mask with all elements false
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto mfalse(Tag<T, N, P> t) -> MaskOf(t) {
  return mfill(t, false);
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
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param a Starting index
 * @param b Upper bound (exclusive)
 * @return Mask where lanes [a, b) are true
 * 
 * @example
 *   Tag<float32_t, 4> t;
 *   auto m = mwhilelt(t, 0, 3);  // m = [true, true, true, false]
 *   
 *   // Useful for processing partial data:
 *   auto m = mwhilelt(t, 0, n);  // Process n elements
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto mwhilelt(Tag<T, N, P> t, nint_t a, nint_t b) {
  using namespace details;
  nint_t ws = word_size(t);
  return vectorized_map_m_indexed(
      t, [=] <nint_t I>(auto tt) { return word::mwhilelt(tt, a + I * ws, b); }
  );
}

/**
 * @brief Create a mask where lanes i are true if (a + i) <= b.
 * 
 * Equivalent to mwhilelt(t, a, b + 1).
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param a Starting index
 * @param b Upper bound (inclusive)
 * @return Mask where lanes [a, b] are true
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto mwhilele(Tag<T, N, P> t, nint_t a, nint_t b) {
  return mwhilelt(t, a, b + 1);
}

/**
 * @brief Create a mask where lanes i are true if (a + i) > b.
 * 
 * Equivalent to mwhilege(t, a, b + 1).
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param a Starting index
 * @param b Lower bound (exclusive)
 * @return Mask where lanes > b are true
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto mwhilegt(Tag<T, N, P> t, nint_t a, nint_t b) {
  return mwhilege(t, a, b + 1);
}

/**
 * @brief Create a mask where lanes i are true if (a + i) >= b.
 * 
 * For a vector of size N:
 *   result[i] = (a + i) >= b
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param a Starting index
 * @param b Lower bound (inclusive)
 * @return Mask where lanes >= b are true
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto mwhilege(Tag<T, N, P> t, nint_t a, nint_t b) {
  using namespace details;
  nint_t ws = word_size(t);
  return vectorized_map_m_indexed(
      t, [=] <nint_t I>(auto tt) { return word::mwhilege(tt, a + I * ws, b); }
  );
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
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Pointer to source memory
 * @return Loaded vector
 * 
 * @note For multi-word vectors, each word is loaded from consecutive addresses
 *       (p, p + word_size, p + 2*word_size, etc.)
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto loadu(Tag<T, N, P> t, const T* p) -> VecOf(t) {
  using namespace details;
  return vectorized_map_v(
      t, [=](auto tt, const T* pp) { return word::loadu(tt, pp); },
      StepPointer(t, p)
  );
}

/**
 * @brief Load a vector from an initializer list.
 * 
 * Convenience overload for creating vectors from brace-enclosed values.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param list Initializer list with at least size(t) elements
 * @return Loaded vector
 * 
 * @example
 *   Tag<float32_t, 4> t;
 *   auto v = loadu(t, {1.0f, 2.0f, 3.0f, 4.0f});
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto loadu(Tag<T, N, P> t, std::initializer_list<T> list) -> VecOf(t) {
  CT_ASSERT(list.size() >= size(t), "insufficient elements: %zd v.s. %zd", (nint_t) list.size(), size(t));
  return loadu(t, (const T*) list.begin());
}

/**
 * @brief Load a vector from aligned memory.
 * 
 * Loads size(t) consecutive elements from memory starting at address p.
 * The pointer p must be aligned to DEFAULT_ALIGNMENT bytes.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Pointer to source memory (must be aligned)
 * @return Loaded vector
 * 
 * @warning Passing an unaligned pointer may cause crashes or performance issues.
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto load(Tag<T, N, P> t, const T* p) -> VecOf(t) {
  using namespace details;
  return vectorized_map_v(
      t, [=](auto tt, const T* pp) { return word::load(tt, pp); },
      StepPointer(t, p)
  );
}

/**
 * @brief Load a vector from an aligned initializer list.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param list Initializer list
 * @return Loaded vector
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto load(Tag<T, N, P> t, std::initializer_list<T> list) -> VecOf(t) {
  return load(t, (const T*) list.begin());
}

/**
 * @brief Load first n elements from unaligned memory, with default for rest.
 * 
 * Loads n consecutive elements from memory. Elements beyond n are filled
 * from default_v. This is useful for processing partial data at the end
 * of an array.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Pointer to source memory
 * @param n Number of elements to load (0 <= n <= size(t))
 * @param default_v Default values for elements beyond n
 * @return Vector with first n elements from memory, rest from default_v
 * 
 * @example
 *   Tag<float32_t, 4> t;
 *   float data[4] = {1, 2, 3, 4};
 *   auto v = loadu(t, data, 2, zeros(t));  // v = [1, 2, 0, 0]
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto loadu(Tag<T, N, P> t, const T* p, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  using namespace details;
  return vectorized_map_v(
      t, n, [=](auto tt, const T* p, auto v_d) { return word::loadu(tt, p); },
      [=](auto tt, nint_t rem, const T* p, auto v_d) { return word::loadu(tt, p, rem, v_d); },
      StepPointer(t, p), ShardVec(t, default_v)
  );
}

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto loadu(Tag<T, N, P> t, const T* p, nint_t n, T default_v = T()) -> VecOf(t) {
  return loadu(t, p, n, fill(t, default_v));
}

/**
 * @brief Load first n elements from aligned memory, with default for rest.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Pointer to source memory (must be aligned)
 * @param n Number of elements to load
 * @param default_v Default values for elements beyond n
 * @return Vector with loaded elements
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto load(Tag<T, N, P> t, const T* p, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  using namespace details;
  return vectorized_map_v(
      t, n, [=](auto tt, const T* p, auto v_d) { return word::load(tt, p); },
      [=](auto tt, nint_t rem, const T* p, auto v_d) { return word::load(tt, p, rem, v_d); },
      StepPointer(t, p), ShardVec(t, default_v)
  );
}

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto load(Tag<T, N, P> t, const T* p, nint_t n, T default_v = T()) {
  return load(t, p, n, fill(t, default_v));
}

/**
 * @brief Masked load from unaligned memory.
 * 
 * For each lane i where mask[i] is true, loads from p[i]. For lanes
 * where mask is false, takes the value from default_v[i].
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Pointer to source memory
 * @param m Mask indicating which lanes to load
 * @param default_v Default values for masked-out lanes
 * @return Vector with masked-loaded elements
 * 
 * @note This operation may load from masked-out addresses; ensure
 *       those addresses are valid even if the values won't be used.
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto loadu(Tag<T, N, P> t, const T* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) {
  using namespace details;
  return vectorized_map_v(
      t, [=](auto tt, const T* p, auto mm, auto v_d) { return word::loadu(tt, p, mm, v_d); },
      StepPointer(t, p), ShardMask(t, m), ShardVec(t, default_v)
  );
}

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto loadu(Tag<T, N, P> t, const T* p, MaskOf(t) m, T default_v = T()) -> VecOf(t) {
  return loadu(t, p, m, fill(t, default_v));
}

/**
 * @brief Masked load from aligned memory.
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
CT_ALWAYS_FORCEINLINE
auto load(Tag<T, N, P> t, const T* p, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) {
  using namespace details;
  return vectorized_map_v(
      t, [=](auto tt, const T* p, auto mm, auto v_d) { return word::load(tt, p, mm, v_d); },
      StepPointer(t, p), ShardMask(t, m), ShardVec(t, default_v)
  );
}

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto load(Tag<T, N, P> t, const T* p, MaskOf(t) m, T default_v = T()) -> VecOf(t) {
  return load(t, p, m, fill(t, default_v));
}

/**
 * @brief Store a vector to unaligned memory.
 * 
 * Stores size(t) consecutive elements to memory starting at address p.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Pointer to destination memory
 * @param v The vector to store
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
void storeu(Tag<T, N, P> t, T* p, VecOf(t) v) {
  using namespace details;
  return vectorized_foreach(
      t, [=](auto tt, T* pp, auto&& vv) { word::storeu(tt, pp, vv); },
      StepPointer(t, p), ShardVec(t, v)
  );
}

/**
 * @brief Store a vector to aligned memory.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Pointer to destination memory (must be aligned)
 * @param v The vector to store
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
void store(Tag<T, N, P> t, T* p, VecOf(t) v) {
  using namespace details;
  return vectorized_foreach(
      t, [=](auto tt, T* pp, auto&& vv) { word::store(tt, pp, vv); },
      StepPointer(t, p), ShardVec(t, v)
  );
}

/**
 * @brief Store first n elements of a vector to unaligned memory.
 * 
 * Only the first n elements are written to memory.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Pointer to destination memory
 * @param n Number of elements to store (0 <= n <= size(t))
 * @param v The vector to store
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
void storeu(Tag<T, N, P> t, T* p, nint_t n, VecOf(t) v) {
  using namespace details;
  return vectorized_foreach(
      t, n, [=](auto tt, T* pp, auto&& vv) { word::storeu(tt, pp, vv); },
      [=](auto tt, nint_t rem, T* pp, auto&& vv) { word::storeu(tt, pp, rem, vv); },
      StepPointer(t, p), ShardVec(t, v)
  );
}

/**
 * @brief Store first n elements of a vector to aligned memory.
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
CT_ALWAYS_FORCEINLINE
void store(Tag<T, N, P> t, T* p, nint_t n, VecOf(t) v) {
  using namespace details;
  return vectorized_foreach(
      t, n, [=](auto tt, T* pp, auto&& vv) { word::store(tt, pp, vv); },
      [=](auto tt, nint_t rem, T* pp, auto&& vv) { word::store(tt, pp, rem, vv); },
      StepPointer(t, p), ShardVec(t, v)
  );
}

/**
 * @brief Masked store to unaligned memory.
 * 
 * For each lane i where mask[i] is true, stores v[i] to p[i].
 * Masked-out lanes are not written.
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
CT_ALWAYS_FORCEINLINE
void storeu(Tag<T, N, P> t, T* p, MaskOf(t) m, VecOf(t) v) {
  using namespace details;
  return vectorized_foreach(
      t, [=](auto tt, T* pp, auto&& mm, auto&& vv) { word::storeu(tt, pp, mm, vv); },
      StepPointer(t, p), ShardMask(t, m), ShardVec(t, v)
  );
}

/**
 * @brief Masked store to aligned memory.
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
CT_ALWAYS_FORCEINLINE
void store(Tag<T, N, P> t, T* p, MaskOf(t) m, VecOf(t) v) {
  using namespace details;
  return vectorized_foreach(
      t, [=](auto tt, T* pp, auto&& mm, auto&& vv) { word::store(tt, pp, mm, vv); },
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
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Base pointer for gather
 * @param i Index vector (indices are signed integers of same size as T)
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
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto gather(Tag<T, N, P> t, const T* p, Vec<Tag<Index<T>, N, P>> i) -> VecOf(t) {
  using namespace details;
  Tag<Index<T>, N, P> it;
  return vectorized_map_v(
      t, [=](auto tt, T* pp, auto&& ii) { return word::gather(tt, pp, ii); },
      StepPointer(t, p), ShardVec(it, i)
  );
}

/**
 * @brief Gather first n elements using an index vector, with default for rest.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Base pointer for gather
 * @param i Index vector
 * @param n Number of elements to gather (0 <= n <= size(t))
 * @param default_v Default values for elements beyond n
 * @return Gathered vector
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto gather(Tag<T, N, P> t, const T* p, Vec<Tag<Index<T>, N, P>> i, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  using namespace details;
  Tag<Index<T>, N, P> it;
  return vectorized_map_v(
      t, n, [=](auto tt, T* pp, auto&& ii, auto&& vv) { return word::gather(tt, pp, ii); },
      [=](auto tt, nint_t rem, T* pp, auto&& ii, auto&& vv) { return word::gather(tt, pp, ii, rem, vv); },
      StepPointer(t, p), ShardVec(it, i), ShardVec(t, default_v)
  );
}

/**
 * @brief Gather first n elements using an index vector, with scalar default.
 * 
 * Convenience overload that broadcasts a scalar default value.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Base pointer for gather
 * @param i Index vector
 * @param n Number of elements to gather
 * @param default_v Scalar default value for elements beyond n
 * @return Gathered vector
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto gather(Tag<T, N, P> t, const T* p, Vec<Tag<Index<T>, N, P>> i, nint_t n, T default_v) -> VecOf(t) {
  return gather(t, p, i, n, fill(t, default_v));
}

/**
 * @brief Masked gather from memory using an index vector.
 * 
 * For each lane i where mask[i] is true, loads from p[index[i]].
 * For masked-out lanes, takes the value from default_v[i].
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
 * 
 * @note May access p[index[i]] for masked-out lanes; ensure indices are valid.
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto gather(Tag<T, N, P> t, const T* p, Vec<Tag<Index<T>, N, P>> i, MaskOf(t) m, VecOf(t) default_v) -> VecOf(t) {
  using namespace details;
  Tag<Index<T>, N, P> it;
  return vectorized_map_v(
      t, [=](auto tt, T* pp, auto&& ii, auto&& mm, auto&& vv) { return word::gather(tt, pp, ii, mm, vv); },
      StepPointer(t, p), ShardVec(it, i), ShardMask(t, m), ShardVec(t, default_v)
  );
}

/**
 * @brief Masked gather with scalar default.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Base pointer for gather
 * @param i Index vector
 * @param m Mask indicating which lanes to gather
 * @param default_v Scalar default value for masked-out lanes
 * @return Gathered vector
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto gather(Tag<T, N, P> t, const T* p, Vec<Tag<Index<T>, N, P>> i, MaskOf(t) m, T default_v) -> VecOf(t) {
  return gather(t, p, i, m, fill(t, default_v));
}

/**
 * @brief Scatter elements to memory using an index vector.
 * 
 * For each lane i, stores v[i] to p[index[i]].
 * This is the vectorized equivalent of:
 *   p[index[i]] = v[i]
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param p Base pointer for scatter
 * @param i Index vector
 * @param v The vector to scatter
 * 
 * @warning If indices are not unique, the result depends on execution order.
 *          Multiple writes to the same location may race.
 * 
 * @note Scatter operations can be significantly slower than consecutive stores.
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
void scatter(Tag<T, N, P> t, T* p, Vec<Tag<Index<T>, N, P>> i, VecOf(t) v) {
  using namespace details;
  Tag<Index<T>, N, P> it;
  return vectorized_foreach(
      t, [=](auto tt, T* pp, auto&& ii, auto&& vv) { word::scatter(tt, pp, ii, vv); },
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
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
void scatter(Tag<T, N, P> t, T* p, Vec<Tag<Index<T>, N, P>> i, VecOf(t) v, nint_t n) {
  using namespace details;
  Tag<Index<T>, N, P> it;
  return vectorized_foreach(
      t, n, [=](auto tt, T* pp, auto&& ii, auto&& vv) { word::scatter(tt, pp, ii, vv); },
      [=](auto tt, nint_t rem, T* pp, auto&& ii, auto&& vv) { word::scatter(tt, pp, ii, vv, rem); },
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
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
void scatter(Tag<T, N, P> t, T* p, Vec<Tag<Index<T>, N, P>> i, VecOf(t) v, MaskOf(t) m) {
  using namespace details;
  Tag<Index<T>, N, P> it;
  return vectorized_foreach(
      t, [=](auto tt, T* pp, auto&& ii, auto&& vv, auto&& mm) { word::scatter(tt, pp, ii, vv, mm); },
      StepPointer(t, p), ShardVec(it, i), ShardVec(t, v), ShardMask(t, m)
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
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param v The vector
 * @param index Element index (0 <= index < size(t))
 * @return The element at the specified index
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
T get(Tag<T, N, P> t, VecOf(t) v, nint_t index) {
  nint_t ws = word_size(t);
  nint_t ord = index / ws, off = index % ws;
  auto word = get_word(t, v, ord);
  return word::get(word_tag(t), word, off);
}

/**
 * @brief Get a single element from a mask by index.
 * 
 * @warning This operation is relatively slow. Avoid using in hot loops.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param m The mask
 * @param index Element index (0 <= index < size(t))
 * @return The boolean value at the specified index
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
T get(Tag<T, N, P> t, MaskOf(t) m, nint_t index) {
  nint_t ws = word_size(t);
  nint_t ord = index / ws, off = index % ws;
  auto word = get_word_mask(t, m, ord);
  return word::get(word_tag(t), word, off);
}

/**
 * @brief Set a single element in a vector by index.
 * 
 * @warning This operation is relatively slow as it requires modifying
 *          the SIMD register. Avoid using in performance-critical loops.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param v The original vector
 * @param index Element index (0 <= index < size(t))
 * @param x The new value
 * @return Vector with the element at index set to x
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto set(Tag<T, N, P> t, VecOf(t) v, nint_t index, T x) -> VecOf(t) {
  nint_t ws = word_size(t);
  nint_t ord = index / ws, off = index % ws;
  auto word = get_word(t, v, ord);
  return set_word(word_tag(t), v, ord, word::set(t, word, off, x));
}

/**
 * @brief Set a single element in a mask by index.
 * 
 * @warning This operation is relatively slow. Avoid using in hot loops.
 * 
 * @tparam T Element type
 * @tparam N Nominal size
 * @tparam P Size multiplier
 * @param t The vector tag
 * @param m The original mask
 * @param index Element index (0 <= index < size(t))
 * @param x The new boolean value
 * @return Mask with the element at index set to x
 */
template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto set(Tag<T, N, P> t, MaskOf(t) m, nint_t index, bool x) -> MaskOf(t) {
  nint_t ws = word_size(t);
  nint_t ord = index / ws, off = index % ws;
  auto word = get_word_mask(t, m, ord);
  return set_word_mask(word_tag(t), m, ord, word::set(t, word, off, x));
}

} // namespace ct::tl::vec

#endif //CTORCH_VEC_H
