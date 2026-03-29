//
// Created by renyz on 2026/3/18.
//

#ifndef CTORCH_VECTORIZEDUTIL_H
#define CTORCH_VECTORIZEDUTIL_H

#include "tl/cpu/VecBase.h"
#include "CoreDefs.h"

namespace ct::tl::vec::details {

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

  TLV_INLINE
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

  TLV_INLINE
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

  TLV_INLINE
  constexpr StepPointer(const T* p, nint_t step) : p(const_cast<T*>(p)), step(step) {}

  template <nint_t N, int P>
  TLV_INLINE
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
  TLV_INLINE
  constexpr decltype(auto) operator()(T&& a) {
    return std::forward<T>(a);
  }

  TLV_INLINE
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
  TLV_INLINE
  constexpr decltype(auto) operator()(ShardVec<T, N, P>&& a) {
    return get_word<Index>(a.tag, a.v);
  }

  TLV_INLINE
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
  TLV_INLINE
  constexpr decltype(auto) operator()(ShardMask<T, N, P>&& a) {
    return get_word_mask<Index>(a.tag, a.m);
  }

  TLV_INLINE
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
  TLV_INLINE
  constexpr decltype(auto) operator()(StepPointer<T>&& a) {
    return this->operator()(std::forward<StepPointer<T>>(a), Index);
  }

  TLV_INLINE
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
  TLV_INLINE
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
  TLV_INLINE
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
  TLV_INLINE
  constexpr void operator()(F&& f, TArgs&& ... args) {}

  template <typename F, typename... TArgs>
  TLV_INLINE
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
TLV_INLINE
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
TLV_INLINE
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
TLV_INLINE
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
TLV_INLINE
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
TLV_INLINE
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
TLV_INLINE
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
TLV_INLINE
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
TLV_INLINE
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
TLV_INLINE
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
} // namespace ct::tl::vec::details

#endif //CTORCH_VECTORIZEDUTIL_H
