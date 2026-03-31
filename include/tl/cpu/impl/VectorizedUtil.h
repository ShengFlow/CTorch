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
 * @tparam T Tag type
 */
template <TLV_DECL_TAG(T)>
struct ShardVec {
  using Tag = T;
  static constexpr T tag{};
  Vec<T>& v;

  TLV_INLINE
  constexpr ShardVec(T t, Vec<T>& v) : v(v) {}
};

/**
 * @brief Wrapper for a mask that should be split into words.
 *
 * Used internally to pass masks to vectorized_map functions where
 * each word should be processed separately.
 *
 * @tparam T Tag type
 */
template <TLV_DECL_TAG(T)>
struct ShardMask {
  using Tag = T;
  static constexpr T tag{};
  Mask<T>& m;

  TLV_INLINE
  constexpr ShardMask(T t, Mask<T>& m) : m(m) {}
};

/**
 * @brief A pointer with a non-contiguous stride.
 *
 * Used internally for operations that need to access data at fixed intervals,
 * such as when processing multiple words of a multi-word vector.
 *
 * @tparam E Element type
 */
template <typename E>
struct StepPointer {
  E* p;
  nint_t step;

  TLV_INLINE
  constexpr StepPointer(const E* p, nint_t step) : p(const_cast<E*>(p)), step(step) {}

  template <nint_t N, int P>
  TLV_INLINE
  constexpr StepPointer(Tag<E, N, P> t, const E* p) : StepPointer(p, word_size(t)) {}
};


template <nint_t NLoop, nint_t Step = 1, nint_t I = 0, typename = void /* SFINAE*/>
struct ForEach {
  static_assert((Step > 0 && I < NLoop) || (Step < 0 && I > NLoop));

  template <typename F, typename... Args>
  TLV_INLINE
  constexpr void operator()(F&& f, Args&& ... args) {
    f.template operator()<I>(std::forward<Args>(args)...);
    ForEach<NLoop, Step, I + Step>()(std::forward<F>(f), std::forward<Args>(args)...);
  }

  template <typename F, typename... Args>
  TLV_INLINE
  constexpr void operator()(nint_t n, F&& f, Args&& ... args) {
    if (I >= n - (Step - 1)) return;
    f.template operator()<I>(std::forward<Args>(args)...);
    ForEach<NLoop, Step, I + Step>()(n, std::forward<F>(f), std::forward<Args>(args)...);
  }
};

template <nint_t NLoop, nint_t Step, nint_t I>
struct ForEach<NLoop, Step, I, std::enable_if_t<(I >= NLoop)>> {
  template <typename F, typename... Args>
  TLV_INLINE
  constexpr void operator()(F&& f, Args&& ... args) {}

  template <typename F, typename... Args>
  TLV_INLINE
  constexpr void operator()(nint_t n, F&& f, Args&& ... args) {}
};

template <nint_t NLoop, nint_t Step = 1, nint_t I = 0, typename F, typename... Args>
TLV_INLINE constexpr void foreach(F&& f, Args&&... args) {
  ForEach<NLoop, Step, I>()(std::forward<F>(f), std::forward<Args>(args)...);
}

template <nint_t NLoop, nint_t Step = 1, nint_t I = 0, typename F, typename... Args>
TLV_INLINE constexpr void foreach(nint_t n, F&& f, Args&&... args) {
  ForEach<NLoop, Step, I>()(n, std::forward<F>(f), std::forward<Args>(args)...);
}

/**
 * @brief Default argument transformer that passes arguments through unchanged.
 *
 * Used by the vectorized_map infrastructure to transform arguments based on
 * their type. This base case does no transformation.
 *
 * @tparam Index The iteration index (unused in this case)
 * @tparam Batch The batching size (unused in this case)
 * @tparam T The argument type
 */
template <nint_t Index, nint_t Batch, typename T>
struct ArgTransform {
  static_assert((Batch & (Batch - 1)) == 0, "Batch is not power of 2");

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
template <nint_t Index, nint_t Batch, TLV_DECL_TAG(T)>
struct ArgTransform<Index, Batch, ShardVec<T>> {
  static_assert((Batch & (Batch - 1)) == 0, "Batch is not power of 2");

  TLV_INLINE
  constexpr decltype(auto) operator()(ShardVec<T>&& a) {
    if constexpr (Batch == 1) {
      return get_word<Index>(T(), a.v);
    } else {
      using Tw = decltype(word_tag(T()));
      static_assert(Tw::POW2 == 0);
      using Tb = Tag<TypeOf<Tw>, Tw::N, log2_floor(Batch)>;

      Vec<Tb> v_batch;
      foreach<Batch, 1>([&]<nint_t I>{
        v_batch = set_word<I>(Tb(), v_batch, get_word<Index * Batch + I>(T(), a.v));
      });
      return v_batch;
    }
  }

  TLV_INLINE
  constexpr decltype(auto) operator()(ShardVec<T>&& a, nint_t index) {
    if constexpr (Batch == 1) {
      return get_word(T(), a.v, index);
    } else {
      using Tw = decltype(word_tag(T()));
      static_assert(Tw::POW2 == 0);
      using Tb = Tag<TypeOf<Tw>, Tw::N, log2_floor(Batch)>;

      Vec<Tb> v_batch;
      for (nint_t i = 0; i < Batch; ++i) {
        v_batch = set_word(Tb(), v_batch, i, get_word(T(), a.v, index * Batch + i));
      }
      return v_batch;
    }
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
template <nint_t Index, nint_t Batch, TLV_DECL_TAG(T)>
struct ArgTransform<Index, Batch, ShardMask<T>> {
  static_assert((Batch & (Batch - 1)) == 0, "Batch is not power of 2");

  TLV_INLINE
  constexpr decltype(auto) operator()(ShardMask<T>&& a) {
    if constexpr (Batch == 1) {
      return get_word_mask<Index>(T(), a.m);
    } else {
      using Tw = decltype(word_tag(T()));
      static_assert(Tw::POW2 == 0);
      using Tb = Tag<TypeOf<Tw>, Tw::N, log2_floor(Batch)>;

      Mask<Tb> m_batch;
      foreach<Batch, 1>([&]<nint_t I>{
        m_batch = set_word_mask<I>(Tb(), m_batch, get_word_mask<Index * Batch + I>(T(), a.m));
      });
      return m_batch;
    }
  }

  TLV_INLINE
  constexpr decltype(auto) operator()(ShardMask<T>&& a, nint_t index) {
    if constexpr (Batch == 1) {
      return get_word_mask(T(), a.m, index);
    } else {
      using Tw = decltype(word_tag(T()));
      static_assert(Tw::POW2 == 0);
      using Tb = Tag<TypeOf<Tw>, Tw::N, log2_floor(Batch)>;

      Mask<Tb> m_batch;
      for (nint_t i = 0; i < Batch; ++i) {
        m_batch = set_word_mask(Tb(), m_batch, i, get_word_mask(T(), a.m, index * Batch + i));
      }
      return m_batch;
    }
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
template <nint_t Index, nint_t Batch, typename T>
struct ArgTransform<Index, Batch, StepPointer<T>> {
  TLV_INLINE
  constexpr decltype(auto) operator()(StepPointer<T>&& a) {
    return this->operator()(std::forward<StepPointer<T>>(a), Index);
  }

  TLV_INLINE
  constexpr decltype(auto) operator()(StepPointer<T>&& a, nint_t index) {
    return a.p + index * Batch * a.step;
  }
};

template <nint_t Index, nint_t Batch = 1, typename T>
TLV_INLINE constexpr decltype(auto) transform(T&& t) {
  return ArgTransform<Index, Batch, T>()(std::forward<T>(t));
}

template <nint_t Index = -1, nint_t Batch = 1, typename T>
TLV_INLINE constexpr decltype(auto) transform(T&& t, nint_t index) {
  return ArgTransform<Index, Batch, T>()(std::forward<T>(t), index);
}

template <int I, typename Fn>
struct IndexedFn {
  Fn& fn;

  template <typename U, TL_IF(!std::is_same_v<std::decay_t<U>, IndexedFn>)>
  explicit IndexedFn(U& u) : fn(u) {}

private:
  template <typename F, typename... Args>
  static auto test_template_call(int)
  -> decltype(std::declval<F>().template operator()<I>(std::declval<Args>()...), std::true_type{});

  template <typename F, typename... Args>
  static auto test_template_call(...) -> std::false_type;

  template <typename F, typename... Args>
  static constexpr bool has_template_call_v =
      decltype(test_template_call<F, Args...>(0))::value;

  template <typename... Args>
  auto call_impl(std::true_type, Args&&... args) {
    return fn.template operator()<I>(std::forward<Args>(args)...);
  }

  template <typename... Args>
  auto call_impl(std::true_type, Args&&... args) const {
    return fn.template operator()<I>(std::forward<Args>(args)...);
  }

  template <typename... Args>
  auto call_impl(std::false_type, Args&&... args) {
    return fn(std::forward<Args>(args)...);
  }

  template <typename... Args>
  auto call_impl(std::false_type, Args&&... args) const {
    return fn(std::forward<Args>(args)...);
  }

public:
  template <typename... Args>
  auto operator()(Args&&... args) {
    return call_impl(std::bool_constant<has_template_call_v<Fn&, Args...>>{},
                     std::forward<Args>(args)...);
  }

  template <typename... Args>
  auto operator()(Args&&... args) const {
    return call_impl(std::bool_constant<has_template_call_v<const Fn&, Args...>>{},
                     std::forward<Args>(args)...);
  }
};

template <typename F, typename... Args>
using TransformedReturn = decltype(std::declval<IndexedFn<0, F>>()(transform<0>(std::forward<Args>(std::declval<Args>()))...));

/**
 * @brief Apply a word-level operation without returning a result.
 *
 * Used for operations like store that don't produce a value.
 *
 * @tparam T The vector tag type
 * @tparam Fn Function type
 * @tparam Args Argument types
 * @param t The vector tag
 * @param f The operation to apply
 * @param args Arguments to pass
 */
template <TLV_DECL_TAG(T), typename Fn, typename... Args>
requires (!is_tag<std::remove_cvref_t<Fn>>)
TLV_INLINE auto vmap(T t, Fn&& f, Args&& ... args) -> std::enable_if_t<
  std::is_void_v<TransformedReturn<Fn, WordOf<T>, Args...>>,
  void>
{
  constexpr auto wt = word_tag(t);
  if constexpr (is_word_vec(t)) {
    IndexedFn<0, Fn>{f}(wt, transform<0>(std::forward<Args>(args))...);
  } else {
    constexpr nint_t nloop = num_words(t);
    foreach<nloop>([&]<nint_t I>{
      IndexedFn<I, Fn>{f}(wt, transform<I>(std::forward<Args>(args))...);
    });
  }
}

/**
 * @brief Apply word-level operations with a partial (tail) handling.
 *
 * This variant handles operations that may only process n elements,
 * where n <= size(t). It uses f_complete for full words and f_tail
 * for the partial last word.
 *
 * @tparam T The vector tag type
 * @tparam FnC Function type for complete words
 * @tparam FnT Function type for tail (partial) word
 * @tparam Args Argument types
 * @param t The vector tag
 * @param n Number of elements to process (0 <= n <= size(t))
 * @param f_complete Function for full words: f(word_tag, args...), does not support constexpr index I.
 * @param f_tail Function for tail: f(word_tag, remaining, args...), does not support constexpr index I.
 * @param args Arguments to pass
 * @return The result vector
 */
template <TLV_DECL_TAG(T), typename FnC, typename FnT, typename... Args>
TLV_INLINE auto vmap(T t, nint_t n, FnC&& f_complete, FnT&& f_tail, Args&& ... args) -> std::enable_if_t<
    std::is_void_v<TransformedReturn<FnC, WordOf<T>, Args...>>,
    void>
{
  // FnT should also returns void
  static_assert(std::is_same_v<TransformedReturn<FnT, WordOf<T>, nint_t, Args...>, void>);
  constexpr auto wt = word_tag(t);
  nint_t L = size(t);
  CT_ASSERT(0 <= n && n <= L, "%zd !in 0..%zd", n, L);

  if constexpr (is_word_vec(t)) {
    std::forward<FnT>(f_tail)(wt, n, transform<0>(std::forward<Args>(args))...);
  } else {
    constexpr nint_t nloop = num_words(t);
    nint_t ws = word_size(t);
    nint_t full_nloop = n / ws;
    nint_t rem = n % ws;

    foreach<nloop>(full_nloop, [&]<nint_t I>{
      f_complete(wt, transform<I>(std::forward<Args>(args))...);
    });
    if (rem > 0) {
      std::forward<FnT>(f_tail)(wt, rem, transform(std::forward<Args>(args), full_nloop)...);
    }
    return;
  }
}

/**
 * @brief Apply a word-level operation to produce a vector result.
 *
 * This is the core infrastructure for vector operations. It:
 * 1. Checks if the vector is a single word (fast path)
 * 2. If multi-word, iterates over all words, applying f to each
 * 3. Collects results into the output vector
 *
 * @tparam T The vector tag type
 * @tparam Fn Function type with
 *   - Vec<decltype(word_tag)> operator()(word_tag, args...), or
 *   - Vec<decltype(word_tag)> template operator()<Index>(word_tag, args...)
 * @tparam Args Argument types (may include ShardVec, ShardMask, StepPointer)
 * @param t The vector tag
 * @param f The operation to apply to each word
 * @param args Arguments to pass (will be transformed per-word)
 * @return The result vector
 */
template <TLV_DECL_TAG(T), typename Fn, typename... Args>
requires (!is_tag<std::remove_cvref_t<Fn>>)
TLV_INLINE auto vmap(T t, Fn&& f, Args&& ... args) -> std::enable_if_t<
    vec::is_vec<TransformedReturn<Fn, WordOf<T>, Args...>>,
    Vec<T>>
{
  constexpr auto wt = word_tag(t);
  if constexpr (is_word_vec(t)) {
    return IndexedFn<0, Fn>{f}(wt, transform<0>(std::forward<Args>(args))...);
  } else {
    constexpr nint_t nloop = num_words(t);
    Vec<T> r;
    foreach<nloop>([&]<nint_t I>{
      auto out = IndexedFn<I, Fn>{f}(wt, transform<I>(std::forward<Args>(args))...);
      r = set_word<I>(t, r, std::move(out));
    });
    return r;
  }
}

/**
 * @brief Apply word-level operations with partial handling, without returning a result.
 *
 * Used for store operations with partial element counts.
 *
 * @tparam T The vector tag type
 * @tparam FnC Function type for complete words
 * @tparam FnT Function type for tail word
 * @tparam Args Argument types
 * @param t The vector tag
 * @param n Number of elements to process
 * @param f_complete Function for full words
 * @param f_tail Function for tail word
 * @param args Arguments to pass
 */
template <TLV_DECL_TAG(T), typename FnC, typename FnT, typename... Args>
TLV_INLINE auto vmap(T t, nint_t n, FnC&& f_complete, FnT&& f_tail, Args&& ... args) -> std::enable_if_t<
    is_vec<TransformedReturn<FnC, WordOf<T>, Args...>>,
    Vec<T>>
{
  // FnT should also returns void
  static_assert(std::is_same_v<TransformedReturn<FnT, WordOf<T>, nint_t, Args...>, void>);
  constexpr auto wt = word_tag(t);
  nint_t L = size(t);
  CT_ASSERT(0 <= n && n <= L, "%zd !in 0..%zd", n, L);

  if constexpr (is_word_vec(t)) {
    return std::forward<FnT>(f_tail)(wt, n, transform<0>(std::forward<Args>(args))...);
  } else {
    constexpr nint_t nloop = num_words(t);
    nint_t ws = word_size(t);
    nint_t full_nloop = n / ws;
    nint_t rem = n % ws;

    Vec<T> r;
    foreach<nloop>(full_nloop, [&]<nint_t I>{
      r = set_word<I>(t, r, f_complete(wt, transform<I>(std::forward<Args>(args))...));
    });
    if (rem > 0) {
      r = set_word(
          t, r, full_nloop,
          std::forward<FnT>(f_tail)(wt, rem, transform(std::forward<Args>(args), full_nloop)...)
      );
    }
    return r;
  }
}

/**
 * @brief Apply a word-level operation to produce a mask result.
 *
 * Similar to vectorized_map_v, but for operations that return masks.
 *
 * @tparam T The vector tag type
 * @tparam Fn Function type with
 *   - Mask<decltype(word_tag)> operator()(word_tag, args...), or
 *   - Mask<decltype(word_tag)> template operator()<Index>(word_tag, args...)
 * @tparam Args Argument types
 * @param t The vector tag
 * @param f The operation to apply to each word
 * @param args Arguments to pass
 * @return The result mask
 */
template <TLV_DECL_TAG(T), typename Fn, typename... Args>
requires (!is_tag<std::remove_cvref_t<Fn>>)
TLV_INLINE auto vmap(T t, Fn&& f, Args&& ... args) -> std::enable_if_t<
    vec::is_mask<TransformedReturn<Fn, WordOf<T>, Args...>>,
    Mask<T>>
{
  constexpr auto wt = word_tag(t);
  if constexpr (is_word_vec(t)) {
    return IndexedFn<0, Fn>{f}(wt, transform<0>(std::forward<Args>(args))...);
  } else {
    constexpr nint_t nloop = num_words(t);
    Mask<T> r;
    foreach<nloop>([&]<nint_t I>{
      auto out = IndexedFn<I, Fn>{f}(wt, transform<I>(std::forward<Args>(args))...);
      r = set_word_mask<I>(t, r, std::move(out));
    });
    return r;
  }
}
} // namespace ct::tl::vec::details

#endif //CTORCH_VECTORIZEDUTIL_H
