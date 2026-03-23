//
// Created by renyz on 2026/3/21.
//

#ifndef CTORCH_SCALAR_CONVERSIONS_H
#define CTORCH_SCALAR_CONVERSIONS_H

#include "tl/cpu/Vec.h"
#include "tl/util/ScalarConvert.h"

namespace ct::tl::vec {
namespace word {
namespace details {
template <typename T, typename V, TL_IF(is_word_vec(T())), TL_IF(is_default_impl(T())), TL_IF(is_default_impl(Vec2Tag<V>()))>
CT_ALWAYS_FORCEINLINE Vec<T> convert_impl(T t, V v) {
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

template <typename T, typename V, TL_IF(is_word_vec(T())), TL_IF(is_default_impl(T())), TL_IF(is_default_impl(Vec2Tag<V>())), TL_IF(sizeof(TypeOf<T>) > sizeof(TypeOf<Vec2Tag<V>>))>
CT_ALWAYS_FORCEINLINE Vec<T> promote(T t, V v) {
  return details::convert_impl(t, v);
}

template <typename T, typename V, TL_IF(is_word_vec(T())), TL_IF(is_default_impl(T())), TL_IF(is_default_impl(Vec2Tag<V>())), TL_IF(sizeof(TypeOf<T>) < sizeof(TypeOf<Vec2Tag<V>>))>
CT_ALWAYS_FORCEINLINE Vec<T> demote(T t, V v) {
  return details::convert_impl(t, v);
}

template <typename T, typename V, TL_IF(is_word_vec(T())), TL_IF(is_default_impl(T())), TL_IF(is_default_impl(Vec2Tag<V>())), TL_IF(sizeof(TypeOf<T>) == sizeof(TypeOf<Vec2Tag<V>>))>
CT_ALWAYS_FORCEINLINE Vec<T> convert(T t, V v) {
  return details::convert_impl(t, v);
}

} // namespace word

template <typename T, nint_t NOut, int POut, nint_t NIn, int PIn>
Vec<Tag<T, NOut, POut>> VecReshape<T, NOut, POut, NIn, PIn>::reshape(Tag<T, NIn, PIn> t_in, Vec<Tag<T, NIn, PIn>> v_in) const {
  Tag<T, NOut, POut> t_out;
  if constexpr (std::is_same_v<Vec<Tag<T, NOut, POut>>, decltype(v_in)>) {
    return v_in;
  }
  Vec<Tag<T, NOut, POut>> u;
  if constexpr (is_default_impl(t_in) && is_default_impl(t_out)) {
    if constexpr (is_word_vec(t_in)) {
      if constexpr (is_word_vec(t_out)) {
        static_assert(NIn == NOut);
        return v_in;
      } else {
        const T* p = v_in.data();
        for (nint_t i = 0; i < num_words(t_out); ++i, p += word_size(t_out)) {
          std::copy(p, p + word_size(t_out), u[i].data());
        }
      }
    } else if constexpr (is_word_vec(t_out)) {
      T* p = u.data();
      for (nint_t i = 0; i < num_words(t_in); ++i, p += word_size(t_in)) {
        std::copy(t_in[i].data(), t_in[i].data() + word_size(t_in), p);
      }
    } else {
      for (nint_t i = 0; i < size(t_in); ++i) {
        t_out[i / word_size(t_out)][i % word_size(t_out)] = t_in[i / word_size(t_in)][i % word_size(t_in)];
      }
    }
  } else if constexpr (is_default_impl(t_out)) {
    for (nint_t i = 0, j = size(t_in); i < j; ++i) {
      u[i] = vec::get(t_in, v_in, i);
    }
  } else {
    for (nint_t i = 0, j = size(t_in); i < j; ++i) {
      u = vec::set(t_out, u, i, vec::get(t_in, v_in, i));
    }
  }
  return u;
}

} // ct::tl::vec

#endif //CTORCH_SCALAR_CONVERSIONS_H
