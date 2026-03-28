//
// Created by renyz on 2026/3/21.
//

#ifndef CTORCH_SCALAR_CONVERSIONS_H
#define CTORCH_SCALAR_CONVERSIONS_H

#include "tl/cpu/Vec.h"
#include "tl/util/ScalarConvert.h"
#include "tl/util/TypeTraits.h"

namespace ct::tl::vec::CPU_CAPABILITY {
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

template <typename To, typename Vi, typename Ti = Vec2Tag<Vi>>
CT_ALWAYS_FORCEINLINE Vec<To> reshape(To t_out, Vi v_in) {
  static_assert(std::is_same_v<TypeOf<To>, TypeOf<Ti>>, "Not same type");
  if constexpr (std::is_same_v<Vec<To>, Vi>) {
    return v_in;
  }
  constexpr Ti t_in;
  constexpr nint_t NIn = Ti::N;
  constexpr nint_t NOut = To::N;
  using T = TypeOf<Ti>;
  Vec<To> v_out;
  if constexpr (is_default_impl(t_in) && is_default_impl(t_out)) {
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
  } else if constexpr (is_default_impl(t_out)) {
    for (nint_t i = 0, j = size(t_in); i < j; ++i) {
      v_out[i] = vec::get(t_in, v_in, i);
    }
  } else {
    for (nint_t i = 0, j = size(t_in); i < j; ++i) {
      v_out = vec::set(t_out, v_out, i, vec::get(t_in, v_in, i));
    }
  }
  return v_out;
}

} // namespace word
} // ct::tl::vec::CPU_CAPABILITY

#endif //CTORCH_SCALAR_CONVERSIONS_H
