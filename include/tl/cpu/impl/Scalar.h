//
// Created by renyz on 2026/3/17.
//

#ifndef CTORCH_SCALAR_H
#define CTORCH_SCALAR_H

#include "tl/cpu/VecBase.h"

namespace ct::tl::vec {
namespace word {
/* ************************************************************************** */
//                               Constructors                                 //
/* ************************************************************************** */

template <typename T, nint_t N, int P>
auto fill(Tag<T, N, P> t, T value) -> VecOf(t) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  VecOf(t) v;
  v.fill(value);
  return v;
}

template <typename T, nint_t N, int P>
auto mfill(Tag<T, N, P> t, bool value) -> MaskOf(t) {
  static_assert(is_word_vec(t));
  MaskOf(t) m; // default inits to zero
  if (value) m.set();
  return m;
}

template <typename T, nint_t N, int P>
auto mwhilelt(Tag<T, N, P> t, nint_t a, nint_t b) -> MaskOf(t) {
  static_assert(is_word_vec(t));
  // Fast path
  if constexpr (size(t) <= sizeof(unsigned long long) * CHAR_BIT) {
    nint_t end = std::max(b - a, nint_t(0));
    // 64 = (0 - 1) -> 0xffffffff_ffffffffuLL
    auto bits = ((1uLL) << end) - 1;
    return { bits };
  } else {
    nint_t end = std::min(b - a, size(t));
    MaskOf(t) m;
    for (nint_t i = 0; i < end; ++i) {
      m.set(i);
    }
    return m;
  }
}

template <typename T, nint_t N, int P>
auto mwhilege(Tag<T, N, P> t, nint_t a, nint_t b) -> MaskOf(t) {
  static_assert(is_word_vec(t));
  // Fast path
  if constexpr (size(t) <= sizeof(unsigned long long) * CHAR_BIT) {
    nint_t end = std::max(b - a, nint_t(0));
    // 64 = (0 - 1) -> 0xffffffff_ffffffffuLL
    auto bits = (1uLL << end) - 1;
    return { ~bits };
  } else {
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

template <typename T, nint_t N, int P>
auto load(Tag<T, N, P> t, const T* p) -> VecOf(t) {
  CT_ASSERT(((nuint_t)(p) & (DEFAULT_ALIGNMENT - 1)) == 0, "Not aligned");
  return word::loadu(t, p);
}

template <typename T, nint_t N, int P>
auto loadu(Tag<T, N, P> t, const T* p, nint_t n, VecOf(t) default_v) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  VecOf(t) v;
  nint_t i;
  for (i = 0; i < n; ++i) {
    v[i] = p[i];
  }
  for (; i < size(t); ++i) {
    v[i] = default_v[i];
  }
  return v;
}

template <typename T, nint_t N, int P>
auto load(Tag<T, N, P> t, const T* p, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  CT_ASSERT(((nuint_t)(p) & (DEFAULT_ALIGNMENT - 1)) == 0, "Not aligned");
  return word::loadu(t, p, n, default_v);
}

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

template <typename T, nint_t N, int P>
auto load(Tag<T, N, P> t, const T* p, MaskOf(t) m, VecOf(t) default_v) {
  CT_ASSERT(((nuint_t)(p) & (DEFAULT_ALIGNMENT - 1)) == 0, "Not aligned");
  return word::loadu(t, p, m, default_v);
}

template <typename T, nint_t N, int P>
void storeu(Tag<T, N, P> t, T* p, VecOf(t) v) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  for (nint_t i = 0; i < size(t); ++i) {
    p[i] = v[i];
  }
}

template <typename T, nint_t N, int P>
void store(Tag<T, N, P> t, T* p, VecOf(t) v) {
  CT_ASSERT(((nuint_t)(p) & (DEFAULT_ALIGNMENT - 1)) == 0, "Not aligned");
  return word::storeu(t, p, v);
}

template <typename T, nint_t N, int P>
void storeu(Tag<T, N, P> t, T* p, nint_t n, VecOf(t) v) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  for (nint_t i = 0; i < n; ++i) {
    p[i] = v[i];
  }
}

template <typename T, nint_t N, int P>
void store(Tag<T, N, P> t, T* p, nint_t n, VecOf(t) v) {
  CT_ASSERT(((nuint_t)(p) & (DEFAULT_ALIGNMENT - 1)) == 0, "Not aligned");
  return word::storeu(t, p, n, v);
}

template <typename T, nint_t N, int P>
void storeu(Tag<T, N, P> t, T* p, MaskOf(t) m, VecOf(t) v) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  for (nint_t i = 0; i < size(t); ++i) {
    if (m[i]) p[i] = v[i];
  }
}

template <typename T, nint_t N, int P>
void store(Tag<T, N, P> t, T* p, MaskOf(t) m, VecOf(t) v) {
  CT_ASSERT(((nuint_t)(p) & (DEFAULT_ALIGNMENT - 1)) == 0, "Not aligned");
  return word::storeu(t, p, m, v);
}

/* ************************************************************************** */
//                         Indexed gather & scatter                           //
/* ************************************************************************** */
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

template <typename T, nint_t N, int P>
auto gather(Tag<T, N, P> t, const T* p, Vec<Tag<Index<T>, N, P>> i, nint_t n, VecOf(t) default_v) -> VecOf(t) {
  static_assert(is_default_impl(t));
  static_assert(is_default_impl(Tag<Index<T>, N, P>()));
  static_assert(is_word_vec(t));
  static_assert(is_word_vec(Tag<Index<T>, N, P>()));
  CT_ASSERT(0 <= n && n <= size(t), "%zd !in 0..%zd", n, size(t));
  VecOf(t) v;
  nint_t j;
  for (j = 0; j < n; ++j) {
    v[j] = p[(nint_t) i[j]];
  }
  for(; j < size(t); ++j) {
    v[j] = default_v[j];
  }
  return v;
}

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

template <typename T, nint_t N, int P>
T get(Tag<T, N, P> t, VecOf(t) v, nint_t index) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  return v[index];
}

template <typename T, nint_t N, int P>
bool get(Tag<T, N, P> t, MaskOf(t) v, nint_t index) {
  static_assert(is_word_vec(t));
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  return v[index];
}

template <typename T, nint_t N, int P>
auto set(Tag<T, N, P> t, VecOf(t) v, nint_t index, T x) -> VecOf(t) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  VecOf(t) u = v;
  u[index] = x;
  return u;
}

template <typename T, nint_t N, int P>
auto set(Tag<T, N, P> t, MaskOf(t) v, nint_t index, bool x) -> MaskOf(t) {
  static_assert(is_default_impl(t));
  static_assert(is_word_vec(t));
  CT_ASSERT(0 <= index && index < size(t), "%zd !in 0..%zd", index, size(t));
  MaskOf(t) u = v;
  u[index] = x;
  return u;
}

} // namespace word
} // namespace ct::tl::vec

#endif //CTORCH_SCALAR_H
