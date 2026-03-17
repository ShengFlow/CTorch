//
// Created by renyz on 2026/3/16.
//

#ifndef CTORCH_VEC_H
#define CTORCH_VEC_H

#include "VecBase.h"

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

template <typename T, nint_t N, int P>
struct ShardVec {
  Tag<T, N, P> tag;
  Vec<Tag<T, N, P>>& v;

  CT_ALWAYS_FORCEINLINE
  constexpr ShardVec(Tag<T, N, P> t, Vec<Tag<T, N, P>>& v) : tag(t), v(v) {}
};

template <typename T, nint_t N, int P>
struct ShardMask {
  Tag<T, N, P> tag;
  Mask<Tag<T, N, P>>& m;

  CT_ALWAYS_FORCEINLINE
  constexpr ShardMask(Tag<T, N, P> t, Mask<Tag<T, N, P>>& m) : tag(t), m(m) {}
};

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

template <nint_t NLoop, nint_t Step = 1, nint_t I = 0, typename = void /* SFINAE*/>
struct ForEachTransformed {
  static_assert((Step > 0 && I < NLoop) || (Step < 0 && I > NLoop));

  template <typename F, typename... TArgs>
  CT_ALWAYS_FORCEINLINE
  constexpr void operator()(F&& f, TArgs&& ... args) {
    f.template operator()<I>(ArgTransform<I, std::remove_cvref_t<TArgs>>()(std::forward<TArgs>(args))...);
    ForEachTransformed<NLoop, Step, I + Step>()(std::forward<F>(f), std::forward<TArgs>(args)...);
  }

  template <typename F, typename... TArgs>
  CT_ALWAYS_FORCEINLINE
  constexpr void operator()(nint_t n, F&& f, TArgs&& ... args) {
    if (I >= n - (Step - 1)) return;
    f.template operator()<I>(ArgTransform<I, std::remove_cvref_t<TArgs>>()(std::forward<TArgs>(args))...);
    ForEachTransformed<NLoop, Step, I + Step>()(n, std::forward<F>(f), std::forward<TArgs>(args)...);
  }
};

template <nint_t NLoop, nint_t Step, nint_t I>
struct ForEachTransformed<NLoop, Step, I, std::enable_if_t<(I >= NLoop)>> {
  template <typename F, typename... TArgs>
  CT_ALWAYS_FORCEINLINE
  constexpr void operator()(F&& f, TArgs&& ... args) {}

  template <typename F, typename... TArgs>
  CT_ALWAYS_FORCEINLINE
  constexpr void operator()(nint_t n, F&& f, TArgs&& ... args) {}
};

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

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto fill(Tag<T, N, P> t, T value) -> VecOf(t) {
  using namespace details;
  return vectorized_map_v(
      t, [=](auto tt) { return word::fill(tt, value); }
  );
}

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto zeros(Tag<T, N, P> t) -> VecOf(t) {
  return fill(t, T());
}

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto mfill(Tag<T, N, P> t, bool value) -> MaskOf(t) {
  using namespace details;
  return vectorized_map_m(
      t, [=](auto tt) { return word::mfill(tt, value); }
  );
}

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto mtrue(Tag<T, N, P> t) -> MaskOf(t) {
  return mfill(t, true);
}

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto mfalse(Tag<T, N, P> t) -> MaskOf(t) {
  return mfill(t, false);
}

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto mwhilelt(Tag<T, N, P> t, nint_t a, nint_t b) {
  using namespace details;
  nint_t ws = word_size(t);
  return vectorized_map_m_indexed(
      t, [=] <nint_t I>(auto tt) { return word::mwhilelt(tt, a + I * ws, b); }
  );
}

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto mwhilele(Tag<T, N, P> t, nint_t a, nint_t b) {
  return mwhilelt(t, a, b + 1);
}

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto mwhilegt(Tag<T, N, P> t, nint_t a, nint_t b) {
  return mwhilege(t, a, b + 1);
}

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
 * Unaligned vector laod
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

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto loadu(Tag<T, N, P> t, std::initializer_list<T> list) -> VecOf(t) {
  CT_ASSERT(list.size() >= size(t), "insufficient elements: %zd v.s. %zd", (nint_t) list.size(), size(t));
  return loadu(t, (const T*) list.begin());
}

/**
 * Aligned vector load
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

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto load(Tag<T, N, P> t, std::initializer_list<T> list) -> VecOf(t) {
  return load(t, (const T*) list.begin());
}

/**
 * Unaligned load first n elements
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

/**
 * Aligned load first n elements
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

/**
 * Unaligned masked load elements
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

/**
 * Aligned masked load elements
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

/**
 * Unaligned vector store
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
 * Aligned vector store
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
 * Unaligned store first n elements
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
 * Aligned store first n elements
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
 * Unaligned masked vector store
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
 * Aligned masked vector store
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
 * Gather vector elements from index
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
 * Gather first n elements from index
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

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto gather(Tag<T, N, P> t, const T* p, Vec<Tag<Index<T>, N, P>> i, nint_t n, T default_v) -> VecOf(t) {
  return gather(t, p, i, n, fill(t, default_v));
}

/**
 * Masked gather vector elements from index
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

template <typename T, nint_t N, int P>
CT_ALWAYS_FORCEINLINE
auto gather(Tag<T, N, P> t, const T* p, Vec<Tag<Index<T>, N, P>> i, MaskOf(t) m, T default_v) -> VecOf(t) {
  return gather(t, p, i, m, fill(t, default_v));
}

/**
 * Scatter vector elements to index
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
 * Scatter first n elements to index
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
 * Masked scatter elements to index
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
 * Get element at specified index
 * Note: slow
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
 * Get element at specified index
 * Note: slow
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
 * Set element at specified index
 * Note: slow
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
 * Set element at specified index
 * Note: slow
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
