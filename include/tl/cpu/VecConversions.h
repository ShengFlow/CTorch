//
// Created by renyz on 2026/3/21.
//

#ifndef CTORCH_VECCONVERSIONS_H
#define CTORCH_VECCONVERSIONS_H

#include "tl/cpu/VecBase.h"
#include "tl/cpu/Capabilities.h"

#if defined(CPU_CAPABILITY_AVX) || defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)
  #include "tl/cpu/impl/x86_Conversions.h"
#elif defined(CPU_CAPABILITY_NEON)
  #include "tl/cpu/impl/ARM_NEON_Conversions.h"
#elif defined(CPU_CAPABILITY_SVE)
  #include "tl/cpu/impl/ARM_SVE_Conversions.h"
#endif
// Always include scalar impl as fallback option
#include "tl/cpu/impl/Scalar_Conversions.h"

namespace ct::tl::vec {
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
template <typename TTag, typename TVec>
Vec<TTag> promote(TTag t, TVec v) {
  constexpr auto t_in = Vec2Tag<TVec>();
  constexpr auto t_out= t;
  constexpr nint_t nw_in = num_words(t_in);
  constexpr nint_t nw_out = num_words(t_out);
  constexpr auto wt_in = word_tag(t_in);
  constexpr auto wt_out = word_tag(t_out);
  using TIn = typename decltype(t_in)::Type;
  constexpr nint_t WNIn = decltype(wt_in)::N;
  using TOut = typename decltype(t_out)::Type;
  constexpr nint_t ANOut = decltype(t_out)::AdjustedN;
  constexpr nint_t WNOut = decltype(wt_out)::N;

  static_assert(sizeof(TIn) < sizeof(TOut));
  static_assert(!(is_scalable(t_in) ^ is_scalable(t_out)));

  constexpr nint_t factor = sizeof(TOut) / sizeof(TIn);
  constexpr nint_t nw_in_required = is_scalable(t_in)
      ? (nw_out + factor - 1) / factor
      : (ANOut + WNIn - 1) / WNIn;
  static_assert(nw_in_required <= nw_in, "Insufficient elements");
  using BatchTag = Tag<TOut, WNOut, log2_floor(factor)>;
  static_assert(num_words(BatchTag()) == factor, "Output element count mismatch");

  Vec<TTag> out;
  if constexpr (nw_out > 1) {
    details::ForEachTransformed<nw_in_required>()(
        [&]<nint_t I>() {
          auto v_in = get_word<I>(t_in, v);
          auto u0 = word::promote(Tag<TOut, WNIn>(), v_in);
          auto u = VecReshape<typename BatchTag::Type, BatchTag::N, BatchTag::POW2, WNIn, 0>().reshape({}, u0);

          details::ForEachTransformed<factor>()(
              [&]<nint_t J>() {
                out = set_word<I * factor + J>(t_out, out, get_word<J>(BatchTag(), u));
              }
          );
        }
    );
  } else {
    static_assert(nw_in_required == 1);
    auto batch_v = get_word<0>(t_in, v);
    auto u = word::promote(Tag<TOut, ANOut>(), batch_v);
    out = set_word<0>(t, out, u);
  }
  return out;
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
template <typename TTag, typename TVec>
Vec<TTag> demote(TTag t, TVec v) {
  constexpr auto t_in = Vec2Tag<TVec>();
  constexpr auto t_out= t;
  constexpr nint_t nw_in = num_words(t_in);
  constexpr nint_t nw_out = num_words(t_out);
  constexpr auto wt_in = word_tag(t_in);
  constexpr auto wt_out = word_tag(t_out);
  using TIn = typename decltype(t_in)::Type;
  constexpr nint_t WNIn = decltype(wt_in)::N;
  using TOut = typename decltype(t_out)::Type;
  constexpr nint_t ANOut = decltype(t_out)::AdjustedN;
  constexpr nint_t WNOut = decltype(wt_out)::N;

  static_assert(sizeof(TIn) > sizeof(TOut));
  static_assert(!(is_scalable(t_in) ^ is_scalable(t_out)));

  constexpr nint_t factor = sizeof(TIn) / sizeof(TOut);
  constexpr nint_t nw_in_required = is_scalable(t_in)
      ? (nw_out + factor - 1) / factor
      : (ANOut + WNIn - 1) / WNIn;
  static_assert(nw_in_required <= nw_in, "Insufficient elements");
  using BatchTag = Tag<TIn, WNIn, log2_floor(factor)>;

  Vec<TTag> out;
  if constexpr (nw_out > 1) {
    static_assert (nw_in_required == nw_out * factor);
    details::ForEachTransformed<nw_out>()(
        [&]<nint_t I>() {
          Vec<BatchTag> batch_v;
          details::ForEachTransformed<factor>()(
              [&]<nint_t J>() {
                batch_v = set_word<J>(BatchTag(), batch_v, get_word<I * factor + J>(t_in, v));
              }
          );
          auto u = word::demote(Tag<TOut, WNOut>(), batch_v);
          out = set_word<I>(t_out, out, u);
        }
    );
  } else {
    static_assert(nw_in_required <= factor);
    using MinibatchTag = Tag<TIn, WNIn, log2_floor(nw_in_required)>;
    Vec<MinibatchTag> batch_v;
    details::ForEachTransformed<nw_in_required>()(
        [&]<nint_t I>() {
          batch_v = set_word<I>(MinibatchTag(), batch_v, get_word<I>(t_in, v));
        }
    );
    auto u = word::demote(Tag<TOut, ANOut>(), batch_v);
    out = set_word<0>(t, out, u);
  }
  return out;
}

/**
 * Size of input vector must not smaller than requested output
 * Size of output dtype must be equals to size of input dtype.
 * Conversion rules:
 *   - all type conversions conform C++ standard.
 */
template <typename TTag, typename TVec>
Vec<TTag> convert(TTag t, TVec v) {
  constexpr auto t_in = Vec2Tag<TVec>();
  constexpr auto t_out= t;
  constexpr nint_t nw_in = num_words(t_in);
  constexpr nint_t nw_out = num_words(t_out);
  constexpr auto wt_in = word_tag(t_in);
  constexpr auto wt_out = word_tag(t_out);
  using TIn = typename decltype(t_in)::Type;
  constexpr nint_t WNIn = decltype(wt_in)::N;
  using TOut = typename decltype(t_out)::Type;
  constexpr nint_t ANOut = decltype(t_out)::AdjustedN;
  constexpr nint_t WNOut = decltype(wt_out)::N;

  static_assert(sizeof(TIn) == sizeof(TOut));
  static_assert(nw_in == nw_out);
  static_assert(!(is_scalable(t_in) ^ is_scalable(t_out)));

  Vec<TTag> out;
  if constexpr (nw_in > 1) {
    details::ForEachTransformed<nw_in>()(
        [&]<nint_t I>() {
          auto v_in = get_word<I>(t_in, v);
          auto u = word::convert(Tag<TOut, WNOut>(), v_in);
          out = set_word<I>(t_out, out, u);
        }
    );
  } else {
    auto v_in = get_word<0>(t_in, v);
    auto u = word::convert(Tag<TOut, ANOut>(), v_in);
    out = set_word<0>(t_out, out, u);
  }
  return out;
}

//template <typename TTag, typename TVec>
//Vec<TTag> bitcast(TTag t, TVec v) {
//  return {};
//}
//
//template <typename TTag, typename TVec>
//Vec<TTag> resize_bitcast(TTag t, TVec v) {
//  return {};
//}
//
//
//template <typename TTagTo, typename TTagFrom, typename TVec>
//Vec<TTagTo> zero_extend_resize_bitcast(TTagTo t_to, TTagFrom t_from, TVec v) {
//  return {};
//}

} // ct::tl::vec

#endif //CTORCH_VECCONVERSIONS_H
