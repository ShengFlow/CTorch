//
// Created by renyz on 2026/3/14.
//

#ifndef CTORCH_VEC_H
#define CTORCH_VEC_H

#include <cstdint>
#include <array>

#include "Assertion.h"
#include "CoreDefs.h"

namespace ct { namespace vec {
/**
 * Default bytes length of a vector, only used for default vec impl.
 */
static constexpr int64_t DEFAULT_LENGTH = 64;

template <typename T>
class Vec {
  static constexpr int64_t NUMEL = DEFAULT_LENGTH / sizeof(T);
  static_assert(DEFAULT_LENGTH % sizeof(T) == 0 && NUMEL > 0, "Bad element type");

  alignas(DEFAULT_LENGTH) T _data[NUMEL];
public:
  /**
   * Default constructor gives vector of undefined values
   */
  CT_ALWAYS_FORCEINLINE
  Vec() = default;

  CT_ALWAYS_FORCEINLINE
  Vec(T fill) {
    for (int i = 0; i < size(); ++i) _data[i] = fill;
  }

  template <typename... Args>
  CT_ALWAYS_FORCEINLINE
  Vec(Args... vals) : _data{vals...} {
    CT_ASSERT(sizeof...(vals) == size(), "Number of given values (%lld) != vector size (%lld)", sizeof...(vals), size());
  }

  CT_ALWAYS_FORCEINLINE
  Vec(std::initializer_list<T> vals) {
    CT_ASSERT(vals.size() == size(), "Number of given values (%lld) != vector size (%lld)", sizeof...(vals), size());
    for (int i = 0; i < size(); ++i) _data[i] = vals.begin()[i];
  }



  /**
   * Note: does not guarantee constexpr in specialized impl (i.e. SVE)
   */
  CT_ALWAYS_FORCEINLINE
  constexpr int64_t size() const {
    return NUMEL;
  }
}; // class Vec<T>
}} // namespace ct::vec

#endif //CTORCH_VEC_H
