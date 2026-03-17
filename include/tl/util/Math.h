//
// Created by renyz on 2026/3/17.
//

#ifndef CTORCH_MATH_H
#define CTORCH_MATH_H

#include <bit>

namespace ct::tl {

template <typename T>
constexpr int log2_floor(T x) noexcept {
  if (x == 0) return -1;
  return std::bit_width(std::make_unsigned_t<T>(x)) - 1;
}

} // ct::tl

#endif //CTORCH_MATH_H
