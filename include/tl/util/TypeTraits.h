//
// Created by renyz on 2026/3/23.
//

#ifndef CTORCH_TYPETRAITS_H
#define CTORCH_TYPETRAITS_H

#include <type_traits>

#include "CoreDefs.h"

namespace ct::tl {
namespace details {

template <typename T, typename... TArgs>
struct IsAnyHelper {};

// Empty input type list defaults to false
template <typename T>
struct IsAnyHelper<T> {
  static constexpr bool value = false;
};

template <typename T, typename T1, typename... TArgs>
struct IsAnyHelper<T, T1, TArgs...> {
  static constexpr bool value = std::is_same_v<T, T1> || IsAnyHelper<T, TArgs...>::value;
};

} // namespace details

template <typename T, typename... TArgs>
static constexpr bool is_any = details::IsAnyHelper<T, TArgs...>::value;

template <typename T, typename ... TArgs>
static constexpr bool is_none = !is_any<T, TArgs...> || sizeof...(TArgs) == 0;

} // namespace ct::tl

#endif //CTORCH_TYPETRAITS_H
