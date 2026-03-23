//
// Created by renyz on 2026/3/21.
//

#ifndef CTORCH_SCALARCONVERT_H
#define CTORCH_SCALARCONVERT_H

#include <type_traits>
#include <numeric>
#include <cmath>
#include "CoreDefs.h"

namespace ct::tl {

template <typename To, typename From>
CT_ALWAYS_FORCEINLINE constexpr To BitCast(const From& from) {
  static_assert(sizeof(To) == sizeof(From), "Size mismatch");
  union { From f; To t; } u {.f = from};
  return u.t;
}

// ========== bf16 ↔ float 转换 ==========

CT_ALWAYS_FORCEINLINE constexpr uint16_t Bf16ToBits(bfloat16_t bf) {
  return static_cast<uint16_t>(bf);
}

CT_ALWAYS_FORCEINLINE constexpr bfloat16_t BitsToBf16(uint16_t bits) {
  return static_cast<bfloat16_t>(bits);
}

CT_ALWAYS_FORCEINLINE constexpr float Bf16ToFloat(bfloat16_t bf) {
  uint32_t bits = static_cast<uint32_t>(Bf16ToBits(bf)) << 16;
  return BitCast<float>(bits);
}

CT_ALWAYS_FORCEINLINE constexpr bfloat16_t FloatToBf16(float f) {
  uint32_t bits = BitCast<uint32_t>(f);
  // 舍入到最近偶数
  uint32_t lsb = (bits >> 16) & 1;
  uint32_t rounding_bias = 0x7FFF + lsb;
  bits += rounding_bias;
  return BitsToBf16(static_cast<uint16_t>(bits >> 16));
}

// ========== float16 辅助 ==========

CT_ALWAYS_FORCEINLINE constexpr uint16_t Float16ToBits(float16_t f) {
  return BitCast<uint16_t>(f);
}

CT_ALWAYS_FORCEINLINE constexpr float16_t BitsToFloat16(uint16_t bits) {
  return BitCast<float16_t>(bits);
}

// ========== 整数饱和转换 ==========

template <typename TOut, typename TIn>
CT_ALWAYS_FORCEINLINE constexpr TOut SaturateIntToInt(TIn t) {
  using OutLimits = std::numeric_limits<TOut>;
  constexpr TOut out_min = OutLimits::lowest();
  constexpr TOut out_max = OutLimits::max();

  if constexpr (IsSignedV<TIn> && IsSignedV<TOut>) {
    // 有符号 → 有符号
    if (t < static_cast<TIn>(out_min)) return out_min;
    if (t > static_cast<TIn>(out_max)) return out_max;
  } else if constexpr (IsSignedV<TIn> && !IsSignedV<TOut>) {
    // 有符号 → 无符号
    if (t < 0) return 0;
    if (static_cast<uint64_t>(t) > static_cast<uint64_t>(out_max)) return out_max;
  } else if constexpr (!IsSignedV<TIn> && IsSignedV<TOut>) {
    // 无符号 → 有符号
    if (static_cast<uint64_t>(t) > static_cast<uint64_t>(out_max)) return out_max;
  } else {
    // 无符号 → 无符号
    if (t > static_cast<TIn>(out_max)) return out_max;
  }
  return static_cast<TOut>(t);
}

// ========== 浮点到整数饱和转换 ==========

template <typename TOut, typename TIn>
CT_ALWAYS_FORCEINLINE constexpr TOut SaturateFloatToInt(TIn t) {
  using OutLimits = std::numeric_limits<TOut>;
  constexpr TOut out_min = OutLimits::lowest();
  constexpr TOut out_max = OutLimits::max();

  // 处理特殊值
  if (!std::isfinite(t)) {
    return (std::isnan(t) || t > 0) ? out_max : out_min;
  }

  // 饱和边界检查
  if (t < static_cast<TIn>(out_min)) return out_min;
  if (t > static_cast<TIn>(out_max)) return out_max;

  // 向零舍入
  return static_cast<TOut>(t);
}

// ================== ScalarConvert 主模板 ==================

// SFINAE 条件类型
template <typename TOut, typename TIn>
using EnableIfIntPromote = std::enable_if_t<
    IsIntV<TIn> && IsIntV<TOut> && (TypeBitsV<TOut> > TypeBitsV<TIn>)>;

template <typename TOut, typename TIn>
using EnableIfIntDemote = std::enable_if_t<
    IsIntV<TIn> && IsIntV<TOut> && (TypeBitsV<TOut> < TypeBitsV<TIn>)>;

template <typename TOut, typename TIn>
using EnableIfIntConvert = std::enable_if_t<
    IsIntV<TIn> && IsIntV<TOut> && (TypeBitsV<TOut> == TypeBitsV<TIn>)>;

template <typename TOut, typename TIn>
using EnableIfIntToFloat = std::enable_if_t<
    IsIntV<TIn> && IsFloatV<TOut> && !IsBfloat16V<TOut>>;

template <typename TOut, typename TIn>
using EnableIfFloatToInt = std::enable_if_t<
    IsFloatV<TIn> && !IsBfloat16V<TIn> && IsIntV<TOut>>;

template <typename TOut, typename TIn>
using EnableIfFloatPromote = std::enable_if_t<
    IsFloatV<TIn> && IsFloatV<TOut> && !IsBfloat16V<TIn> && !IsBfloat16V<TOut> &&
    (TypeBitsV<TOut> > TypeBitsV<TIn>)>;

template <typename TOut, typename TIn>
using EnableIfFloatDemote = std::enable_if_t<
    IsFloatV<TIn> && IsFloatV<TOut> && !IsBfloat16V<TIn> && !IsBfloat16V<TOut> &&
    (TypeBitsV<TOut> < TypeBitsV<TIn>)>;

// ================== 基础模板 ==================

template <typename TOut, typename TIn, typename = void>
struct ScalarConvert {
  CT_ALWAYS_FORCEINLINE constexpr TOut operator()(TIn t) const {
    return static_cast<TOut>(t);
  }
};

// ================== 整数扩展 ==================

template <typename TOut, typename TIn>
struct ScalarConvert<TOut, TIn, EnableIfIntPromote<TOut, TIn>> {
  CT_ALWAYS_FORCEINLINE constexpr TOut operator()(TIn t) const {
    // static_cast 自动处理零扩展(无符号)或符号扩展(有符号)
    return static_cast<TOut>(t);
  }
};

// ================== 整数缩减 ==================

template <typename TOut, typename TIn>
struct ScalarConvert<TOut, TIn, EnableIfIntDemote<TOut, TIn>> {
  CT_ALWAYS_FORCEINLINE constexpr TOut operator()(TIn t) const {
    return SaturateIntToInt<TOut>(t);
  }
};

// ================== 整数等宽转换 ==================

template <typename TOut, typename TIn>
struct ScalarConvert<TOut, TIn, EnableIfIntConvert<TOut, TIn>> {
  CT_ALWAYS_FORCEINLINE constexpr TOut operator()(TIn t) const {
    // 等宽转换保持位模式，但可能改变解释
    return static_cast<TOut>(t);
  }
};

// ================== 整数 → 浮点 (非 bf16) ==================

template <typename TOut, typename TIn>
struct ScalarConvert<TOut, TIn, EnableIfIntToFloat<TOut, TIn>> {
  CT_ALWAYS_FORCEINLINE constexpr TOut operator()(TIn t) const {
    return static_cast<TOut>(t);
  }
};

// ================== 浮点 (非 bf16) → 整数 ==================

template <typename TOut, typename TIn>
struct ScalarConvert<TOut, TIn, EnableIfFloatToInt<TOut, TIn>> {
  CT_ALWAYS_FORCEINLINE constexpr TOut operator()(TIn t) const {
    return SaturateFloatToInt<TOut>(t);
  }
};

// ================== 浮点扩展 ==================

template <typename TOut, typename TIn>
struct ScalarConvert<TOut, TIn, EnableIfFloatPromote<TOut, TIn>> {
  CT_ALWAYS_FORCEINLINE constexpr TOut operator()(TIn t) const {
    return static_cast<TOut>(t);
  }
};

// ================== 浮点缩减 ==================

template <typename TOut, typename TIn>
struct ScalarConvert<TOut, TIn, EnableIfFloatDemote<TOut, TIn>> {
  CT_ALWAYS_FORCEINLINE constexpr TOut operator()(TIn t) const {
    // 编译器会处理舍入，需要手动处理溢出
    if (!std::isfinite(t)) {
      if (std::isnan(t)) return std::numeric_limits<TOut>::quiet_NaN();
      return t > 0 ? std::numeric_limits<TOut>::infinity()
                   : -std::numeric_limits<TOut>::infinity();
    }
    return static_cast<TOut>(t);
  }
};

// ================== bf16 特化 ==================

// bf16 → float
template <>
struct ScalarConvert<float, bfloat16_t, void> {
  CT_ALWAYS_FORCEINLINE constexpr float operator()(bfloat16_t t) const {
    return Bf16ToFloat(t);
  }
};

// float → bf16
template <>
struct ScalarConvert<bfloat16_t, float, void> {
  CT_ALWAYS_FORCEINLINE constexpr bfloat16_t operator()(float t) const {
    if (!std::isfinite(t)) {
      if (std::isnan(t)) return BitsToBf16(0x7FC0);  // bf16 NaN
      return t > 0 ? BitsToBf16(0x7F80) : BitsToBf16(0xFF80);
    }
    return FloatToBf16(t);
  }
};

// bf16 → double
template <>
struct ScalarConvert<double, bfloat16_t, void> {
  CT_ALWAYS_FORCEINLINE constexpr double operator()(bfloat16_t t) const {
    return static_cast<double>(Bf16ToFloat(t));
  }
};

// double → bf16
template <>
struct ScalarConvert<bfloat16_t, double, void> {
  CT_ALWAYS_FORCEINLINE constexpr bfloat16_t operator()(double t) const {
    return ScalarConvert<bfloat16_t, float>{}(static_cast<float>(t));
  }
};

// 整数 → bf16
template <typename TIn>
struct ScalarConvert<bfloat16_t, TIn, std::enable_if_t<IsIntV<TIn>>> {
  CT_ALWAYS_FORCEINLINE constexpr bfloat16_t operator()(TIn t) const {
    return FloatToBf16(static_cast<float>(t));
  }
};

// bf16 → 整数
template <typename TOut>
struct ScalarConvert<TOut, bfloat16_t, std::enable_if_t<IsIntV<TOut>>> {
  CT_ALWAYS_FORCEINLINE constexpr TOut operator()(bfloat16_t t) const {
    return SaturateFloatToInt<TOut>(Bf16ToFloat(t));
  }
};

// float16 ↔ bf16
template <>
struct ScalarConvert<bfloat16_t, float16_t, void> {
  CT_ALWAYS_FORCEINLINE constexpr bfloat16_t operator()(float16_t t) const {
    return FloatToBf16(static_cast<float>(t));
  }
};

template <>
struct ScalarConvert<float16_t, bfloat16_t, void> {
  CT_ALWAYS_FORCEINLINE constexpr float16_t operator()(bfloat16_t t) const {
    return static_cast<float16_t>(Bf16ToFloat(t));
  }
};

template <typename TOut, typename TIn>
CT_FORCEINLINE constexpr TOut convert(TIn t) {
  return ScalarConvert<TOut, TIn>()(t);
}

} // namespace ct::tl

#endif //CTORCH_SCALARCONVERT_H
