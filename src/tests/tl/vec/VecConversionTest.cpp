//
// VecConvertTest.cpp
// Comprehensive test for vector type conversions
// Covers: promote, demote, convert
//

#include <gtest/gtest.h>
#include <cstring>
#include <cmath>
#include <limits>
#include <type_traits>

#include "tl/cpu/Vec.h"
#include "tl/util/ScalarConvert.h"

using namespace ct;
using namespace ct::tl;
using namespace ct::tl::vec;

// ============================================================================
// Helper utilities
// ============================================================================

namespace test_utils {

template <typename T>
constexpr T get_test_value(int idx) {
  if constexpr (std::is_same_v<T, bfloat16_t>) {
    return static_cast<bfloat16_t>(static_cast<float>(idx * 1.5f + 0.5f));
  } else if constexpr (std::is_same_v<T, float16_t>) {
    return static_cast<float16_t>(static_cast<float>(idx * 1.5f + 0.5f));
  } else if constexpr (std::is_same_v<T, float32_t>) {
    return static_cast<float32_t>(idx * 1.5f + 0.5f);
  } else if constexpr (std::is_same_v<T, float64_t>) {
    return static_cast<float64_t>(idx * 1.5 + 0.5);
  } else if constexpr (std::is_same_v<T, int8_t>) {
    return static_cast<int8_t>((idx * 7 + 3) % 127 - 64);
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return static_cast<uint8_t>((idx * 7 + 3) % 256);
  } else if constexpr (std::is_same_v<T, int16_t>) {
    return static_cast<int16_t>((idx * 100 + 50) % 32767 - 16384);
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    return static_cast<uint16_t>((idx * 100 + 50) % 65536);
  } else if constexpr (std::is_same_v<T, int32_t>) {
    return static_cast<int32_t>(idx * 1000 + 500);
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    return static_cast<uint32_t>(idx * 1000 + 500);
  } else if constexpr (std::is_same_v<T, int64_t>) {
    return static_cast<int64_t>(idx * 100000LL + 50000LL);
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    return static_cast<uint64_t>(idx * 100000ULL + 50000ULL);
  }
}

template <typename T>
::testing::AssertionResult values_equal(T expected, T actual, double tolerance = 0.01) {
  if constexpr (std::is_same_v<T, bfloat16_t>) {
    float e = static_cast<float>(expected);
    float a = static_cast<float>(actual);
    if (std::abs(e - a) <= std::max(std::abs(e), std::abs(a)) * tolerance)
      return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "Expected " << e << ", got " << a;
  } else if constexpr (std::is_same_v<T, float16_t>) {
    float e = static_cast<float>(expected);
    float a = static_cast<float>(actual);
    if (std::abs(e - a) <= std::max(std::abs(e), std::abs(a)) * tolerance)
      return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "Expected " << e << ", got " << a;
  } else if constexpr (std::is_same_v<T, float32_t>) {
    if (std::abs(expected - actual) <= std::max(std::abs(expected), std::abs(actual)) * tolerance)
      return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "Expected " << expected << ", got " << actual;
  } else if constexpr (std::is_same_v<T, float64_t>) {
    if (std::abs(expected - actual) <= std::max(std::abs(expected), std::abs(actual)) * tolerance)
      return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "Expected " << expected << ", got " << actual;
  } else {
    if (expected == actual) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
        << "Expected " << static_cast<long long>(expected)
        << ", got " << static_cast<long long>(actual);
  }
}

template <typename T>
constexpr nint_t full_vec_size() {
  return VEC_WIDTH / 8 / sizeof(T);
}

template <typename T>
T* alloc_aligned(size_t count) {
  void* ptr = std::aligned_alloc(DEFAULT_ALIGNMENT, count * sizeof(T));
  return static_cast<T*>(ptr);
}

} // namespace test_utils

template <typename T1_, typename T2_>
struct Pair {
  using T1 = T1_;
  using T2 = T2_;
};

// ============================================================================
// Promote Tests: smaller type -> larger type
// ============================================================================

// Test fixture for promote operations
template <typename TPair>
class VecPromoteTest : public ::testing::Test {
protected:
  using TIn = typename TPair::T1;
  using TOut = typename TPair::T2;
  using InType = TIn;
  using OutType = TOut;
  static constexpr nint_t IN_SIZE = test_utils::full_vec_size<TIn>();
  static constexpr nint_t OUT_SIZE = test_utils::full_vec_size<TOut>();

  void SetUp() override {
    in_data_ = test_utils::alloc_aligned<TIn>(256);
    for (size_t i = 0; i < 256; ++i) {
      in_data_[i] = test_utils::get_test_value<TIn>(i);
    }
  }

  void TearDown() override {
    std::free(in_data_);
  }

  TIn* in_data_{};
};

// Define type pairs for promote tests
using PromoteTypes = ::testing::Types<
    // 8-bit -> 16-bit
    Pair<int8_t, int16_t>,
    Pair<uint8_t, uint16_t>,
    Pair<int8_t, uint16_t>,   // signed -> unsigned (sign extension)
    Pair<uint8_t, int16_t>,   // unsigned -> signed (zero extension)
    // 8-bit -> 32-bit
    Pair<int8_t, int32_t>,
    Pair<uint8_t, uint32_t>,
    // 8-bit -> 64-bit
    Pair<int8_t, int64_t>,
    Pair<uint8_t, uint64_t>,
    // 16-bit -> 32-bit
    Pair<int16_t, int32_t>,
    Pair<uint16_t, uint32_t>,
    Pair<int16_t, uint32_t>,
    Pair<uint16_t, int32_t>,
    // 16-bit -> 64-bit
    Pair<int16_t, int64_t>,
    Pair<uint16_t, uint64_t>,
    // 32-bit -> 64-bit
    Pair<int32_t, int64_t>,
    Pair<uint32_t, uint64_t>,
    Pair<int32_t, uint64_t>,
    Pair<uint32_t, int64_t>,
    // Float promote
    Pair<float32_t, float64_t>
>;

TYPED_TEST_SUITE(VecPromoteTest, PromoteTypes);

TYPED_TEST(VecPromoteTest, BasicPromote) {
  using TIn = typename TestFixture::InType;
  using TOut = typename TestFixture::OutType;
  constexpr nint_t N = std::min(TestFixture::IN_SIZE, TestFixture::OUT_SIZE);

  FixedTag<TIn, N> t_in;
  FixedTag<TOut, N> t_out;

  auto v_in = loadu(t_in, this->in_data_);
  auto v_out = promote(t_out, v_in);

  for (nint_t i = 0; i < N; ++i) {
    TOut expected = tl::convert<TOut>(this->in_data_[i]);
    TOut actual = get(t_out, v_out, i);
    EXPECT_TRUE(test_utils::values_equal(expected, actual))
        << "i=" << i << " input=" << static_cast<long long>(this->in_data_[i]);
  }
}

TYPED_TEST(VecPromoteTest, PromoteWithZeroValues) {
  using TIn = typename TestFixture::InType;
  using TOut = typename TestFixture::OutType;
  constexpr nint_t N_OUT = TestFixture::OUT_SIZE;

  FixedTag<TIn, N_OUT> t_in;
  FixedTag<TOut, N_OUT> t_out;

  auto v_in = zeros(t_in);
  auto v_out = promote(t_out, v_in);

  for (nint_t i = 0; i < N_OUT; ++i) {
    EXPECT_EQ(TOut(0), get(t_out, v_out, i)) << "i=" << i;
  }
}

TYPED_TEST(VecPromoteTest, PromoteWithMaxMinValues) {
  using TIn = typename TestFixture::InType;
  using TOut = typename TestFixture::OutType;
  constexpr nint_t N = std::min(TestFixture::IN_SIZE, TestFixture::OUT_SIZE);

  FixedTag<TIn, N> t_in;
  FixedTag<TOut, N> t_out;

  alignas(16) TIn data[N];
  for (nint_t i = 0; i < N; ++i) {
    if (i % 4 == 0) data[i] = std::numeric_limits<TIn>::max();
    else if (i % 4 == 1) data[i] = std::numeric_limits<TIn>::min();
    else if (i % 4 == 2) data[i] = TIn(0);
    else data[i] = TIn(-1);
  }

  auto v_in = loadu(t_in, data);
  auto v_out = promote(t_out, v_in);

  for (nint_t i = 0; i < N; ++i) {
    TOut expected = tl::convert<TOut>(data[i]);
    TOut actual = get(t_out, v_out, i);
    EXPECT_TRUE(test_utils::values_equal(expected, actual))
        << "i=" << i << " input=" << static_cast<long long>(data[i]);
  }
}

// Sign extension test for signed types
TYPED_TEST(VecPromoteTest, SignExtensionTest) {
  using TIn = typename TestFixture::InType;
  using TOut = typename TestFixture::OutType;
  constexpr nint_t N = std::min(TestFixture::IN_SIZE, TestFixture::OUT_SIZE);

  if constexpr (std::is_signed_v<TIn>) {
    FixedTag<TIn, N> t_in;
    FixedTag<TOut, N> t_out;

    alignas(16) TIn data[N];
    // Create negative values
    for (nint_t i = 0; i < N; ++i) {
      data[i] = static_cast<TIn>(-(i + 1));
    }

    auto v_in = loadu(t_in, data);
    auto v_out = promote(t_out, v_in);

    for (nint_t i = 0; i < N; ++i) {
      TOut actual = get(t_out, v_out, i);
      // For signed-to-signed, should preserve sign
      // For signed-to-unsigned, sign extension still applies then reinterpreted
      TIn original = data[i];
      // Verify that promotion preserves the value
      EXPECT_TRUE(test_utils::values_equal(static_cast<TOut>(original), actual))
          << "i=" << i << " original=" << static_cast<long long>(original)
          << " actual=" << static_cast<long long>(actual);
    }
  }
}

// ============================================================================
// Demote Tests: larger type -> smaller type
// ============================================================================

template <typename TPair>
class VecDemoteTest : public ::testing::Test {
protected:
  using TIn = typename TPair::T1;
  using TOut = typename TPair::T2;
  using InType = TIn;
  using OutType = TOut;
  static constexpr nint_t IN_SIZE = test_utils::full_vec_size<TIn>();
  static constexpr nint_t OUT_SIZE = test_utils::full_vec_size<TOut>();

  void SetUp() override {
    in_data_ = test_utils::alloc_aligned<TIn>(256);
    // Use smaller values to avoid overflow issues in demote tests
    for (size_t i = 0; i < 256; ++i) {
      if constexpr (std::is_floating_point_v<TIn> ||
                    std::is_same_v<TIn, float16_t> ||
                    std::is_same_v<TIn, bfloat16_t>) {
        in_data_[i] = static_cast<TIn>((i % 20) - 10 + 0.5);
      } else if constexpr (std::is_signed_v<TIn>) {
        in_data_[i] = static_cast<TIn>((i % 100) - 50);
      } else {
        in_data_[i] = static_cast<TIn>(i % 100);
      }
    }
  }

  void TearDown() override {
    std::free(in_data_);
  }

  TIn* in_data_{};
};

// Define type pairs for demote tests
using DemoteTypes = ::testing::Types<
    // 16-bit -> 8-bit
    Pair<int16_t, int8_t>,
    Pair<uint16_t, uint8_t>,
    Pair<int16_t, uint8_t>,
    Pair<uint16_t, int8_t>,
    // 32-bit -> 16-bit
    Pair<int32_t, int16_t>,
    Pair<uint32_t, uint16_t>,
    Pair<int32_t, uint16_t>,
    Pair<uint32_t, int16_t>,
    // 32-bit -> 8-bit
    Pair<int32_t, int8_t>,
    Pair<uint32_t, uint8_t>,
    // 64-bit -> 32-bit
    Pair<int64_t, int32_t>,
    Pair<uint64_t, uint32_t>,
    Pair<int64_t, uint32_t>,
    Pair<uint64_t, int32_t>,
    // 64-bit -> 16-bit
    Pair<int64_t, int16_t>,
    Pair<uint64_t, uint16_t>,
    // 64-bit -> 8-bit
    Pair<int64_t, int8_t>,
    Pair<uint64_t, uint8_t>,
    // Float demote
    Pair<float64_t, float32_t>
>;

TYPED_TEST_SUITE(VecDemoteTest, DemoteTypes);

TYPED_TEST(VecDemoteTest, BasicDemote) {
  using TIn = typename TestFixture::InType;
  using TOut = typename TestFixture::OutType;
  constexpr nint_t N = std::min(TestFixture::IN_SIZE, TestFixture::OUT_SIZE);

  FixedTag<TIn, N> t_in;
  FixedTag<TOut, N> t_out;

  auto v_in = loadu(t_in, this->in_data_);
  auto v_out = demote(t_out, v_in);

  for (nint_t i = 0; i < N; ++i) {
    TOut expected = tl::convert<TOut>(this->in_data_[i]);
    TOut actual = get(t_out, v_out, i);
    EXPECT_TRUE(test_utils::values_equal(expected, actual))
              << "i=" << i << " input=" << static_cast<long long>(this->in_data_[i]);
  }
}

TYPED_TEST(VecDemoteTest, DemoteWithZeroValues) {
  using TIn = typename TestFixture::InType;
  using TOut = typename TestFixture::OutType;
  constexpr nint_t N = std::min(TestFixture::IN_SIZE, TestFixture::OUT_SIZE);

  FixedTag<TIn, N> t_in;
  FixedTag<TOut, N> t_out;

  auto v_in = zeros(t_in);
  auto v_out = demote(t_out, v_in);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_EQ(TOut(0), get(t_out, v_out, i)) << "i=" << i;
  }
}

// Truncation behavior test for overflow cases
TYPED_TEST(VecDemoteTest, TruncationBehavior) {
  using TIn = typename TestFixture::InType;
  using TOut = typename TestFixture::OutType;
  constexpr nint_t N = std::min(TestFixture::IN_SIZE, TestFixture::OUT_SIZE);

  // Only test if output type can represent values within its range
  FixedTag<TIn, N> t_in;
  FixedTag<TOut, N> t_out;

  // Create values that are within representable range of output type
  alignas(16) TIn data[N];
  TIn max_out = static_cast<TIn>(std::numeric_limits<TOut>::max());
  TIn min_out = std::is_signed_v<TOut>
                    ? static_cast<TIn>(std::numeric_limits<TOut>::min())
                    : TIn(0);

  for (nint_t i = 0; i < N; ++i) {
    // Alternate between min, max, and middle values
    if (i % 3 == 0) data[i] = max_out;
    else if (i % 3 == 1 && std::is_signed_v<TOut>) data[i] = min_out;
    else data[i] = TIn(0);
  }

  auto v_in = loadu(t_in, data);
  auto v_out = demote(t_out, v_in);

  for (nint_t i = 0; i < N; ++i) {
    TOut expected = tl::convert<TOut>(data[i]);
    TOut actual = get(t_out, v_out, i);
    EXPECT_TRUE(test_utils::values_equal(expected, actual))
        << "i=" << i << " input=" << static_cast<long long>(data[i]);
  }
}

// ============================================================================
// Convert Tests: same-size type conversions
// ============================================================================

template <typename TPair>
class VecConvertTest : public ::testing::Test {
protected:
  using TIn = typename TPair::T1;
  using TOut = typename TPair::T2;
  using InType = TIn;
  using OutType = TOut;
  static_assert(sizeof(TIn) == sizeof(TOut), "Convert requires same-size types");
  static constexpr nint_t SIZE = test_utils::full_vec_size<TIn>();

  void SetUp() override {
    in_data_ = test_utils::alloc_aligned<TIn>(256);
    for (size_t i = 0; i < 256; ++i) {
      in_data_[i] = test_utils::get_test_value<TIn>(i);
    }
  }

  void TearDown() override {
    std::free(in_data_);
  }

  TIn* in_data_{};
};

// Define type pairs for convert tests (same size)
using ConvertTypes = ::testing::Types<
    // 8-bit conversions
    Pair<int8_t, uint8_t>,
    Pair<uint8_t, int8_t>,
    // 16-bit conversions
    Pair<int16_t, uint16_t>,
    Pair<uint16_t, int16_t>,
    // 32-bit conversions
    Pair<int32_t, uint32_t>,
    Pair<uint32_t, int32_t>,
    Pair<int32_t, float32_t>,
    Pair<float32_t, int32_t>,
    Pair<uint32_t, float32_t>,
    Pair<float32_t, uint32_t>,
    // 64-bit conversions
    Pair<int64_t, uint64_t>,
    Pair<uint64_t, int64_t>,
    Pair<int64_t, float64_t>,
    Pair<float64_t, int64_t>,
    Pair<uint64_t, float64_t>,
    Pair<float64_t, uint64_t>
>;

TYPED_TEST_SUITE(VecConvertTest, ConvertTypes);

TYPED_TEST(VecConvertTest, BasicConvert) {
  using TIn = typename TestFixture::InType;
  using TOut = typename TestFixture::OutType;
  constexpr nint_t N = TestFixture::SIZE;

  FixedTag<TIn, N> t_in;
  FixedTag<TOut, N> t_out;

  auto v_in = loadu(t_in, this->in_data_);
  auto v_out = convert(t_out, v_in);

  for (nint_t i = 0; i < N; ++i) {
    TOut expected = tl::convert<TOut>(this->in_data_[i]);
    TOut actual = get(t_out, v_out, i);
    EXPECT_TRUE(test_utils::values_equal(expected, actual))
        << "i=" << i << " input=" << static_cast<long long>(this->in_data_[i]);
  }
}

TYPED_TEST(VecConvertTest, ConvertWithZeroValues) {
  using TIn = typename TestFixture::InType;
  using TOut = typename TestFixture::OutType;
  constexpr nint_t N = TestFixture::SIZE;

  FixedTag<TIn, N> t_in;
  FixedTag<TOut, N> t_out;

  auto v_in = zeros(t_in);
  auto v_out = convert(t_out, v_in);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_EQ(TOut(0), get(t_out, v_out, i)) << "i=" << i;
  }
}

// Test signed/unsigned reinterpretation
TYPED_TEST(VecConvertTest, SignedUnsignedConversion) {
  using TIn = typename TestFixture::InType;
  using TOut = typename TestFixture::OutType;
  constexpr nint_t N = TestFixture::SIZE;

  if constexpr (!std::is_floating_point_v<TIn>)
  if constexpr ((std::is_signed_v<TIn> && std::is_unsigned_v<TOut>) ||
                (std::is_unsigned_v<TIn> && std::is_signed_v<TOut>)) {
    FixedTag<TIn, N> t_in;
    FixedTag<TOut, N> t_out;

    alignas(16) TIn data[N];
    // Create values with high bit set (negative if signed)
    for (nint_t i = 0; i < N; ++i) {
      data[i] = static_cast<TIn>(~TIn(0) - i);
    }

    auto v_in = loadu(t_in, data);
    auto v_out = convert(t_out, v_in);

    for (nint_t i = 0; i < N; ++i) {
      TOut expected = tl::convert<TOut>(data[i]);
      TOut actual = get(t_out, v_out, i);
      EXPECT_TRUE(test_utils::values_equal(expected, actual))
          << "i=" << i << " input=" << static_cast<long long>(data[i])
          << " expected=" << static_cast<long long>(expected);
    }
  }
}

// Float to int and int to float conversion test
TYPED_TEST(VecConvertTest, FloatIntConversion) {
  using TIn = typename TestFixture::InType;
  using TOut = typename TestFixture::OutType;
  constexpr nint_t N = TestFixture::SIZE;

  if constexpr ((std::is_floating_point_v<TIn> && std::is_integral_v<TOut>) ||
                (std::is_integral_v<TIn> && std::is_floating_point_v<TOut>)) {
    FixedTag<TIn, N> t_in;
    FixedTag<TOut, N> t_out;

    alignas(16) TIn data[N];
    // Use values that can be exactly represented
    for (nint_t i = 0; i < N; ++i) {
      if constexpr (std::is_floating_point_v<TIn>) {
        if constexpr (std::is_unsigned_v<TOut>) {
          // note: conversion from negative float to unsigned int is undefined
          data[i] = static_cast<TIn>(i * 10 + 33);
        } else {
          data[i] = static_cast<TIn>(i * 10 - 50);
        }
      } else {
        data[i] = static_cast<TIn>(i * 100);
      }
    }

    auto v_in = loadu(t_in, data);
    auto v_out = convert(t_out, v_in);

    for (nint_t i = 0; i < N; ++i) {
      TOut expected = tl::convert<TOut>(data[i]);
      TOut actual = get(t_out, v_out, i);
      EXPECT_TRUE(test_utils::values_equal(expected, actual))
          << "i=" << i << " input=" << static_cast<long long>(data[i]);
    }
  }
}

// ============================================================================
// Corner Case Tests
// ============================================================================

// Test fixture for corner cases
class VecConvertCornerCaseTest : public ::testing::Test {
protected:
  void SetUp() override {
    data_256_ = test_utils::alloc_aligned<uint8_t>(256);
    memset(data_256_, 0, 256);
  }

  void TearDown() override {
    std::free(data_256_);
  }

  uint8_t* data_256_{};
};

// Test: int8_t (-1) -> uint16_t (should be 65535 via sign extension)
TEST_F(VecConvertCornerCaseTest, PromoteNegativeToUnsigned) {
  FixedTag<int8_t, 16> t_in;
  FixedTag<uint16_t, 16> t_out;

  alignas(16) int8_t data[16];
  for (int i = 0; i < 16; ++i) {
    data[i] = -1;
  }

  auto v_in = loadu(t_in, data);
  auto v_out = promote(t_out, v_in);

  for (nint_t i = 0; i < 16; ++i) {
    uint16_t actual = get(t_out, v_out, i);
    // -1 as int8_t promoted to uint16_t should be 0xFFFF (65535)
    EXPECT_EQ(uint16_t(65535), actual) << "i=" << i;
  }
}

// Test: uint8_t (255) -> int16_t (should be 255, not -1)
TEST_F(VecConvertCornerCaseTest, PromoteUnsignedToSigned) {
  FixedTag<uint8_t, 16> t_in;
  FixedTag<int16_t, 16> t_out;

  alignas(16) uint8_t data[16];
  for (int i = 0; i < 16; ++i) {
    data[i] = 255;
  }

  auto v_in = loadu(t_in, data);
  auto v_out = promote(t_out, v_in);

  for (nint_t i = 0; i < 16; ++i) {
    int16_t actual = get(t_out, v_out, i);
    // 255 as uint8_t promoted to int16_t should be 255
    EXPECT_EQ(int16_t(255), actual) << "i=" << i;
  }
}

// Test: int16_t (-1) -> int8_t (truncation to -1)
TEST_F(VecConvertCornerCaseTest, DemoteNegativeValue) {
  FixedTag<int16_t, 8> t_in;
  FixedTag<int8_t, 8> t_out;

  alignas(16) int16_t data[8];
  for (int i = 0; i < 8; ++i) {
    data[i] = -1;
  }

  auto v_in = loadu(t_in, data);
  auto v_out = demote(t_out, v_in);

  for (nint_t i = 0; i < 8; ++i) {
    int8_t actual = get(t_out, v_out, i);
    // -1 truncated should still be -1 (0xFF)
    EXPECT_EQ(int8_t(-1), actual) << "i=" << i;
  }
}

// Test: int32_t <-> float32_t for boundary values
TEST_F(VecConvertCornerCaseTest, Int32Float32Boundary) {
  FixedTag<int32_t, 4> t_int;
  FixedTag<float32_t, 4> t_float;

  alignas(16) int32_t int_data[4] = {
    0,
    1,
    -1,
    123456
  };

  // int32 -> float32
  auto v_int = loadu(t_int, int_data);
  auto v_float = convert(t_float, v_int);

  for (nint_t i = 0; i < 4; ++i) {
    float32_t expected = static_cast<float32_t>(int_data[i]);
    float32_t actual = get(t_float, v_float, i);
    EXPECT_FLOAT_EQ(expected, actual) << "i=" << i;
  }

  // float32 -> int32
  alignas(16) float32_t float_data[4] = {
    0.0f,
    1.0f,
    -1.0f,
    123456.0f
  };

  auto v_f = loadu(t_float, float_data);
  auto v_i = convert(t_int, v_f);

  for (nint_t i = 0; i < 4; ++i) {
    int32_t expected = static_cast<int32_t>(float_data[i]);
    int32_t actual = get(t_int, v_i, i);
    EXPECT_EQ(expected, actual) << "i=" << i;
  }
}

// Test: Large int32_t values to float32_t (precision loss expected)
TEST_F(VecConvertCornerCaseTest, LargeInt32ToFloat32) {
  FixedTag<int32_t, 4> t_int;
  FixedTag<float32_t, 4> t_float;

  alignas(16) int32_t int_data[4] = {
    0x7FFFFFFF,  // max int32
    (int)0x80000000,  // min int32 (as unsigned, will be -2147483648 as signed)
    1234567890,
    -1234567890
  };

  auto v_int = loadu(t_int, int_data);
  auto v_float = convert(t_float, v_int);

  for (nint_t i = 0; i < 4; ++i) {
    float32_t actual = get(t_float, v_float, i);
    float32_t expected = static_cast<float32_t>(int_data[i]);
    // Allow some precision difference for large values
    EXPECT_NEAR(expected, actual, std::abs(expected) * 1e-6f)
        << "i=" << i << " input=" << int_data[i];
  }
}

// Test: Multi-word vector conversions
//TEST_F(VecConvertCornerCaseTest, MultiWordPromote) {
//  // 2-word int8_t vector -> int16_t
//  FixedTag<int8_t, 32> t_in;   // 2 words (256-bit total)
//  FixedTag<int16_t, 16> t_out; // 1 word (256-bit total)
//
//  alignas(16) int8_t data[32];
//  for (int i = 0; i < 32; ++i) {
//    data[i] = static_cast<int8_t>(i - 16);
//  }
//
//  auto v_in = loadu(t_in, data);
//  auto v_out = promote(t_out, v_in);
//
//  for (nint_t i = 0; i < 16; ++i) {
//    int16_t expected = static_cast<int16_t>(data[i]);
//    int16_t actual = get(t_out, v_out, i);
//    EXPECT_EQ(expected, actual) << "i=" << i;
//  }
//}

TEST_F(VecConvertCornerCaseTest, MultiWordDemote) {
  FixedTag<int16_t, 16> t_in;
  FixedTag<int8_t, 16> t_out;

  alignas(16) int16_t data[16];
  for (int i = 0; i < 16; ++i) {
    data[i] = static_cast<int16_t>((i - 8) * 10);
  }

  auto v_in = loadu(t_in, data);
  auto v_out = demote(t_out, v_in);

  for (nint_t i = 0; i < 16; ++i) {
    int8_t expected = static_cast<int8_t>(data[i]);
    int8_t actual = get(t_out, v_out, i);
    EXPECT_EQ(expected, actual) << "i=" << i;
  }
}

TEST_F(VecConvertCornerCaseTest, MultiWordConvert) {
  FixedTag<int32_t, 8> t_in;
  FixedTag<uint32_t, 8> t_out;

  alignas(16) int32_t data[8];
  for (int i = 0; i < 8; ++i) {
    data[i] = (i - 4) * 1000;
  }

  auto v_in = loadu(t_in, data);
  auto v_out = convert(t_out, v_in);

  for (nint_t i = 0; i < 8; ++i) {
    uint32_t expected = static_cast<uint32_t>(data[i]);
    uint32_t actual = get(t_out, v_out, i);
    EXPECT_EQ(expected, actual) << "i=" << i;
  }
}

// Test: Float64 <-> Float32 conversion
TEST_F(VecConvertCornerCaseTest, Float64Float32RoundTrip) {
  FixedTag<float64_t, 2> t_f64;
  FixedTag<float32_t, 2> t_f32;

  // Float64 -> Float32 (demote)
  alignas(16) float64_t f64_data[2] = {1.5, -2.5};
  auto v_f64 = loadu(t_f64, f64_data);
  auto v_f32 = demote(t_f32, v_f64);

  for (nint_t i = 0; i < 2; ++i) {
    float32_t expected = static_cast<float32_t>(f64_data[i]);
    float32_t actual = get(t_f32, v_f32, i);
    EXPECT_FLOAT_EQ(expected, actual) << "i=" << i;
  }
}

TEST_F(VecConvertCornerCaseTest, Float32Float64RoundTrip) {
  FixedTag<float32_t, 4> t_f32;
  FixedTag<float64_t, 2> t_f64;

  // Float32 -> Float64 (promote)
  alignas(16) float32_t f32_data[4] = {1.5f, -2.5f, 0.0f, 100.5f};
  auto v_f32 = loadu(t_f32, f32_data);
  auto v_f64 = promote(t_f64, v_f32);

  for (nint_t i = 0; i < 2; ++i) {
    float64_t expected = static_cast<float64_t>(f32_data[i]);
    float64_t actual = get(t_f64, v_f64, i);
    EXPECT_DOUBLE_EQ(expected, actual) << "i=" << i;
  }
}

// Test: All bits set patterns
TEST_F(VecConvertCornerCaseTest, AllBitsSetPattern) {
  // int8_t all 0xFF -> uint16_t should be 0xFFFF
  FixedTag<int8_t, 16> t_i8;
  FixedTag<uint16_t, 8> t_u16;

  alignas(16) int8_t data[16];
  memset(data, 0xFF, 16);

  auto v_in = loadu(t_i8, data);
  auto v_out = promote(t_u16, v_in);

  for (nint_t i = 0; i < 8; ++i) {
    uint16_t actual = get(t_u16, v_out, i);
    // Each pair of 0xFF becomes 0xFFFF
    EXPECT_EQ(uint16_t(0xFFFF), actual) << "i=" << i;
  }
}

// ============================================================================
// Bitcast Tests: same-size type reinterpretation
// ============================================================================

// Helper to compare bit patterns (handles NaN correctly)
template <typename T1, typename T2>
::testing::AssertionResult bits_equal(T1 expected, T2 actual) {
  static_assert(sizeof(T1) == sizeof(T2), "Size mismatch");
  if (std::memcmp(&expected, &actual, sizeof(T1)) == 0) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure()
      << "Bit pattern mismatch: expected " << static_cast<long long>(expected)
      << ", got " << static_cast<long long>(actual);
}

template <typename TPair>
class VecBitcastTest : public ::testing::Test {
protected:
  using TIn = typename TPair::T1;
  using TOut = typename TPair::T2;
  using InType = TIn;
  using OutType = TOut;
  static_assert(sizeof(TIn) == sizeof(TOut), "Bitcast requires same-size types");

  void SetUp() override {
    in_data_ = test_utils::alloc_aligned<TIn>(256);
    for (size_t i = 0; i < 256; ++i) {
      in_data_[i] = test_utils::get_test_value<TIn>(i);
    }
  }

  void TearDown() override {
    std::free(in_data_);
  }

  TIn* in_data_{};
};

// Define type pairs for bitcast tests (same size)
using BitcastTypes = ::testing::Types<
    // 8-bit bitcast
    Pair<int8_t, uint8_t>,
    Pair<uint8_t, int8_t>,
    // 16-bit bitcast
    Pair<int16_t, uint16_t>,
    Pair<uint16_t, int16_t>,
    // 32-bit bitcast
    Pair<int32_t, uint32_t>,
    Pair<uint32_t, int32_t>,
    Pair<int32_t, float32_t>,
    Pair<float32_t, int32_t>,
    Pair<uint32_t, float32_t>,
    Pair<float32_t, uint32_t>,
    // 64-bit bitcast
    Pair<int64_t, uint64_t>,
    Pair<uint64_t, int64_t>,
    Pair<int64_t, float64_t>,
    Pair<float64_t, int64_t>,
    Pair<uint64_t, float64_t>,
    Pair<float64_t, uint64_t>
>;

TYPED_TEST_SUITE(VecBitcastTest, BitcastTypes);

TYPED_TEST(VecBitcastTest, BasicBitcast) {
  using TIn = typename TestFixture::InType;
  using TOut = typename TestFixture::OutType;

  ScalableTag<TIn> t_in;
  ScalableTag<TOut> t_out;
  constexpr nint_t N = size(t_in);

  auto v_in = loadu(t_in, this->in_data_);
  auto v_out = bitcast(t_out, v_in);

  // Bitcast should preserve bit pattern exactly
  for (nint_t i = 0; i < N; ++i) {
    TIn original = this->in_data_[i];
    TOut actual = get(t_out, v_out, i);
    // Reinterpret bits: use memcpy for type-punning
    TOut expected;
    std::memcpy(&expected, &original, sizeof(TOut));
    EXPECT_TRUE(bits_equal(expected, actual))
        << "i=" << i << " input=" << static_cast<long long>(original);
  }
}

TYPED_TEST(VecBitcastTest, BitcastRoundTrip) {
  using TIn = typename TestFixture::InType;
  using TOut = typename TestFixture::OutType;

  ScalableTag<TIn> t_in;
  ScalableTag<TOut> t_out;
  constexpr nint_t N = size(t_in);

  auto v_in = loadu(t_in, this->in_data_);
  auto v_mid = bitcast(t_out, v_in);
  auto v_out = bitcast(t_in, v_mid);

  // Round-trip should preserve original values
  for (nint_t i = 0; i < N; ++i) {
    TIn expected = this->in_data_[i];
    TIn actual = get(t_in, v_out, i);
    EXPECT_TRUE(bits_equal(expected, actual)) << "i=" << i;
  }
}

TYPED_TEST(VecBitcastTest, BitcastWithZeroValues) {
  using TIn = typename TestFixture::InType;
  using TOut = typename TestFixture::OutType;

  ScalableTag<TIn> t_in;
  ScalableTag<TOut> t_out;
  constexpr nint_t N = size(t_in);

  auto v_in = zeros(t_in);
  auto v_out = bitcast(t_out, v_in);

  for (nint_t i = 0; i < N; ++i) {
    TOut actual = get(t_out, v_out, i);
    EXPECT_TRUE(bits_equal(TOut(0), actual)) << "i=" << i;
  }
}

// Test special bit patterns (all ones = NaN for floats)
TYPED_TEST(VecBitcastTest, BitcastWithAllOnesPattern) {
  using TIn = typename TestFixture::InType;
  using TOut = typename TestFixture::OutType;

  ScalableTag<TIn> t_in;
  ScalableTag<TOut> t_out;
  constexpr nint_t N = size(t_in);

  // Use aligned buffer with sufficient size
  auto data = test_utils::alloc_aligned<TIn>(N);
  // Fill with all 1s
  std::memset(data, 0xFF, N * sizeof(TIn));

  auto v_in = loadu(t_in, data);
  auto v_out = bitcast(t_out, v_in);

  // All bits should be 1 - compare as bits to handle NaN correctly
  for (nint_t i = 0; i < N; ++i) {
    TOut actual = get(t_out, v_out, i);
    TOut expected;
    std::memset(&expected, 0xFF, sizeof(TOut));
    EXPECT_TRUE(bits_equal(expected, actual)) << "i=" << i;
  }

  std::free(data);
}

// Corner case tests for bitcast using ScalableTag
TEST_F(VecConvertCornerCaseTest, BitcastFloat32ToInt32) {
  ScalableTag<float32_t> t_f32;
  ScalableTag<int32_t> t_i32;
  constexpr nint_t N = size(t_f32);

  // Test specific float values with known bit representations
  auto f32_data = test_utils::alloc_aligned<float32_t>(N);
  f32_data[0] = 0.0f;           // 0x00000000
  f32_data[1] = -0.0f;          // 0x80000000
  f32_data[2] = 1.0f;           // 0x3F800000
  f32_data[3] = -1.0f;          // 0xBF800000

  auto v_f32 = loadu(t_f32, f32_data);
  auto v_i32 = bitcast(t_i32, v_f32);

  EXPECT_EQ(0x00000000, uint32_t(get(t_i32, v_i32, 0)));
  EXPECT_EQ(0x80000000, uint32_t(get(t_i32, v_i32, 1)));
  EXPECT_EQ(0x3F800000, uint32_t(get(t_i32, v_i32, 2)));
  EXPECT_EQ(0xBF800000, uint32_t(get(t_i32, v_i32, 3)));

  std::free(f32_data);
}

TEST_F(VecConvertCornerCaseTest, BitcastInt32ToFloat32) {
  ScalableTag<int32_t> t_i32;
  ScalableTag<float32_t> t_f32;
  constexpr nint_t N = size(t_i32);

  auto i32_data = test_utils::alloc_aligned<int32_t>(N);
  i32_data[0] = 0x00000000;        // 0.0f
  i32_data[1] = (int)0x80000000;   // -0.0f
  i32_data[2] = 0x3F800000;        // 1.0f
  i32_data[3] = (int)0xBF800000;   // -1.0f

  auto v_i32 = loadu(t_i32, i32_data);
  auto v_f32 = bitcast(t_f32, v_i32);

  EXPECT_FLOAT_EQ(0.0f, get(t_f32, v_f32, 0));
  EXPECT_FLOAT_EQ(-0.0f, get(t_f32, v_f32, 1));
  EXPECT_FLOAT_EQ(1.0f, get(t_f32, v_f32, 2));
  EXPECT_FLOAT_EQ(-1.0f, get(t_f32, v_f32, 3));

  std::free(i32_data);
}

TEST_F(VecConvertCornerCaseTest, BitcastFloat64ToInt64) {
  ScalableTag<float64_t> t_f64;
  ScalableTag<int64_t> t_i64;
  constexpr nint_t N = size(t_f64);

  auto f64_data = test_utils::alloc_aligned<float64_t>(N);
  f64_data[0] = 0.0;   // 0x0000000000000000
  f64_data[1] = 1.0;   // 0x3FF0000000000000

  auto v_f64 = loadu(t_f64, f64_data);
  auto v_i64 = bitcast(t_i64, v_f64);

  EXPECT_EQ(int64_t(0x0000000000000000LL), get(t_i64, v_i64, 0));
  EXPECT_EQ(int64_t(0x3FF0000000000000LL), get(t_i64, v_i64, 1));

  std::free(f64_data);
}

TEST_F(VecConvertCornerCaseTest, BitcastSignedUnsigned) {
  ScalableTag<int32_t> t_i32;
  ScalableTag<uint32_t> t_u32;
  constexpr nint_t N = size(t_i32);

  auto i32_data = test_utils::alloc_aligned<int32_t>(N);
  i32_data[0] = -1;                // 0xFFFFFFFF
  i32_data[1] = -2;                // 0xFFFFFFFE
  i32_data[2] = 0x7FFFFFFF;        // max int32
  i32_data[3] = (int)0x80000000;   // min int32

  auto v_i32 = loadu(t_i32, i32_data);
  auto v_u32 = bitcast(t_u32, v_i32);

  EXPECT_EQ(uint32_t(0xFFFFFFFF), get(t_u32, v_u32, 0));
  EXPECT_EQ(uint32_t(0xFFFFFFFE), get(t_u32, v_u32, 1));
  EXPECT_EQ(uint32_t(0x7FFFFFFF), get(t_u32, v_u32, 2));
  EXPECT_EQ(uint32_t(0x80000000), get(t_u32, v_u32, 3));

  std::free(i32_data);
}

// Test bitcast with multi-word vectors (same size)
TEST_F(VecConvertCornerCaseTest, BitcastMultiWordSameSize) {
  // ScalableTag<T, 1> = 2 words
  ScalableTag<int32_t, 1> t_i32;
  ScalableTag<uint32_t, 1> t_u32;
  constexpr nint_t N = size(t_i32);

  auto i32_data = test_utils::alloc_aligned<int32_t>(N);
  for (nint_t i = 0; i < N; ++i) {
    i32_data[i] = (i - N/2) * 0x11111111;
  }

  auto v_i32 = loadu(t_i32, i32_data);
  auto v_u32 = bitcast(t_u32, v_i32);

  for (nint_t i = 0; i < N; ++i) {
    uint32_t expected;
    std::memcpy(&expected, &i32_data[i], sizeof(uint32_t));
    uint32_t actual = get(t_u32, v_u32, i);
    EXPECT_EQ(expected, actual) << "i=" << i;
  }

  std::free(i32_data);
}

// Test bitcast with larger input vector (input words > output words)
TEST_F(VecConvertCornerCaseTest, BitcastInputLarger) {
  // ScalableTag<T, 1> = 2 words, ScalableTag<T, 0> = 1 word
  ScalableTag<int32_t, 1> t_i32_in;   // 2 words input
  ScalableTag<uint32_t, 0> t_u32_out; // 1 word output
  constexpr nint_t N_IN = size(t_i32_in);
  constexpr nint_t N_OUT = size(t_u32_out);

  auto i32_data = test_utils::alloc_aligned<int32_t>(N_IN);
  for (nint_t i = 0; i < N_IN; ++i) {
    i32_data[i] = i * 12345;
  }

  auto v_in = loadu(t_i32_in, i32_data);
  auto v_out = bitcast(t_u32_out, v_in);

  // Only first N_OUT elements should be preserved
  for (nint_t i = 0; i < N_OUT; ++i) {
    uint32_t expected;
    std::memcpy(&expected, &i32_data[i], sizeof(uint32_t));
    uint32_t actual = get(t_u32_out, v_out, i);
    EXPECT_EQ(expected, actual) << "i=" << i;
  }

  std::free(i32_data);
}

// Test bitcast with larger output vector (input words < output words)
TEST_F(VecConvertCornerCaseTest, BitcastOutputLarger) {
  // ScalableTag<T, 0> = 1 word, ScalableTag<T, 1> = 2 words
  ScalableTag<float32_t, 0> t_f32_in;   // 1 word input
  ScalableTag<uint32_t, 1> t_u32_out;   // 2 words output
  constexpr nint_t N_IN = size(t_f32_in);
  constexpr nint_t N_OUT = size(t_u32_out);

  auto f32_data = test_utils::alloc_aligned<float32_t>(N_IN);
  for (nint_t i = 0; i < N_IN; ++i) {
    f32_data[i] = static_cast<float32_t>(i * 1.5f + 0.5f);
  }

  auto v_in = loadu(t_f32_in, f32_data);
  auto v_out = bitcast(t_u32_out, v_in);

  // First N_IN elements should be correct
  for (nint_t i = 0; i < N_IN; ++i) {
    uint32_t expected;
    std::memcpy(&expected, &f32_data[i], sizeof(uint32_t));
    uint32_t actual = get(t_u32_out, v_out, i);
    EXPECT_EQ(expected, actual) << "i=" << i;
  }
  // Elements beyond N_IN are undefined, so we don't test them

  std::free(f32_data);
}

// Test bitcast with 4-word vectors
TEST_F(VecConvertCornerCaseTest, BitcastFourWords) {
  // ScalableTag<T, 2> = 4 words
  ScalableTag<float64_t, 2> t_f64;
  ScalableTag<uint64_t, 2> t_u64;
  constexpr nint_t N = size(t_f64);

  auto f64_data = test_utils::alloc_aligned<float64_t>(N);
  for (nint_t i = 0; i < N; ++i) {
    f64_data[i] = static_cast<float64_t>(i * 2.5 + 1.0);
  }

  auto v_f64 = loadu(t_f64, f64_data);
  auto v_u64 = bitcast(t_u64, v_f64);

  for (nint_t i = 0; i < N; ++i) {
    uint64_t expected;
    std::memcpy(&expected, &f64_data[i], sizeof(uint64_t));
    uint64_t actual = get(t_u64, v_u64, i);
    EXPECT_EQ(expected, actual) << "i=" << i;
  }

  std::free(f64_data);
}

//// Test: Alternating bit patterns
//TEST_F(VecConvertCornerCaseTest, AlternatingBitPattern) {
//  FixedTag<uint8_t, 16> t_u8;
//  FixedTag<uint16_t, 8> t_u16;
//
//  alignas(16) uint8_t data[16];
//  for (int i = 0; i < 16; ++i) {
//    data[i] = (i % 2 == 0) ? 0xAA : 0x55;
//  }
//
//  auto v_in = loadu(t_u8, data);
//  auto v_out = promote(t_u16, v_in);
//
//  // Check that patterns are correctly combined
//  for (nint_t i = 0; i < 8; ++i) {
//    uint16_t actual = get(t_u16, v_out, i);
//    uint16_t expected = (uint16_t(data[i * 2]) | (uint16_t(data[i * 2 + 1]) << 8));
//    // Note: endianness may affect this test
//    // Just verify it's consistent
//    EXPECT_EQ(actual, tl::convert<uint16_t>(data[i * 2]))
//        << "i=" << i << " low byte should be " << (int)data[i * 2];
//  }
//}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
