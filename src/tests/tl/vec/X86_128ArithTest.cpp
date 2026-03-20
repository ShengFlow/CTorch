//
// X86_128ArithTest.cpp
// Comprehensive test for x86_128.h Basic arithmetic operations
// Covers: add, sub, mul, div, rcp, max, min,
//         bit_and, bit_or, bit_xor, bit_andnot, bit_not,
//         bit_shl, bit_shr,
//         neg, abs, sqrt, rsqrt,
//         cmpeq, cmpne, cmplt, cmpgt, cmple, cmpge,
//         isnan, isposinf, isneginf, isinf
//

#include <gtest/gtest.h>
#include <cstring>
#include <cmath>
#include <limits>
#include <type_traits>

#include "Features.h"

#if !defined(ARCH_X86_FAMILY)
  #define ARCH_X86_FAMILY 1
#endif

#define SIMD_WIDTH 128

#include "tl/cpu/Vec.h"

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
constexpr T get_test_value_b(int idx) {
  // Second series of values for constructing different vectors
  return get_test_value<T>(idx + 50);
}

template <typename T>
::testing::AssertionResult values_equal(T expected, T actual) {
  if constexpr (std::is_same_v<T, bfloat16_t>) {
    float e = static_cast<float>(expected);
    float a = static_cast<float>(actual);
    if (std::abs(e - a) < 0.01f) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "Expected " << e << ", got " << a;
  } else if constexpr (std::is_same_v<T, float16_t>) {
    float e = static_cast<float>(expected);
    float a = static_cast<float>(actual);
    if (std::abs(e - a) < 0.01f) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "Expected " << e << ", got " << a;
  } else if constexpr (std::is_same_v<T, float32_t>) {
    if (std::abs(expected - actual) < 1e-5f) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "Expected " << expected << ", got " << actual;
  } else if constexpr (std::is_same_v<T, float64_t>) {
    if (std::abs(expected - actual) < 1e-10) return ::testing::AssertionSuccess();
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
  return 16 / sizeof(T);
}

template <typename T>
T* alloc_aligned(size_t count) {
  void* ptr = std::aligned_alloc(DEFAULT_ALIGNMENT, count * sizeof(T));
  return static_cast<T*>(ptr);
}

// Arithmetic helper: compute expected result element-wise
template <typename T>
T scalar_add(T a, T b) { return a + b; }

template <typename T>
T scalar_sub(T a, T b) { return a - b; }

template <typename T>
T scalar_mul(T a, T b) { return a * b; }

template <typename T>
T scalar_div(T a, T b) {
  if constexpr (std::is_floating_point_v<T>) {
    return a / b;
  } else {
    return static_cast<T>(0); // not used
  }
}

template <typename T>
T scalar_max(T a, T b) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::fmax(a, b);
  } else {
    return (a > b) ? a : b;
  }
}

template <typename T>
T scalar_min(T a, T b) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::fmin(a, b);
  } else {
    return (a < b) ? a : b;
  }
}

template <typename T>
T scalar_neg(T a) { return -a; }

template <typename T>
T scalar_abs(T a) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::fabs(a);
  } else {
    return (a < T{}) ? -a : a;
  }
}

template <typename T>
T scalar_bit_and(T a, T b) {
  using U = std::make_unsigned_t<T>;
  return static_cast<T>(static_cast<U>(a) & static_cast<U>(b));
}

template <typename T>
T scalar_bit_or(T a, T b) {
  using U = std::make_unsigned_t<T>;
  return static_cast<T>(static_cast<U>(a) | static_cast<U>(b));
}

template <typename T>
T scalar_bit_xor(T a, T b) {
  using U = std::make_unsigned_t<T>;
  return static_cast<T>(static_cast<U>(a) ^ static_cast<U>(b));
}

template <typename T>
T scalar_bit_andnot(T a, T b) {
  // andnot(a, b) = (~a) & b
  using U = std::make_unsigned_t<T>;
  return static_cast<T>((~static_cast<U>(a)) & static_cast<U>(b));
}

template <typename T>
T scalar_bit_not(T a) {
  using U = std::make_unsigned_t<T>;
  return static_cast<T>(~static_cast<U>(a));
}

template <typename T>
T scalar_bit_shl(T a, int count) {
  using U = std::make_unsigned_t<T>;
  if (count >= static_cast<int>(sizeof(T) * 8)) return T{0};
  if (count < 0) return a;
  return static_cast<T>(static_cast<U>(a) << count);
}

template <typename T>
T scalar_bit_shr(T a, int count) {
  if (count >= static_cast<int>(sizeof(T) * 8)) {
    // For signed types: all sign bits (0 or 1 depending on sign)
    // For unsigned types: 0
    if constexpr (std::is_signed_v<T>) {
      return (a < 0) ? static_cast<T>(-1) : T{0};
    } else {
      return T{0};
    }
  }
  if (count < 0) return a;
  if constexpr (std::is_signed_v<T>) {
    // Arithmetic right shift for signed types
    return a >> count;
  } else {
    // Logical right shift for unsigned types
    return a >> count;
  }
}

template <typename T>
T scalar_sqrt(T a) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::sqrt(a);
  } else {
    return static_cast<T>(0);
  }
}

// Check if an arithmetic type supports a given operation at compile time
// (Used to skip tests for types that don't support the operation)
template <typename T, typename = void>
struct has_add : std::true_type {};

template <typename T>
struct has_add<T, std::enable_if_t<std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float16_t>>>
    : std::false_type {};

} // namespace test_utils

// ============================================================================
// Test Fixture
// ============================================================================

template <typename T>
class X86_128ArithTest : public ::testing::Test {
protected:
  using Type = T;
  static constexpr nint_t FULL_SIZE = test_utils::full_vec_size<T>();

  void SetUp() override {
    a_data_ = test_utils::alloc_aligned<T>(256);
    b_data_ = test_utils::alloc_aligned<T>(256);
    for (size_t i = 0; i < 256; ++i) {
      a_data_[i] = test_utils::get_test_value<T>(i);
      b_data_[i] = test_utils::get_test_value_b<T>(i);
    }
  }

  void TearDown() override {
    std::free(a_data_);
    std::free(b_data_);
  }

  T* a_data_{};
  T* b_data_{};
};

// All tested types (arithmetic ops don't support bf16/fp16, but we include them
// so the fixture can be used; we'll skip with if constexpr where needed)
using AllTypes = ::testing::Types<
    float32_t, float64_t,
    int8_t, uint8_t, int16_t, uint16_t,
    int32_t, uint32_t, int64_t, uint64_t
>;

TYPED_TEST_SUITE(X86_128ArithTest, AllTypes);

// ============================================================================
// add
// ============================================================================

TYPED_TEST(X86_128ArithTest, AddBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  FixedTag<T, N> t;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto vr = add(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    T expected = test_utils::scalar_add(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
        << "i=" << i << " type=" << typeid(T).name();
  }
}

TYPED_TEST(X86_128ArithTest, AddWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  FixedTag<T, N> t;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto m = mwhilelt(t, 0, N / 2);
  auto vr = add(va, vb, m);

  for (nint_t i = 0; i < N / 2; ++i) {
    T expected = test_utils::scalar_add(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
        << "i=" << i;
  }
  for (nint_t i = N / 2; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t, vr, i)))
        << "i=" << i << " (masked out, should be a)";
  }
}

// ============================================================================
// sub
// ============================================================================

TYPED_TEST(X86_128ArithTest, SubBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  FixedTag<T, N> t;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto vr = sub(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    T expected = test_utils::scalar_sub(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
        << "i=" << i;
  }
}

TYPED_TEST(X86_128ArithTest, SubWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  FixedTag<T, N> t;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto m = mwhilelt(t, 0, N / 2);
  auto vr = sub(va, vb, m);

  for (nint_t i = 0; i < N / 2; ++i) {
    T expected = test_utils::scalar_sub(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
        << "i=" << i;
  }
  for (nint_t i = N / 2; i < N; ++i) {
    EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t, vr, i)));
  }
}

// ============================================================================
// mul
// ============================================================================

TYPED_TEST(X86_128ArithTest, MulBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  FixedTag<T, N> t;

  // Use smaller values to avoid overflow for integer types
  alignas(16) T small_a[N], small_b[N];
  for (nint_t i = 0; i < N; ++i) {
    small_a[i] = test_utils::get_test_value<T>(i % 5);
    small_b[i] = test_utils::get_test_value<T>((i + 2) % 5);
  }

  auto va = loadu(t, small_a);
  auto vb = loadu(t, small_b);
  auto vr = mul(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    T expected = test_utils::scalar_mul(small_a[i], small_b[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
        << "i=" << i;
  }
}

TYPED_TEST(X86_128ArithTest, MulWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (N >= 2) {
    FixedTag<T, N> t;

    alignas(16) T small_a[N], small_b[N];
    for (nint_t i = 0; i < N; ++i) {
      small_a[i] = test_utils::get_test_value<T>(i % 5);
      small_b[i] = test_utils::get_test_value<T>((i + 2) % 5);
    }

    auto va = loadu(t, small_a);
    auto vb = loadu(t, small_b);
    auto m = mwhilelt(t, 0, N / 2);
    auto vr = mul(va, vb, m);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_mul(small_a[i], small_b[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(small_a[i], get(t, vr, i)));
    }
  }
}

// ============================================================================
// div (float32/64 only)
// ============================================================================

TYPED_TEST(X86_128ArithTest, DivBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_floating_point_v<T>) {
    FixedTag<T, N> t;

    alignas(16) T a[N], b[N];
    for (nint_t i = 0; i < N; ++i) {
      a[i] = static_cast<T>((i + 1) * 10.0 + 1.0);
      b[i] = static_cast<T>((i + 1) * 3.0 + 1.0);
    }

    auto va = loadu(t, a);
    auto vb = loadu(t, b);
    auto vr = div(va, vb);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_div(a[i], b[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
          << "i=" << i;
    }
  }
}

TYPED_TEST(X86_128ArithTest, DivWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_floating_point_v<T> && N >= 2) {
    FixedTag<T, N> t;

    alignas(16) T a[N], b[N];
    for (nint_t i = 0; i < N; ++i) {
      a[i] = static_cast<T>((i + 1) * 10.0 + 1.0);
      b[i] = static_cast<T>((i + 1) * 3.0 + 1.0);
    }

    auto va = loadu(t, a);
    auto vb = loadu(t, b);
    auto m = mwhilelt(t, 0, N / 2);
    auto vr = div(va, vb, m);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_div(a[i], b[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(a[i], get(t, vr, i)));
    }
  }
}

// ============================================================================
// rcp (float32/64 only)
// ============================================================================

TYPED_TEST(X86_128ArithTest, RcpBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_floating_point_v<T>) {
    FixedTag<T, N> t;

    alignas(16) T a[N];
    for (nint_t i = 0; i < N; ++i) {
      a[i] = static_cast<T>((i + 1) * 2.0 + 1.0);
    }

    auto va = loadu(t, a);
    auto vr = rcp(va);

    for (nint_t i = 0; i < N; ++i) {
      // rcp is approximate — rcp guarantees <= 2^-12 relative error
      T expected = T{1} / a[i];
      T actual = get(t, vr, i);
      EXPECT_LT(std::abs(expected - actual) / std::abs(expected), T(0.00025))
          << "i=" << i << " expected=" << expected << " got=" << actual;
    }
  }
}

TYPED_TEST(X86_128ArithTest, RcpWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_floating_point_v<T> && N >= 2) {
    FixedTag<T, N> t;

    alignas(16) T a[N];
    for (nint_t i = 0; i < N; ++i) {
      a[i] = static_cast<T>((i + 1) * 2.0 + 1.0);
    }

    auto va = loadu(t, a);
    auto m = mwhilelt(t, 0, N / 2);
    auto default_v = fill(t, T(999));
    auto vr = rcp(va, m, default_v);

    // Masked region: check approximate
    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = T{1} / a[i];
      T actual = get(t, vr, i);
      EXPECT_LT(std::abs(expected - actual) / std::abs(expected), 0.01);
    }
    // Unmasked region: should be default
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(T(999), get(t, vr, i)));
    }
  }
}

// ============================================================================
// max / min
// ============================================================================

TYPED_TEST(X86_128ArithTest, MaxBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  FixedTag<T, N> t;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto vr = vec::max(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    T expected = test_utils::scalar_max(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
        << "i=" << i;
  }
}

TYPED_TEST(X86_128ArithTest, MaxWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (N >= 2) {
    FixedTag<T, N> t;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto m = mwhilelt(t, 0, N / 2);
    auto vr = vec::max(va, vb, m);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_max(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t, vr, i)));
    }
  }
}

TYPED_TEST(X86_128ArithTest, MinBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  FixedTag<T, N> t;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto vr = vec::min(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    T expected = test_utils::scalar_min(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
        << "i=" << i;
  }
}

TYPED_TEST(X86_128ArithTest, MinWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (N >= 2) {
    FixedTag<T, N> t;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto m = mwhilelt(t, 0, N / 2);
    auto vr = vec::min(va, vb, m);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_min(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t, vr, i)));
    }
  }
}

// ============================================================================
// neg / abs
// ============================================================================

TYPED_TEST(X86_128ArithTest, NegBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  FixedTag<T, N> t;

  auto va = loadu(t, this->a_data_);
  auto vr = neg(va);

  for (nint_t i = 0; i < N; ++i) {
    T expected = test_utils::scalar_neg(this->a_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
        << "i=" << i;
  }
}

TYPED_TEST(X86_128ArithTest, NegWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (N >= 2) {
    FixedTag<T, N> t;

    auto va = loadu(t, this->a_data_);
    auto m = mwhilelt(t, 0, N / 2);
    auto default_v = fill(t, T(999));
    auto vr = neg(va, m, default_v);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_neg(this->a_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(T(999), get(t, vr, i)));
    }
  }
}

TYPED_TEST(X86_128ArithTest, AbsBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  FixedTag<T, N> t;

  auto va = loadu(t, this->a_data_);
  auto vr = abs(va);

  for (nint_t i = 0; i < N; ++i) {
    T expected = test_utils::scalar_abs(this->a_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
        << "i=" << i;
  }
}

TYPED_TEST(X86_128ArithTest, AbsWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (N >= 2) {
    FixedTag<T, N> t;

    auto va = loadu(t, this->a_data_);
    auto m = mwhilelt(t, 0, N / 2);
    auto default_v = fill(t, T(999));
    auto vr = abs(va, m, default_v);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_abs(this->a_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(T(999), get(t, vr, i)));
    }
  }
}

// ============================================================================
// sqrt / rsqrt (float only)
// ============================================================================

TYPED_TEST(X86_128ArithTest, SqrtBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_floating_point_v<T>) {
    FixedTag<T, N> t;

    alignas(16) T a[N];
    for (nint_t i = 0; i < N; ++i) {
      a[i] = static_cast<T>((i + 1) * 4.0 + 1.0); // positive values
    }

    auto va = loadu(t, a);
    auto vr = sqrt(va);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_sqrt(a[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
          << "i=" << i;
    }
  }
}

TYPED_TEST(X86_128ArithTest, SqrtWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_floating_point_v<T> && N >= 2) {
    FixedTag<T, N> t;

    alignas(16) T a[N];
    for (nint_t i = 0; i < N; ++i) {
      a[i] = static_cast<T>((i + 1) * 4.0 + 1.0);
    }

    auto va = loadu(t, a);
    auto m = mwhilelt(t, 0, N / 2);
    auto default_v = fill(t, T(999));
    auto vr = sqrt(va, m, default_v);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_sqrt(a[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(T(999), get(t, vr, i)));
    }
  }
}

TYPED_TEST(X86_128ArithTest, RsqrtBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_floating_point_v<T>) {
    FixedTag<T, N> t;

    alignas(16) T a[N];
    for (nint_t i = 0; i < N; ++i) {
      a[i] = static_cast<T>((i + 1) * 4.0 + 1.0);
    }

    auto va = loadu(t, a);
    auto vr = rsqrt(va);

    for (nint_t i = 0; i < N; ++i) {
      T expected = T{1} / std::sqrt(a[i]);
      T actual = get(t, vr, i);
      // rsqrt guarantees <= 2^-12 relative error
      EXPECT_LT(std::abs(expected - actual) / std::abs(expected), T(0.00025))
          << "i=" << i;
    }
  }
}

TYPED_TEST(X86_128ArithTest, RsqrtWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_floating_point_v<T> && N >= 2) {
    FixedTag<T, N> t;

    alignas(16) T a[N];
    for (nint_t i = 0; i < N; ++i) {
      a[i] = static_cast<T>((i + 1) * 4.0 + 1.0);
    }

    auto va = loadu(t, a);
    auto m = mwhilelt(t, 0, N / 2);
    auto default_v = fill(t, T(999));
    auto vr = rsqrt(va, m, default_v);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = T{1} / std::sqrt(a[i]);
      T actual = get(t, vr, i);
      EXPECT_LT(std::abs(expected - actual) / std::abs(expected), 0.01);
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(T(999), get(t, vr, i)));
    }
  }
}

// ============================================================================
// Bitwise operations (integral types only)
// ============================================================================

TYPED_TEST(X86_128ArithTest, BitAndBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_integral_v<T>) {
    FixedTag<T, N> t;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto vr = bit_and(va, vb);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_bit_and(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
          << "i=" << i;
    }
  }
}

TYPED_TEST(X86_128ArithTest, BitAndWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_integral_v<T> && N >= 2) {
    FixedTag<T, N> t;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto m = mwhilelt(t, 0, N / 2);
    auto vr = bit_and(va, vb, m);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_bit_and(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t, vr, i)));
    }
  }
}

TYPED_TEST(X86_128ArithTest, BitOrBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_integral_v<T>) {
    FixedTag<T, N> t;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto vr = bit_or(va, vb);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_bit_or(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
          << "i=" << i;
    }
  }
}

TYPED_TEST(X86_128ArithTest, BitOrWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_integral_v<T> && N >= 2) {
    FixedTag<T, N> t;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto m = mwhilelt(t, 0, N / 2);
    auto vr = bit_or(va, vb, m);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_bit_or(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t, vr, i)));
    }
  }
}

TYPED_TEST(X86_128ArithTest, BitXorBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_integral_v<T>) {
    FixedTag<T, N> t;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto vr = bit_xor(va, vb);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_bit_xor(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
          << "i=" << i;
    }
  }
}

TYPED_TEST(X86_128ArithTest, BitXorWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_integral_v<T> && N >= 2) {
    FixedTag<T, N> t;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto m = mwhilelt(t, 0, N / 2);
    auto vr = bit_xor(va, vb, m);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_bit_xor(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t, vr, i)));
    }
  }
}

TYPED_TEST(X86_128ArithTest, BitAndnotBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_integral_v<T>) {
    FixedTag<T, N> t;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto vr = bit_andnot(va, vb);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_bit_andnot(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
          << "i=" << i;
    }
  }
}

TYPED_TEST(X86_128ArithTest, BitAndnotWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_integral_v<T> && N >= 2) {
    FixedTag<T, N> t;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto m = mwhilelt(t, 0, N / 2);
    auto vr = bit_andnot(va, vb, m);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_bit_andnot(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t, vr, i)));
    }
  }
}

TYPED_TEST(X86_128ArithTest, BitNotBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_integral_v<T>) {
    FixedTag<T, N> t;

    auto va = loadu(t, this->a_data_);
    auto vr = bit_not(va);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_bit_not(this->a_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
          << "i=" << i;
    }
  }
}

TYPED_TEST(X86_128ArithTest, BitNotWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_integral_v<T> && N >= 2) {
    FixedTag<T, N> t;

    auto va = loadu(t, this->a_data_);
    auto m = mwhilelt(t, 0, N / 2);
    auto default_v = fill(t, T{0x42});
    auto vr = bit_not(va, m, default_v);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_bit_not(this->a_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_TRUE(test_utils::values_equal(T{0x42}, get(t, vr, i)));
    }
  }
}

// ============================================================================
// Bitwise shift operations (integral types only)
// ============================================================================

TYPED_TEST(X86_128ArithTest, BitShlBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_integral_v<T>) {
    FixedTag<T, N> t;

    // Test various shift counts
    int shift_counts[] = {0, 1, 2, 3, 4, 7, 8};
    
    for (int shift : shift_counts) {
      auto va = loadu(t, this->a_data_);
      auto vr = bit_shl(va, shift);

      for (nint_t i = 0; i < N; ++i) {
        T expected = test_utils::scalar_bit_shl(this->a_data_[i], shift);
        EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
            << "i=" << i << " shift=" << shift
            << " a=" << static_cast<long long>(this->a_data_[i])
            << " expected=" << static_cast<long long>(expected)
            << " actual=" << static_cast<long long>(get(t, vr, i));
      }
    }
  }
}

TYPED_TEST(X86_128ArithTest, BitShlWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_integral_v<T> && N >= 2) {
    FixedTag<T, N> t;

    int shift = 2;
    auto va = loadu(t, this->a_data_);
    auto m = mwhilelt(t, 0, N / 2);
    auto vr = bit_shl(va, shift, m);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_bit_shl(this->a_data_[i], shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      // Masked out positions should retain original value
      EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t, vr, i)));
    }
  }
}

TYPED_TEST(X86_128ArithTest, BitShlLargeShift) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_integral_v<T>) {
    FixedTag<T, N> t;

    // Test shift >= type width (should result in 0)
    int large_shift = sizeof(T) * 8;  // Exactly type width
    
    auto va = loadu(t, this->a_data_);
    auto vr = bit_shl(va, large_shift);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_bit_shl(this->a_data_[i], large_shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
          << "i=" << i << " shift=" << large_shift;
    }
  }
}

TYPED_TEST(X86_128ArithTest, BitShrBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_integral_v<T>) {
    FixedTag<T, N> t;

    // Test various shift counts
    int shift_counts[] = {0, 1, 2, 3, 4, 7, 8};
    
    for (int shift : shift_counts) {
      auto va = loadu(t, this->a_data_);
      auto vr = bit_shr(va, shift);

      for (nint_t i = 0; i < N; ++i) {
        T expected = test_utils::scalar_bit_shr(this->a_data_[i], shift);
        EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
            << "i=" << i << " shift=" << shift
            << " a=" << static_cast<long long>(this->a_data_[i])
            << " expected=" << static_cast<long long>(expected)
            << " actual=" << static_cast<long long>(get(t, vr, i));
      }
    }
  }
}

TYPED_TEST(X86_128ArithTest, BitShrWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_integral_v<T> && N >= 2) {
    FixedTag<T, N> t;

    int shift = 2;
    auto va = loadu(t, this->a_data_);
    auto m = mwhilelt(t, 0, N / 2);
    auto vr = bit_shr(va, shift, m);

    for (nint_t i = 0; i < N / 2; ++i) {
      T expected = test_utils::scalar_bit_shr(this->a_data_[i], shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      // Masked out positions should retain original value
      EXPECT_TRUE(test_utils::values_equal(this->a_data_[i], get(t, vr, i)));
    }
  }
}

// Test arithmetic right shift for signed types (sign extension)
TYPED_TEST(X86_128ArithTest, BitShrArithmeticSignExtension) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
    FixedTag<T, N> t;

    // Create test data with negative values
    alignas(16) T test_data[N];
    for (nint_t i = 0; i < N; ++i) {
      // Mix of positive and negative values
      if (i % 2 == 0) {
        test_data[i] = static_cast<T>(-1 - i);  // Negative values
      } else {
        test_data[i] = static_cast<T>(1 + i);   // Positive values
      }
    }

    int shift = 1;
    auto va = loadu(t, test_data);
    auto vr = bit_shr(va, shift);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_bit_shr(test_data[i], shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
          << "i=" << i << " input=" << static_cast<long long>(test_data[i])
          << " expected=" << static_cast<long long>(expected)
          << " actual=" << static_cast<long long>(get(t, vr, i));
      
      // Verify sign extension: negative >> 1 should still be negative
      if (test_data[i] < 0) {
        EXPECT_LT(get(t, vr, i), T{0})
            << "Arithmetic right shift should preserve sign for negative values";
      }
    }
  }
}

// Test logical right shift for unsigned types (zero fill)
TYPED_TEST(X86_128ArithTest, BitShrLogicalZeroFill) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
    FixedTag<T, N> t;

    // Create test data with high bits set
    alignas(16) T test_data[N];
    for (nint_t i = 0; i < N; ++i) {
      test_data[i] = static_cast<T>(~T{0} - i);  // High bits set
    }

    int shift = 1;
    auto va = loadu(t, test_data);
    auto vr = bit_shr(va, shift);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_bit_shr(test_data[i], shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
          << "i=" << i << " input=" << static_cast<long long>(test_data[i])
          << " expected=" << static_cast<long long>(expected)
          << " actual=" << static_cast<long long>(get(t, vr, i));
    }
  }
}

TYPED_TEST(X86_128ArithTest, BitShrLargeShift) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_integral_v<T>) {
    FixedTag<T, N> t;

    // Test shift >= type width
    int large_shift = sizeof(T) * 8;  // Exactly type width
    
    // Create test data with both positive and negative values
    alignas(16) T test_data[N];
    for (nint_t i = 0; i < N; ++i) {
      if constexpr (std::is_signed_v<T>) {
        test_data[i] = (i % 2 == 0) ? static_cast<T>(-1 - i) : static_cast<T>(1 + i);
      } else {
        test_data[i] = static_cast<T>(i + 1);
      }
    }

    auto va = loadu(t, test_data);
    auto vr = bit_shr(va, large_shift);

    for (nint_t i = 0; i < N; ++i) {
      T expected = test_utils::scalar_bit_shr(test_data[i], large_shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
          << "i=" << i << " shift=" << large_shift;
    }
  }
}

// Test shift with various patterns to ensure correctness
TYPED_TEST(X86_128ArithTest, BitShlPattern) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_integral_v<T>) {
    FixedTag<T, N> t;

    // Test with specific bit patterns
    alignas(16) T test_data[N];
    for (nint_t i = 0; i < N; ++i) {
      test_data[i] = static_cast<T>(1);  // Single bit set
    }

    for (int shift = 0; shift < static_cast<int>(sizeof(T) * 8); ++shift) {
      auto va = loadu(t, test_data);
      auto vr = bit_shl(va, shift);

      for (nint_t i = 0; i < N; ++i) {
        T expected = test_utils::scalar_bit_shl(T{1}, shift);
        EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
            << "shift=" << shift
            << " expected=" << static_cast<long long>(expected)
            << " actual=" << static_cast<long long>(get(t, vr, i));
      }
    }
  }
}

TYPED_TEST(X86_128ArithTest, BitShrPattern) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (std::is_integral_v<T>) {
    FixedTag<T, N> t;

    // Test with high bit set
    alignas(16) T test_data[N];
    for (nint_t i = 0; i < N; ++i) {
      // Set the highest bit
      test_data[i] = static_cast<T>(T{1} << (sizeof(T) * 8 - 1));
    }

    for (int shift = 0; shift < static_cast<int>(sizeof(T) * 8); ++shift) {
      auto va = loadu(t, test_data);
      auto vr = bit_shr(va, shift);

      for (nint_t i = 0; i < N; ++i) {
        T expected = test_utils::scalar_bit_shr(
            static_cast<T>(T{1} << (sizeof(T) * 8 - 1)), shift);
        EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
            << "shift=" << shift
            << " expected=" << static_cast<long long>(expected)
            << " actual=" << static_cast<long long>(get(t, vr, i));
      }
    }
  }
}

// Half-size vector shift tests
TYPED_TEST(X86_128ArithTest, HalfSizeBitShl) {
  using T = typename TestFixture::Type;
  constexpr nint_t HALF = TestFixture::FULL_SIZE / 2;
  if constexpr (std::is_integral_v<T> && HALF >= 1) {
    FixedTag<T, HALF> t;

    int shift = 3;
    auto va = loadu(t, this->a_data_);
    auto vr = bit_shl(va, shift);

    for (nint_t i = 0; i < HALF; ++i) {
      T expected = test_utils::scalar_bit_shl(this->a_data_[i], shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
  }
}

TYPED_TEST(X86_128ArithTest, HalfSizeBitShr) {
  using T = typename TestFixture::Type;
  constexpr nint_t HALF = TestFixture::FULL_SIZE / 2;
  if constexpr (std::is_integral_v<T> && HALF >= 1) {
    FixedTag<T, HALF> t;

    int shift = 3;
    auto va = loadu(t, this->a_data_);
    auto vr = bit_shr(va, shift);

    for (nint_t i = 0; i < HALF; ++i) {
      T expected = test_utils::scalar_bit_shr(this->a_data_[i], shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
  }
}

// Multi-word vector shift tests
TYPED_TEST(X86_128ArithTest, MultiWordBitShl) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL = TestFixture::FULL_SIZE;
  constexpr nint_t MULTI = FULL * 2;

  if constexpr (std::is_integral_v<T>) {
    FixedTag<T, MULTI> t;

    int shift = 3;
    auto va = loadu(t, this->a_data_);
    auto vr = bit_shl(va, shift);

    for (nint_t i = 0; i < MULTI; ++i) {
      T expected = test_utils::scalar_bit_shl(this->a_data_[i], shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
  }
}

TYPED_TEST(X86_128ArithTest, MultiWordBitShr) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL = TestFixture::FULL_SIZE;
  constexpr nint_t MULTI = FULL * 2;

  if constexpr (std::is_integral_v<T>) {
    FixedTag<T, MULTI> t;

    int shift = 3;
    auto va = loadu(t, this->a_data_);
    auto vr = bit_shr(va, shift);

    for (nint_t i = 0; i < MULTI; ++i) {
      T expected = test_utils::scalar_bit_shr(this->a_data_[i], shift);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)));
    }
  }
}

// ============================================================================
// Comparison operations → Mask
// ============================================================================

// Helper: verify a comparison mask element by element
template <typename T, typename CompareFn>
void verify_cmp_mask(FixedTag<T, 16 / sizeof(T)> t, nint_t N,
                     const T* a, const T* b,
                     typename FixedTag<T, 16 / sizeof(T)>::MaskType m,
                     CompareFn cmp_fn) {
  for (nint_t i = 0; i < N; ++i) {
    bool expected = cmp_fn(a[i], b[i]);
    bool actual = get(t, m, i);
    EXPECT_EQ(expected, actual)
        << "i=" << i << " a=" << static_cast<long long>(a[i])
        << " b=" << static_cast<long long>(b[i]);
  }
}

TYPED_TEST(X86_128ArithTest, CmpeqBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  FixedTag<T, N> t;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->a_data_); // same data → all true
  auto m = cmpeq(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(get(t, m, i)) << "i=" << i;
  }

  // Different data → all false
  auto vb2 = loadu(t, this->b_data_);
  auto m2 = cmpeq(va, vb2);
  for (nint_t i = 0; i < N; ++i) {
    EXPECT_FALSE(get(t, m2, i)) << "i=" << i;
  }
}

TYPED_TEST(X86_128ArithTest, CmpeqWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  FixedTag<T, N> t;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->a_data_); // all equal
  auto m_pred = mwhilelt(t, 0, N / 2);
  auto m_result = cmpeq(va, vb, m_pred);

  for (nint_t i = 0; i < N / 2; ++i) {
    EXPECT_TRUE(get(t, m_result, i)) << "i=" << i;
  }
  for (nint_t i = N / 2; i < N; ++i) {
    EXPECT_FALSE(get(t, m_result, i)) << "i=" << i;
  }
}

TYPED_TEST(X86_128ArithTest, CmpneBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  FixedTag<T, N> t;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_); // different → all true
  auto m = cmpne(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    EXPECT_TRUE(get(t, m, i)) << "i=" << i;
  }

  auto vsame = loadu(t, this->a_data_);
  auto m2 = cmpne(va, vsame);
  for (nint_t i = 0; i < N; ++i) {
    EXPECT_FALSE(get(t, m2, i)) << "i=" << i;
  }
}

TYPED_TEST(X86_128ArithTest, CmpneWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  FixedTag<T, N> t;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto m_pred = mwhilelt(t, 0, N / 2);
  auto m_result = cmpne(va, vb, m_pred);

  for (nint_t i = 0; i < N / 2; ++i) {
    EXPECT_TRUE(get(t, m_result, i));
  }
  for (nint_t i = N / 2; i < N; ++i) {
    EXPECT_FALSE(get(t, m_result, i));
  }
}

TYPED_TEST(X86_128ArithTest, CmpltBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  FixedTag<T, N> t;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto m = cmplt(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    bool expected;
    if constexpr (std::is_floating_point_v<T>) {
      expected = this->a_data_[i] < this->b_data_[i];
    } else {
      expected = this->a_data_[i] < this->b_data_[i];
    }
    EXPECT_EQ(expected, get(t, m, i))
        << "i=" << i << " a=" << static_cast<long long>(this->a_data_[i])
        << " b=" << static_cast<long long>(this->b_data_[i]);
  }
}

TYPED_TEST(X86_128ArithTest, CmpltWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (N >= 2) {
    FixedTag<T, N> t;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto m_pred = mwhilelt(t, 0, N / 2);
    auto m_result = cmplt(va, vb, m_pred);

    // Within pred range: verify correctness
    for (nint_t i = 0; i < N / 2; ++i) {
      bool expected = this->a_data_[i] < this->b_data_[i];
      EXPECT_EQ(expected, get(t, m_result, i)) << "i=" << i;
    }
    // Outside pred range: should be false
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_FALSE(get(t, m_result, i)) << "i=" << i;
    }
  }
}

TYPED_TEST(X86_128ArithTest, CmpgtBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  FixedTag<T, N> t;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto m = cmpgt(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    bool expected = this->a_data_[i] > this->b_data_[i];
    EXPECT_EQ(expected, get(t, m, i))
        << "i=" << i << " a=" << static_cast<long long>(this->a_data_[i])
        << " b=" << static_cast<long long>(this->b_data_[i]);
  }
}

TYPED_TEST(X86_128ArithTest, CmpgtWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (N >= 2) {
    FixedTag<T, N> t;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto m_pred = mwhilelt(t, 0, N / 2);
    auto m_result = cmpgt(va, vb, m_pred);

    for (nint_t i = 0; i < N / 2; ++i) {
      bool expected = this->a_data_[i] > this->b_data_[i];
      EXPECT_EQ(expected, get(t, m_result, i));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_FALSE(get(t, m_result, i));
    }
  }
}

TYPED_TEST(X86_128ArithTest, CmpleBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  FixedTag<T, N> t;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto m = cmple(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    bool expected = this->a_data_[i] <= this->b_data_[i];
    EXPECT_EQ(expected, get(t, m, i))
        << "i=" << i;
  }
}

TYPED_TEST(X86_128ArithTest, CmpleWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (N >= 2) {
    FixedTag<T, N> t;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto m_pred = mwhilelt(t, 0, N / 2);
    auto m_result = cmple(va, vb, m_pred);

    for (nint_t i = 0; i < N / 2; ++i) {
      bool expected = this->a_data_[i] <= this->b_data_[i];
      EXPECT_EQ(expected, get(t, m_result, i));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_FALSE(get(t, m_result, i));
    }
  }
}

TYPED_TEST(X86_128ArithTest, CmpgeBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  FixedTag<T, N> t;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto m = cmpge(va, vb);

  for (nint_t i = 0; i < N; ++i) {
    bool expected = this->a_data_[i] >= this->b_data_[i];
    EXPECT_EQ(expected, get(t, m, i))
        << "i=" << i;
  }
}

TYPED_TEST(X86_128ArithTest, CmpgeWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (N >= 2) {
    FixedTag<T, N> t;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto m_pred = mwhilelt(t, 0, N / 2);
    auto m_result = cmpge(va, vb, m_pred);

    for (nint_t i = 0; i < N / 2; ++i) {
      bool expected = this->a_data_[i] >= this->b_data_[i];
      EXPECT_EQ(expected, get(t, m_result, i));
    }
    for (nint_t i = N / 2; i < N; ++i) {
      EXPECT_FALSE(get(t, m_result, i));
    }
  }
}

// ============================================================================
// Float-specific classification: isnan, isposinf, isneginf, isinf
// ============================================================================

// Specialize test fixture for float-only tests
using FloatTypes = ::testing::Types<float32_t, float64_t>;

template <typename T>
class X86_128FloatClassifyTest : public ::testing::Test {
protected:
  using Type = T;
  static constexpr nint_t FULL_SIZE = test_utils::full_vec_size<T>();
};

TYPED_TEST_SUITE(X86_128FloatClassifyTest, FloatTypes);

TYPED_TEST(X86_128FloatClassifyTest, IsNanBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  FixedTag<T, N> t;

  alignas(16) T a[N];
  for (nint_t i = 0; i < N; ++i) {
    a[i] = static_cast<T>(i + 1.0);
  }
  // Inject NaN at index 0 and N-1
  if constexpr (std::is_same_v<T, float32_t>) {
    a[0] = std::numeric_limits<float>::quiet_NaN();
    a[N - 1] = std::numeric_limits<float>::quiet_NaN();
  } else {
    a[0] = std::numeric_limits<double>::quiet_NaN();
    a[N - 1] = std::numeric_limits<double>::quiet_NaN();
  }

  auto va = loadu(t, a);
  auto m = isnan(va);

  EXPECT_TRUE(get(t, m, 0));
  EXPECT_TRUE(get(t, m, N - 1));
  for (nint_t i = 1; i < N - 1; ++i) {
    EXPECT_FALSE(get(t, m, i)) << "i=" << i;
  }
}

TYPED_TEST(X86_128FloatClassifyTest, IsNanWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (N >= 2) {
    FixedTag<T, N> t;

    alignas(16) T a[N];
    for (nint_t i = 0; i < N; ++i) a[i] = static_cast<T>(i + 1.0);
    a[0] = static_cast<T>(std::numeric_limits<double>::quiet_NaN());
    a[N / 2] = static_cast<T>(std::numeric_limits<double>::quiet_NaN());

    auto va = loadu(t, a);
    auto m_pred = mwhilelt(t, 0, N / 2);
    auto m_result = isnan(va, m_pred);

    // Only i=0 is in range and is NaN
    EXPECT_TRUE(get(t, m_result, 0));
    // i=N/2 is NaN but outside pred mask → false
    EXPECT_FALSE(get(t, m_result, N / 2));
    for (nint_t i = 1; i < N / 2; ++i) {
      EXPECT_FALSE(get(t, m_result, i));
    }
  }
}

TYPED_TEST(X86_128FloatClassifyTest, IsPosInfBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  FixedTag<T, N> t;

  alignas(16) T a[N];
  for (nint_t i = 0; i < N; ++i) a[i] = static_cast<T>(i + 1.0);
  a[0] = static_cast<T>(INFINITY);
  a[N - 1] = static_cast<T>(-INFINITY); // negative inf, should NOT match

  auto va = loadu(t, a);
  auto m = isposinf(va);

  EXPECT_TRUE(get(t, m, 0));
  EXPECT_FALSE(get(t, m, N - 1));
  for (nint_t i = 1; i < N - 1; ++i) {
    EXPECT_FALSE(get(t, m, i)) << "i=" << i;
  }
}

TYPED_TEST(X86_128FloatClassifyTest, IsPosInfWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (N >= 2) {
    FixedTag<T, N> t;

    alignas(16) T a[N];
    for (nint_t i = 0; i < N; ++i) a[i] = static_cast<T>(i + 1.0);
    a[0] = static_cast<T>(INFINITY);
    a[N - 1] = static_cast<T>(INFINITY);

    auto va = loadu(t, a);
    auto m_pred = mwhilelt(t, 0, N / 2);
    auto m_result = isposinf(va, m_pred);

    EXPECT_TRUE(get(t, m_result, 0));
    EXPECT_FALSE(get(t, m_result, N - 1)); // outside pred mask
  }
}

TYPED_TEST(X86_128FloatClassifyTest, IsNegInfBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  FixedTag<T, N> t;

  alignas(16) T a[N];
  for (nint_t i = 0; i < N; ++i) a[i] = static_cast<T>(i + 1.0);
  a[0] = static_cast<T>(-INFINITY);
  a[N - 1] = static_cast<T>(INFINITY); // positive inf, should NOT match

  auto va = loadu(t, a);
  auto m = isneginf(va);

  EXPECT_TRUE(get(t, m, 0));
  EXPECT_FALSE(get(t, m, N - 1));
  for (nint_t i = 1; i < N - 1; ++i) {
    EXPECT_FALSE(get(t, m, i)) << "i=" << i;
  }
}

TYPED_TEST(X86_128FloatClassifyTest, IsNegInfWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (N >= 2) {
    FixedTag<T, N> t;

    alignas(16) T a[N];
    for (nint_t i = 0; i < N; ++i) a[i] = static_cast<T>(i + 1.0);
    a[0] = static_cast<T>(-INFINITY);
    a[N - 1] = static_cast<T>(-INFINITY);

    auto va = loadu(t, a);
    auto m_pred = mwhilelt(t, 0, N / 2);
    auto m_result = isneginf(va, m_pred);

    EXPECT_TRUE(get(t, m_result, 0));
    EXPECT_FALSE(get(t, m_result, N - 1)); // outside pred mask
  }
}

TYPED_TEST(X86_128FloatClassifyTest, IsInfBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  FixedTag<T, N> t;

  alignas(16) T a[N];
  for (nint_t i = 0; i < N; ++i) a[i] = static_cast<T>(i + 1.0);
  a[0] = static_cast<T>(INFINITY);
  a[1] = static_cast<T>(-INFINITY);
  if (N > 2)
    a[2] = static_cast<T>(std::numeric_limits<double>::quiet_NaN()); // NaN is not inf

  auto va = loadu(t, a);
  auto m = isinf(va);

  EXPECT_TRUE(get(t, m, 0));
  EXPECT_TRUE(get(t, m, 1));
  if (N > 2)
    EXPECT_FALSE(get(t, m, 2)); // NaN
  for (nint_t i = 3; i < N; ++i) {
    EXPECT_FALSE(get(t, m, i)) << "i=" << i;
  }
}

TYPED_TEST(X86_128FloatClassifyTest, IsInfWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t N = TestFixture::FULL_SIZE;
  if constexpr (N >= 4) {
    FixedTag<T, N> t;

    alignas(16) T a[N];
    for (nint_t i = 0; i < N; ++i) a[i] = static_cast<T>(i + 1.0);
    a[0] = static_cast<T>(INFINITY);
    a[N - 1] = static_cast<T>(-INFINITY);

    auto va = loadu(t, a);
    auto m_pred = mwhilelt(t, 0, N / 2);
    auto m_result = isinf(va, m_pred);

    EXPECT_TRUE(get(t, m_result, 0));
    EXPECT_FALSE(get(t, m_result, N - 1)); // outside pred mask
  }
}

// ============================================================================
// Half-size vector arithmetic
// ============================================================================

TYPED_TEST(X86_128ArithTest, HalfSizeAdd) {
  using T = typename TestFixture::Type;
  constexpr nint_t HALF = TestFixture::FULL_SIZE / 2;
  if constexpr (HALF >= 1) {
    FixedTag<T, HALF> t;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto vr = add(va, vb);

    for (nint_t i = 0; i < HALF; ++i) {
      T expected = test_utils::scalar_add(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
          << "i=" << i;
    }
  }
}

TYPED_TEST(X86_128ArithTest, HalfSizeSub) {
  using T = typename TestFixture::Type;
  constexpr nint_t HALF = TestFixture::FULL_SIZE / 2;
  if constexpr (HALF >= 1) {
    FixedTag<T, HALF> t;

    auto va = loadu(t, this->a_data_);
    auto vb = loadu(t, this->b_data_);
    auto vr = sub(va, vb);

    for (nint_t i = 0; i < HALF; ++i) {
      T expected = test_utils::scalar_sub(this->a_data_[i], this->b_data_[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
          << "i=" << i;
    }
  }
}

TYPED_TEST(X86_128ArithTest, HalfSizeMul) {
  using T = typename TestFixture::Type;
  constexpr nint_t HALF = TestFixture::FULL_SIZE / 2;
  if constexpr (HALF >= 1) {
    FixedTag<T, HALF> t;

    alignas(16) T sa[HALF], sb[HALF];
    for (nint_t i = 0; i < HALF; ++i) {
      sa[i] = test_utils::get_test_value<T>(i % 5);
      sb[i] = test_utils::get_test_value<T>((i + 2) % 5);
    }

    auto va = loadu(t, sa);
    auto vb = loadu(t, sb);
    auto vr = mul(va, vb);

    for (nint_t i = 0; i < HALF; ++i) {
      T expected = test_utils::scalar_mul(sa[i], sb[i]);
      EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
          << "i=" << i;
    }
  }
}

// ============================================================================
// Multi-word vector arithmetic
// ============================================================================

TYPED_TEST(X86_128ArithTest, MultiWordAdd) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL = TestFixture::FULL_SIZE;
  constexpr nint_t MULTI = FULL * 2;

  FixedTag<T, MULTI> t;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto vr = add(va, vb);

  for (nint_t i = 0; i < MULTI; ++i) {
    T expected = test_utils::scalar_add(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
        << "i=" << i;
  }
}

TYPED_TEST(X86_128ArithTest, MultiWordSub) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL = TestFixture::FULL_SIZE;
  constexpr nint_t MULTI = FULL * 2;

  FixedTag<T, MULTI> t;

  auto va = loadu(t, this->a_data_);
  auto vb = loadu(t, this->b_data_);
  auto vr = sub(va, vb);

  for (nint_t i = 0; i < MULTI; ++i) {
    T expected = test_utils::scalar_sub(this->a_data_[i], this->b_data_[i]);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, vr, i)))
        << "i=" << i;
  }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
