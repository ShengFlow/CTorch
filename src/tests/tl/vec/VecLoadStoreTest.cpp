//
// VecLoadStoreTest.cpp
// Comprehensive test for word-level SIMD operations
// Covers all 12 element types: bfloat16_t, float16_t, float32_t, float64_t,
//                               int8_t, uint8_t, int16_t, uint16_t,
//                               int32_t, uint32_t, int64_t, uint64_t
//

#include <gtest/gtest.h>
#include <cstring>
#include <memory>
#include <limits>
#include <type_traits>

#include "tl/cpu/Vec.h"

using namespace ct;
using namespace ct::tl::vec;

// ============================================================================
// Helper utilities for type-specific operations
// ============================================================================

namespace test_utils {

// Helper to get a test value for each type
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

// Helper to check if two values are approximately equal
template <typename T>
::testing::AssertionResult values_equal(T expected, T actual) {
  if constexpr (std::is_same_v<T, bfloat16_t>) {
    float e = static_cast<float>(expected);
    float a = static_cast<float>(actual);
    if (std::abs(e - a) < 0.01f) {
      return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() 
        << "Expected " << e << ", got " << a;
  } else if constexpr (std::is_same_v<T, float16_t>) {
    float e = static_cast<float>(expected);
    float a = static_cast<float>(actual);
    if (std::abs(e - a) < 0.01f) {
      return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() 
        << "Expected " << e << ", got " << a;
  } else if constexpr (std::is_same_v<T, float32_t>) {
    if (std::abs(expected - actual) < 1e-5f) {
      return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() 
        << "Expected " << expected << ", got " << actual;
  } else if constexpr (std::is_same_v<T, float64_t>) {
    if (std::abs(expected - actual) < 1e-10) {
      return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() 
        << "Expected " << expected << ", got " << actual;
  } else {
    if (expected == actual) {
      return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() 
        << "Expected " << static_cast<long long>(expected) 
        << ", got " << static_cast<long long>(actual);
  }
}

// Get the full vector size for a type
template <typename T>
constexpr nint_t full_vec_size() {
  return ScalableTag<T>::N; // runtime scalable not supported currently
}

// Aligned memory allocator
template <typename T>
T* alloc_aligned(size_t count) {
  void* ptr = std::aligned_alloc(DEFAULT_ALIGNMENT, count * sizeof(T));
  return static_cast<T*>(ptr);
}

} // namespace test_utils

// ============================================================================
// Test Fixture Template
// ============================================================================

template <typename T>
class VecLoadStoreTest : public ::testing::Test {
protected:
  using Type = T;
  static constexpr nint_t FULL_SIZE = test_utils::full_vec_size<T>();

  void SetUp() override {
    // Allocate aligned memory for testing
    aligned_data_ = test_utils::alloc_aligned<T>(256);
    aligned_out_ = test_utils::alloc_aligned<T>(256);
    
    // Initialize with test values
    for (size_t i = 0; i < 256; ++i) {
      aligned_data_[i] = test_utils::get_test_value<T>(i);
      aligned_out_[i] = T{};
    }
  }
  
  void TearDown() override {
    std::free(aligned_data_);
    std::free(aligned_out_);
  }
  
  T* aligned_data_{};
  T* aligned_out_{};
};

// List of all tested types
using TestedTypes = ::testing::Types<
    bfloat16_t, float16_t, float32_t, float64_t,
    int8_t, uint8_t, int16_t, uint16_t,
    int32_t, uint32_t, int64_t, uint64_t
>;

TYPED_TEST_SUITE(VecLoadStoreTest, TestedTypes);

// ============================================================================
// Size Verification Tests
// ============================================================================

TYPED_TEST(VecLoadStoreTest, VerifySize) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  EXPECT_EQ(size(t), FULL_SIZE);
  EXPECT_EQ(word_size(t), FULL_SIZE);
  EXPECT_EQ(num_words(t), 1);
  EXPECT_TRUE(is_word_vec(t));
}

TYPED_TEST(VecLoadStoreTest, VerifyHalfSize) {
  using T = typename TestFixture::Type;
  constexpr nint_t HALF_SIZE = TestFixture::FULL_SIZE / 2;
  
  if constexpr (HALF_SIZE >= 1) {
    FixedTag<T, HALF_SIZE> t;
    EXPECT_EQ(size(t), HALF_SIZE);
    // Half-size vectors use full-size registers
    EXPECT_EQ(word_size(t), HALF_SIZE);
  }
}

// ============================================================================
// fill / zeros Tests
// ============================================================================

TYPED_TEST(VecLoadStoreTest, FillBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  T fill_val = test_utils::get_test_value<T>(42);
  
  auto v = fill(t, fill_val);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, get(t, v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, FillZero) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  T zero_val = T{};
  
  auto v = fill(t, zero_val);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(zero_val, get(t, v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, ZerosBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  auto v = zeros(t);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(T{}, get(t, v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, FillWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  T fill_val = test_utils::get_test_value<T>(100);
  T default_val = test_utils::get_test_value<T>(200);
  
  // Create mask for first half elements
  auto m = mwhilelt(t, 0, FULL_SIZE / 2);
  
  auto default_v = fill(t, default_val);
  auto v = fill(t, fill_val, m, default_v);
  
  for (nint_t i = 0; i < FULL_SIZE / 2; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, get(t, v, i)));
  }
  for (nint_t i = FULL_SIZE / 2; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(t, v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, FillWithMaskAll) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  T fill_val = test_utils::get_test_value<T>(100);
  T default_val = test_utils::get_test_value<T>(200);
  
  auto m = mtrue(t);
  auto default_v = fill(t, default_val);
  auto v = fill(t, fill_val, m, default_v);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, get(t, v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, FillWithMaskNone) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  T fill_val = test_utils::get_test_value<T>(100);
  T default_val = test_utils::get_test_value<T>(200);
  
  auto m = mfalse(t);
  auto default_v = fill(t, default_val);
  auto v = fill(t, fill_val, m, default_v);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(t, v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, FillWithN) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  T fill_val = test_utils::get_test_value<T>(100);
  T default_val = test_utils::get_test_value<T>(200);
  
  nint_t n = FULL_SIZE / 2;
  auto default_v = fill(t, default_val);
  auto v = fill(t, fill_val, n, default_v);
  
  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, get(t, v, i)));
  }
  for (nint_t i = n; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(t, v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, FillWithNZero) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  T fill_val = test_utils::get_test_value<T>(100);
  T default_val = test_utils::get_test_value<T>(200);
  
  auto default_v = fill(t, default_val);
  auto v = fill(t, fill_val, 0, default_v);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(t, v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, FillWithNFull) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  T fill_val = test_utils::get_test_value<T>(100);
  T default_val = test_utils::get_test_value<T>(200);
  
  auto default_v = fill(t, default_val);
  auto v = fill(t, fill_val, FULL_SIZE, default_v);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, get(t, v, i)));
  }
}

// ============================================================================
// mfill / mtrue / mfalse Tests
// ============================================================================

TYPED_TEST(VecLoadStoreTest, MfillTrue) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  auto m = mfill(t, true);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(get(t, m, i));
  }
}

TYPED_TEST(VecLoadStoreTest, MfillFalse) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  auto m = mfill(t, false);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_FALSE(get(t, m, i));
  }
}

TYPED_TEST(VecLoadStoreTest, MtrueMfalse) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  auto m_true = mtrue(t);
  auto m_false = mfalse(t);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(get(t, m_true, i));
    EXPECT_FALSE(get(t, m_false, i));
  }
}

// ============================================================================
// mwhilelt / mwhilege Tests
// ============================================================================

TYPED_TEST(VecLoadStoreTest, MwhileltBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  // Test first half
  auto m = mwhilelt(t, 0, FULL_SIZE / 2);
  
  for (nint_t i = 0; i < FULL_SIZE / 2; ++i) {
    EXPECT_TRUE(get(t, m, i)) << "i = " << i;
  }
  for (nint_t i = FULL_SIZE / 2; i < FULL_SIZE; ++i) {
    EXPECT_FALSE(get(t, m, i)) << "i = " << i;
  }
}

TYPED_TEST(VecLoadStoreTest, MwhileltAll) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  auto m = mwhilelt(t, 0, FULL_SIZE);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(get(t, m, i)) << "i = " << i;
  }
}

TYPED_TEST(VecLoadStoreTest, MwhileltNone) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  auto m = mwhilelt(t, 0, 0);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_FALSE(get(t, m, i)) << "i = " << i;
  }
}

TYPED_TEST(VecLoadStoreTest, MwhileltVariousRanges) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  // Test various ranges
  for (nint_t end = 0; end <= FULL_SIZE; ++end) {
    auto m = mwhilelt(t, 0, end);
    
    for (nint_t i = 0; i < FULL_SIZE; ++i) {
      EXPECT_EQ(get(t, m, i), i < end) 
          << "end = " << end << ", i = " << i;
    }
  }
}

TYPED_TEST(VecLoadStoreTest, MwhilegeBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  // Test second half
  auto m = mwhilege(t, 0, FULL_SIZE / 2);
  
  for (nint_t i = 0; i < FULL_SIZE / 2; ++i) {
    EXPECT_FALSE(get(t, m, i)) << "i = " << i;
  }
  for (nint_t i = FULL_SIZE / 2; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(get(t, m, i)) << "i = " << i;
  }
}

TYPED_TEST(VecLoadStoreTest, MwhilegeAll) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  auto m = mwhilege(t, 0, 0);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(get(t, m, i)) << "i = " << i;
  }
}

TYPED_TEST(VecLoadStoreTest, MwhilegeNone) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  auto m = mwhilege(t, 0, FULL_SIZE);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_FALSE(get(t, m, i)) << "i = " << i;
  }
}

TYPED_TEST(VecLoadStoreTest, MwhilegeVariousRanges) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  // Test various ranges
  for (nint_t start = 0; start <= FULL_SIZE; ++start) {
    auto m = mwhilege(t, 0, start);
    
    for (nint_t i = 0; i < FULL_SIZE; ++i) {
      EXPECT_EQ(get(t, m, i), i >= start) 
          << "start = " << start << ", i = " << i;
    }
  }
}

TYPED_TEST(VecLoadStoreTest, MwhileleMwhilegt) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  nint_t mid = FULL_SIZE / 2;
  
  // mwhilele: [0, mid] true
  auto m_le = mwhilele(t, 0, mid);
  for (nint_t i = 0; i <= mid; ++i) {
    EXPECT_TRUE(get(t, m_le, i)) << "mwhilele: i = " << i;
  }
  for (nint_t i = mid + 1; i < FULL_SIZE; ++i) {
    EXPECT_FALSE(get(t, m_le, i)) << "mwhilele: i = " << i;
  }
  
  // mwhilegt: (mid, FULL_SIZE) true
  auto m_gt = mwhilegt(t, 0, mid);
  for (nint_t i = 0; i <= mid; ++i) {
    EXPECT_FALSE(get(t, m_gt, i)) << "mwhilegt: i = " << i;
  }
  for (nint_t i = mid + 1; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(get(t, m_gt, i)) << "mwhilegt: i = " << i;
  }
}

// ============================================================================
// loadu / storeu Tests
// ============================================================================

TYPED_TEST(VecLoadStoreTest, LoaduBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  auto v = loadu(t, this->aligned_data_);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], get(t, v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, StoreuBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  auto v = loadu(t, this->aligned_data_);
  storeu(t, this->aligned_out_, v);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], this->aligned_out_[i]));
  }
}

TYPED_TEST(VecLoadStoreTest, LoaduStoreuRoundTrip) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  T fill_val = test_utils::get_test_value<T>(123);
  auto v = fill(t, fill_val);
  storeu(t, this->aligned_out_, v);
  
  auto v2 = loadu(t, this->aligned_out_);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, get(t, v2, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, LoaduWithN) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  T default_val = test_utils::get_test_value<T>(999);
  auto default_v = fill(t, default_val);
  
  nint_t n = FULL_SIZE / 2;
  auto v = loadu(t, this->aligned_data_, n, default_v);
  
  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], get(t, v, i)));
  }
  for (nint_t i = n; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(t, v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, LoaduWithNZero) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  T default_val = test_utils::get_test_value<T>(999);
  auto default_v = fill(t, default_val);
  
  auto v = loadu(t, this->aligned_data_, 0, default_v);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(t, v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, LoaduWithNFull) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  T default_val = test_utils::get_test_value<T>(999);
  auto default_v = fill(t, default_val);
  
  auto v = loadu(t, this->aligned_data_, FULL_SIZE, default_v);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], get(t, v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, StoreuWithN) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  // Reset output
  for (int i = 0; i < FULL_SIZE; ++i) {
    this->aligned_out_[i] = test_utils::get_test_value<T>(-1);
  }
  
  T fill_val = test_utils::get_test_value<T>(42);
  auto v = fill(t, fill_val);
  
  nint_t n = FULL_SIZE / 2;
  storeu(t, this->aligned_out_, n, v);
  
  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, this->aligned_out_[i]));
  }
  // Rest should be unchanged (not the fill_val)
}

TYPED_TEST(VecLoadStoreTest, StoreuWithNZero) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < FULL_SIZE; ++i) {
    this->aligned_out_[i] = sentinel;
  }
  
  T fill_val = test_utils::get_test_value<T>(42);
  auto v = fill(t, fill_val);
  
  storeu(t, this->aligned_out_, 0, v);
  
  // Nothing should be stored
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(sentinel, this->aligned_out_[i]));
  }
}

// ============================================================================
// load / store (aligned) Tests
// ============================================================================

TYPED_TEST(VecLoadStoreTest, LoadAligned) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  auto v = load(t, this->aligned_data_);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], get(t, v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, StoreAligned) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  auto v = load(t, this->aligned_data_);
  store(t, this->aligned_out_, v);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], this->aligned_out_[i]));
  }
}

TYPED_TEST(VecLoadStoreTest, LoadWithN) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  T default_val = test_utils::get_test_value<T>(999);
  auto default_v = fill(t, default_val);
  
  nint_t n = FULL_SIZE / 2;
  auto v = load(t, this->aligned_data_, n, default_v);
  
  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], get(t, v, i)));
  }
  for (nint_t i = n; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(t, v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, StoreWithN) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  // Reset output
  for (int i = 0; i < FULL_SIZE; ++i) {
    this->aligned_out_[i] = test_utils::get_test_value<T>(-1);
  }
  
  T fill_val = test_utils::get_test_value<T>(42);
  auto v = fill(t, fill_val);
  
  nint_t n = FULL_SIZE / 2;
  store(t, this->aligned_out_, n, v);
  
  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, this->aligned_out_[i]));
  }
}

// ============================================================================
// Masked load/store Tests
// ============================================================================

TYPED_TEST(VecLoadStoreTest, LoaduWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  auto m = mwhilelt(t, 0, FULL_SIZE / 2);
  T default_val = test_utils::get_test_value<T>(999);
  auto default_v = fill(t, default_val);
  
  auto v = loadu(t, this->aligned_data_, m, default_v);
  
  for (nint_t i = 0; i < FULL_SIZE / 2; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], get(t, v, i)));
  }
  for (nint_t i = FULL_SIZE / 2; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(t, v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, LoadWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  auto m = mwhilelt(t, 0, FULL_SIZE / 2);
  T default_val = test_utils::get_test_value<T>(999);
  auto default_v = fill(t, default_val);
  
  auto v = load(t, this->aligned_data_, m, default_v);
  
  for (nint_t i = 0; i < FULL_SIZE / 2; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], get(t, v, i)));
  }
  for (nint_t i = FULL_SIZE / 2; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(t, v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, StoreuWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < FULL_SIZE; ++i) {
    this->aligned_out_[i] = sentinel;
  }
  
  auto m = mwhilelt(t, 0, FULL_SIZE / 2);
  T fill_val = test_utils::get_test_value<T>(42);
  auto v = fill(t, fill_val);
  
  storeu(t, this->aligned_out_, m, v);
  
  for (nint_t i = 0; i < FULL_SIZE / 2; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, this->aligned_out_[i]));
  }
  for (nint_t i = FULL_SIZE / 2; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(sentinel, this->aligned_out_[i]));
  }
}

TYPED_TEST(VecLoadStoreTest, StoreWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < FULL_SIZE; ++i) {
    this->aligned_out_[i] = sentinel;
  }
  
  auto m = mwhilelt(t, 0, FULL_SIZE / 2);
  T fill_val = test_utils::get_test_value<T>(42);
  auto v = fill(t, fill_val);
  
  store(t, this->aligned_out_, m, v);
  
  for (nint_t i = 0; i < FULL_SIZE / 2; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, this->aligned_out_[i]));
  }
  for (nint_t i = FULL_SIZE / 2; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(sentinel, this->aligned_out_[i]));
  }
}

TYPED_TEST(VecLoadStoreTest, StoreuWithMaskAll) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  auto m = mtrue(t);
  T fill_val = test_utils::get_test_value<T>(42);
  auto v = fill(t, fill_val);
  
  storeu(t, this->aligned_out_, m, v);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, this->aligned_out_[i]));
  }
}

TYPED_TEST(VecLoadStoreTest, StoreuWithMaskNone) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < FULL_SIZE; ++i) {
    this->aligned_out_[i] = sentinel;
  }
  
  auto m = mfalse(t);
  T fill_val = test_utils::get_test_value<T>(42);
  auto v = fill(t, fill_val);
  
  storeu(t, this->aligned_out_, m, v);
  
  // Nothing should be stored
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(sentinel, this->aligned_out_[i]));
  }
}

// ============================================================================
// get / set element Tests
// ============================================================================

TYPED_TEST(VecLoadStoreTest, GetElement) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  auto v = loadu(t, this->aligned_data_);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], get(t, v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, SetElement) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  auto v = zeros(t);
  T new_val = test_utils::get_test_value<T>(123);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    v = set(t, v, i, new_val);
    EXPECT_TRUE(test_utils::values_equal(new_val, get(t, v, i)));
    
    // Other elements should still be zero (except previously set ones)
    for (nint_t j = 0; j <= i; ++j) {
      EXPECT_TRUE(test_utils::values_equal(new_val, get(t, v, j)));
    }
  }
}

TYPED_TEST(VecLoadStoreTest, SetElementIndividual) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  auto v = zeros(t);
  
  // Set each element to a unique value
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    T val = test_utils::get_test_value<T>(i * 10 + 5);
    v = set(t, v, i, val);
  }
  
  // Verify each element
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    T expected = test_utils::get_test_value<T>(i * 10 + 5);
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, GetMaskElement) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  auto m = mwhilelt(t, 0, FULL_SIZE / 2);
  
  for (nint_t i = 0; i < FULL_SIZE / 2; ++i) {
    EXPECT_TRUE(get(t, m, i));
  }
  for (nint_t i = FULL_SIZE / 2; i < FULL_SIZE; ++i) {
    EXPECT_FALSE(get(t, m, i));
  }
}

TYPED_TEST(VecLoadStoreTest, SetMaskElement) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  auto m = mfalse(t);
  
  // Set individual bits
  for (nint_t i = 0; i < FULL_SIZE; i += 2) {
    m = set(t, m, i, true);
  }
  
  // Verify
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_EQ(get(t, m, i), (i % 2 == 0)) << "i = " << i;
  }
}

TYPED_TEST(VecLoadStoreTest, SetMaskElementToggle) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  auto m = mtrue(t);
  
  // Toggle off
  for (nint_t i = 0; i < FULL_SIZE; i += 2) {
    m = set(t, m, i, false);
  }
  
  // Verify
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_EQ(get(t, m, i), (i % 2 != 0)) << "i = " << i;
  }
}

// ============================================================================
// Half-size vector Tests
// ============================================================================

TYPED_TEST(VecLoadStoreTest, HalfSizeFill) {
  using T = typename TestFixture::Type;
  constexpr nint_t HALF_SIZE = TestFixture::FULL_SIZE / 2;
  
  if constexpr (HALF_SIZE >= 1) {
    FixedTag<T, HALF_SIZE> t;
    
    T fill_val = test_utils::get_test_value<T>(42);
    auto v = fill(t, fill_val);
    
    for (nint_t i = 0; i < HALF_SIZE; ++i) {
      EXPECT_TRUE(test_utils::values_equal(fill_val, get(t, v, i)));
    }
  }
}

TYPED_TEST(VecLoadStoreTest, HalfSizeLoadStore) {
  using T = typename TestFixture::Type;
  constexpr nint_t HALF_SIZE = TestFixture::FULL_SIZE / 2;
  
  if constexpr (HALF_SIZE >= 1) {
    FixedTag<T, HALF_SIZE> t;
    
    auto v = loadu(t, this->aligned_data_);
    
    for (nint_t i = 0; i < HALF_SIZE; ++i) {
      EXPECT_TRUE(test_utils::values_equal(
          this->aligned_data_[i], get(t, v, i)));
    }
    
    storeu(t, this->aligned_out_, v);
    
    for (nint_t i = 0; i < HALF_SIZE; ++i) {
      EXPECT_TRUE(test_utils::values_equal(
          this->aligned_data_[i], this->aligned_out_[i]));
    }
  }
}

TYPED_TEST(VecLoadStoreTest, HalfSizeMwhilelt) {
  using T = typename TestFixture::Type;
  constexpr nint_t HALF_SIZE = TestFixture::FULL_SIZE / 2;
  
  if constexpr (HALF_SIZE >= 1) {
    FixedTag<T, HALF_SIZE> t;
    
    auto m = mwhilelt(t, 0, HALF_SIZE / 2);
    
    for (nint_t i = 0; i < HALF_SIZE / 2; ++i) {
      EXPECT_TRUE(get(t, m, i));
    }
    for (nint_t i = HALF_SIZE / 2; i < HALF_SIZE; ++i) {
      EXPECT_FALSE(get(t, m, i));
    }
  }
}

TYPED_TEST(VecLoadStoreTest, HalfSizeLoadWithN) {
  using T = typename TestFixture::Type;
  constexpr nint_t HALF_SIZE = TestFixture::FULL_SIZE / 2;
  
  if constexpr (HALF_SIZE >= 2) {
    FixedTag<T, HALF_SIZE> t;
    
    T default_val = test_utils::get_test_value<T>(999);
    auto default_v = fill(t, default_val);
    
    nint_t n = HALF_SIZE / 2;
    auto v = loadu(t, this->aligned_data_, n, default_v);
    
    for (nint_t i = 0; i < n; ++i) {
      EXPECT_TRUE(test_utils::values_equal(
          this->aligned_data_[i], get(t, v, i)));
    }
    for (nint_t i = n; i < HALF_SIZE; ++i) {
      EXPECT_TRUE(test_utils::values_equal(default_val, get(t, v, i)));
    }
  }
}

// ============================================================================
// Edge Cases
// ============================================================================

TYPED_TEST(VecLoadStoreTest, VariousMaskPatterns) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  // Test every possible pattern count
  for (nint_t count = 0; count <= FULL_SIZE; ++count) {
    auto m = mwhilelt(t, 0, count);
    
    for (nint_t i = 0; i < FULL_SIZE; ++i) {
      EXPECT_EQ(get(t, m, i), i < count) 
          << "count=" << count << ", i=" << i;
    }
  }
}

TYPED_TEST(VecLoadStoreTest, LoadStoreSequential) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  // Load and store sequentially
  for (int offset = 0; offset < 64; offset += FULL_SIZE) {
    auto v = loadu(t, this->aligned_data_ + offset);
    storeu(t, this->aligned_out_ + offset, v);
  }
  
  for (int i = 0; i < 64; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], this->aligned_out_[i]));
  }
}

TYPED_TEST(VecLoadStoreTest, ExtremeValues) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  // Test with extreme values if applicable
  if constexpr (std::is_integral_v<T>) {
    T min_val = std::numeric_limits<T>::min();
    T max_val = std::numeric_limits<T>::max();
    
    auto v_min = fill(t, min_val);
    auto v_max = fill(t, max_val);
    
    for (nint_t i = 0; i < FULL_SIZE; ++i) {
      EXPECT_TRUE(test_utils::values_equal(min_val, get(t, v_min, i)));
      EXPECT_TRUE(test_utils::values_equal(max_val, get(t, v_max, i)));
    }
  }
}

TYPED_TEST(VecLoadStoreTest, InitializerListLoad) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  
  // Create initializer list with FULL_SIZE elements
  // Using a workaround since we can't easily create initializer lists dynamically
  alignas(16) T data[FULL_SIZE];
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    data[i] = test_utils::get_test_value<T>(i + 100);
  }
  
  auto v = loadu(t, data);
  
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(data[i], get(t, v, i)));
  }
}

// ============================================================================
// Multi-word Vector Tests (N > FULL_SIZE)
// ============================================================================

TYPED_TEST(VecLoadStoreTest, MultiWordSize) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  constexpr nint_t MULTI_SIZE = FULL_SIZE * 2;
  
  FixedTag<T, MULTI_SIZE> t;
  
  EXPECT_EQ(size(t), MULTI_SIZE);
  #ifndef CPU_CAPABILITY_GENERIC
  EXPECT_EQ(word_size(t), FULL_SIZE);
  EXPECT_EQ(num_words(t), 2);
  EXPECT_FALSE(is_word_vec(t));
  #endif // CPU_CAPABILITY_GENERIC
}

TYPED_TEST(VecLoadStoreTest, MultiWordFill) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  constexpr nint_t MULTI_SIZE = FULL_SIZE * 2;
  
  FixedTag<T, MULTI_SIZE> t;
  
  T fill_val = test_utils::get_test_value<T>(77);
  auto v = fill(t, fill_val);
  
  for (nint_t i = 0; i < MULTI_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, get(t, v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, MultiWordLoadStore) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  constexpr nint_t MULTI_SIZE = FULL_SIZE * 2;
  
  FixedTag<T, MULTI_SIZE> t;
  
  auto v = loadu(t, this->aligned_data_);
  storeu(t, this->aligned_out_, v);
  
  for (nint_t i = 0; i < MULTI_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], this->aligned_out_[i]));
  }
}

TYPED_TEST(VecLoadStoreTest, MultiWordMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  constexpr nint_t MULTI_SIZE = FULL_SIZE * 2;
  
  FixedTag<T, MULTI_SIZE> t;
  
  auto m = mwhilelt(t, 0, FULL_SIZE + FULL_SIZE / 2);
  
  for (nint_t i = 0; i < FULL_SIZE + FULL_SIZE / 2; ++i) {
    EXPECT_TRUE(get(t, m, i)) << "i = " << i;
  }
  for (nint_t i = FULL_SIZE + FULL_SIZE / 2; i < MULTI_SIZE; ++i) {
    EXPECT_FALSE(get(t, m, i)) << "i = " << i;
  }
}

TYPED_TEST(VecLoadStoreTest, MultiWordLoadWithN) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  constexpr nint_t MULTI_SIZE = FULL_SIZE * 2;
  
  FixedTag<T, MULTI_SIZE> t;
  
  T default_val = test_utils::get_test_value<T>(999);
  auto default_v = fill(t, default_val);
  
  nint_t n = FULL_SIZE + FULL_SIZE / 2;
  auto v = loadu(t, this->aligned_data_, n, default_v);
  
  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], get(t, v, i)));
  }
  for (nint_t i = n; i < MULTI_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(t, v, i)));
  }
}

TYPED_TEST(VecLoadStoreTest, MultiWordStoreWithN) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  constexpr nint_t MULTI_SIZE = FULL_SIZE * 2;
  
  FixedTag<T, MULTI_SIZE> t;
  
  // Reset output
  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < MULTI_SIZE; ++i) {
    this->aligned_out_[i] = sentinel;
  }
  
  T fill_val = test_utils::get_test_value<T>(42);
  auto v = fill(t, fill_val);
  
  nint_t n = FULL_SIZE + FULL_SIZE / 2;
  storeu(t, this->aligned_out_, n, v);
  
  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, this->aligned_out_[i]));
  }
  for (nint_t i = n; i < MULTI_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(sentinel, this->aligned_out_[i]));
  }
}

TYPED_TEST(VecLoadStoreTest, MultiWordSetElement) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  constexpr nint_t MULTI_SIZE = FULL_SIZE * 4;
  
  FixedTag<T, MULTI_SIZE> t;
  
  auto v = zeros(t);
  
  // Set elements across word boundaries
  T val1 = test_utils::get_test_value<T>(111);
  T val2 = test_utils::get_test_value<T>(222);
  
  v = set(t, v, FULL_SIZE - 1, val1);
  v = set(t, v, FULL_SIZE, val2);
  
  EXPECT_TRUE(test_utils::values_equal(val1, get(t, v, FULL_SIZE - 1)));
  EXPECT_TRUE(test_utils::values_equal(val2, get(t, v, FULL_SIZE)));
}

// ============================================================================
// Gather / Scatter Tests
// ============================================================================


template <typename T>
class VecGatherScatterTest : public ::testing::Test {
protected:
  using Type = T;
  static constexpr nint_t FULL_SIZE = test_utils::full_vec_size<T>();

  void SetUp() override {
    // Allocate aligned memory for testing
    aligned_data_ = test_utils::alloc_aligned<T>(256);
    aligned_out_ = test_utils::alloc_aligned<T>(256);

    // Initialize with test values
    for (size_t i = 0; i < 256; ++i) {
      aligned_data_[i] = test_utils::get_test_value<T>(i);
      aligned_out_[i] = T{};
    }
  }

  void TearDown() override {
    std::free(aligned_data_);
    std::free(aligned_out_);
  }

  T* aligned_data_{};
  T* aligned_out_{};
};

// List of all tested types
using GatherScatterTypes = ::testing::Types<
    float32_t, float64_t, int32_t, uint32_t, int64_t, uint64_t
>;

TYPED_TEST_SUITE(VecGatherScatterTest, GatherScatterTypes);

TYPED_TEST(VecGatherScatterTest, GatherBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  using IndexT = Index<T>;
  constexpr FixedTag<IndexT, FULL_SIZE> it;
  
  // Create index vector: indices = [0, 2, 4, 6, ...]
  alignas(16) IndexT indices[FULL_SIZE];
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    indices[i] = static_cast<IndexT>(i * 2);
  }
  
  auto idx = loadu(it, indices);
  auto v = gather(t, this->aligned_data_, idx);
  
  // Verify: v[i] should equal aligned_data_[indices[i]] = aligned_data_[i * 2]
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    T expected = this->aligned_data_[i * 2];
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, v, i)));
  }
}

TYPED_TEST(VecGatherScatterTest, GatherWithN) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  FixedTag<T, FULL_SIZE> t;
  using IndexT = Index<T>;
  constexpr FixedTag<IndexT, FULL_SIZE> it;
  
  // Create index vector
  alignas(16) IndexT indices[FULL_SIZE];
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    indices[i] = static_cast<IndexT>((i * 3) % 128);  // Keep indices in bounds
  }
  
  auto idx = loadu(it, indices);
  T default_val = test_utils::get_test_value<T>(999);
  
  nint_t n = FULL_SIZE / 2;
  auto v = gather(t, this->aligned_data_, idx, n, default_val);
  
  // First n elements should be gathered
  for (nint_t i = 0; i < n; ++i) {
    T expected = this->aligned_data_[indices[i]];
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, v, i)));
  }
  // Rest should be default
  for (nint_t i = n; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(t, v, i)));
  }
}

TYPED_TEST(VecGatherScatterTest, GatherWithNZero) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;

  FixedTag<T, FULL_SIZE> t;
  using IndexT = Index<T>;
  constexpr FixedTag<IndexT, FULL_SIZE> it;

  alignas(16) IndexT indices[FULL_SIZE];
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    indices[i] = static_cast<IndexT>(i);
  }

  auto idx = loadu(it, indices);
  T default_val = test_utils::get_test_value<T>(999);

  auto v = gather(t, this->aligned_data_, idx, 0, default_val);

  // All elements should be default
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(t, v, i)));
  }
}

TYPED_TEST(VecGatherScatterTest, GatherWithNFull) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;

  FixedTag<T, FULL_SIZE> t;
  using IndexT = Index<T>;
  constexpr FixedTag<IndexT, FULL_SIZE> it;

  alignas(16) IndexT indices[FULL_SIZE];
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    indices[i] = static_cast<IndexT>(i);
  }

  auto idx = loadu(it, indices);
  T default_val = test_utils::get_test_value<T>(999);

  auto v = gather(t, this->aligned_data_, idx, FULL_SIZE, default_val);

  // All elements should be gathered
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    T expected = this->aligned_data_[i];
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, v, i)));
  }
}

TYPED_TEST(VecGatherScatterTest, GatherWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;

  FixedTag<T, FULL_SIZE> t;
  using IndexT = Index<T>;
  constexpr FixedTag<IndexT, FULL_SIZE> it;

  alignas(16) IndexT indices[FULL_SIZE];
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    indices[i] = static_cast<IndexT>((i * 5 + 10) % 128);
  }

  auto idx = loadu(it, indices);
  auto m = mwhilelt(t, 0, FULL_SIZE / 2);
  T default_val = test_utils::get_test_value<T>(777);

  auto v = gather(t, this->aligned_data_, idx, m, default_val);

  // First half should be gathered
  for (nint_t i = 0; i < FULL_SIZE / 2; ++i) {
    T expected = this->aligned_data_[indices[i]];
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, v, i)));
  }
  // Second half should be default
  for (nint_t i = FULL_SIZE / 2; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(t, v, i)));
  }
}

TYPED_TEST(VecGatherScatterTest, GatherWithMaskAll) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;

  FixedTag<T, FULL_SIZE> t;
  using IndexT = Index<T>;
  constexpr FixedTag<IndexT, FULL_SIZE> it;

  alignas(16) IndexT indices[FULL_SIZE];
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    indices[i] = static_cast<IndexT>(i);
  }

  auto idx = loadu(it, indices);
  auto m = mtrue(t);
  T default_val = test_utils::get_test_value<T>(777);

  auto v = gather(t, this->aligned_data_, idx, m, default_val);

  // All elements should be gathered
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    T expected = this->aligned_data_[i];
    EXPECT_TRUE(test_utils::values_equal(expected, get(t, v, i)));
  }
}

TYPED_TEST(VecGatherScatterTest, GatherWithMaskNone) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;

  FixedTag<T, FULL_SIZE> t;
  using IndexT = Index<T>;
  constexpr FixedTag<IndexT, FULL_SIZE> it;

  alignas(16) IndexT indices[FULL_SIZE];
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    indices[i] = static_cast<IndexT>(i);
  }

  auto idx = loadu(it, indices);
  auto m = mfalse(t);
  T default_val = test_utils::get_test_value<T>(777);

  auto v = gather(t, this->aligned_data_, idx, m, default_val);

  // All elements should be default
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(default_val, get(t, v, i)));
  }
}

TYPED_TEST(VecGatherScatterTest, ScatterBasic) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;

  FixedTag<T, FULL_SIZE> t;
  using IndexT = Index<T>;
  constexpr FixedTag<IndexT, FULL_SIZE> it;

  // Initialize output with sentinel values
  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < 256; ++i) {
    this->aligned_out_[i] = sentinel;
  }

  // Create index vector: scatter to positions [0, 2, 4, 6, ...]
  alignas(16) IndexT indices[FULL_SIZE];
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    indices[i] = static_cast<IndexT>(i * 2);
  }

  auto idx = loadu(it, indices);
  auto v = loadu(t, this->aligned_data_);

  scatter(t, this->aligned_out_, idx, v);

  // Verify: aligned_out_[indices[i]] should equal aligned_data_[i]
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(this->aligned_data_[i], this->aligned_out_[indices[i]]));
  }
}

TYPED_TEST(VecGatherScatterTest, ScatterWithN) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;

  FixedTag<T, FULL_SIZE> t;
  using IndexT = Index<T>;
  constexpr FixedTag<IndexT, FULL_SIZE> it;

  // Initialize output with sentinel values
  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < 256; ++i) {
    this->aligned_out_[i] = sentinel;
  }

  alignas(16) IndexT indices[FULL_SIZE];
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    indices[i] = static_cast<IndexT>((i * 3) % 128);
  }

  auto idx = loadu(it, indices);
  auto v = loadu(t, this->aligned_data_);

  nint_t n = FULL_SIZE / 2;
  scatter(t, this->aligned_out_, idx, n, v);

  // Only first n elements should be scattered
  for (nint_t i = 0; i < n; ++i) {
    EXPECT_TRUE(test_utils::values_equal(this->aligned_data_[i], this->aligned_out_[indices[i]]));
  }
}

TYPED_TEST(VecGatherScatterTest, ScatterWithNZero) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;

  FixedTag<T, FULL_SIZE> t;
  using IndexT = Index<T>;
  constexpr FixedTag<IndexT, FULL_SIZE> it;

  // Initialize output with sentinel values
  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < 256; ++i) {
    this->aligned_out_[i] = sentinel;
  }

  alignas(16) IndexT indices[FULL_SIZE];
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    indices[i] = static_cast<IndexT>(i);
  }

  auto idx = loadu(it, indices);
  auto v = loadu(t, this->aligned_data_);

  scatter(t, this->aligned_out_, idx, 0, v);

  // Nothing should be scattered
  for (int i = 0; i < 256; ++i) {
    EXPECT_TRUE(test_utils::values_equal(sentinel, this->aligned_out_[i]));
  }
}

TYPED_TEST(VecGatherScatterTest, ScatterWithNFull) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;

  FixedTag<T, FULL_SIZE> t;
  using IndexT = Index<T>;
  constexpr FixedTag<IndexT, FULL_SIZE> it;

  // Initialize output with sentinel values
  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < 256; ++i) {
    this->aligned_out_[i] = sentinel;
  }

  alignas(16) IndexT indices[FULL_SIZE];
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    indices[i] = static_cast<IndexT>(i + 50);  // Offset to avoid overlap
  }

  auto idx = loadu(it, indices);
  auto v = loadu(t, this->aligned_data_);

  scatter(t, this->aligned_out_, idx, FULL_SIZE, v);

  // All elements should be scattered
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(this->aligned_data_[i], this->aligned_out_[indices[i]]));
  }
}

TYPED_TEST(VecGatherScatterTest, ScatterWithMask) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;

  FixedTag<T, FULL_SIZE> t;
  using IndexT = Index<T>;
  constexpr FixedTag<IndexT, FULL_SIZE> it;

  // Initialize output with sentinel values
  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < 256; ++i) {
    this->aligned_out_[i] = sentinel;
  }

  alignas(16) IndexT indices[FULL_SIZE];
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    indices[i] = static_cast<IndexT>((i * 5 + 10) % 128);
  }

  auto idx = loadu(it, indices);
  auto v = loadu(t, this->aligned_data_);
  auto m = mwhilelt(t, 0, FULL_SIZE / 2);

  scatter(t, this->aligned_out_, idx, m, v);

  // Only first half should be scattered
  for (nint_t i = 0; i < FULL_SIZE / 2; ++i) {
    EXPECT_TRUE(test_utils::values_equal(this->aligned_data_[i], this->aligned_out_[indices[i]]));
  }
}

TYPED_TEST(VecGatherScatterTest, ScatterWithMaskAll) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;

  FixedTag<T, FULL_SIZE> t;
  using IndexT = Index<T>;
  constexpr FixedTag<IndexT, FULL_SIZE> it;

  // Initialize output with sentinel values
  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < 256; ++i) {
    this->aligned_out_[i] = sentinel;
  }

  alignas(16) IndexT indices[FULL_SIZE];
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    indices[i] = static_cast<IndexT>(i + 100);
  }

  auto idx = loadu(it, indices);
  auto v = loadu(t, this->aligned_data_);
  auto m = mtrue(t);

  scatter(t, this->aligned_out_, idx, m, v);

  // All elements should be scattered
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    EXPECT_TRUE(test_utils::values_equal(this->aligned_data_[i], this->aligned_out_[indices[i]]));
  }
}

TYPED_TEST(VecGatherScatterTest, ScatterWithMaskNone) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;

  FixedTag<T, FULL_SIZE> t;
  using IndexT = Index<T>;
  constexpr FixedTag<IndexT, FULL_SIZE> it;

  // Initialize output with sentinel values
  T sentinel = test_utils::get_test_value<T>(-1);
  for (int i = 0; i < 256; ++i) {
    this->aligned_out_[i] = sentinel;
  }

  alignas(16) IndexT indices[FULL_SIZE];
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    indices[i] = static_cast<IndexT>(i);
  }

  auto idx = loadu(it, indices);
  auto v = loadu(t, this->aligned_data_);
  auto m = mfalse(t);

  scatter(t, this->aligned_out_, idx, m, v);

  // Nothing should be scattered
  for (int i = 0; i < 256; ++i) {
    EXPECT_TRUE(test_utils::values_equal(sentinel, this->aligned_out_[i]));
  }
}

TYPED_TEST(VecGatherScatterTest, GatherScatterRoundTrip) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;

  FixedTag<T, FULL_SIZE> t;
  using IndexT = Index<T>;
  constexpr FixedTag<IndexT, FULL_SIZE> it;

  // Use identity indices for round-trip
  alignas(16) IndexT indices[FULL_SIZE];
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    indices[i] = static_cast<IndexT>(i + 128);  // Offset to different region
  }

  auto idx = loadu(it, indices);

  // Initialize source regionw
  for (int i = 128; i < 128 + FULL_SIZE; ++i) {
    this->aligned_data_[i] = test_utils::get_test_value<T>(i);
  }

  // Clear output
  for (int i = 0; i < 256; ++i) {
    this->aligned_out_[i] = T{};
  }

  // Gather from source
  auto v = gather(t, this->aligned_data_, idx);

  // Create new indices for scatter (back to beginning)
  alignas(16) IndexT scatter_indices[FULL_SIZE];
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    scatter_indices[i] = static_cast<IndexT>(i);
  }
  auto scatter_idx = loadu(it, scatter_indices);

  // Scatter to output
  scatter(t, this->aligned_out_, scatter_idx, v);

  // Verify round-trip
  for (nint_t i = 0; i < FULL_SIZE; ++i) {
    T expected = test_utils::get_test_value<T>(static_cast<int>(indices[i]));
    EXPECT_TRUE(test_utils::values_equal(expected, this->aligned_out_[i]));
  }
}

// ============================================================================
// ScalableTag Tests
// ============================================================================

TYPED_TEST(VecLoadStoreTest, ScalableTagTest) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  ScalableTag<T> t;
  
  EXPECT_EQ(size(t), FULL_SIZE);
  #ifndef CPU_CAPABILITY_GENERIC
  EXPECT_FALSE(is_default_impl(t));
  EXPECT_FALSE(is_scalable(t));  // TODO: currently only support fixed-size
  #endif
}

TYPED_TEST(VecLoadStoreTest, ScalableTagFill) {
  using T = typename TestFixture::Type;
  
  ScalableTag<T> t;
  
  T fill_val = test_utils::get_test_value<T>(123);
  auto v = fill(t, fill_val);
  
  for (nint_t i = 0; i < size(t); ++i) {
    EXPECT_TRUE(test_utils::values_equal(fill_val, get(t, v, i)));
  }
}

// ============================================================================
// Utility function test (similar to copy function in original test)
// ============================================================================

template <typename T>
CT_NOINLINE
static void vectorized_copy(const T* from, T* to, nint_t len) {
  ScalableTag<T> t;
  nint_t vec_size = size(t);
  nint_t i;
  
  for (i = 0; i <= len - vec_size; i += vec_size) {
    auto v = loadu(t, from + i);
    storeu(t, to + i, v);
  }
  
  // Handle remaining elements
  if (i < len) {
    auto m = mwhilelt(t, i, len);
    auto v = loadu(t, from + i, m, zeros(t));
    storeu(t, to + i, m, v);
  }
}

TYPED_TEST(VecLoadStoreTest, VectorizedCopyFunction) {
  using T = typename TestFixture::Type;
  
  vectorized_copy(this->aligned_data_, this->aligned_out_, 100);
  
  for (int i = 0; i < 100; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], this->aligned_out_[i]));
  }
}

TYPED_TEST(VecLoadStoreTest, VectorizedCopyOddLength) {
  using T = typename TestFixture::Type;
  constexpr nint_t FULL_SIZE = TestFixture::FULL_SIZE;
  
  // Test with odd length that doesn't align with vector size
  nint_t test_len = FULL_SIZE * 3 + FULL_SIZE / 2;
  
  vectorized_copy(this->aligned_data_, this->aligned_out_, test_len);
  
  for (nint_t i = 0; i < test_len; ++i) {
    EXPECT_TRUE(test_utils::values_equal(
        this->aligned_data_[i], this->aligned_out_[i]));
  }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
