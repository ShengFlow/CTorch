//
// VecScalarTest.cpp
// Test file for Vec.h using Scalar implementation
// Forces scalar implementation by undefining architecture macros
//

#include <gtest/gtest.h>
#include <cstring>
#include <memory>

// Force Scalar implementation
#include "Features.h"

#undef ARCH_X86_FAMILY
#undef ARCH_ARM_FAMILY
#undef SIMD_WIDTH
#define SIMD_WIDTH 0

#include "tl/cpu/Vec.h"

using namespace ct;
using namespace ct::tl::vec;

// ============================================================================
// Test Fixtures
// ============================================================================

class VecScalarTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Allocate aligned memory for testing
    aligned_data_ = static_cast<float*>(std::aligned_alloc(DEFAULT_ALIGNMENT, 256 * sizeof(float)));
    aligned_out_ = static_cast<float*>(std::aligned_alloc(DEFAULT_ALIGNMENT, 256 * sizeof(float)));

    for (int i = 0; i < 256; ++i) {
      aligned_data_[i] = static_cast<float>(i);
      aligned_out_[i] = -1.0f;
    }
  }

  void TearDown() override {
    std::free(aligned_data_);
    std::free(aligned_out_);
  }

  float* aligned_data_{};
  float* aligned_out_{};
};

// ============================================================================
// Constructor Tests - fill/zeros
// ============================================================================

TEST_F(VecScalarTest, FillFloat32) {
  FixedTag<float32_t, 16> t;

  auto v = fill(t, 3.14f);

  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(v[i], 3.14f);
  }
}

TEST_F(VecScalarTest, ZerosFloat32) {
  FixedTag<float32_t, 16> t;

  auto v = zeros(t);

  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(v[i], 0.0f);
  }
}

TEST_F(VecScalarTest, FillFloat64) {
  FixedTag<float64_t, 8> t;

  auto v = fill(t, 2.718);

  for (int i = 0; i < 8; ++i) {
    EXPECT_DOUBLE_EQ(v[i], 2.718);
  }
}

TEST_F(VecScalarTest, FillWithPOW2) {
  Tag<float32_t, 4, 1> t; // size = 8

  auto v = fill(t, 1.5f);

  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 4; ++i) {
      EXPECT_FLOAT_EQ(v[j][i], 1.5f);
    }
  }
}

// ============================================================================
// Mask Constructor Tests
// ============================================================================

TEST_F(VecScalarTest, MfillTrue) {
  FixedTag<float32_t, 16> t;

  auto m = mfill(t, true);

  for (int i = 0; i < 16; ++i) {
    EXPECT_TRUE(m[i]);
  }
}

TEST_F(VecScalarTest, MfillFalse) {
  FixedTag<float32_t, 16> t;

  auto m = mfill(t, false);

  for (int i = 0; i < 16; ++i) {
    EXPECT_FALSE(m[i]);
  }
}

TEST_F(VecScalarTest, MtrueMfalse) {
  FixedTag<float32_t, 16> t;

  auto m_true = mtrue(t);
  auto m_false = mfalse(t);

  for (int i = 0; i < 16; ++i) {
    EXPECT_TRUE(m_true[i]);
    EXPECT_FALSE(m_false[i]);
  }
}

// ============================================================================
// mwhilelt/mwhilele/mwhilegt/mwhilege Tests
// ============================================================================

TEST_F(VecScalarTest, MwhileltBasic) {
  FixedTag<float32_t, 16> t;

  auto m = mwhilelt(t, 0, 5);

  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(m[i]) << "i=" << i;
  }
  for (int i = 5; i < 16; ++i) {
    EXPECT_FALSE(m[i]) << "i=" << i;
  }
}

TEST_F(VecScalarTest, MwhileltAll) {
  FixedTag<float32_t, 16> t;

  auto m = mwhilelt(t, 0, 16);

  for (int i = 0; i < 16; ++i) {
    EXPECT_TRUE(m[i]) << "i=" << i;
  }
}

TEST_F(VecScalarTest, MwhileltNone) {
  FixedTag<float32_t, 16> t;

  auto m = mwhilelt(t, 5, 5);

  for (int i = 0; i < 16; ++i) {
    EXPECT_FALSE(m[i]) << "i=" << i;
  }
}

TEST_F(VecScalarTest, MwhileltNegative) {
  FixedTag<float32_t, 16> t;

  // b < a should give all false
  auto m = mwhilelt(t, 10, 5);

  for (int i = 0; i < 16; ++i) {
    EXPECT_FALSE(m[i]) << "i=" << i;
  }
}

TEST_F(VecScalarTest, MwhileltWithOffset) {
  FixedTag<float32_t, 16> t;

  // a=3, b=7 => bits 0, 1, 2, 3 should be true
  auto m = mwhilelt(t, 3, 7);

  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(m[i]) << "i=" << i;
  }
  for (int i = 4; i < 16; ++i) {
    EXPECT_FALSE(m[i]) << "i=" << i;
  }
}

TEST_F(VecScalarTest, MwhilegeBasic) {
  FixedTag<float32_t, 16> t;

  // a=0, b=5 => bits 5..15 should be true
  auto m = mwhilege(t, 0, 5);

  for (int i = 0; i < 5; ++i) {
    EXPECT_FALSE(m[i]) << "i=" << i;
  }
  for (int i = 5; i < 16; ++i) {
    EXPECT_TRUE(m[i]) << "i=" << i;
  }
}

TEST_F(VecScalarTest, MwhilegeAll) {
  FixedTag<float32_t, 16> t;

  // b=0 => all bits should be true
  auto m = mwhilege(t, 0, 0);

  for (int i = 0; i < 16; ++i) {
    EXPECT_TRUE(m[i]) << "i=" << i;
  }
}

TEST_F(VecScalarTest, Mwhilele) {
  FixedTag<float32_t, 16> t;

  // mwhilele(t, a, b) = mwhilelt(t, a, b+1)
  auto m = mwhilele(t, 0, 4);

  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(m[i]) << "i=" << i;
  }
  for (int i = 5; i < 16; ++i) {
    EXPECT_FALSE(m[i]) << "i=" << i;
  }
}

TEST_F(VecScalarTest, Mwhilegt) {
  FixedTag<float32_t, 16> t;

  // mwhilegt(t, a, b) = mwhilege(t, a, b+1)
  auto m = mwhilegt(t, 0, 4);

  for (int i = 0; i < 5; ++i) {
    EXPECT_FALSE(m[i]) << "i=" << i;
  }
  for (int i = 5; i < 16; ++i) {
    EXPECT_TRUE(m[i]) << "i=" << i;
  }
}

// ============================================================================
// loadu/storeu Tests
// ============================================================================

TEST_F(VecScalarTest, LoaduBasic) {
  FixedTag<float32_t, 16> t;

  auto v = loadu(t, aligned_data_);

  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(v[i], static_cast<float>(i));
  }
}

TEST_F(VecScalarTest, StoreuBasic) {
  FixedTag<float32_t, 16> t;

  auto v = loadu(t, aligned_data_);
  storeu(t, aligned_out_, v);

  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], static_cast<float>(i));
  }
}

TEST_F(VecScalarTest, LoaduInitializerList) {
  FixedTag<float32_t, 4> t;

  auto v = loadu(t, {1.0f, 2.0f, 3.0f, 4.0f});

  EXPECT_FLOAT_EQ(v[0], 1.0f);
  EXPECT_FLOAT_EQ(v[1], 2.0f);
  EXPECT_FLOAT_EQ(v[2], 3.0f);
  EXPECT_FLOAT_EQ(v[3], 4.0f);
}

// ============================================================================
// load/store Tests (aligned)
// ============================================================================

TEST_F(VecScalarTest, LoadAlignedBasic) {
  FixedTag<float32_t, 16> t;

  auto v = load(t, aligned_data_);

  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(v[i], static_cast<float>(i));
  }
}

TEST_F(VecScalarTest, StoreAlignedBasic) {
  FixedTag<float32_t, 16> t;

  auto v = load(t, aligned_data_);
  store(t, aligned_out_, v);

  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], static_cast<float>(i));
  }
}

TEST_F(VecScalarTest, LoadAlignedInitializerList) {
  alignas(16) float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  FixedTag<float32_t, 4> t;

  auto v = load(t, data);

  EXPECT_FLOAT_EQ(v[0], 1.0f);
  EXPECT_FLOAT_EQ(v[1], 2.0f);
  EXPECT_FLOAT_EQ(v[2], 3.0f);
  EXPECT_FLOAT_EQ(v[3], 4.0f);
}

// ============================================================================
// load/store with n parameter Tests
// ============================================================================

TEST_F(VecScalarTest, LoaduWithN) {
  FixedTag<float32_t, 16> t;

  auto default_v = fill(t, -99.0f);
  auto v = loadu(t, aligned_data_, 5, default_v);

  // First 5 elements from memory
  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(v[i], static_cast<float>(i));
  }
  // Remaining from default
  for (int i = 5; i < 16; ++i) {
    EXPECT_FLOAT_EQ(v[i], -99.0f);
  }
}

TEST_F(VecScalarTest, LoaduWithNZero) {
  FixedTag<float32_t, 16> t;

  auto default_v = fill(t, -99.0f);
  auto v = loadu(t, aligned_data_, 0, default_v);

  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(v[i], -99.0f);
  }
}

TEST_F(VecScalarTest, LoaduWithNFull) {
  FixedTag<float32_t, 16> t;

  auto default_v = fill(t, -99.0f);
  auto v = loadu(t, aligned_data_, 16, default_v);

  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(v[i], static_cast<float>(i));
  }
}

TEST_F(VecScalarTest, StoreuWithN) {
  FixedTag<float32_t, 16> t;

  auto v = fill(t, 42.0f);
  storeu(t, aligned_out_, 5, v);

  // First 5 elements stored
  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], 42.0f);
  }
  // Remaining unchanged
  for (int i = 5; i < 16; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], -1.0f);
  }
}

TEST_F(VecScalarTest, StoreuWithNZero) {
  FixedTag<float32_t, 16> t;

  // Reset output
  for (int i = 0; i < 16; ++i) aligned_out_[i] = -1.0f;

  auto v = fill(t, 42.0f);
  storeu(t, aligned_out_, 0, v);

  // Nothing should be stored
  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], -1.0f);
  }
}

// ============================================================================
// load/store with mask Tests
// ============================================================================

TEST_F(VecScalarTest, LoaduWithMask) {
  FixedTag<float32_t, 16> t;

  auto m = mwhilelt(t, 0, 5); // First 5 bits true
  auto default_v = fill(t, -99.0f);

  auto v = loadu(t, aligned_data_, m, default_v);

  // First 5 from memory
  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(v[i], static_cast<float>(i));
  }
  // Rest from default
  for (int i = 5; i < 16; ++i) {
    EXPECT_FLOAT_EQ(v[i], -99.0f);
  }
}

TEST_F(VecScalarTest, StoreuWithMask) {
  FixedTag<float32_t, 16> t;

  auto m = mwhilelt(t, 0, 5);
  auto v = fill(t, 42.0f);

  storeu(t, aligned_out_, m, v);

  // First 5 stored
  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], 42.0f);
  }
  // Rest unchanged
  for (int i = 5; i < 16; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], -1.0f);
  }
}

TEST_F(VecScalarTest, StoreuWithMaskAll) {
  FixedTag<float32_t, 16> t;

  auto m = mtrue(t);
  auto v = fill(t, 42.0f);

  storeu(t, aligned_out_, m, v);

  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], 42.0f);
  }
}

TEST_F(VecScalarTest, StoreuWithMaskNone) {
  FixedTag<float32_t, 16> t;

  // Reset output
  for (int i = 0; i < 16; ++i) aligned_out_[i] = -1.0f;

  auto m = mfalse(t);
  auto v = fill(t, 42.0f);

  storeu(t, aligned_out_, m, v);

  // Nothing should be stored
  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], -1.0f);
  }
}

// ============================================================================
// gather/scatter Tests
// ============================================================================

TEST_F(VecScalarTest, GatherBasic) {
  FixedTag<float32_t, 16> t;
  using IndexVec = Vec<Tag<Index<float32_t>, 16>>;

  // Create indices: gather every other element
  IndexVec indices{};
  for (int i = 0; i < 16; ++i) {
    indices[i] = i * 2;
  }

  auto v = gather(t, aligned_data_, indices);

  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(v[i], static_cast<float>(i * 2));
  }
}

TEST_F(VecScalarTest, ScatterBasic) {
  FixedTag<float32_t, 16> t;
  using IndexVec = Vec<Tag<Index<float32_t>, 16>>;

  // Clear output
  for (int i = 0; i < 32; ++i) aligned_out_[i] = -1.0f;

  // Scatter to every other position
  IndexVec indices{};
  auto values = fill(t, 42.0f);
  for (int i = 0; i < 16; ++i) {
    indices[i] = i * 2;
  }

  scatter(t, aligned_out_, indices, values);

  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i * 2], 42.0f);
    if (i > 0) {
      EXPECT_FLOAT_EQ(aligned_out_[i * 2 - 1], -1.0f);
    }
  }
}

TEST_F(VecScalarTest, GatherWithN) {
  FixedTag<float32_t, 16> t;
  using IndexVec = Vec<Tag<Index<float32_t>, 16>>;

  IndexVec indices{};
  for (int i = 0; i < 16; ++i) {
    indices[i] = i;
  }

  auto default_v = fill(t, -99.0f);
  auto v = gather(t, aligned_data_, indices, 5, default_v);

  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(v[i], static_cast<float>(i));
  }
  for (int i = 5; i < 16; ++i) {
    EXPECT_FLOAT_EQ(v[i], -99.0f);
  }
}

TEST_F(VecScalarTest, GatherWithMask) {
  FixedTag<float32_t, 16> t;
  using IndexVec = Vec<Tag<Index<float32_t>, 16>>;

  IndexVec indices{};
  for (int i = 0; i < 16; ++i) {
    indices[i] = i;
  }

  auto m = mwhilelt(t, 0, 5);
  auto default_v = fill(t, -99.0f);

  auto v = gather(t, aligned_data_, indices, m, default_v);

  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(v[i], static_cast<float>(i));
  }
  for (int i = 5; i < 16; ++i) {
    EXPECT_FLOAT_EQ(v[i], -99.0f);
  }
}

TEST_F(VecScalarTest, ScatterWithN) {
  FixedTag<float32_t, 16> t;
  using IndexVec = Vec<Tag<Index<float32_t>, 16>>;

  // Clear output
  for (int i = 0; i < 16; ++i) aligned_out_[i] = -1.0f;

  IndexVec indices{};
  auto values = fill(t, 42.0f);
  for (int i = 0; i < 16; ++i) {
    indices[i] = i;
  }

  scatter(t, aligned_out_, indices, values, 5);

  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], 42.0f);
  }
  for (int i = 5; i < 16; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], -1.0f);
  }
}

TEST_F(VecScalarTest, ScatterWithMask) {
  FixedTag<float32_t, 16> t;
  using IndexVec = Vec<Tag<Index<float32_t>, 16>>;

  // Clear output
  for (int i = 0; i < 16; ++i) aligned_out_[i] = -1.0f;

  IndexVec indices{};
  auto values = fill(t, 42.0f);
  for (int i = 0; i < 16; ++i) {
    indices[i] = i;
  }

  auto m = mwhilelt(t, 0, 5);

  scatter(t, aligned_out_, indices, values, m);

  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], 42.0f);
  }
  for (int i = 5; i < 16; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], -1.0f);
  }
}

// ============================================================================
// get/set element Tests
// ============================================================================

TEST_F(VecScalarTest, GetElement) {
  FixedTag<float32_t, 16> t;

  auto v = fill(t, 3.14f);

  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(get(t, v, i), 3.14f);
  }
}

TEST_F(VecScalarTest, SetElement) {
  FixedTag<float32_t, 16> t;

  auto v = zeros(t);

  v = set(t, v, 5, 42.0f);

  EXPECT_FLOAT_EQ(get(t, v, 5), 42.0f);
  EXPECT_FLOAT_EQ(get(t, v, 4), 0.0f);
  EXPECT_FLOAT_EQ(get(t, v, 6), 0.0f);
}

TEST_F(VecScalarTest, GetMaskElement) {
  FixedTag<float32_t, 16> t;

  auto m = mwhilelt(t, 0, 5);

  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(get(t, m, i));
  }
  for (int i = 5; i < 16; ++i) {
    EXPECT_FALSE(get(t, m, i));
  }
}

TEST_F(VecScalarTest, SetMaskElement) {
  FixedTag<float32_t, 16> t;

  auto m = mfalse(t);

  m = set(t, m, 5, true);

  EXPECT_FALSE(get(t, m, 4));
  EXPECT_TRUE(get(t, m, 5));
  EXPECT_FALSE(get(t, m, 6));
}

// ============================================================================
// Different Type Tests
// ============================================================================

TEST_F(VecScalarTest, Float64Operations) {
  FixedTag<float64_t, 8> t;

  auto v = fill(t, 2.718);

  for (int i = 0; i < 8; ++i) {
    EXPECT_DOUBLE_EQ(v[i], 2.718);
  }
}

TEST_F(VecScalarTest, Int32Operations) {
  FixedTag<int32_t, 16> t;

  auto v = fill(t, 42);

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(v[i], 42);
  }
}

// ============================================================================
// POW2 Tests
// ============================================================================

TEST_F(VecScalarTest, POW2PositiveLoadStore) {
  Tag<float32_t, 4, 1> t; // size = 8

  auto v = loadu(t, aligned_data_);

  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 4; ++i) {
      EXPECT_FLOAT_EQ(v[j][i], static_cast<float>(j * 4 + i));
    }
  }

  storeu(t, aligned_out_, v);

  for (int i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], static_cast<float>(i));
  }
}

TEST_F(VecScalarTest, POW2NegativeLoadStore) {
  Tag<float32_t, 8, -1> t; // size = 4

  auto v = loadu(t, aligned_data_);

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(v[i], static_cast<float>(i));
  }
}

// ============================================================================
// Copy Function Test (like the original VecTest)
// ============================================================================

CT_NOINLINE
static void scalar_copy(const float* from, float* to, nint_t len) {
  FixedTag<float32_t, 16> t;
  nint_t i;
  for (i = 0; i <= len - size(t); i += size(t)) {
    auto v = loadu(t, from + i);
    storeu(t, to + i, v);
  }
  for (; i < len; ++i) {
    to[i] = from[i];
  }
}

TEST_F(VecScalarTest, CopyFunction) {
  scalar_copy(aligned_data_, aligned_out_, 128);

  for (int i = 0; i < 128; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], static_cast<float>(i));
  }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(VecScalarTest, SmallVector) {
  FixedTag<float32_t, 2> t;

  auto v = fill(t, 1.5f);

  EXPECT_FLOAT_EQ(v[0], 1.5f);
  EXPECT_FLOAT_EQ(v[1], 1.5f);
}

TEST_F(VecScalarTest, LargeVector) {
  FixedTag<float32_t, 64> t;

  auto v = fill(t, 3.14f);

  for (int i = 0; i < 64; ++i) {
    EXPECT_FLOAT_EQ(v[i], 3.14f);
  }
}

TEST_F(VecScalarTest, UnalignedPointer) {
  // Test with unaligned pointer
  float* unaligned = aligned_data_ + 1; // Not 16-byte aligned

  FixedTag<float32_t, 4> t;

  auto v = loadu(t, unaligned);

  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(v[i], static_cast<float>(i + 1));
  }
}

// ============================================================================
// Arithmetic operation test
// ============================================================================
TEST_F(VecScalarTest, Add) {
  FixedTag<float32_t, 16> t;
  auto a = loadu(t, aligned_data_);
  auto b = loadu(t, aligned_data_ + 1);
  auto m = mwhilelt(t, 0, 8);
  static_assert(std::is_same_v<decltype(a), ScalarArray<float, 16>>);
  static_assert(is_element_type<float>);
  auto c = add(a, b, m);

  for (int i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(c[i], 2 * i + 1);
  }
  for (int i = 8; i < 16; ++i) {
    EXPECT_FLOAT_EQ(c[i], a[i]);
  }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
