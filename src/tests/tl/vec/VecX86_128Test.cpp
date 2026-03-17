//
// VecX86_128Test.cpp
// Test file for Vec.h using x86_128 implementation
// Forces x86_128 implementation by defining architecture macros
//

#include <gtest/gtest.h>
#include <cstring>
#include <memory>

// Force x86_128 implementation
#include "Features.h"

// Ensure x86 architecture
#if !defined(ARCH_X86_FAMILY)
  #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define ARCH_X86_FAMILY 1
  #else
    // Force it anyway for testing purposes
    #define ARCH_X86_FAMILY 1
  #endif
#endif

#define SIMD_WIDTH 128

#include "tl/cpu/Vec.h"

using namespace ct;
using namespace ct::tl::vec;

// ============================================================================
// Test Fixtures
// ============================================================================

class VecX86_128Test : public ::testing::Test {
protected:
  void SetUp() override {
    // Allocate aligned memory for testing (16-byte aligned for SSE)
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
// Architecture Verification Tests
// ============================================================================

TEST_F(VecX86_128Test, VerifyX86Implementation) {
  // Verify we're using x86 implementation
  FixedTag<float32_t, 4> t;
  
  // x86 implementation should not be default impl for N=4
  EXPECT_FALSE(is_default_impl(t));
}

TEST_F(VecX86_128Test, VerifySize) {
  FixedTag<float32_t, 4> t;
  
  EXPECT_EQ(size(t), 4);
  EXPECT_EQ(word_size(t), 4);
}

// ============================================================================
// Constructor Tests - fill/zeros
// ============================================================================

TEST_F(VecX86_128Test, FillFloat32_4) {
  FixedTag<float32_t, 4> t;
  
  auto v = fill(t, 3.14f);
  
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(get(t, v, i), 3.14f);
  }
}

TEST_F(VecX86_128Test, ZerosFloat32_4) {
  FixedTag<float32_t, 4> t;
  
  auto v = zeros(t);
  
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(get(t, v, i), 0.0f);
  }
}

TEST_F(VecX86_128Test, FillFloat32_2) {
  FixedTag<float32_t, 2> t;
  
  auto v = fill(t, 2.71f);
  
  EXPECT_FLOAT_EQ(get(t, v, 0), 2.71f);
  EXPECT_FLOAT_EQ(get(t, v, 1), 2.71f);
}

// ============================================================================
// Mask Constructor Tests
// ============================================================================

TEST_F(VecX86_128Test, MfillTrue) {
  FixedTag<float32_t, 4> t;
  
  auto m = mfill(t, true);
  
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(get(t, m, i));
  }
}

TEST_F(VecX86_128Test, MfillFalse) {
  FixedTag<float32_t, 4> t;
  
  auto m = mfill(t, false);
  
  for (int i = 0; i < 4; ++i) {
    EXPECT_FALSE(get(t, m, i));
  }
}

TEST_F(VecX86_128Test, MtrueMfalse) {
  FixedTag<float32_t, 4> t;
  
  auto m_true = mtrue(t);
  auto m_false = mfalse(t);
  
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(get(t, m_true, i));
    EXPECT_FALSE(get(t, m_false, i));
  }
}

// ============================================================================
// mwhilelt/mwhilege Tests
// ============================================================================

TEST_F(VecX86_128Test, MwhileltBasic) {
  FixedTag<float32_t, 4> t;
  
  auto m = mwhilelt(t, 0, 2);
  
  EXPECT_TRUE(get(t, m, 0));
  EXPECT_TRUE(get(t, m, 1));
  EXPECT_FALSE(get(t, m, 2));
  EXPECT_FALSE(get(t, m, 3));
}

TEST_F(VecX86_128Test, MwhileltAll) {
  FixedTag<float32_t, 4> t;
  
  auto m = mwhilelt(t, 0, 4);
  
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(get(t, m, i));
  }
}

TEST_F(VecX86_128Test, MwhileltNone) {
  FixedTag<float32_t, 4> t;
  
  auto m = mwhilelt(t, 2, 2);
  
  for (int i = 0; i < 4; ++i) {
    EXPECT_FALSE(get(t, m, i));
  }
}

TEST_F(VecX86_128Test, MwhilegeBasic) {
  FixedTag<float32_t, 4> t;
  
  auto m = mwhilege(t, 0, 2);
  
  EXPECT_FALSE(get(t, m, 0));
  EXPECT_FALSE(get(t, m, 1));
  EXPECT_TRUE(get(t, m, 2));
  EXPECT_TRUE(get(t, m, 3));
}

TEST_F(VecX86_128Test, MwhilegeAll) {
  FixedTag<float32_t, 4> t;
  
  auto m = mwhilege(t, 0, 0);
  
  for (int i = 0; i < 4; ++i) {
    EXPECT_TRUE(get(t, m, i));
  }
}

TEST_F(VecX86_128Test, MwhileleMwhilegt) {
  FixedTag<float32_t, 4> t;
  
  auto m_le = mwhilele(t, 0, 2);
  auto m_gt = mwhilegt(t, 0, 2);
  
  // mwhilele: [0,1,2] true
  EXPECT_TRUE(get(t, m_le, 0));
  EXPECT_TRUE(get(t, m_le, 1));
  EXPECT_TRUE(get(t, m_le, 2));
  EXPECT_FALSE(get(t, m_le, 3));
  
  // mwhilegt: [3] true
  EXPECT_FALSE(get(t, m_gt, 0));
  EXPECT_FALSE(get(t, m_gt, 1));
  EXPECT_FALSE(get(t, m_gt, 2));
  EXPECT_TRUE(get(t, m_gt, 3));
}

// ============================================================================
// loadu/storeu Tests
// ============================================================================

TEST_F(VecX86_128Test, LoaduBasic) {
  FixedTag<float32_t, 4> t;
  
  auto v = loadu(t, aligned_data_);
  
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(get(t, v, i), static_cast<float>(i));
  }
}

TEST_F(VecX86_128Test, StoreuBasic) {
  FixedTag<float32_t, 4> t;
  
  auto v = loadu(t, aligned_data_);
  storeu(t, aligned_out_, v);
  
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], static_cast<float>(i));
  }
}

TEST_F(VecX86_128Test, LoaduUnaligned) {
  // Test with unaligned pointer
  float* unaligned = aligned_data_ + 1;
  
  FixedTag<float32_t, 4> t;
  
  auto v = loadu(t, unaligned);
  
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(get(t, v, i), static_cast<float>(i + 1));
  }
}

TEST_F(VecX86_128Test, LoaduStoreu_2) {
  FixedTag<float32_t, 2> t;
  
  auto v = loadu(t, aligned_data_);
  storeu(t, aligned_out_, v);
  
  EXPECT_FLOAT_EQ(aligned_out_[0], 0.0f);
  EXPECT_FLOAT_EQ(aligned_out_[1], 1.0f);
}

// ============================================================================
// load/store Tests (aligned)
// ============================================================================

TEST_F(VecX86_128Test, LoadAlignedBasic) {
  FixedTag<float32_t, 4> t;
  
  auto v = load(t, aligned_data_);
  
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(get(t, v, i), static_cast<float>(i));
  }
}

TEST_F(VecX86_128Test, StoreAlignedBasic) {
  FixedTag<float32_t, 4> t;
  
  auto v = load(t, aligned_data_);
  store(t, aligned_out_, v);
  
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], static_cast<float>(i));
  }
}

TEST_F(VecX86_128Test, LoadAlignedInitializerList) {
  alignas(16) float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  FixedTag<float32_t, 4> t;
  
  auto v = load(t, data);
  
  EXPECT_FLOAT_EQ(get(t, v, 0), 1.0f);
  EXPECT_FLOAT_EQ(get(t, v, 1), 2.0f);
  EXPECT_FLOAT_EQ(get(t, v, 2), 3.0f);
  EXPECT_FLOAT_EQ(get(t, v, 3), 4.0f);
}

// ============================================================================
// load/store with n parameter Tests
// ============================================================================

TEST_F(VecX86_128Test, LoaduWithN) {
  FixedTag<float32_t, 4> t;
  
  auto default_v = fill(t, -99.0f);
  auto v = loadu(t, aligned_data_, 2, default_v);
  
  EXPECT_FLOAT_EQ(get(t, v, 0), 0.0f);
  EXPECT_FLOAT_EQ(get(t, v, 1), 1.0f);
  EXPECT_FLOAT_EQ(get(t, v, 2), -99.0f);
  EXPECT_FLOAT_EQ(get(t, v, 3), -99.0f);
}

TEST_F(VecX86_128Test, LoaduWithNZero) {
  FixedTag<float32_t, 4> t;
  
  auto default_v = fill(t, -99.0f);
  auto v = loadu(t, aligned_data_, 0, default_v);
  
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(get(t, v, i), -99.0f);
  }
}

TEST_F(VecX86_128Test, LoaduWithNFull) {
  FixedTag<float32_t, 4> t;
  
  auto default_v = fill(t, -99.0f);
  auto v = loadu(t, aligned_data_, 4, default_v);
  
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(get(t, v, i), static_cast<float>(i));
  }
}

TEST_F(VecX86_128Test, StoreuWithN) {
  FixedTag<float32_t, 4> t;
  
  auto v = fill(t, 42.0f);
  storeu(t, aligned_out_, 2, v);
  
  EXPECT_FLOAT_EQ(aligned_out_[0], 42.0f);
  EXPECT_FLOAT_EQ(aligned_out_[1], 42.0f);
  EXPECT_FLOAT_EQ(aligned_out_[2], -1.0f); // Unchanged
  EXPECT_FLOAT_EQ(aligned_out_[3], -1.0f); // Unchanged
}

TEST_F(VecX86_128Test, StoreuWithNZero) {
  FixedTag<float32_t, 4> t;
  
  // Reset
  for (int i = 0; i < 4; ++i) aligned_out_[i] = -1.0f;
  
  auto v = fill(t, 42.0f);
  storeu(t, aligned_out_, 0, v);
  
  // Nothing should be stored
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], -1.0f);
  }
}

// ============================================================================
// load/store with mask Tests
// ============================================================================

TEST_F(VecX86_128Test, LoaduWithMask) {
  FixedTag<float32_t, 4> t;
  
  auto m = mwhilelt(t, 0, 2);
  auto default_v = fill(t, -99.0f);
  
  auto v = loadu(t, aligned_data_, m, default_v);
  
  EXPECT_FLOAT_EQ(get(t, v, 0), 0.0f);
  EXPECT_FLOAT_EQ(get(t, v, 1), 1.0f);
  EXPECT_FLOAT_EQ(get(t, v, 2), -99.0f);
  EXPECT_FLOAT_EQ(get(t, v, 3), -99.0f);
}

TEST_F(VecX86_128Test, StoreuWithMask) {
  FixedTag<float32_t, 4> t;
  
  auto m = mwhilelt(t, 0, 2);
  auto v = fill(t, 42.0f);
  
  storeu(t, aligned_out_, m, v);
  
  EXPECT_FLOAT_EQ(aligned_out_[0], 42.0f);
  EXPECT_FLOAT_EQ(aligned_out_[1], 42.0f);
  EXPECT_FLOAT_EQ(aligned_out_[2], -1.0f);
  EXPECT_FLOAT_EQ(aligned_out_[3], -1.0f);
}

TEST_F(VecX86_128Test, StoreuWithMaskAll) {
  FixedTag<float32_t, 4> t;
  
  auto m = mtrue(t);
  auto v = fill(t, 42.0f);
  
  storeu(t, aligned_out_, m, v);
  
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], 42.0f);
  }
}

TEST_F(VecX86_128Test, StoreuWithMaskNone) {
  FixedTag<float32_t, 4> t;
  
  // Reset
  for (int i = 0; i < 4; ++i) aligned_out_[i] = -1.0f;
  
  auto m = mfalse(t);
  auto v = fill(t, 42.0f);
  
  storeu(t, aligned_out_, m, v);
  
  // Nothing should be stored
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], -1.0f);
  }
}

// ============================================================================
// gather/scatter Tests
// ============================================================================

// TODO impl it
//TEST_F(VecX86_128Test, GatherBasic) {
//  FixedTag<float32_t, 4> t;
//  using IndexVec = Vec<Tag<Index<float32_t>, 4>>;
//
//  IndexVec indices{};
//  for (int i = 0; i < 4; ++i) {
//    indices[i] = i * 2; // Every other element
//  }
//
//  auto v = gather(t, aligned_data_, indices);
//
//  for (int i = 0; i < 4; ++i) {
//    EXPECT_FLOAT_EQ(get(t, v, i), static_cast<float>(i * 2));
//  }
//}
//
//TEST_F(VecX86_128Test, ScatterBasic) {
//  FixedTag<float32_t, 4> t;
//  using IndexVec = Vec<Tag<Index<float32_t>, 4>>;
//
//  // Clear output
//  for (int i = 0; i < 8; ++i) aligned_out_[i] = -1.0f;
//
//  IndexVec indices{};
//  auto values = fill(t, 42.0f);
//  for (int i = 0; i < 4; ++i) {
//    indices[i] = i * 2;
//  }
//
//  scatter(t, aligned_out_, indices, values);
//
//  for (int i = 0; i < 4; ++i) {
//    EXPECT_FLOAT_EQ(aligned_out_[i * 2], 42.0f);
//    if (i > 0) {
//      EXPECT_FLOAT_EQ(aligned_out_[i * 2 - 1], -1.0f);
//    }
//  }
//}
//
//TEST_F(VecX86_128Test, GatherWithN) {
//  FixedTag<float32_t, 4> t;
//  using IndexVec = Vec<Tag<Index<float32_t>, 4>>;
//
//  IndexVec indices{};
//  for (int i = 0; i < 4; ++i) {
//    indices[i] = i;
//  }
//
//  auto default_v = fill(t, -99.0f);
//  auto v = gather(t, aligned_data_, indices, 2, default_v);
//
//  EXPECT_FLOAT_EQ(get(t, v, 0), 0.0f);
//  EXPECT_FLOAT_EQ(get(t, v, 1), 1.0f);
//  EXPECT_FLOAT_EQ(get(t, v, 2), -99.0f);
//  EXPECT_FLOAT_EQ(get(t, v, 3), -99.0f);
//}
//
//TEST_F(VecX86_128Test, GatherWithMask) {
//  FixedTag<float32_t, 4> t;
//  using IndexVec = Vec<Tag<Index<float32_t>, 4>>;
//
//  IndexVec indices{};
//  for (int i = 0; i < 4; ++i) {
//    indices[i] = i;
//  }
//
//  auto m = mwhilelt(t, 0, 2);
//  auto default_v = fill(t, -99.0f);
//
//  auto v = gather(t, aligned_data_, indices, m, default_v);
//
//  EXPECT_FLOAT_EQ(get(t, v, 0), 0.0f);
//  EXPECT_FLOAT_EQ(get(t, v, 1), 1.0f);
//  EXPECT_FLOAT_EQ(get(t, v, 2), -99.0f);
//  EXPECT_FLOAT_EQ(get(t, v, 3), -99.0f);
//}
//
//TEST_F(VecX86_128Test, ScatterWithN) {
//  FixedTag<float32_t, 4> t;
//  using IndexVec = Vec<Tag<Index<float32_t>, 4>>;
//
//  // Clear output
//  for (int i = 0; i < 4; ++i) aligned_out_[i] = -1.0f;
//
//  IndexVec indices{};
//  auto values = fill(t, 42.0f);
//  for (int i = 0; i < 4; ++i) {
//    indices[i] = i;
//  }
//
//  scatter(t, aligned_out_, indices, values, 2);
//
//  EXPECT_FLOAT_EQ(aligned_out_[0], 42.0f);
//  EXPECT_FLOAT_EQ(aligned_out_[1], 42.0f);
//  EXPECT_FLOAT_EQ(aligned_out_[2], -1.0f);
//  EXPECT_FLOAT_EQ(aligned_out_[3], -1.0f);
//}
//
//TEST_F(VecX86_128Test, ScatterWithMask) {
//  FixedTag<float32_t, 4> t;
//  using IndexVec = Vec<Tag<Index<float32_t>, 4>>;
//
//  // Clear output
//  for (int i = 0; i < 4; ++i) aligned_out_[i] = -1.0f;
//
//  IndexVec indices{};
//  auto values = fill(t, 42.0f);
//  for (int i = 0; i < 4; ++i) {
//    indices[i] = i;
//  }
//
//  auto m = mwhilelt(t, 0, 2);
//
//  scatter(t, aligned_out_, indices, values, m);
//
//  EXPECT_FLOAT_EQ(aligned_out_[0], 42.0f);
//  EXPECT_FLOAT_EQ(aligned_out_[1], 42.0f);
//  EXPECT_FLOAT_EQ(aligned_out_[2], -1.0f);
//  EXPECT_FLOAT_EQ(aligned_out_[3], -1.0f);
//}

// ============================================================================
// get/set element Tests
// ============================================================================

TEST_F(VecX86_128Test, GetElement) {
  FixedTag<float32_t, 4> t;
  
  auto v = fill(t, 3.14f);
  
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(get(t, v, i), 3.14f);
  }
}

TEST_F(VecX86_128Test, SetElement) {
  FixedTag<float32_t, 4> t;
  
  auto v = zeros(t);
  
  v = set(t, v, 2, 42.0f);
  
  EXPECT_FLOAT_EQ(get(t, v, 1), 0.0f);
  EXPECT_FLOAT_EQ(get(t, v, 2), 42.0f);
  EXPECT_FLOAT_EQ(get(t, v, 3), 0.0f);
}

TEST_F(VecX86_128Test, GetMaskElement) {
  FixedTag<float32_t, 4> t;
  
  auto m = mwhilelt(t, 0, 2);
  
  EXPECT_TRUE(get(t, m, 0));
  EXPECT_TRUE(get(t, m, 1));
  EXPECT_FALSE(get(t, m, 2));
  EXPECT_FALSE(get(t, m, 3));
}

TEST_F(VecX86_128Test, SetMaskElement) {
  FixedTag<float32_t, 4> t;
  
  auto m = mfalse(t);
  
  m = set(t, m, 2, true);
  
  EXPECT_FALSE(get(t, m, 1));
  EXPECT_TRUE(get(t, m, 2));
  EXPECT_FALSE(get(t, m, 3));
}

// ============================================================================
// Multi-word Vector Tests (N > 4)
// ============================================================================

TEST_F(VecX86_128Test, MultiWordSize8) {
  FixedTag<float32_t, 8> t;
  
  EXPECT_EQ(size(t), 8);
  EXPECT_EQ(word_size(t), 4);
  EXPECT_EQ(num_words(t), 2);
  EXPECT_FALSE(is_word_vec(t));
}

TEST_F(VecX86_128Test, MultiWordLoadStore_8) {
  FixedTag<float32_t, 8> t;
  
  auto v = loadu(t, aligned_data_);
  
  for (int i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(get(t, v, i), static_cast<float>(i));
  }
  
  storeu(t, aligned_out_, v);
  
  for (int i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], static_cast<float>(i));
  }
}

TEST_F(VecX86_128Test, MultiWordFill_8) {
  FixedTag<float32_t, 8> t;
  
  auto v = fill(t, 2.5f);
  
  for (int i = 0; i < 8; ++i) {
    EXPECT_FLOAT_EQ(get(t, v, i), 2.5f);
  }
}

TEST_F(VecX86_128Test, MultiWordMask_8) {
  FixedTag<float32_t, 8> t;
  
  auto m = mwhilelt(t, 0, 5);
  
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(get(t, m, i));
  }
  for (int i = 5; i < 8; ++i) {
    EXPECT_FALSE(get(t, m, i));
  }
}

TEST_F(VecX86_128Test, MultiWordLoadWithN_8) {
  FixedTag<float32_t, 8> t;
  
  auto default_v = fill(t, -99.0f);
  auto v = loadu(t, aligned_data_, 5, default_v);
  
  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(get(t, v, i), static_cast<float>(i));
  }
  for (int i = 5; i < 8; ++i) {
    EXPECT_FLOAT_EQ(get(t, v, i), -99.0f);
  }
}

TEST_F(VecX86_128Test, MultiWordStoreWithN_8) {
  FixedTag<float32_t, 8> t;
  
  auto v = fill(t, 42.0f);
  storeu(t, aligned_out_, 5, v);
  
  for (int i = 0; i < 5; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], 42.0f);
  }
  for (int i = 5; i < 8; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], -1.0f);
  }
}

TEST_F(VecX86_128Test, MultiWordSize16) {
  FixedTag<float32_t, 16> t;
  
  EXPECT_EQ(size(t), 16);
  EXPECT_EQ(word_size(t), 4);
  EXPECT_EQ(num_words(t), 4);
}

TEST_F(VecX86_128Test, MultiWordLoadStore_16) {
  FixedTag<float32_t, 16> t;
  
  auto v = loadu(t, aligned_data_);
  storeu(t, aligned_out_, v);
  
  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], static_cast<float>(i));
  }
}

// ============================================================================
// Copy Function Test (like the original VecTest)
// ============================================================================

CT_NOINLINE
static void x86_copy(const float* from, float* to, nint_t len) {
  ScalableTag<float32_t> t; // Should be 4 elements with SIMD_WIDTH=128
  nint_t i;
  for (i = 0; i <= len - size(t); i += size(t)) {
    auto v = loadu(t, from + i);
    storeu(t, to + i, v);
  }
  for (; i < len; ++i) {
    to[i] = from[i];
  }
}

TEST_F(VecX86_128Test, CopyFunction) {
  x86_copy(aligned_data_, aligned_out_, 128);
  
  for (int i = 0; i < 128; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], static_cast<float>(i));
  }
}

// ============================================================================
// ScalableTag Test
// ============================================================================

TEST_F(VecX86_128Test, ScalableTagTest) {
  ScalableTag<float32_t> t;
  
  EXPECT_EQ(size(t), 4);
  EXPECT_FALSE(is_default_impl(t));
  EXPECT_FALSE(is_scalable(t));
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(VecX86_128Test, LoadStoreSequential) {
  FixedTag<float32_t, 4> t;
  
  // Load and store sequentially
  for (int offset = 0; offset < 16; offset += 4) {
    auto v = loadu(t, aligned_data_ + offset);
    storeu(t, aligned_out_ + offset, v);
  }
  
  for (int i = 0; i < 16; ++i) {
    EXPECT_FLOAT_EQ(aligned_out_[i], static_cast<float>(i));
  }
}

TEST_F(VecX86_128Test, VariousMaskPatterns) {
  FixedTag<float32_t, 4> t;
  
  // Test various mask patterns
  auto m0 = mwhilelt(t, 0, 0);
  auto m1 = mwhilelt(t, 0, 1);
  auto m2 = mwhilelt(t, 0, 2);
  auto m3 = mwhilelt(t, 0, 3);
  auto m4 = mwhilelt(t, 0, 4);
  
  // m0: none
  for (int i = 0; i < 4; ++i) EXPECT_FALSE(get(t, m0, i));
  
  // m1: [0]
  EXPECT_TRUE(get(t, m1, 0));
  for (int i = 1; i < 4; ++i) EXPECT_FALSE(get(t, m1, i));
  
  // m2: [0,1]
  for (int i = 0; i < 2; ++i) EXPECT_TRUE(get(t, m2, i));
  for (int i = 2; i < 4; ++i) EXPECT_FALSE(get(t, m2, i));
  
  // m3: [0,1,2]
  for (int i = 0; i < 3; ++i) EXPECT_TRUE(get(t, m3, i));
  EXPECT_FALSE(get(t, m3, 3));
  
  // m4: all
  for (int i = 0; i < 4; ++i) EXPECT_TRUE(get(t, m4, i));
}

// TODO impl it
//TEST_F(VecX86_128Test, GatherScatterRoundTrip) {
//  FixedTag<float32_t, 4> t;
//  using IndexVec = Vec<Tag<Index<float32_t>, 4>>;
//
//  // Clear output
//  for (int i = 0; i < 8; ++i) aligned_out_[i] = -1.0f;
//
//  // Create permutation indices
//  IndexVec indices{};
//  indices[0] = 3;
//  indices[1] = 1;
//  indices[2] = 2;
//  indices[3] = 0;
//
//  auto v = gather(t, aligned_data_, indices);
//  scatter(t, aligned_out_, indices, v);
//
//  // Check permuted values
//  EXPECT_FLOAT_EQ(aligned_out_[0], 3.0f); // v[3] = 3
//  EXPECT_FLOAT_EQ(aligned_out_[1], 1.0f); // v[1] = 1
//  EXPECT_FLOAT_EQ(aligned_out_[2], 2.0f); // v[2] = 2
//  EXPECT_FLOAT_EQ(aligned_out_[3], 0.0f); // v[0] = 0
//}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
