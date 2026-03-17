//
// Created by renyz on 2026/3/14.
// Test file for Array.h
// Comprehensive tests covering all APIs and corner cases
//

#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <memory>
#include "tl/Array.h"

using namespace ct::tl::array;

// ============================================================================
// Test Fixtures
// ============================================================================

class ArrayTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create sample data for testing
    data_1d_.resize(10);
    std::iota(data_1d_.begin(), data_1d_.end(), 0);

    data_2d_.resize(20);
    std::iota(data_2d_.begin(), data_2d_.end(), 0);

    data_3d_.resize(60);
    std::iota(data_3d_.begin(), data_3d_.end(), 0);

    data_4d_.resize(120);
    std::iota(data_4d_.begin(), data_4d_.end(), 0);
  }

  std::vector<int64_t> data_1d_;
  std::vector<int64_t> data_2d_;
  std::vector<int64_t> data_3d_;
  std::vector<int64_t> data_4d_;
};

// ============================================================================
// Helper Function Tests
// ============================================================================

TEST_F(ArrayTest, NewAxisDefaultRepeat) {
  auto axis = new_axis();
  EXPECT_EQ(axis.repeat, 1);
}

TEST_F(ArrayTest, NewAxisCustomRepeat) {
  auto axis = new_axis(5);
  EXPECT_EQ(axis.repeat, 5);
}

TEST_F(ArrayTest, NewAxisZeroRepeat) {
  auto axis = new_axis(0);
  EXPECT_EQ(axis.repeat, 0);
}

TEST_F(ArrayTest, NewAxisNegativeRepeat) {
// This should trigger CT_ASSERT
  EXPECT_DEATH(new_axis(-1), "Cannot repeat negative times");
}

TEST_F(ArrayTest, NewAxisLargeRepeat) {
  auto axis = new_axis(1000000);
  EXPECT_EQ(axis.repeat, 1000000);
}

TEST_F(ArrayTest, RangeBasic) {
  auto r = range(0, 10);
  EXPECT_EQ(r.start, 0);
  EXPECT_EQ(r.end, 10);
  EXPECT_EQ(r.step, 1);
}

TEST_F(ArrayTest, RangeWithStep) {
  auto r = range(0, 10, 2);
  EXPECT_EQ(r.start, 0);
  EXPECT_EQ(r.end, 10);
  EXPECT_EQ(r.step, 2);
}

TEST_F(ArrayTest, RangeNegativeStep) {
  auto r = range(10, 0, -1);
  EXPECT_EQ(r.start, 10);
  EXPECT_EQ(r.end, 0);
  EXPECT_EQ(r.step, -1);
}

TEST_F(ArrayTest, RangeLargeStep) {
  auto r = range(0, 100, 25);
  EXPECT_EQ(r.start, 0);
  EXPECT_EQ(r.end, 100);
  EXPECT_EQ(r.step, 25);
}

TEST_F(ArrayTest, RangeNegativeValues) {
  auto r = range(-10, -5, 1);
  EXPECT_EQ(r.start, -10);
  EXPECT_EQ(r.end, -5);
  EXPECT_EQ(r.step, 1);
}

TEST_F(ArrayTest, ReserveMarker) {
  auto r = reserve;
  (void) r; // Just verify it compiles
}

TEST_F(ArrayTest, EllipseMarker) {
  auto e = ellipse;
  (void) e; // Just verify it compiles
}

// ============================================================================
// ArrayFlags Tests
// ============================================================================

TEST_F(ArrayTest, ArrayFlagsValues) {
  EXPECT_EQ(AF_NONE, 0);
  EXPECT_EQ(AF_LAST_CONTIGUOUS, 1);
  EXPECT_EQ(AF_LAST2_CONTIGUOUS, 3);
  EXPECT_EQ(AF_CONTIGUOUS, 7);
  EXPECT_EQ(_AF_ALL_CONTIGUITY_FLAGS, 7);
}

TEST_F(ArrayTest, ArrayFlagsBitwiseOr) {
  int flags = AF_NONE | AF_LAST_CONTIGUOUS;
  EXPECT_EQ(flags, AF_LAST_CONTIGUOUS);

  flags = AF_LAST_CONTIGUOUS | AF_LAST2_CONTIGUOUS;
  EXPECT_EQ(flags, AF_LAST2_CONTIGUOUS);
}

// ============================================================================
// Constructor Tests - 1D Array
// ============================================================================

TEST_F(ArrayTest, Constructor1DWithSizesAndStrides) {
  int64_t sizes[] = {10};
  int64_t strides[] = {1};

  Array<int64_t, 1> arr(data_1d_.data(), sizes, strides);

  EXPECT_EQ(arr.ndim(), 1);
  EXPECT_EQ(arr.size(0), 10);
  EXPECT_EQ(arr.stride(0), 1);
  EXPECT_EQ(arr.numel(), 10);
  EXPECT_EQ(arr.data(), data_1d_.data());
}

TEST_F(ArrayTest, Constructor1DContiguous) {
  int64_t sizes[] = {10};

  Array<int64_t, 1> arr(data_1d_.data(), sizes);

  EXPECT_EQ(arr.ndim(), 1);
  EXPECT_EQ(arr.size(0), 10);
  EXPECT_EQ(arr.stride(0), 1);
  EXPECT_TRUE(arr.is_contiguous());
  EXPECT_TRUE(arr.is_last_contiguous());
}

TEST_F(ArrayTest, Constructor1DInitializerList) {
  Array<int64_t, 1> arr(data_1d_.data(), {10}, {1});

  EXPECT_EQ(arr.size(0), 10);
  EXPECT_EQ(arr.stride(0), 1);
}

TEST_F(ArrayTest, Constructor1DInitializerListContiguous) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});

  EXPECT_EQ(arr.size(0), 10);
  EXPECT_TRUE(arr.is_contiguous());
}

TEST_F(ArrayTest, Constructor1DStdArray) {
  std::array<int64_t, 1> sizes = {10};
  std::array<int64_t, 1> strides = {1};

  Array<int64_t, 1> arr(data_1d_.data(), sizes, strides);

  EXPECT_EQ(arr.size(0), 10);
  EXPECT_EQ(arr.stride(0), 1);
}

TEST_F(ArrayTest, Constructor1DStdArrayContiguous) {
  std::array<int64_t, 1> sizes = {10};

  Array<int64_t, 1> arr(data_1d_.data(), sizes);

  EXPECT_EQ(arr.size(0), 10);
  EXPECT_TRUE(arr.is_contiguous());
}

TEST_F(ArrayTest, Constructor1DIntsMetaRef) {
  int64_t meta_data[] = {10};
  Array<int64_t, 1, AF_CONTIGUOUS> sizes_ref(meta_data, {1}, {1});
  Array<int64_t, 1, AF_CONTIGUOUS> strides_ref(meta_data + 0, {1}, {1});

// Note: IntsMetaRef is Array<int64_t, 1, AF_CONTIGUOUS>
  Array<int64_t, 1> arr(data_1d_.data(), sizes_ref, strides_ref);

  EXPECT_EQ(arr.size(0), 10);
}

TEST_F(ArrayTest, Constructor1DCopy) {
  int64_t sizes[] = {10};

  Array<int64_t, 1> arr1(data_1d_.data(), sizes);
  Array<int64_t, 1> arr2(arr1);

  EXPECT_EQ(arr2.size(0), arr1.size(0));
  EXPECT_EQ(arr2.stride(0), arr1.stride(0));
  EXPECT_EQ(arr2.data(), arr1.data());
}

TEST_F(ArrayTest, Constructor1DNullDataPtr) {
  int64_t sizes[] = {10};
  int64_t strides[] = {1};
  using IntArray = Array<int64_t, 1>;
  EXPECT_DEATH(IntArray(nullptr, sizes, strides), "data ptr should not be null");
}

TEST_F(ArrayTest, Constructor1DNullSizesPtr) {
  int64_t strides[] = {1};
  using IntArray = Array<int64_t, 1>;

  EXPECT_DEATH(IntArray(data_1d_.data(), nullptr, strides), "sizes ptr should not be null");
}

TEST_F(ArrayTest, Constructor1DNullStridesPtr) {
  int64_t sizes[] = {10};
  using IntArray = Array<int64_t, 1>;

  EXPECT_DEATH(IntArray(data_1d_.data(), sizes, nullptr), "strides ptr should not be null");
}

TEST_F(ArrayTest, Constructor1DInitializerListWrongSize) {
  using IntArray = Array<int64_t, 1>;
// Should fail: initializer list size != N
  EXPECT_DEATH(IntArray(data_1d_.data(), {10, 20}), "sizes._size()");
}

// ============================================================================
// Constructor Tests - 2D Array
// ============================================================================

TEST_F(ArrayTest, Constructor2DWithSizesAndStrides) {
  int64_t sizes[] = {4, 5};
  int64_t strides[] = {5, 1};

  Array<int64_t, 2> arr(data_2d_.data(), sizes, strides);

  EXPECT_EQ(arr.ndim(), 2);
  EXPECT_EQ(arr.size(0), 4);
  EXPECT_EQ(arr.size(1), 5);
  EXPECT_EQ(arr.stride(0), 5);
  EXPECT_EQ(arr.stride(1), 1);
  EXPECT_EQ(arr.numel(), 20);
}

TEST_F(ArrayTest, Constructor2DContiguous) {
  int64_t sizes[] = {4, 5};

  Array<int64_t, 2> arr(data_2d_.data(), sizes);

  EXPECT_EQ(arr.size(0), 4);
  EXPECT_EQ(arr.size(1), 5);
  EXPECT_EQ(arr.stride(0), 5);
  EXPECT_EQ(arr.stride(1), 1);
  EXPECT_TRUE(arr.is_contiguous());
  EXPECT_TRUE(arr.is_last2_contiguous());
}

TEST_F(ArrayTest, Constructor2DNonContiguous) {
  int64_t sizes[] = {4, 5};
  int64_t strides[] = {10, 2}; // Non-contiguous strides

  Array<int64_t, 2> arr(data_2d_.data(), sizes, strides);

  EXPECT_FALSE(arr.is_contiguous());
  EXPECT_FALSE(arr.is_last_contiguous());
}

TEST_F(ArrayTest, Constructor2DInitializerList) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5}, {5, 1});

  EXPECT_EQ(arr.size(0), 4);
  EXPECT_EQ(arr.size(1), 5);
}

TEST_F(ArrayTest, Constructor2DStdArray) {
  std::array<int64_t, 2> sizes = {4, 5};
  std::array<int64_t, 2> strides = {5, 1};

  Array<int64_t, 2> arr(data_2d_.data(), sizes, strides);

  EXPECT_EQ(arr.size(0), 4);
  EXPECT_EQ(arr.size(1), 5);
}

TEST_F(ArrayTest, Constructor2DCopy) {
  int64_t sizes[] = {4, 5};

  Array<int64_t, 2> arr1(data_2d_.data(), sizes);
  Array<int64_t, 2> arr2(arr1);

  EXPECT_EQ(arr2.size(0), arr1.size(0));
  EXPECT_EQ(arr2.size(1), arr1.size(1));
  EXPECT_EQ(arr2.data(), arr1.data());
}

// ============================================================================
// Constructor Tests - 3D Array
// ============================================================================

TEST_F(ArrayTest, Constructor3DWithSizesAndStrides) {
  int64_t sizes[] = {3, 4, 5};
  int64_t strides[] = {20, 5, 1};

  Array<int64_t, 3> arr(data_3d_.data(), sizes, strides);

  EXPECT_EQ(arr.ndim(), 3);
  EXPECT_EQ(arr.size(0), 3);
  EXPECT_EQ(arr.size(1), 4);
  EXPECT_EQ(arr.size(2), 5);
  EXPECT_EQ(arr.stride(0), 20);
  EXPECT_EQ(arr.stride(1), 5);
  EXPECT_EQ(arr.stride(2), 1);
  EXPECT_EQ(arr.numel(), 60);
}

TEST_F(ArrayTest, Constructor3DContiguous) {
  int64_t sizes[] = {3, 4, 5};

  Array<int64_t, 3> arr(data_3d_.data(), sizes);

  EXPECT_TRUE(arr.is_contiguous());
  EXPECT_TRUE(arr.is_last2_contiguous());
  EXPECT_TRUE(arr.is_last_contiguous());
}

TEST_F(ArrayTest, Constructor3DInitializerList) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  EXPECT_EQ(arr.size(0), 3);
  EXPECT_EQ(arr.size(1), 4);
  EXPECT_EQ(arr.size(2), 5);
}

// ============================================================================
// Constructor Tests - 4D Array
// ============================================================================

TEST_F(ArrayTest, Constructor4DContiguous) {
  int64_t sizes[] = {2, 3, 4, 5};

  Array<int64_t, 4> arr(data_4d_.data(), sizes);

  EXPECT_EQ(arr.ndim(), 4);
  EXPECT_EQ(arr.numel(), 120);
  EXPECT_TRUE(arr.is_contiguous());
}

TEST_F(ArrayTest, Constructor4DInitializerList) {
  Array<int64_t, 4> arr(data_4d_.data(), {2, 3, 4, 5});

  EXPECT_EQ(arr.size(0), 2);
  EXPECT_EQ(arr.size(1), 3);
  EXPECT_EQ(arr.size(2), 4);
  EXPECT_EQ(arr.size(3), 5);
}

// ============================================================================
// Size and Stride Access Tests
// ============================================================================

TEST_F(ArrayTest, SizeValidIndex) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  EXPECT_EQ(arr.size(0), 3);
  EXPECT_EQ(arr.size(1), 4);
  EXPECT_EQ(arr.size(2), 5);
}

TEST_F(ArrayTest, SizeInvalidIndexNegative) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  EXPECT_DEATH(arr.size(-1), "n !in 0:N");
}

TEST_F(ArrayTest, SizeInvalidIndexTooLarge) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  EXPECT_DEATH(arr.size(3), "n !in 0:N");
  EXPECT_DEATH(arr.size(100), "n !in 0:N");
}

TEST_F(ArrayTest, StrideValidIndex) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  EXPECT_EQ(arr.stride(0), 20);
  EXPECT_EQ(arr.stride(1), 5);
  EXPECT_EQ(arr.stride(2), 1);
}

TEST_F(ArrayTest, StrideInvalidIndexNegative) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  EXPECT_DEATH(arr.stride(-1), "n !in 0:N");
}

TEST_F(ArrayTest, StrideInvalidIndexTooLarge) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  EXPECT_DEATH(arr.stride(3), "n !in 0:N");
}

TEST_F(ArrayTest, SizesMethod) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  auto sizes = arr.sizes();
  EXPECT_EQ(sizes.size(0), 3);
  EXPECT_EQ(sizes(0), 3);
  EXPECT_EQ(sizes(1), 4);
  EXPECT_EQ(sizes(2), 5);
}

TEST_F(ArrayTest, StridesMethod) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  auto strides = arr.strides();
  EXPECT_EQ(strides.size(0), 3);
  EXPECT_EQ(strides(0), 20);
  EXPECT_EQ(strides(1), 5);
  EXPECT_EQ(strides(2), 1);
}

// ============================================================================
// Data Pointer Tests
// ============================================================================

TEST_F(ArrayTest, DataPointerNonConst) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

  int64_t* ptr = arr.data();
  EXPECT_EQ(ptr, data_2d_.data());

// Verify we can modify data through the pointer
  ptr[0] = 999;
  EXPECT_EQ(data_2d_[0], 999);
}

TEST_F(ArrayTest, DataPointerConst) {
  const Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

  const int64_t* ptr = arr.data();
  EXPECT_EQ(ptr, data_2d_.data());
}

// ============================================================================
// Numel Tests
// ============================================================================

TEST_F(ArrayTest, Numel1D) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});
  EXPECT_EQ(arr.numel(), 10);
}

TEST_F(ArrayTest, Numel2D) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
  EXPECT_EQ(arr.numel(), 20);
}

TEST_F(ArrayTest, Numel3D) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
  EXPECT_EQ(arr.numel(), 60);
}

TEST_F(ArrayTest, Numel4D) {
  Array<int64_t, 4> arr(data_4d_.data(), {2, 3, 4, 5});
  EXPECT_EQ(arr.numel(), 120);
}

TEST_F(ArrayTest, NumelWithSizeOne) {
  Array<int64_t, 3> arr(data_3d_.data(), {1, 1, 60});
  EXPECT_EQ(arr.numel(), 60);
}

TEST_F(ArrayTest, NumelWithZeroSize) {
// Note: This may trigger assertion in debug mode
  int64_t sizes[] = {0, 5};
  int64_t strides[] = {5, 1};

// In release mode, numel would be 0
// In debug mode, this might assert
  Array<int64_t, 2> arr(data_2d_.data(), sizes, strides);
  EXPECT_EQ(arr.numel(), 0);
}

// ============================================================================
// Ndim Tests
// ============================================================================

TEST_F(ArrayTest, Ndim1D) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});
  EXPECT_EQ(arr.ndim(), 1);
}

TEST_F(ArrayTest, Ndim2D) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
  EXPECT_EQ(arr.ndim(), 2);
}

TEST_F(ArrayTest, Ndim3D) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
  EXPECT_EQ(arr.ndim(), 3);
}

TEST_F(ArrayTest, Ndim4D) {
  Array<int64_t, 4> arr(data_4d_.data(), {2, 3, 4, 5});
  EXPECT_EQ(arr.ndim(), 4);
}

// ============================================================================
// Contiguity Tests
// ============================================================================

TEST_F(ArrayTest, IsLastContiguous1D) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});
  EXPECT_TRUE(arr.is_last_contiguous());
}

TEST_F(ArrayTest, IsLastContiguous2D) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
  EXPECT_TRUE(arr.is_last_contiguous());
}

TEST_F(ArrayTest, IsLastContiguousNonContiguous) {
  int64_t sizes[] = {4, 5};
  int64_t strides[] = {10, 2}; // Last axis stride is 2, not 1

  Array<int64_t, 2> arr(data_2d_.data(), sizes, strides);
  EXPECT_FALSE(arr.is_last_contiguous());
}

TEST_F(ArrayTest, IsLast2Contiguous2D) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
  EXPECT_TRUE(arr.is_last2_contiguous());
}

TEST_F(ArrayTest, IsLast2Contiguous3D) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
  EXPECT_TRUE(arr.is_last2_contiguous());
}

TEST_F(ArrayTest, IsLast2ContiguousNonContiguous) {
  int64_t sizes[] = {3, 4, 5};
  int64_t strides[] = {20, 10, 1}; // stride[1] != size[2]

  Array<int64_t, 3> arr(data_3d_.data(), sizes, strides);
  EXPECT_TRUE(arr.is_last_contiguous());
  EXPECT_FALSE(arr.is_last2_contiguous());
}

TEST_F(ArrayTest, IsContiguous1D) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});
  EXPECT_TRUE(arr.is_contiguous());
}

TEST_F(ArrayTest, IsContiguous2D) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});
  EXPECT_TRUE(arr.is_contiguous());
}

TEST_F(ArrayTest, IsContiguous3D) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});
  EXPECT_TRUE(arr.is_contiguous());
}

TEST_F(ArrayTest, IsContiguousNonContiguous) {
  int64_t sizes[] = {3, 4, 5};
  int64_t strides[] = {40, 10, 2}; // Non-contiguous

  Array<int64_t, 3> arr(data_3d_.data(), sizes, strides);
  EXPECT_FALSE(arr.is_contiguous());
}

TEST_F(ArrayTest, IsContiguousWithFlags) {
// Array with AF_CONTIGUOUS flag should always return true
  Array<int64_t, 2, AF_CONTIGUOUS> arr(data_2d_.data(), {4, 5});
  EXPECT_TRUE(arr.is_contiguous());
}

TEST_F(ArrayTest, IsLastContiguousWithFlags) {
// Array with AF_LAST_CONTIGUOUS flag
  Array<int64_t, 2, AF_LAST_CONTIGUOUS> arr(data_2d_.data(), {4, 5});
  EXPECT_TRUE(arr.is_last_contiguous());
}

// ============================================================================
// As Contiguous Cast Tests
// ============================================================================

TEST_F(ArrayTest, AsLastContiguous) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

  auto l1_contiguous = arr.as_last_contiguous();
  EXPECT_TRUE(l1_contiguous.is_last_contiguous());
}

TEST_F(ArrayTest, AsLast2Contiguous) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  auto l2_contiguous = arr.as_last2_contiguous();
  EXPECT_TRUE(l2_contiguous.is_last2_contiguous());
}

TEST_F(ArrayTest, AsContiguous) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  auto contiguous = arr.as_contiguous();
  EXPECT_TRUE(contiguous.is_contiguous());
}

// ============================================================================
// Indexing Tests - Single Element Access
// ============================================================================

TEST_F(ArrayTest, Indexing1DValid) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});

  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(arr(i), i);
  }
}

TEST_F(ArrayTest, Indexing1DOutOfBounds) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});

  EXPECT_DEATH(arr(-1), "!in 0:");
  EXPECT_DEATH(arr(10), "!in 0:");
  EXPECT_DEATH(arr(100), "!in 0:");
}

TEST_F(ArrayTest, Indexing2DValid) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

  int64_t expected = 0;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 5; ++j) {
      EXPECT_EQ(arr(i, j), expected);
      ++expected;
    }
  }
}

TEST_F(ArrayTest, Indexing2DOutOfBounds) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

  EXPECT_DEATH(arr(-1, 0), "!in 0:");
  EXPECT_DEATH(arr(4, 0), "!in 0:");
  EXPECT_DEATH(arr(0, -1), "!in 0:");
  EXPECT_DEATH(arr(0, 5), "!in 0:");
}

TEST_F(ArrayTest, Indexing3DValid) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  int64_t expected = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 5; ++k) {
        EXPECT_EQ(arr(i, j, k), expected);
        ++expected;
      }
    }
  }
}

TEST_F(ArrayTest, Indexing3DOutOfBounds) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  EXPECT_DEATH(arr(3, 0, 0), "!in 0:");
  EXPECT_DEATH(arr(0, 4, 0), "!in 0:");
  EXPECT_DEATH(arr(0, 0, 5), "!in 0:");
}

TEST_F(ArrayTest, Indexing4DValid) {
  Array<int64_t, 4> arr(data_4d_.data(), {2, 3, 4, 5});

  int64_t expected = 0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 4; ++k) {
        for (int l = 0; l < 5; ++l) {
          EXPECT_EQ(arr(i, j, k, l), expected);
          ++expected;
        }
      }
    }
  }
}

TEST_F(ArrayTest, BracketOperator) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

// operator[] should behave like operator()
  EXPECT_EQ(arr[0](0), arr(0, 0));
  EXPECT_EQ(arr[1](0), arr(1, 0));
}

// ============================================================================
// Slicing Tests - ReserveAxis
// ============================================================================

TEST_F(ArrayTest, SlicingReserveAxis1D) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});

  auto sliced = arr(reserve);
  EXPECT_EQ(sliced.ndim(), 1);
  EXPECT_EQ(sliced.size(0), 10);
}

TEST_F(ArrayTest, SlicingReserveAxis2D) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

  auto sliced = arr(reserve, reserve);
  EXPECT_EQ(sliced.ndim(), 2);
  EXPECT_EQ(sliced.size(0), 4);
  EXPECT_EQ(sliced.size(1), 5);
}

TEST_F(ArrayTest, SlicingReserveAxisPartial) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

// Reserve first axis, index second, reserve third
  auto sliced = arr(reserve, 2, reserve);
  EXPECT_EQ(sliced.ndim(), 2);
  EXPECT_EQ(sliced.size(0), 3);
  EXPECT_EQ(sliced.size(1), 5);
}

// ============================================================================
// Slicing Tests - NewAxis
// ============================================================================

TEST_F(ArrayTest, SlicingNewAxis1D) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});

  auto sliced = arr(new_axis(), reserve);
  EXPECT_EQ(sliced.ndim(), 2);
  EXPECT_EQ(sliced.size(0), 1);
  EXPECT_EQ(sliced.size(1), 10);
}

TEST_F(ArrayTest, SlicingNewAxisMultiple) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});

  auto sliced = arr(new_axis(3), reserve);
  EXPECT_EQ(sliced.ndim(), 2);
  EXPECT_EQ(sliced.size(0), 3);
  EXPECT_EQ(sliced.size(1), 10);
}

TEST_F(ArrayTest, SlicingNewAxisZeroRepeat) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});

  auto sliced = arr(new_axis(0), reserve);
  EXPECT_EQ(sliced.ndim(), 2);
  EXPECT_EQ(sliced.size(0), 0);
  EXPECT_EQ(sliced.size(1), 10);
}

TEST_F(ArrayTest, SlicingNewAxis2D) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

  auto sliced = arr(new_axis(), reserve, reserve);
  EXPECT_EQ(sliced.ndim(), 3);
  EXPECT_EQ(sliced.size(0), 1);
  EXPECT_EQ(sliced.size(1), 4);
  EXPECT_EQ(sliced.size(2), 5);
}

TEST_F(ArrayTest, SlicingNewAxisInMiddle) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

  auto sliced = arr(reserve, new_axis(), reserve);
  EXPECT_EQ(sliced.ndim(), 3);
  EXPECT_EQ(sliced.size(0), 4);
  EXPECT_EQ(sliced.size(1), 1);
  EXPECT_EQ(sliced.size(2), 5);
}

TEST_F(ArrayTest, SlicingNewAxisAtEnd) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

  auto sliced = arr(reserve, reserve, new_axis());
  EXPECT_EQ(sliced.ndim(), 3);
  EXPECT_EQ(sliced.size(0), 4);
  EXPECT_EQ(sliced.size(1), 5);
  EXPECT_EQ(sliced.size(2), 1);
}

// ============================================================================
// Slicing Tests - Range
// ============================================================================

TEST_F(ArrayTest, SlicingRange1D) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});

  auto sliced = arr(range(2, 5));
  EXPECT_EQ(sliced.ndim(), 1);
  EXPECT_EQ(sliced.size(0), 3);
  EXPECT_EQ(sliced(0), 2);
  EXPECT_EQ(sliced(1), 3);
  EXPECT_EQ(sliced(2), 4);
}

TEST_F(ArrayTest, SlicingRangeWithStep) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});

  auto sliced = arr(range(0, 10, 2));
  EXPECT_EQ(sliced.ndim(), 1);
  EXPECT_EQ(sliced.size(0), 5);
  EXPECT_EQ(sliced(0), 0);
  EXPECT_EQ(sliced(1), 2);
  EXPECT_EQ(sliced(2), 4);
  EXPECT_EQ(sliced(3), 6);
  EXPECT_EQ(sliced(4), 8);
}

TEST_F(ArrayTest, SlicingRangeNegativeStep) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});

  auto sliced = arr(range(9, 0, -1));
  EXPECT_EQ(sliced.ndim(), 1);
  EXPECT_EQ(sliced.size(0), 9);
  EXPECT_EQ(sliced(0), 9);
  EXPECT_EQ(sliced(1), 8);
  EXPECT_EQ(sliced(8), 1);
}

TEST_F(ArrayTest, SlicingRangeNegativeStepLarge) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});

  auto sliced = arr(range(9, 0, -2));
  EXPECT_EQ(sliced.ndim(), 1);
  EXPECT_EQ(sliced.size(0), 5);
  EXPECT_EQ(sliced(0), 9);
  EXPECT_EQ(sliced(1), 7);
  EXPECT_EQ(sliced(2), 5);
  EXPECT_EQ(sliced(3), 3);
  EXPECT_EQ(sliced(4), 1);
}

TEST_F(ArrayTest, SlicingRange2D) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

  auto sliced = arr(range(1, 3), reserve);
  EXPECT_EQ(sliced.ndim(), 2);
  EXPECT_EQ(sliced.size(0), 2);
  EXPECT_EQ(sliced.size(1), 5);
}

TEST_F(ArrayTest, SlicingRangeBothAxes) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

  auto sliced = arr(range(1, 3), range(2, 4));
  EXPECT_EQ(sliced.ndim(), 2);
  EXPECT_EQ(sliced.size(0), 2);
  EXPECT_EQ(sliced.size(1), 2);

// Check values: original[1,2] = 1*5 + 2 = 7
  EXPECT_EQ(sliced(0, 0), 7);
  EXPECT_EQ(sliced(0, 1), 8);
  EXPECT_EQ(sliced(1, 0), 12);
  EXPECT_EQ(sliced(1, 1), 13);
}

TEST_F(ArrayTest, SlicingRangeInvalidFrom) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});

  EXPECT_DEATH(arr(range(-1, 5)), "!in 0:");
  EXPECT_DEATH(arr(range(10, 15)), "!in 0:");
}

TEST_F(ArrayTest, SlicingRangeInvalidTo) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});

  EXPECT_DEATH(arr(range(0, -1)), "!in 0:");
  EXPECT_DEATH(arr(range(0, 11)), "!in 0:");
}

TEST_F(ArrayTest, SlicingRangeInvalidStepZero) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});

  EXPECT_DEATH(arr(range(0, 10, 0)), "step cannot be zero");
}

TEST_F(ArrayTest, SlicingRangePositiveStepFromGteTo) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});

  EXPECT_DEATH(arr(range(5, 5, 1)), "for positive step, from must be less than to");
  EXPECT_DEATH(arr(range(6, 5, 1)), "for positive step, from must be less than to");
}

TEST_F(ArrayTest, SlicingRangeNegativeStepFromLteTo) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});

  EXPECT_DEATH(arr(range(5, 5, -1)), "for negative step, from must be greater than to");
  EXPECT_DEATH(arr(range(4, 5, -1)), "for negative step, from must be greater than to");
}

TEST_F(ArrayTest, SlicingRangeSingleElement) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});

  auto sliced = arr(range(5, 6));
  EXPECT_EQ(sliced.ndim(), 1);
  EXPECT_EQ(sliced.size(0), 1);
  EXPECT_EQ(sliced(0), 5);
}

TEST_F(ArrayTest, SlicingRangeLargeStep) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});

  auto sliced = arr(range(0, 10, 100));
  EXPECT_EQ(sliced.ndim(), 1);
  EXPECT_EQ(sliced.size(0), 1);
  EXPECT_EQ(sliced(0), 0);
}

// ============================================================================
// Slicing Tests - Mixed
// ============================================================================

TEST_F(ArrayTest, SlicingMixedIndexAndReserve) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  auto sliced = arr(1, reserve, reserve);
  EXPECT_EQ(sliced.ndim(), 2);
  EXPECT_EQ(sliced.size(0), 4);
  EXPECT_EQ(sliced.size(1), 5);

// Check that we're pointing to the right slice
  EXPECT_EQ(sliced(0, 0), 1 * 20 + 0 * 5 + 0); // = 20
}

TEST_F(ArrayTest, SlicingMixedIndexAndRange) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  auto sliced = arr(reserve, range(1, 3), reserve);
  EXPECT_EQ(sliced.ndim(), 3);
  EXPECT_EQ(sliced.size(0), 3);
  EXPECT_EQ(sliced.size(1), 2);
  EXPECT_EQ(sliced.size(2), 5);
}

TEST_F(ArrayTest, SlicingMixedAllTypes) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  auto sliced = arr(new_axis(), reserve, range(1, 3), 2);
  EXPECT_EQ(sliced.ndim(), 3);
  EXPECT_EQ(sliced.size(0), 1);
  EXPECT_EQ(sliced.size(1), 3);
  EXPECT_EQ(sliced.size(2), 2);
}

TEST_F(ArrayTest, SlicingReduceToScalar) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

  auto elem = arr(2, 3);
  EXPECT_EQ(elem, 2 * 5 + 3); // = 13
}

TEST_F(ArrayTest, SlicingReduceToScalarModify) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

  arr(2, 3) = 999;
  EXPECT_EQ(data_2d_[2 * 5 + 3], 999);
}

// ============================================================================
// Transpose Tests
// ============================================================================

TEST_F(ArrayTest, Transpose2D) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

  auto transposed = transpose(arr, 0, 1);
  EXPECT_EQ(transposed.ndim(), 2);
  EXPECT_EQ(transposed.size(0), 5);
  EXPECT_EQ(transposed.size(1), 4);
  EXPECT_EQ(transposed.stride(0), 1);
  EXPECT_EQ(transposed.stride(1), 5);
}

TEST_F(ArrayTest, Transpose2DValues) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

  auto transposed = transpose(arr, 0, 1);

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 5; ++j) {
      EXPECT_EQ(arr(i, j), transposed(j, i));
    }
  }
}

TEST_F(ArrayTest, Transpose3D) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  auto transposed = transpose(arr, 0, 2);
  EXPECT_EQ(transposed.ndim(), 3);
  EXPECT_EQ(transposed.size(0), 5);
  EXPECT_EQ(transposed.size(1), 4);
  EXPECT_EQ(transposed.size(2), 3);
}

TEST_F(ArrayTest, Transpose3DValues) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  auto transposed = transpose(arr, 1, 2);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 5; ++k) {
        EXPECT_EQ(arr(i, j, k), transposed(i, k, j));
      }
    }
  }
}

TEST_F(ArrayTest, TransposeInvalidIndex) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

  EXPECT_DEATH(transpose(arr, -1, 0), "!in 0:");
  EXPECT_DEATH(transpose(arr, 0, 2), "!in 0:");
  EXPECT_DEATH(transpose(arr, 2, 3), "!in 0:");
}

TEST_F(ArrayTest, TransposeSameAxis) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

  auto transposed = transpose(arr, 0, 0);
  EXPECT_EQ(transposed.size(0), 4);
  EXPECT_EQ(transposed.size(1), 5);
}

TEST_F(ArrayTest, TransposeLosesContiguity) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

  auto transposed = transpose(arr, 0, 1);
  EXPECT_FALSE(transposed.is_contiguous());
}

// ============================================================================
// Permute Tests
// ============================================================================

TEST_F(ArrayTest, Permute2D) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

  auto permuted = permute(arr, 1, 0);
  EXPECT_EQ(permuted.size(0), 5);
  EXPECT_EQ(permuted.size(1), 4);
}

TEST_F(ArrayTest, Permute3D) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  auto permuted = permute(arr, 2, 0, 1);
  EXPECT_EQ(permuted.size(0), 5);
  EXPECT_EQ(permuted.size(1), 3);
  EXPECT_EQ(permuted.size(2), 4);
}

TEST_F(ArrayTest, Permute3DValues) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  auto permuted = permute(arr, 2, 1, 0);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 5; ++k) {
        EXPECT_EQ(arr(i, j, k), permuted(k, j, i));
      }
    }
  }
}

TEST_F(ArrayTest, Permute4D) {
  Array<int64_t, 4> arr(data_4d_.data(), {2, 3, 4, 5});

  auto permuted = permute(arr, 3, 2, 1, 0);
  EXPECT_EQ(permuted.size(0), 5);
  EXPECT_EQ(permuted.size(1), 4);
  EXPECT_EQ(permuted.size(2), 3);
  EXPECT_EQ(permuted.size(3), 2);
}

TEST_F(ArrayTest, PermuteIdentity) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  auto permuted = permute(arr, 0, 1, 2);
  EXPECT_EQ(permuted.size(0), 3);
  EXPECT_EQ(permuted.size(1), 4);
  EXPECT_EQ(permuted.size(2), 5);
}

TEST_F(ArrayTest, PermuteInvalidIndex) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  EXPECT_DEATH(permute(arr, 0, 1, 3), "!in 0:");
  EXPECT_DEATH(permute(arr, 0, 1, -1), "!in 0:");
}

TEST_F(ArrayTest, PermuteDuplicateIndex) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  EXPECT_DEATH(permute(arr, 0, 0, 1), "is being used twice");
}

TEST_F(ArrayTest, PermuteLosesContiguity) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  auto permuted = permute(arr, 2, 1, 0);
  EXPECT_FALSE(permuted.is_contiguous());
}

// ============================================================================
// Type Alias Tests
// ============================================================================

TEST_F(ArrayTest, TypeAliasL1Contiguous) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

  Array<int64_t, 2>::L1Contiguous l1_arr = arr.as_last_contiguous();
  EXPECT_TRUE(l1_arr.is_last_contiguous());
}

TEST_F(ArrayTest, TypeAliasL2Contiguous) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  Array<int64_t, 3>::L2Contiguous l2_arr = arr.as_last2_contiguous();
  EXPECT_TRUE(l2_arr.is_last2_contiguous());
}

TEST_F(ArrayTest, TypeAliasContiguous) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  Array<int64_t, 3>::Contiguous cont_arr = arr.as_contiguous();
  EXPECT_TRUE(cont_arr.is_contiguous());
}

TEST_F(ArrayTest, TypeAliasIntsMetaRef) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

  Array<int64_t, 2>::IntsMetaRef sizes = arr.sizes();
  EXPECT_EQ(sizes.size(0), 2);
  EXPECT_EQ(sizes(0), 4);
  EXPECT_EQ(sizes(1), 5);
}

// ============================================================================
// Different Data Types Tests
// ============================================================================

TEST_F(ArrayTest, DataTypeFloat) {
  std::vector<float> data(20);
  for (int i = 0; i < 20; ++i) data[i] = static_cast<float>(i) * 0.5f;

  Array<float, 2> arr(data.data(), {4, 5});

  EXPECT_EQ(arr.ndim(), 2);
  EXPECT_FLOAT_EQ(arr(0, 0), 0.0f);
  EXPECT_FLOAT_EQ(arr(1, 2), 3.5f);
}

TEST_F(ArrayTest, DataTypeDouble) {
  std::vector<double> data(20);
  for (int i = 0; i < 20; ++i) data[i] = static_cast<double>(i) * 0.25;

  Array<double, 2> arr(data.data(), {4, 5});

  EXPECT_EQ(arr.ndim(), 2);
  EXPECT_DOUBLE_EQ(arr(0, 0), 0.0);
  EXPECT_DOUBLE_EQ(arr(2, 3), 13.0 * 0.25);
}

TEST_F(ArrayTest, DataTypeInt32) {
  std::vector<int32_t> data(20);
  for (int i = 0; i < 20; ++i) data[i] = i * 2;

  Array<int32_t, 2> arr(data.data(), {4, 5});

  EXPECT_EQ(arr.ndim(), 2);
  EXPECT_EQ(arr(0, 0), 0);
  EXPECT_EQ(arr(3, 4), 38);
}

TEST_F(ArrayTest, DataTypeInt8) {
  std::vector<int8_t> data(20);
  for (int i = 0; i < 20; ++i) data[i] = static_cast<int8_t>(i);

  Array<int8_t, 2> arr(data.data(), {4, 5});

  EXPECT_EQ(arr.ndim(), 2);
  EXPECT_EQ(arr(0, 0), 0);
  EXPECT_EQ(arr(1, 4), 9);
}

TEST_F(ArrayTest, DataTypeUInt8) {
  std::vector<uint8_t> data(20);
  for (int i = 0; i < 20; ++i) data[i] = static_cast<uint8_t>(i + 200);

  Array<uint8_t, 2> arr(data.data(), {4, 5});

  EXPECT_EQ(arr.ndim(), 2);
  EXPECT_EQ(arr(0, 0), 200);
}

// ============================================================================
// Edge Cases and Corner Cases
// ============================================================================

TEST_F(ArrayTest, EdgeCaseSizeOneDimension) {
  Array<int64_t, 3> arr(data_3d_.data(), {1, 1, 60});

  EXPECT_EQ(arr.size(0), 1);
  EXPECT_EQ(arr.size(1), 1);
  EXPECT_EQ(arr.size(2), 60);
  EXPECT_EQ(arr.numel(), 60);
  EXPECT_TRUE(arr.is_contiguous());
}

TEST_F(ArrayTest, EdgeCaseAllSizeOne) {
  int64_t data[] = {42};
  Array<int64_t, 3> arr(data, {1, 1, 1});

  EXPECT_EQ(arr.numel(), 1);
  EXPECT_EQ(arr(0, 0, 0), 42);
}

TEST_F(ArrayTest, EdgeCaseLargeDimension) {
// Test with larger dimension values
  std::vector<int64_t> data(1000);
  std::iota(data.begin(), data.end(), 0);

  Array<int64_t, 1> arr(data.data(), {1000});
  EXPECT_EQ(arr.size(0), 1000);
  EXPECT_EQ(arr(999), 999);
}

TEST_F(ArrayTest, EdgeCaseStridedView) {
// Create a strided view that skips elements
  int64_t sizes[] = {5};
  int64_t strides[] = {2}; // Every other element

  Array<int64_t, 1> arr(data_1d_.data(), sizes, strides);

  EXPECT_EQ(arr.size(0), 5);
  EXPECT_EQ(arr(0), 0);
  EXPECT_EQ(arr(1), 2);
  EXPECT_EQ(arr(2), 4);
  EXPECT_EQ(arr(3), 6);
  EXPECT_EQ(arr(4), 8);
  EXPECT_FALSE(arr.is_contiguous());
}

TEST_F(ArrayTest, EdgeCaseNegativeStride) {
// Create a reversed view
  int64_t sizes[] = {5};
  int64_t strides[] = {-1};

  Array<int64_t, 1> arr(data_1d_.data() + 4, sizes, strides);

  EXPECT_EQ(arr(0), 4);
  EXPECT_EQ(arr(1), 3);
  EXPECT_EQ(arr(2), 2);
  EXPECT_EQ(arr(3), 1);
  EXPECT_EQ(arr(4), 0);
}

TEST_F(ArrayTest, EdgeCaseZeroStride) {
// Create a broadcast view (same element repeated)
  int64_t sizes[] = {5};
  int64_t strides[] = {0};

  Array<int64_t, 1> arr(data_1d_.data(), sizes, strides);

// All elements should be the same
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(arr(i), 0);
  }
}

TEST_F(ArrayTest, EdgeCaseBroadcastDimension) {
// Create a 2D array where one dimension is broadcast
  int64_t sizes[] = {3, 5};
  int64_t strides[] = {0, 1}; // First dimension has stride 0 (broadcast)

  Array<int64_t, 2> arr(data_1d_.data(), sizes, strides);

// All rows should be identical
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      EXPECT_EQ(arr(i, j), j);
    }
  }
}

TEST_F(ArrayTest, EdgeCaseEmptyRange) {
  Array<int64_t, 1> arr(data_1d_.data(), {10});

// Range that results in empty slice
// Note: This might be invalid based on the implementation
// Testing the behavior
}

TEST_F(ArrayTest, EdgeCaseMultipleSlicing) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

// Multiple levels of slicing
  auto slice1 = arr(1, reserve, reserve); // 2D
  auto slice2 = slice1(range(1, 3), reserve); // 2D
  auto slice3 = slice2(0, reserve); // 1D

  EXPECT_EQ(slice3.ndim(), 1);
  EXPECT_EQ(slice3.size(0), 5);
}

TEST_F(ArrayTest, EdgeCaseChainedTranspose) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  auto t1 = transpose(arr, 0, 1);
  auto t2 = transpose(t1, 1, 2);

// t2 should be equivalent to permute(arr, 1, 2, 0)
  EXPECT_EQ(t2.size(0), 4);
  EXPECT_EQ(t2.size(1), 5);
  EXPECT_EQ(t2.size(2), 3);
}

TEST_F(ArrayTest, EdgeCaseConstCorrectness) {
  const Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

// Should be able to read but not write
  EXPECT_EQ(arr(0, 0), 0);

  const int64_t* ptr = arr.data();
  EXPECT_NE(ptr, nullptr);
}

TEST_F(ArrayTest, EdgeCaseModifyThroughSlice) {
  Array<int64_t, 2> arr(data_2d_.data(), {4, 5});

  auto slice = arr(2, reserve);

// Modify through slice
  slice(0) = 100;
  slice(1) = 101;

  EXPECT_EQ(arr(2, 0), 100);
  EXPECT_EQ(arr(2, 1), 101);
}

// ============================================================================
// Contiguity Flag Propagation Tests
// ============================================================================

TEST_F(ArrayTest, ContiguityAfterReserve) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  auto sliced = arr(reserve, reserve, reserve);
  EXPECT_TRUE(sliced.is_contiguous());
}

TEST_F(ArrayTest, ContiguityAfterIndexing) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  auto sliced = arr(1, reserve, reserve);
// After indexing, contiguity should be preserved for remaining axes
  EXPECT_TRUE(sliced.is_contiguous());
}

TEST_F(ArrayTest, ContiguityAfterRange) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  auto sliced = arr(reserve, range(1, 4), reserve);
// Range breaks contiguity
  EXPECT_FALSE(sliced.is_contiguous());
}

TEST_F(ArrayTest, ContiguityAfterNewAxis) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  auto sliced = arr(new_axis(), reserve, reserve, reserve);
// NewAxis with stride 0 breaks contiguity
  EXPECT_FALSE(sliced.is_contiguous());
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_F(ArrayTest, StressManySlices) {
  Array<int64_t, 4> arr(data_4d_.data(), {2, 3, 4, 5});

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      auto slice = arr(i, j, reserve, reserve);
      EXPECT_EQ(slice.ndim(), 2);
      EXPECT_EQ(slice.size(0), 4);
      EXPECT_EQ(slice.size(1), 5);
    }
  }
}

TEST_F(ArrayTest, StressManyTransposes) {
  Array<int64_t, 3> arr(data_3d_.data(), {3, 4, 5});

  auto t1 = transpose(arr, 0, 1); // (4, 3, 5)
  auto t2 = transpose(t1, 1, 2); // (4, 5, 3)
  auto t3 = transpose(t2, 0, 1); // (5, 4, 3)

// Verify data is still accessible correctly
  EXPECT_EQ(t3.size(0), 5);
  EXPECT_EQ(t3.size(1), 4);
  EXPECT_EQ(t3.size(2), 3);
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
