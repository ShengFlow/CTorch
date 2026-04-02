//
// VecShuffleTest.cpp
// Comprehensive tests for local_shuf and shuf operations.
// Covers: local_shuf (compile-time indices, vector indices, scalar indices)
//         shuf   (per-word shuffle, cross-lane)
// Tests all data types, vector widths, and multi-word vectors.
//

#include <gtest/gtest.h>
#include <cmath>

#include "tl/cpu/Vec.h"

using namespace ct;
using namespace ct::tl;
using namespace ct::tl::vec;

// ============================================================================
// Helper utilities
// ============================================================================

namespace test_utils {

/// Number of elements in a full-width single-word vector of type T.
template <typename T>
constexpr nint_t full_vec_size() { return VEC_WIDTH / 8 / sizeof(T); }

/// Number of elements per 16-byte lane (block).
template <typename T>
constexpr nint_t lane_size() { return 16 / sizeof(T); }

/// Number of elements per hardware word.
template <typename T>
constexpr nint_t word_elements() { return VEC_WIDTH / 8 / sizeof(T); }

/// Signed integer type used as shuffle index for element type T.
template <typename T> struct ShuffleIndex     { using type = T; };
template <> struct ShuffleIndex<float32_t>     { using type = int32_t; };
template <> struct ShuffleIndex<float64_t>     { using type = int64_t; };
template <> struct ShuffleIndex<uint32_t>      { using type = int32_t; };
template <> struct ShuffleIndex<uint64_t>      { using type = int64_t; };
template <> struct ShuffleIndex<uint8_t>       { using type = int8_t; };
template <> struct ShuffleIndex<uint16_t>      { using type = int16_t; };
template <typename T> using shuffle_idx_t = typename ShuffleIndex<T>::type;

/// Fill array with distinct sequential values: data[i] = i + 1.
template <typename T>
void fill_seq(T* data, nint_t n) {
  for (nint_t i = 0; i < n; ++i)
    data[i] = static_cast<T>(i + 1);
}

/// Fill identity index array for local_shuf: idx[i] = i % lane_size.
template <typename I>
void fill_local_identity(I* idx, nint_t n) {
  constexpr nint_t M = 16 / sizeof(I);
  for (nint_t i = 0; i < n; ++i)
    idx[i] = static_cast<I>(i % M);
}

/// Fill identity index array for shuf: idx[i] = i % word_size.
template <typename I>
void fill_shuf_identity(I* idx, nint_t n) {
  constexpr nint_t WS = VEC_WIDTH / 8 / sizeof(I);
  for (nint_t i = 0; i < n; ++i)
    idx[i] = static_cast<I>(i % WS);
}

} // namespace test_utils

// ============================================================================
// local_shuf with compile-time indices — 4 elements per lane
// (float32_t, int32_t, uint32_t)
// ============================================================================

using Types4 = ::testing::Types<float32_t, int32_t, uint32_t>;

template <typename T> class LocalShufCT4 : public ::testing::Test {};
TYPED_TEST_SUITE(LocalShufCT4, Types4);

TYPED_TEST(LocalShufCT4, Identity) {
  using T = TypeParam;
  constexpr auto N = test_utils::full_vec_size<T>();
  FixedTag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data[N];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);
  auto r = local_shuf<3, 2, 1, 0>(v);  // I_j = j → identity
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
}

TYPED_TEST(LocalShufCT4, ReverseWithinLane) {
  using T = TypeParam;
  constexpr auto N = test_utils::full_vec_size<T>();
  constexpr auto M = 4;
  FixedTag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data[N];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);
  auto r = local_shuf<0, 1, 2, 3>(v);  // I_j = M-1-j → reverse
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
}

TYPED_TEST(LocalShufCT4, BroadcastFirst) {
  using T = TypeParam;
  constexpr auto N = test_utils::full_vec_size<T>();
  constexpr auto M = 4;
  FixedTag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data[N];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);
  auto r = local_shuf<0, 0, 0, 0>(v);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[(i / M) * M]) << "i=" << i;
}

TYPED_TEST(LocalShufCT4, MixedPermutation) {
  using T = TypeParam;
  constexpr auto N = test_utils::full_vec_size<T>();
  constexpr auto M = 4;
  FixedTag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data[N];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);
  // Permutation: [2, 0, 3, 1] — picks elements {2,0,3,1} from each lane
  auto r = local_shuf<1, 3, 0, 2>(v);  // I_3=1, I_2=3, I_1=0, I_0=2
  // result[0]=v[2], result[1]=v[0], result[2]=v[3], result[3]=v[1]
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    static constexpr int perm[] = {2, 0, 3, 1};
    EXPECT_EQ(get(t, r, i), data[lane * M + perm[pos]]) << "i=" << i;
  }
}

// ============================================================================
// local_shuf with compile-time indices — 2 elements per lane
// (float64_t, int64_t, uint64_t)
// ============================================================================

using Types2 = ::testing::Types<float64_t, int64_t, uint64_t>;

template <typename T> class LocalShufCT2 : public ::testing::Test {};
TYPED_TEST_SUITE(LocalShufCT2, Types2);

TYPED_TEST(LocalShufCT2, Identity) {
  using T = TypeParam;
  constexpr auto N = test_utils::full_vec_size<T>();
  FixedTag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data[N];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);
  auto r = local_shuf<1, 0>(v);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
}

TYPED_TEST(LocalShufCT2, Swap) {
  using T = TypeParam;
  constexpr auto N = test_utils::full_vec_size<T>();
  constexpr auto M = 2;
  FixedTag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data[N];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);
  auto r = local_shuf<0, 1>(v);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
}

// ============================================================================
// local_shuf with compile-time indices — 8 elements per lane
// (int16_t, uint16_t)
// ============================================================================

using Types8 = ::testing::Types<int16_t, uint16_t>;

template <typename T> class LocalShufCT8 : public ::testing::Test {};
TYPED_TEST_SUITE(LocalShufCT8, Types8);

TYPED_TEST(LocalShufCT8, Identity) {
  using T = TypeParam;
  constexpr auto N = test_utils::full_vec_size<T>();
  FixedTag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data[N];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);
  auto r = local_shuf<7, 6, 5, 4, 3, 2, 1, 0>(v);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
}

TYPED_TEST(LocalShufCT8, ReverseWithinLane) {
  using T = TypeParam;
  constexpr auto N = test_utils::full_vec_size<T>();
  constexpr auto M = 8;
  FixedTag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data[N];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);
  auto r = local_shuf<0, 1, 2, 3, 4, 5, 6, 7>(v);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
}

// ============================================================================
// local_shuf with compile-time indices — 16 elements per lane
// (int8_t, uint8_t)
// ============================================================================

using Types16 = ::testing::Types<int8_t, uint8_t>;

template <typename T> class LocalShufCT16 : public ::testing::Test {};
TYPED_TEST_SUITE(LocalShufCT16, Types16);

TYPED_TEST(LocalShufCT16, Identity) {
  using T = TypeParam;
  constexpr auto N = test_utils::full_vec_size<T>();
  FixedTag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data[N];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);
  auto r = local_shuf<15, 14, 13, 12, 11, 10, 9, 8,
                     7,  6,  5,  4,  3,  2,  1,  0>(v);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
}

TYPED_TEST(LocalShufCT16, ReverseWithinLane) {
  using T = TypeParam;
  constexpr auto N = test_utils::full_vec_size<T>();
  constexpr auto M = 16;
  FixedTag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data[N];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);
  auto r = local_shuf<0, 1, 2, 3, 4, 5, 6, 7,
                     8, 9, 10, 11, 12, 13, 14, 15>(v);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
}

TYPED_TEST(LocalShufCT16, BroadcastMiddle) {
  using T = TypeParam;
  constexpr auto N = test_utils::full_vec_size<T>();
  constexpr auto M = 16;
  FixedTag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data[N];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);
  // Broadcast element 7 of each lane to all positions
  auto r = local_shuf<7, 7, 7, 7, 7, 7, 7, 7,
                     7, 7, 7, 7, 7, 7, 7, 7>(v);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[(i / M) * M + 7]) << "i=" << i;
}

// ============================================================================
// local_shuf with vector indices — 4 elements per lane
// (float32_t, int32_t, uint32_t)
// ============================================================================

template <typename T> class LocalShufVI4 : public ::testing::Test {};
TYPED_TEST_SUITE(LocalShufVI4, Types4);

TYPED_TEST(LocalShufVI4, Identity) {
  using T = TypeParam;
  using I = test_utils::shuffle_idx_t<T>;
  constexpr auto N = test_utils::full_vec_size<T>();
  FixedTag<T, N> t;
  FixedTag<I, N> ti;
  alignas(DEFAULT_ALIGNMENT) T data[N];
  alignas(DEFAULT_ALIGNMENT) I idx[N];
  test_utils::fill_seq(data, N);
  test_utils::fill_local_identity(idx, N);
  auto v = loadu(t, data);
  auto vi = loadu(ti, idx);
  auto r = local_shuf(v, vi);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
}

TYPED_TEST(LocalShufVI4, ReverseWithinLane) {
  using T = TypeParam;
  using I = test_utils::shuffle_idx_t<T>;
  constexpr auto N = test_utils::full_vec_size<T>();
  constexpr auto M = 4;
  FixedTag<T, N> t;
  FixedTag<I, N> ti;
  alignas(DEFAULT_ALIGNMENT) T data[N];
  alignas(DEFAULT_ALIGNMENT) I idx[N];
  test_utils::fill_seq(data, N);
  for (nint_t i = 0; i < N; ++i)
    idx[i] = static_cast<I>(M - 1 - (i % M));
  auto v = loadu(t, data);
  auto vi = loadu(ti, idx);
  auto r = local_shuf(v, vi);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
}

// Verify that different lanes can use different index patterns.
TYPED_TEST(LocalShufVI4, DifferentPerLane) {
  using T = TypeParam;
  using I = test_utils::shuffle_idx_t<T>;
  constexpr auto N = test_utils::full_vec_size<T>();
  constexpr auto M = 4;
  FixedTag<T, N> t;
  FixedTag<I, N> ti;
  alignas(DEFAULT_ALIGNMENT) T data[N];
  alignas(DEFAULT_ALIGNMENT) I idx[N];
  test_utils::fill_seq(data, N);
  // Even lanes: identity; odd lanes: reverse
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    idx[i] = (lane % 2 == 0)
        ? static_cast<I>(pos)
        : static_cast<I>(M - 1 - pos);
  }
  auto v = loadu(t, data);
  auto vi = loadu(ti, idx);
  auto r = local_shuf(v, vi);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    T expected = (lane % 2 == 0)
        ? data[lane * M + pos]
        : data[lane * M + (M - 1 - pos)];
    EXPECT_EQ(get(t, r, i), expected) << "i=" << i;
  }
}

// ============================================================================
// local_shuf with vector indices — 2 elements per lane
// ============================================================================

template <typename T> class LocalShufVI2 : public ::testing::Test {};
TYPED_TEST_SUITE(LocalShufVI2, Types2);

TYPED_TEST(LocalShufVI2, Identity) {
  using T = TypeParam;
  using I = test_utils::shuffle_idx_t<T>;
  constexpr auto N = test_utils::full_vec_size<T>();
  FixedTag<T, N> t;
  FixedTag<I, N> ti;
  alignas(DEFAULT_ALIGNMENT) T data[N];
  alignas(DEFAULT_ALIGNMENT) I idx[N];
  test_utils::fill_seq(data, N);
  test_utils::fill_local_identity(idx, N);
  auto v = loadu(t, data);
  auto vi = loadu(ti, idx);
  auto r = local_shuf(v, vi);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
}

TYPED_TEST(LocalShufVI2, Swap) {
  using T = TypeParam;
  using I = test_utils::shuffle_idx_t<T>;
  constexpr auto N = test_utils::full_vec_size<T>();
  constexpr auto M = 2;
  FixedTag<T, N> t;
  FixedTag<I, N> ti;
  alignas(DEFAULT_ALIGNMENT) T data[N];
  alignas(DEFAULT_ALIGNMENT) I idx[N];
  test_utils::fill_seq(data, N);
  for (nint_t i = 0; i < N; ++i)
    idx[i] = static_cast<I>(M - 1 - (i % M));
  auto v = loadu(t, data);
  auto vi = loadu(ti, idx);
  auto r = local_shuf(v, vi);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
}

// ============================================================================
// local_shuf with vector indices — 8 elements per lane
// ============================================================================

template <typename T> class LocalShufVI8 : public ::testing::Test {};
TYPED_TEST_SUITE(LocalShufVI8, Types8);

TYPED_TEST(LocalShufVI8, Identity) {
  using T = TypeParam;
  using I = test_utils::shuffle_idx_t<T>;
  constexpr auto N = test_utils::full_vec_size<T>();
  FixedTag<T, N> t;
  FixedTag<I, N> ti;
  alignas(DEFAULT_ALIGNMENT) T data[N];
  alignas(DEFAULT_ALIGNMENT) I idx[N];
  test_utils::fill_seq(data, N);
  test_utils::fill_local_identity(idx, N);
  auto v = loadu(t, data);
  auto vi = loadu(ti, idx);
  auto r = local_shuf(v, vi);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
}

// ============================================================================
// local_shuf with vector indices — 16 elements per lane
// ============================================================================

template <typename T> class LocalShufVI16 : public ::testing::Test {};
TYPED_TEST_SUITE(LocalShufVI16, Types16);

TYPED_TEST(LocalShufVI16, Identity) {
  using T = TypeParam;
  using I = test_utils::shuffle_idx_t<T>;
  constexpr auto N = test_utils::full_vec_size<T>();
  FixedTag<T, N> t;
  FixedTag<I, N> ti;
  alignas(DEFAULT_ALIGNMENT) T data[N];
  alignas(DEFAULT_ALIGNMENT) I idx[N];
  test_utils::fill_seq(data, N);
  test_utils::fill_local_identity(idx, N);
  auto v = loadu(t, data);
  auto vi = loadu(ti, idx);
  auto r = local_shuf(v, vi);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
}

TYPED_TEST(LocalShufVI16, ReverseWithinLane) {
  using T = TypeParam;
  using I = test_utils::shuffle_idx_t<T>;
  constexpr auto N = test_utils::full_vec_size<T>();
  constexpr auto M = 16;
  FixedTag<T, N> t;
  FixedTag<I, N> ti;
  alignas(DEFAULT_ALIGNMENT) T data[N];
  alignas(DEFAULT_ALIGNMENT) I idx[N];
  test_utils::fill_seq(data, N);
  for (nint_t i = 0; i < N; ++i)
    idx[i] = static_cast<I>(M - 1 - (i % M));
  auto v = loadu(t, data);
  auto vi = loadu(ti, idx);
  auto r = local_shuf(v, vi);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
}

// ============================================================================
// local_shuf with scalar indices (runtime int... parameters)
// ============================================================================

TEST(LocalShufScalar, Float32_Identity) {
  FixedTag<float32_t, 4> t;
  alignas(DEFAULT_ALIGNMENT) float32_t data[4] = {1, 2, 3, 4};
  auto v = loadu(t, data);
  auto r = local_shuf(v, 3, 2, 1, 0);  // I_3=3, I_2=2, I_1=1, I_0=0 → identity
  for (nint_t i = 0; i < 4; ++i)
    EXPECT_EQ(get(t, r, i), data[i]);
}

TEST(LocalShufScalar, Float32_Reverse) {
  FixedTag<float32_t, 4> t;
  alignas(DEFAULT_ALIGNMENT) float32_t data[4] = {1, 2, 3, 4};
  auto v = loadu(t, data);
  auto r = local_shuf(v, 0, 1, 2, 3);
  float32_t expected[] = {4, 3, 2, 1};
  for (nint_t i = 0; i < 4; ++i)
    EXPECT_EQ(get(t, r, i), expected[i]);
}

TEST(LocalShufScalar, Float64_Swap) {
  FixedTag<float64_t, 2> t;
  alignas(DEFAULT_ALIGNMENT) float64_t data[2] = {10.5, 20.5};
  auto v = loadu(t, data);
  auto r = local_shuf(v, 0, 1);
  EXPECT_EQ(get(t, r, 0), data[1]);
  EXPECT_EQ(get(t, r, 1), data[0]);
}

TEST(LocalShufScalar, Int8_Identity) {
  constexpr auto N = test_utils::full_vec_size<int8_t>();
  FixedTag<int8_t, N> t;
  alignas(DEFAULT_ALIGNMENT) int8_t data[64];  // max possible N
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);
  auto r = local_shuf(v, 15, 14, 13, 12, 11, 10, 9, 8,
                         7,  6,  5,  4,  3,  2,  1,  0);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
}

TEST(LocalShufScalar, Int16_Reverse) {
  constexpr auto N = test_utils::full_vec_size<int16_t>();
  FixedTag<int16_t, N> t;
  alignas(DEFAULT_ALIGNMENT) int16_t data[32];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);
  auto r = local_shuf(v, 0, 1, 2, 3, 4, 5, 6, 7);
  constexpr auto M = 8;
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
}

// ============================================================================
// shuf — identity for all type categories
// ============================================================================

// All element types that support shuf
using ShufTypes = ::testing::Types<
    float32_t, int32_t, uint32_t,
    float64_t, int64_t, uint64_t,
    int8_t,   uint8_t,
    int16_t,  uint16_t>;

template <typename T> class ShufIdentity : public ::testing::Test {};
TYPED_TEST_SUITE(ShufIdentity, ShufTypes);

TYPED_TEST(ShufIdentity, Identity) {
  using T = TypeParam;
  using I = test_utils::shuffle_idx_t<T>;
  constexpr auto N = test_utils::full_vec_size<T>();
  FixedTag<T, N> t;
  FixedTag<I, N> ti;
  alignas(DEFAULT_ALIGNMENT) T data[N];
  alignas(DEFAULT_ALIGNMENT) I idx[N];
  test_utils::fill_seq(data, N);
  test_utils::fill_shuf_identity(idx, N);
  auto v = loadu(t, data);
  auto vi = loadu(ti, idx);
  auto r = shuf(v, vi);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
}

// ============================================================================
// shuf vs local_shuf — demonstrate cross-lane capability
// ============================================================================

// On multi-lane words (256-bit+), shuf can swap elements across lanes
// while local_shuf cannot. On single-lane words (128-bit), they behave
// identically.
TEST(ShufVsLocalShuf, Float32_FullWordReverse) {
  using T = float32_t;
  using I = int32_t;
  constexpr auto N = test_utils::full_vec_size<T>();
  constexpr auto M = test_utils::lane_size<T>();  // 4
  FixedTag<T, N> t;
  FixedTag<I, N> ti;
  alignas(DEFAULT_ALIGNMENT) T data[N];
  alignas(DEFAULT_ALIGNMENT) I idx_local[N];
  alignas(DEFAULT_ALIGNMENT) I idx_shuf[N];
  test_utils::fill_seq(data, N);

  // local_shuf: reverse each 4-element lane independently
  for (nint_t i = 0; i < N; ++i)
    idx_local[i] = static_cast<I>(M - 1 - (i % M));
  // shuf: reverse entire word (cross-lane)
  for (nint_t i = 0; i < N; ++i)
    idx_shuf[i] = static_cast<I>(N - 1 - i);

  auto v = loadu(t, data);
  auto r_local = local_shuf(v, loadu(ti, idx_local));
  auto r_shuf  = shuf(v, loadu(ti, idx_shuf));

  for (nint_t i = 0; i < N; ++i) {
    // local_shuf: reversed within each lane
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r_local, i), data[lane * M + (M - 1 - pos)])
        << "local_shuf i=" << i;
    // shuf: fully reversed word
    EXPECT_EQ(get(t, r_shuf, i), data[N - 1 - i])
        << "shuf i=" << i;
  }

  // When N == M (single lane), both produce the same result
  if (N == M) {
    for (nint_t i = 0; i < N; ++i)
      EXPECT_EQ(get(t, r_local, i), get(t, r_shuf, i));
  } else {
    // For multi-lane words, verify they differ at lane boundaries
    bool differs = false;
    for (nint_t i = 0; i < N; ++i) {
      if (get(t, r_local, i) != get(t, r_shuf, i)) {
        differs = true;
        break;
      }
    }
    EXPECT_TRUE(differs) << "local_shuf and shuf should differ on multi-lane words";
  }
}

TEST(ShufVsLocalShuf, Int8_FullWordReverse) {
  using T = int8_t;
  using I = int8_t;
  constexpr auto N = test_utils::full_vec_size<T>();
  constexpr auto M = 16;
  FixedTag<T, N> t;
  FixedTag<I, N> ti;
  alignas(DEFAULT_ALIGNMENT) T data[64];
  alignas(DEFAULT_ALIGNMENT) I idx_local[64];
  alignas(DEFAULT_ALIGNMENT) I idx_shuf[64];
  test_utils::fill_seq(data, N);

  for (nint_t i = 0; i < N; ++i)
    idx_local[i] = static_cast<I>(M - 1 - (i % M));
  for (nint_t i = 0; i < N; ++i)
    idx_shuf[i] = static_cast<I>(N - 1 - i);

  auto v = loadu(t, data);
  auto i_local = loadu(ti, idx_local);
  auto r_local = local_shuf(v, i_local);
  auto i_shuf = loadu(ti, idx_shuf);
  auto r_shuf  = shuf(v, i_shuf);

  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r_local, i), data[lane * M + (M - 1 - pos)])
        << "local_shuf i=" << i;
    EXPECT_EQ(get(t, r_shuf, i), data[N - 1 - i])
        << "shuf i=" << i;
  }
}

// ============================================================================
// shuf — cross-lane swap for float64
// ============================================================================

TEST(Shuf, Float64_CrossLaneSwap) {
  using T = float64_t;
  using I = int64_t;
  constexpr auto N = test_utils::full_vec_size<T>();
  constexpr auto M = 2;  // lane_size
  FixedTag<T, N> t;
  FixedTag<I, N> ti;
  alignas(DEFAULT_ALIGNMENT) T data[8];
  alignas(DEFAULT_ALIGNMENT) I idx[8];
  test_utils::fill_seq(data, N);
  // Swap each pair within the full word
  for (nint_t i = 0; i < N; ++i)
    idx[i] = static_cast<I>((i % M == 0) ? i + 1 : i - 1);
  auto v = loadu(t, data);
  auto vi = loadu(ti, idx);
  auto r = shuf(v, vi);
  for (nint_t i = 0; i < N; ++i) {
    I expected_idx = static_cast<I>((i % M == 0) ? i + 1 : i - 1);
    EXPECT_EQ(get(t, r, i), data[expected_idx]) << "i=" << i;
  }
}

// ============================================================================
// Multi-word vector tests — vectors wider than VEC_WIDTH
// ============================================================================

// --- Multi-word local_shuf with compile-time indices ---

TEST(LocalShufCT_MultiWord, Float32_Identity) {
  using T = float32_t;
  constexpr auto N1 = test_utils::full_vec_size<T>();  // single-word size
  constexpr auto N  = N1 * 2;                          // two-word size
  Tag<T, N> t;  // FixedTag<T, N> works when N is power of 2
  alignas(DEFAULT_ALIGNMENT) T data[N];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);
  auto wt = word_tag(t);
  // Identity pattern applied to every word's every lane
  auto r = local_shuf<3, 2, 1, 0>(v);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
}

TEST(LocalShufCT_MultiWord, Float32_Reverse) {
  using T = float32_t;
  constexpr auto N1 = test_utils::full_vec_size<T>();
  constexpr auto N  = N1 * 2;
  constexpr auto M  = 4;
  Tag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data[N];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);
  auto r = local_shuf<0, 1, 2, 3>(v);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
}

TEST(LocalShufCT_MultiWord, Int8_Identity) {
  using T = int8_t;
  constexpr auto N1 = test_utils::full_vec_size<T>();
  constexpr auto N  = N1 * 2;
  Tag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data[128];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);
  auto r = local_shuf<15, 14, 13, 12, 11, 10, 9, 8,
                     7,  6,  5,  4,  3,  2,  1,  0>(v);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
}

TEST(LocalShufCT_MultiWord, Int64_Swap) {
  using T = int64_t;
  constexpr auto N1 = test_utils::full_vec_size<T>();
  constexpr auto N  = N1 * 2;
  constexpr auto M  = 2;
  Tag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data[16];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);
  auto r = local_shuf<0, 1>(v);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
}

// --- Multi-word local_shuf with vector indices ---

TEST(LocalShufVI_MultiWord, Float32_Identity) {
  using T = float32_t;
  using I = int32_t;
  constexpr auto N1 = test_utils::full_vec_size<T>();
  constexpr auto N  = N1 * 2;
  Tag<T, N> t;
  Tag<I, N> ti;
  alignas(DEFAULT_ALIGNMENT) T data[32];
  alignas(DEFAULT_ALIGNMENT) I idx[32];
  test_utils::fill_seq(data, N);
  test_utils::fill_local_identity(idx, N);
  auto v  = loadu(t, data);
  auto vi = loadu(ti, idx);
  auto r  = local_shuf(v, vi);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
}

TEST(LocalShufVI_MultiWord, Uint8_Reverse) {
  using T = uint8_t;
  using I = int8_t;
  constexpr auto N1 = test_utils::full_vec_size<T>();
  constexpr auto N  = N1 * 2;
  constexpr auto M  = 16;
  Tag<T, N> t;
  Tag<I, N> ti;
  alignas(DEFAULT_ALIGNMENT) T data[128];
  alignas(DEFAULT_ALIGNMENT) I idx[128];
  test_utils::fill_seq(data, N);
  for (nint_t i = 0; i < N; ++i)
    idx[i] = static_cast<I>(M - 1 - (i % M));
  auto v  = loadu(t, data);
  auto vi = loadu(ti, idx);
  auto r  = local_shuf(v, vi);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
}

// --- Multi-word shuf ---

TEST(Shuf_MultiWord, Float32_ReverseWithinWord) {
  using T = float32_t;
  using I = int32_t;
  constexpr auto N1 = test_utils::full_vec_size<T>();
  constexpr auto N  = N1 * 2;
  Tag<T, N1, 1> t;
  Tag<I, N1, 1> ti;
  alignas(DEFAULT_ALIGNMENT) T data[32];
  alignas(DEFAULT_ALIGNMENT) I idx[32];
  test_utils::fill_seq(data, N);
  // Within each word, reverse all elements (cross-lane for multi-lane words)
  for (nint_t i = 0; i < N; ++i) {
    nint_t word = i / N1;
    nint_t pos  = i % N1;
    idx[i] = static_cast<I>(N1 - 1 - pos);
  }
  auto v  = loadu(t, data);
  auto vi = loadu(ti, idx);
  auto r  = shuf(v, vi);
  for (nint_t i = 0; i < N; ++i) {
    nint_t word = i / N1;
    nint_t pos  = i % N1;
    EXPECT_EQ(get(t, r, i), data[word * N1 + (N1 - 1 - pos)]) << "i=" << i;
  }
}

TEST(Shuf_MultiWord, Int16_Identity) {
  using T = int16_t;
  using I = int16_t;
  constexpr auto N1 = test_utils::full_vec_size<T>();
  constexpr auto N  = N1 * 2;
  Tag<T, N1, 1> t;
  Tag<I, N1, 1> ti;
  alignas(DEFAULT_ALIGNMENT) T data[64];
  alignas(DEFAULT_ALIGNMENT) I idx[64];
  test_utils::fill_seq(data, N);
  test_utils::fill_shuf_identity(idx, N);
  auto v  = loadu(t, data);
  auto vi = loadu(ti, idx);
  auto r  = shuf(v, vi);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[i]) << "i=" << i;
}

TEST(Shuf_MultiWord, Int64_Swap) {
  using T = int64_t;
  using I = int64_t;
  constexpr auto N1 = test_utils::full_vec_size<T>();
  constexpr auto N  = N1 * 2;
  constexpr auto M  = 2;
  Tag<T, N1, 1> t;
  Tag<I, N1, 1> ti;
  alignas(DEFAULT_ALIGNMENT) T data[16];
  alignas(DEFAULT_ALIGNMENT) I idx[16];
  test_utils::fill_seq(data, N);
  // Swap each pair within each word
  for (nint_t i = 0; i < N; ++i) {
    nint_t pos = i % N1;
    idx[i] = static_cast<I>((pos % M == 0) ? pos + 1 : pos - 1);
  }
  auto v  = loadu(t, data);
  auto vi = loadu(ti, idx);
  auto r  = shuf(v, vi);
  for (nint_t i = 0; i < N; ++i) {
    nint_t pos = i % N1;
    I expected_idx = static_cast<I>((pos % M == 0) ? pos + 1 : pos - 1);
    EXPECT_EQ(get(t, r, i), data[i / N1 * N1 + expected_idx]) << "i=" << i;
  }
}

// ============================================================================
// Multi-word local_shuf with scalar indices
// ============================================================================

TEST(LocalShufScalar_MultiWord, Float32_Reverse) {
  using T = float32_t;
  constexpr auto N1 = test_utils::full_vec_size<T>();
  constexpr auto N  = N1 * 2;
  constexpr auto M  = 4;
  Tag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data[32];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);
  // Scalar indices: reverse each lane in every word
  auto r = local_shuf(v, 0, 1, 2, 3);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
}

TEST(LocalShufScalar_MultiWord, Float64_Swap) {
  using T = float64_t;
  constexpr auto N1 = test_utils::full_vec_size<T>();
  constexpr auto N  = N1 * 2;
  constexpr auto M  = 2;
  Tag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data[16];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);
  auto r = local_shuf(v, 0, 1);
  for (nint_t i = 0; i < N; ++i) {
    auto lane = i / M, pos = i % M;
    EXPECT_EQ(get(t, r, i), data[lane * M + (M - 1 - pos)]) << "i=" << i;
  }
}

// ============================================================================
// Corner cases
// ============================================================================

// Verify 128-bit shuf degenerates to local_shuf (single lane per word)
TEST(ShufCornerCase, SingleLaneEqualsLocalShuf) {
  // For a 128-bit float32 word (4 elements, 1 lane), shuf == local_shuf
  // because there is only one lane. We test this by applying an arbitrary
  // permutation with both and checking equality.
  using T = float32_t;
  using I = int32_t;
  constexpr auto N = test_utils::full_vec_size<T>();

  // If N == 4 (VEC_WIDTH=128), we have exactly one lane
  if (N == 4) {
    FixedTag<T, N> t;
    FixedTag<I, N> ti;
    alignas(DEFAULT_ALIGNMENT) T data[4] = {10, 20, 30, 40};
    alignas(DEFAULT_ALIGNMENT) I idx[4]  = {2, 0, 3, 1};
    auto v  = loadu(t, data);
    auto vi = loadu(ti, idx);
    auto r_local = local_shuf(v, vi);
    auto r_shuf  = shuf(v, vi);
    for (nint_t i = 0; i < N; ++i)
      EXPECT_EQ(get(t, r_local, i), get(t, r_shuf, i)) << "i=" << i;
  } else {
    // For wider words, we test on a sub-word-sized vector (single lane)
    FixedTag<T, 4> t;
    FixedTag<I, 4> ti;
    alignas(DEFAULT_ALIGNMENT) T data[4] = {10, 20, 30, 40};
    alignas(DEFAULT_ALIGNMENT) I idx[4]  = {2, 0, 3, 1};
    auto v  = loadu(t, data);
    auto vi = loadu(ti, idx);
    auto r_local = local_shuf(v, vi);
    auto r_shuf  = shuf(v, vi);
    for (nint_t i = 0; i < 4; ++i)
      EXPECT_EQ(get(t, r_local, i), get(t, r_shuf, i)) << "i=" << i;
  }
}

// Zero vector shuffle should produce zeros
TEST(ShufCornerCase, ZeroVector) {
  using T = int32_t;
  using I = int32_t;
  constexpr auto N = test_utils::full_vec_size<T>();
  FixedTag<T, N> t;
  FixedTag<I, N> ti;
  auto v = zeros(t);
  alignas(DEFAULT_ALIGNMENT) I idx[16];
  // Arbitrary permutation indices
  for (nint_t i = 0; i < N; ++i) idx[i] = static_cast<I>((N - 1 - i) % N);
  auto vi = loadu(ti, idx);
  auto r = shuf(v, vi);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), T(0)) << "i=" << i;
}

// shuf with broadcast index: all positions select element 0 of each word
TEST(ShufCornerCase, BroadcastFirst) {
  using T = float32_t;
  using I = int32_t;
  constexpr auto N = test_utils::full_vec_size<T>();
  FixedTag<T, N> t;
  FixedTag<I, N> ti;
  alignas(DEFAULT_ALIGNMENT) T data[16];
  alignas(DEFAULT_ALIGNMENT) I idx[16];
  test_utils::fill_seq(data, N);
  // All indices point to position 0 within each word
  for (nint_t i = 0; i < N; ++i) idx[i] = 0;
  auto v  = loadu(t, data);
  auto vi = loadu(ti, idx);
  auto r  = shuf(v, vi);
  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, r, i), data[0]) << "i=" << i;
}

// ============================================================================
// upper / lower tests
// ============================================================================

// All 12 element types
using AllTypes = ::testing::Types<
    float32_t, float64_t,
    int8_t, uint8_t,
    int16_t, uint16_t,
    int32_t, uint32_t,
    int64_t, uint64_t,
    bfloat16_t, float16_t>;

template <typename T> class UpperLower : public ::testing::Test {};
TYPED_TEST_SUITE(UpperLower, AllTypes);

TYPED_TEST(UpperLower, SingleWord) {
  using T = TypeParam;
  constexpr auto N = test_utils::full_vec_size<T>();
  FixedTag<T, N> t;
  Half<decltype(t)> th;
  alignas(DEFAULT_ALIGNMENT) T data[64];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);

  auto lo = lower(t, v);
  auto hi = upper(t, v);

  // Verify lower part
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(th, lo, i), data[i]) << "lower i=" << i;

  // Verify upper part
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(th, hi, i), data[i + N / 2]) << "upper i=" << i;
}

TYPED_TEST(UpperLower, MultiWord) {
  using T = TypeParam;
  constexpr auto N1 = test_utils::full_vec_size<T>();
  constexpr auto N = N1 * 2;
  Tag<T, N> t;
  Half<decltype(t)> th;
  alignas(DEFAULT_ALIGNMENT) T data[128];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);

  auto lo = lower(t, v);
  auto hi = upper(t, v);

  // Verify lower part (first half)
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(th, lo, i), data[i]) << "lower i=" << i;

  // Verify upper part (second half)
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(th, hi, i), data[i + N / 2]) << "upper i=" << i;
}

// ============================================================================
// even / odd tests
// ============================================================================

template <typename T> class EvenOdd : public ::testing::Test {};
TYPED_TEST_SUITE(EvenOdd, AllTypes);

TYPED_TEST(EvenOdd, SingleWord) {
  using T = TypeParam;
  constexpr auto N = test_utils::full_vec_size<T>();
  FixedTag<T, N> t;
  Half<decltype(t)> th;
  alignas(DEFAULT_ALIGNMENT) T data[64];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);

  auto ev = even(t, v);
  auto od = odd(t, v);

  // Verify even elements: [0, 2, 4, ...]
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(th, ev, i), data[2 * i]) << "even i=" << i;

  // Verify odd elements: [1, 3, 5, ...]
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(th, od, i), data[2 * i + 1]) << "odd i=" << i;
}

TYPED_TEST(EvenOdd, MultiWord) {
  using T = TypeParam;
  constexpr auto N1 = test_utils::full_vec_size<T>();
  constexpr auto N = N1 * 2;
  Tag<T, N> t;
  Half<decltype(t)> th;
  alignas(DEFAULT_ALIGNMENT) T data[128];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);

  auto ev = even(t, v);
  auto od = odd(t, v);

  // Verify even elements across all words
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(th, ev, i), data[2 * i]) << "even i=" << i;

  // Verify odd elements across all words
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(th, od, i), data[2 * i + 1]) << "odd i=" << i;
}

// ============================================================================
// concat tests
// ============================================================================

template <typename T> class Concat : public ::testing::Test {};
TYPED_TEST_SUITE(Concat, AllTypes);

TYPED_TEST(Concat, SingleWord) {
  using T = TypeParam;
  constexpr auto N = test_utils::full_vec_size<T>();
  FixedTag<T, N> t;
  Half<decltype(t)> th;
  alignas(DEFAULT_ALIGNMENT) T data_lo[32], data_hi[32];
  test_utils::fill_seq(data_lo, N / 2);
  for (nint_t i = 0; i < N / 2; ++i)
    data_hi[i] = static_cast<T>(data_lo[i] + N / 2);

  auto v_lo = loadu(th, data_lo);
  auto v_hi = loadu(th, data_hi);
  auto v = concat(t, v_lo, v_hi);

  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, v, i), data_lo[i]) << "lower i=" << i;
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, v, i + N / 2), data_hi[i]) << "upper i=" << i;
}

TYPED_TEST(Concat, MultiWord) {
  using T = TypeParam;
  constexpr auto N1 = test_utils::full_vec_size<T>();
  constexpr auto N = N1 * 2;
  Tag<T, N> t;
  Half<decltype(t)> th;
  alignas(DEFAULT_ALIGNMENT) T data_lo[64], data_hi[64];
  test_utils::fill_seq(data_lo, N / 2);
  for (nint_t i = 0; i < N / 2; ++i)
    data_hi[i] = static_cast<T>(data_lo[i] + N / 2);

  auto v_lo = loadu(th, data_lo);
  auto v_hi = loadu(th, data_hi);
  auto v = concat(t, v_lo, v_hi);

  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, v, i), data_lo[i]) << "lower i=" << i;
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, v, i + N / 2), data_hi[i]) << "upper i=" << i;
}

TYPED_TEST(Concat, RoundTrip) {
  using T = TypeParam;
  constexpr auto N1 = test_utils::full_vec_size<T>();
  constexpr auto N = N1 * 2;
  Tag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data[128];
  test_utils::fill_seq(data, N);
  auto v = loadu(t, data);

  auto lo = lower(t, v);
  auto hi = upper(t, v);
  auto v2 = concat(t, lo, hi);

  for (nint_t i = 0; i < N; ++i)
    EXPECT_EQ(get(t, v2, i), data[i]) << "i=" << i;
}

// ============================================================================
// concat_even / concat_odd tests
// ============================================================================

template <typename T> class ConcatEvenOdd : public ::testing::Test {};
TYPED_TEST_SUITE(ConcatEvenOdd, AllTypes);

TYPED_TEST(ConcatEvenOdd, SingleWord) {
  using T = TypeParam;
  constexpr auto N = test_utils::full_vec_size<T>();
  FixedTag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data_a[64], data_b[64];
  test_utils::fill_seq(data_a, N);
  for (nint_t i = 0; i < N; ++i)
    data_b[i] = static_cast<T>(data_a[i] + N);

  auto a = loadu(t, data_a);
  auto b = loadu(t, data_b);

  auto ce = concat_even(t, a, b);
  auto co = concat_odd(t, a, b);

  // concat_even: even elements of a and b
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, ce, i), data_a[2 * i]) << "concat_even lower i=" << i;
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, ce, i + N / 2), data_b[2 * i]) << "concat_even upper i=" << i;

  // concat_odd: odd elements of a and b
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, co, i), data_a[2 * i + 1]) << "concat_odd lower i=" << i;
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, co, i + N / 2), data_b[2 * i + 1]) << "concat_odd upper i=" << i;
}

TYPED_TEST(ConcatEvenOdd, MultiWord) {
  using T = TypeParam;
  constexpr auto N1 = test_utils::full_vec_size<T>();
  constexpr auto N = N1 * 2;
  Tag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data_a[128], data_b[128];
  test_utils::fill_seq(data_a, N);
  for (nint_t i = 0; i < N; ++i)
    data_b[i] = static_cast<T>(data_a[i] + N);

  auto a = loadu(t, data_a);
  auto b = loadu(t, data_b);

  auto ce = concat_even(t, a, b);
  auto co = concat_odd(t, a, b);

  // concat_even: even elements of a and b
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, ce, i), data_a[2 * i]) << "concat_even lower i=" << i;
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, ce, i + N / 2), data_b[2 * i]) << "concat_even upper i=" << i;

  // concat_odd: odd elements of a and b
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, co, i), data_a[2 * i + 1]) << "concat_odd lower i=" << i;
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(t, co, i + N / 2), data_b[2 * i + 1]) << "concat_odd upper i=" << i;
}

// ============================================================================
// local_interleave_lower / local_interleave_upper tests
// ============================================================================

template <typename T> class LocalInterleave : public ::testing::Test {};
TYPED_TEST_SUITE(LocalInterleave, AllTypes);

TYPED_TEST(LocalInterleave, SingleWord) {
  using T = TypeParam;
  constexpr auto N = test_utils::full_vec_size<T>();
  constexpr auto M = test_utils::lane_size<T>();  // elements per 16-byte lane
  FixedTag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data_a[64], data_b[64];
  test_utils::fill_seq(data_a, N);
  for (nint_t i = 0; i < N; ++i)
    data_b[i] = static_cast<T>(data_a[i] + N);

  auto a = loadu(t, data_a);
  auto b = loadu(t, data_b);

  auto lo = local_interleave_lower(a, b);
  auto hi = local_interleave_upper(a, b);

  // local_interleave_lower: interleave lower half of each 16-byte lane
  // result: [b[0], a[0], b[1], a[1], ...] within each lane
  for (nint_t lane = 0; lane < N / M; ++lane) {
    for (nint_t i = 0; i < M / 2; ++i) {
      nint_t idx = lane * M + i;
      nint_t out_idx = lane * M + 2 * i;
      EXPECT_EQ(get(t, lo, out_idx), data_a[idx]) << "lower a lane=" << lane << " i=" << i;
      EXPECT_EQ(get(t, lo, out_idx + 1), data_b[idx]) << "lower b lane=" << lane << " i=" << i;
    }
  }

  // local_interleave_upper: interleave upper half of each 16-byte lane
  for (nint_t lane = 0; lane < N / M; ++lane) {
    for (nint_t i = 0; i < M / 2; ++i) {
      nint_t idx = lane * M + M / 2 + i;
      nint_t out_idx = lane * M + 2 * i;
      EXPECT_EQ(get(t, hi, out_idx), data_a[idx]) << "upper a lane=" << lane << " i=" << i;
      EXPECT_EQ(get(t, hi, out_idx + 1), data_b[idx]) << "upper b lane=" << lane << " i=" << i;
    }
  }
}

TYPED_TEST(LocalInterleave, MultiWord) {
  using T = TypeParam;
  constexpr auto N1 = test_utils::full_vec_size<T>();
  constexpr auto N = N1 * 2;
  constexpr auto M = test_utils::lane_size<T>();
  Tag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data_a[128], data_b[128];
  test_utils::fill_seq(data_a, N);
  for (nint_t i = 0; i < N; ++i)
    data_b[i] = static_cast<T>(data_a[i] + N);

  auto a = loadu(t, data_a);
  auto b = loadu(t, data_b);

  auto lo = local_interleave_lower(a, b);
  auto hi = local_interleave_upper(a, b);

  // Verify interleaving in each lane
  for (nint_t lane = 0; lane < N / M; ++lane) {
    for (nint_t i = 0; i < M / 2; ++i) {
      nint_t idx = lane * M + i;
      nint_t out_idx = lane * M + 2 * i;
      EXPECT_EQ(get(t, lo, out_idx), data_a[idx]) << "lower a lane=" << lane << " i=" << i;
      EXPECT_EQ(get(t, lo, out_idx + 1), data_b[idx]) << "lower b lane=" << lane << " i=" << i;
    }
  }

  for (nint_t lane = 0; lane < N / M; ++lane) {
    for (nint_t i = 0; i < M / 2; ++i) {
      nint_t idx = lane * M + M / 2 + i;
      nint_t out_idx = lane * M + 2 * i;
      EXPECT_EQ(get(t, hi, out_idx), data_a[idx]) << "upper a lane=" << lane << " i=" << i;
      EXPECT_EQ(get(t, hi, out_idx + 1), data_b[idx]) << "upper b lane=" << lane << " i=" << i;
    }
  }
}

// ============================================================================
// interleave tests
// ============================================================================

template <typename T> class Interleave : public ::testing::Test {};
TYPED_TEST_SUITE(Interleave, AllTypes);

TYPED_TEST(Interleave, SingleWord) {
  using T = TypeParam;
  constexpr auto N = test_utils::full_vec_size<T>();
  FixedTag<T, N> t;
  Half<decltype(t)> th;
  alignas(DEFAULT_ALIGNMENT) T data_a[64], data_b[64];
  test_utils::fill_seq(data_a, N / 2);
  for (nint_t i = 0; i < N / 2; ++i)
    data_b[i] = static_cast<T>(data_a[i] + N / 2);

  auto a = loadu(th, data_a);
  auto b = loadu(th, data_b);

  auto v = interleave(t, a, b);

  // interleave: [..., b[1], a[1], b[0], a[0]]
  for (nint_t i = 0; i < N / 2; ++i) {
    EXPECT_EQ(get(t, v, 2 * i), data_a[i]) << "a i=" << i;
    EXPECT_EQ(get(t, v, 2 * i + 1), data_b[i]) << "b i=" << i;
  }
}

TYPED_TEST(Interleave, MultiWord) {
  using T = TypeParam;
  constexpr auto N1 = test_utils::full_vec_size<T>();
  constexpr auto N = N1 * 2;
  Tag<T, N> t;
  Half<decltype(t)> th;
  alignas(DEFAULT_ALIGNMENT) T data_a[64], data_b[64];
  test_utils::fill_seq(data_a, N / 2);
  for (nint_t i = 0; i < N / 2; ++i)
    data_b[i] = static_cast<T>(data_a[i] + N / 2);

  auto a = loadu(th, data_a);
  auto b = loadu(th, data_b);

  auto v = interleave(t, a, b);

  // interleave across entire vector
  for (nint_t i = 0; i < N / 2; ++i) {
    EXPECT_EQ(get(t, v, 2 * i), data_a[i]) << "a i=" << i;
    EXPECT_EQ(get(t, v, 2 * i + 1), data_b[i]) << "b i=" << i;
  }
}

// ============================================================================
// interleave_even / interleave_odd tests
// ============================================================================

template <typename T> class InterleaveEvenOdd : public ::testing::Test {};
TYPED_TEST_SUITE(InterleaveEvenOdd, AllTypes);

TYPED_TEST(InterleaveEvenOdd, SingleWord) {
  using T = TypeParam;
  constexpr auto N = test_utils::full_vec_size<T>();
  FixedTag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data_a[64], data_b[64];
  test_utils::fill_seq(data_a, N);
  for (nint_t i = 0; i < N; ++i)
    data_b[i] = static_cast<T>(data_a[i] + N);

  auto a = loadu(t, data_a);
  auto b = loadu(t, data_b);

  auto ie = interleave_even(a, b);
  auto io = interleave_odd(a, b);

  // interleave_even: interleave even-indexed elements
  // result[2*i] = a[2*i], result[2*i+1] = b[2*i]
  for (nint_t i = 0; i < N / 2; ++i) {
    EXPECT_EQ(get(t, ie, 2 * i), data_a[2 * i]) << "ie a i=" << i;
    EXPECT_EQ(get(t, ie, 2 * i + 1), data_b[2 * i]) << "ie b i=" << i;
  }

  // interleave_odd: interleave odd-indexed elements
  // result[2*i] = a[2*i+1], result[2*i+1] = b[2*i+1]
  for (nint_t i = 0; i < N / 2; ++i) {
    EXPECT_EQ(get(t, io, 2 * i), data_a[2 * i + 1]) << "io a i=" << i;
    EXPECT_EQ(get(t, io, 2 * i + 1), data_b[2 * i + 1]) << "io b i=" << i;
  }
}

TYPED_TEST(InterleaveEvenOdd, MultiWord) {
  using T = TypeParam;
  constexpr auto N1 = test_utils::full_vec_size<T>();
  constexpr auto N = N1 * 2;
  Tag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data_a[128], data_b[128];
  test_utils::fill_seq(data_a, N);
  for (nint_t i = 0; i < N; ++i)
    data_b[i] = static_cast<T>(data_a[i] + N);

  auto a = loadu(t, data_a);
  auto b = loadu(t, data_b);

  auto ie = interleave_even(a, b);
  auto io = interleave_odd(a, b);

  // interleave_even across multi-word vector
  for (nint_t i = 0; i < N / 2; ++i) {
    EXPECT_EQ(get(t, ie, 2 * i), data_a[2 * i]) << "ie a i=" << i;
    EXPECT_EQ(get(t, ie, 2 * i + 1), data_b[2 * i]) << "ie b i=" << i;
  }

  // interleave_odd across multi-word vector
  for (nint_t i = 0; i < N / 2; ++i) {
    EXPECT_EQ(get(t, io, 2 * i), data_a[2 * i + 1]) << "io a i=" << i;
    EXPECT_EQ(get(t, io, 2 * i + 1), data_b[2 * i + 1]) << "io b i=" << i;
  }
}

// ============================================================================
// Combined roundtrip tests
// ============================================================================

TEST(InterleaveRoundTrip, Float32) {
  using T = float32_t;
  constexpr auto N = test_utils::full_vec_size<T>();
  FixedTag<T, N> t;
  Half<decltype(t)> th;
  alignas(DEFAULT_ALIGNMENT) T data_a[16], data_b[16];
  test_utils::fill_seq(data_a, N / 2);
  for (nint_t i = 0; i < N / 2; ++i)
    data_b[i] = static_cast<T>(data_a[i] + N / 2);

  auto a = loadu(th, data_a);
  auto b = loadu(th, data_b);

  // interleave then even/odd extraction
  auto v = interleave(t, a, b);
  auto ev = even(t, v);
  auto od = odd(t, v);

  // even(interleave(a, b)) should give a
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(th, ev, i), data_a[i]) << "even i=" << i;

  // odd(interleave(a, b)) should give b
  for (nint_t i = 0; i < N / 2; ++i)
    EXPECT_EQ(get(th, od, i), data_b[i]) << "odd i=" << i;
}

TEST(LocalInterleaveRoundTrip, Float64) {
  using T = float64_t;
  constexpr auto N = test_utils::full_vec_size<T>();
  FixedTag<T, N> t;
  alignas(DEFAULT_ALIGNMENT) T data_a[8], data_b[8];
  test_utils::fill_seq(data_a, N);
  for (nint_t i = 0; i < N; ++i)
    data_b[i] = static_cast<T>(data_a[i] + N);

  auto a = loadu(t, data_a);
  auto b = loadu(t, data_b);

  // local_interleave_lower + local_interleave_upper = full interleaved vector
  auto lo = local_interleave_lower(a, b);
  auto hi = local_interleave_upper(a, b);

  // For float64 with 2 elements per lane:
  // lo = [b[0], a[0], b[2], a[2], ...] (lower half of each lane)
  // hi = [b[1], a[1], b[3], a[3], ...] (upper half of each lane)
  // These together cover all elements
  constexpr auto M = test_utils::lane_size<T>();
  for (nint_t lane = 0; lane < N / M; ++lane) {
    // Verify lower
    for (nint_t i = 0; i < M / 2; ++i) {
      nint_t idx = lane * M + i;
      EXPECT_EQ(get(t, lo, lane * M + 2 * i), data_a[idx]);
      EXPECT_EQ(get(t, lo, lane * M + 2 * i + 1), data_b[idx]);
    }
    // Verify upper
    for (nint_t i = 0; i < M / 2; ++i) {
      nint_t idx = lane * M + M / 2 + i;
      EXPECT_EQ(get(t, hi, lane * M + 2 * i), data_a[idx]);
      EXPECT_EQ(get(t, hi, lane * M + 2 * i + 1), data_b[idx]);
    }
  }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
