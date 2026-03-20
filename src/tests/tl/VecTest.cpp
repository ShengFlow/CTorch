//
// Created by renyz on 2026/3/14.
// Non-standard test playground.
// For removal once the API is stabled.
//

#include <gtest/gtest.h>
#include "Features.h"
#undef ARCH_X86_FAMILY
#include "tl/cpu/Vec.h"

using namespace ct;
using namespace ct::tl::vec;

CT_NOINLINE
static void cpy(const float * from, float * to, nint_t len) {
  ScalableTag<float32_t> t;
  nint_t i;
  for (i = 0; i < len - (size(t) - 1); i += size(t)) {
    auto v = load(t, from + i);
    store(t, to + i, v);
  }
  for (; i < len; ++i) {
    to[i] = from[i];
  }
}

TEST(VecTest, CopyTest) {
  float* data = new float[128];
  float* out = new float[128];

  for (int i = 0; i < 128; ++i) {
    data[i] = i;
    out[i] = -1;
  }

  cpy(data, out, 128);

  EXPECT_FLOAT_EQ(out[0], 0.0f);
  EXPECT_FLOAT_EQ(out[63], 63.0f);
  EXPECT_FLOAT_EQ(out[64], 64.0f);
  EXPECT_FLOAT_EQ(out[127], 127.0f);

  delete[] data;
  delete[] out;
}

TEST(VecTest, Playground) {
}

TEST(VecTest, TagTest) {
  Tag<float32_t, 8, 1> d;
  // 添加更多测试...
}
