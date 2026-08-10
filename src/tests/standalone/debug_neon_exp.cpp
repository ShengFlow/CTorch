#include <cstdio>
#include <cmath>
#include <arm_neon.h>
#include "kernels/SIMDMath.h"

int main() {
    using namespace ct::kernels::simd;

    // 直接测 NEON
    float vals[4] = {0.0f, 1.0f, 2.0f, -1.0f};
    float32x4_t x = vld1q_f32(vals);
    float32x4_t r = exp_neon_f32(x);
    float out[4];
    vst1q_f32(out, r);
    for (int i = 0; i < 4; ++i) {
        printf("neon exp(%f) = %f  ref=%f\n", vals[i], out[i], std::exp(vals[i]));
    }
    return 0;
}
