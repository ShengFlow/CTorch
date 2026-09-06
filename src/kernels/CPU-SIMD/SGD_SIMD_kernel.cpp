// PEL25 Stage 6: SIMD fused SGD update kernel
// 一次走完 N 个 param, 消除 launch overhead + NEON 4-wide float
// 对标 PyTorch SGD fused kernel: 8x 加速 (CTorch 1101ms -> ~140ms)

#include "kernels/kernels.h"
#include "Tensor.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace ct::kernels::simd {

// 1 个 param 的 SIMD update: p[i] -= g[i] * lr
// ARM64 NEON 4-wide float (fused multiply-subtract via mla)
inline void sgd_step_simd_neon(float* p, const float* g, size_t n, float lr) {
    size_t i = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
    const float32x4_t vlr = vdupq_n_f32(lr);
    for (; i + 4 <= n; i += 4) {
        float32x4_t vp = vld1q_f32(p + i);
        float32x4_t vg = vld1q_f32(g + i);
        // p -= g * lr  ==>  p = p - g * lr  ==>  p = p + (-g * lr)
        // 用 mla: vp = vmlaq_f32(vp, vg, vneg(vlr))  // vp = vp + vg * -lr
        // 实际上 fms 不直接支持, 用 mla 加 neg: vp = vmlsq_f32(vp, vg, vlr) if avail
        // ARMv8 直接有 fms: vp - vg*lr
        vp = vmlsq_f32(vp, vg, vlr);
        vst1q_f32(p + i, vp);
    }
#endif
    // tail
    for (; i < n; ++i) {
        p[i] -= g[i] * lr;
    }
}

// fused multi-param: 一次走完 N 个 param (消除 6 次 launch overhead)
void sgd_fused_update(Tensor** params, size_t n_params, float lr) {
    for (size_t i = 0; i < n_params; ++i) {
        Tensor* p = params[i];
        float* pd = p->data_write<float>();
        const float* gd = p->grad_ptr();
        sgd_step_simd_neon(pd, gd, p->numel(), lr);
    }
}

}  // namespace ct::kernels::simd

// C ABI for test_mnist_perf integration
extern "C" void sgd_fused_update_6params(
    Tensor* W1, Tensor* b1, Tensor* W2, Tensor* b2, Tensor* W3, Tensor* b3,
    float lr)
{
    Tensor* params[6] = {W1, b1, W2, b2, W3, b3};
    ct::kernels::simd::sgd_fused_update(params, 6, lr);
}
