#include "Tensor.h"
#include "AutoGrad.h"
#include <chrono>
#include <iostream>
#include <cstdlib>

static inline double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

int main() {
    using namespace ct;
    const int M = 512, K = 512;

    auto x_base = Tensor(ShapeTag{}, {M, K}, DType::kFloat, DeviceType::kCPU);
    float* xp = x_base.data_write<float>();
    uint32_t rng = 0xdeadbeef;
    for (int i = 0; i < M*K; ++i) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        xp[i] = ((double)rng / 4294967296.0) * 4.0f - 2.0f;
    }

    // ====== CASE A: Forward only (x.requires_grad(false)) ======
    std::cout << "=== [CASE A] Forward only (no autograd) 前10次 ===" << std::endl;
    for (int i = 0; i < 10; ++i) {
        Tensor x = x_base.clone();
        double t0 = now_ms();
        Tensor y = x.tanh().sigmoid().relu();
        double t1 = now_ms();
        std::cout << "  iter" << i << ": fwd_only=" << (t1-t0) << " ms" << std::endl;
        (void)y.numel();
    }

    // ====== CASE B: Forward + Backward split time 前10次 ======
    std::cout << "\n=== [CASE B] Forward + Backward, split time 前10次 ===" << std::endl;
    for (int i = 0; i < 10; ++i) {
        Tensor x = x_base.clone();
        x.requires_grad(true);
        double t0 = now_ms();
        Tensor y = x.tanh().sigmoid().relu();
        double t1 = now_ms();
        AutoGrad::backward(y.getRelatedNode(), false);
        double t2 = now_ms();
        double fwd_ms = t1 - t0;
        double bwd_ms = t2 - t1;
        double tot_ms = t2 - t0;
        printf("  iter%d: fwd=%.3f ms  bwd=%.3f ms  total=%.3f ms\n",
               i, fwd_ms, bwd_ms, tot_ms);
    }

    return 0;
}
