#include "AutoGrad.h"
#include "Tensor.h"
#include "CtorchError.h"
#include "src/kernels/kernels.h"
#include <iostream>
#include <cmath>

static bool near(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

static bool run_sgd_sync_test(bool flush_before_sgd) {
    const size_t N = 500000;
    Tensor a(ShapeTag{}, {N}, DType::kFloat, DeviceType::kMPS);
    Tensor b(ShapeTag{}, {N}, DType::kFloat, DeviceType::kMPS);
    Tensor param(ShapeTag{}, {N}, DType::kFloat, DeviceType::kMPS);

    float* ap = a.data_write<float>();
    float* bp = b.data_write<float>();
    float* pp = param.data_write<float>();
    for (size_t i = 0; i < N; ++i) {
        ap[i] = 1.0f;
        bp[i] = 2.0f;
        pp[i] = 10.0f;
    }

    // grad = a + b = 3.0，通过 accumulator 异步派发
    Tensor grad = a + b;

    if (flush_before_sgd) {
        MPS_flush_wait(true);
    }

    // SGD 使用独立 command buffer；若未 flush accumulator，grad 可能尚未写入
    SGD_Step_MPS_kernel(param, grad, 1.0f);

    // 读回验证
    Tensor param_cpu = param.to(DeviceType::kCPU);
    const float* pcpu = param_cpu.data_read<float>();

    float expected = 10.0f - 3.0f * 1.0f; // 7.0
    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        if (!near(pcpu[i], expected)) {
            ok = false;
            if (i < 5) {
                std::cout << "  idx " << i << ": got " << pcpu[i] << ", expected " << expected << std::endl;
            }
            break;
        }
    }
    return ok;
}

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);
    CtorchScheduler::getInstance();

    std::cout << "=== SGD sync test: WITHOUT explicit flush ===" << std::endl;
    bool without_flush = run_sgd_sync_test(false);
    std::cout << "Result: " << (without_flush ? "PASS" : "FAIL") << std::endl;

    std::cout << "\n=== SGD sync test: WITH explicit flush ===" << std::endl;
    bool with_flush = run_sgd_sync_test(true);
    std::cout << "Result: " << (with_flush ? "PASS" : "FAIL") << std::endl;

    if (with_flush && !without_flush) {
        std::cout << "\nEvidence: SGD_Step_MPS_kernel reads grad before accumulator completes." << std::endl;
        return 0;
    }
    if (with_flush && without_flush) {
        std::cout << "\nBoth passed; no sync issue reproduced in this test." << std::endl;
        return 0;
    }
    std::cout << "\nUnexpected result: flush case failed." << std::endl;
    return 1;
}
