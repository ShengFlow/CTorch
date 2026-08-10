#include "AutoGrad.h"
#include "Tensor.h"
#include "CtorchError.h"
#include <iostream>
#include <cmath>

static float grad_l2(const Tensor& t) {
    const float* p = t.data_read<float>();
    float s = 0.0f;
    for (size_t i = 0; i < t.numel(); ++i) s += p[i] * p[i];
    return std::sqrt(s);
}

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);
    CtorchScheduler::getInstance();

    const size_t N = 6;
    Tensor x_cpu(ShapeTag{}, {N}, DType::kFloat, DeviceType::kCPU);
    float* xp = x_cpu.data_write<float>();
    for (size_t i = 0; i < N; ++i) xp[i] = static_cast<float>(i) - 2.5f; // -2.5,-1.5,-0.5,0.5,1.5,2.5

    Tensor x_mps = x_cpu.to(DeviceType::kMPS);
    x_cpu.requires_grad(true);
    x_mps.requires_grad(true);

    Tensor y_cpu = x_cpu.relu();
    Tensor y_mps = x_mps.relu();

    std::cout << "CPU relu: ";
    for (size_t i = 0; i < N; ++i) std::cout << y_cpu.data_read<float>()[i] << " ";
    std::cout << std::endl;

    MPS_flush_wait(true);
    Tensor y_mps_cpu = y_mps.to(DeviceType::kCPU);
    std::cout << "MPS relu: ";
    for (size_t i = 0; i < N; ++i) std::cout << y_mps_cpu.data_read<float>()[i] << " ";
    std::cout << std::endl;

    // 反事实实验：单独检查比较掩码
    Tensor mask_mps = (x_mps > 0);
    MPS_flush_wait(true);
    Tensor mask_cpu = mask_mps.to(DeviceType::kCPU);
    std::cout << "MPS mask (x > 0): ";
    for (size_t i = 0; i < N; ++i) std::cout << mask_cpu.data_read<float>()[i] << " ";
    std::cout << std::endl;

    // 反事实实验：用 CPU 掩码与 MPS grad_out 相乘
    Tensor ones_mps(ShapeTag{}, {N}, DType::kFloat, DeviceType::kMPS);
    ones_mps.ones();
    Tensor grad_mps_manual = mask_mps * ones_mps;
    MPS_flush_wait(true);
    Tensor grad_mps_manual_cpu = grad_mps_manual.to(DeviceType::kCPU);
    std::cout << "MPS grad (mask * ones): ";
    for (size_t i = 0; i < N; ++i) std::cout << grad_mps_manual_cpu.data_read<float>()[i] << " ";
    std::cout << std::endl;

    // 反事实实验：检查 AutoGrad 种子梯度是否被正确传递到 MPS
    {
        Tensor seed_cpu(ShapeTag{}, {N}, DType::kFloat, DeviceType::kCPU, false);
        float* sp = seed_cpu.data_write<float>();
        for (size_t i = 0; i < N; ++i) sp[i] = 1.0f;
        Tensor seed_mps = seed_cpu.to(DeviceType::kMPS);
        MPS_flush_wait(true);
        Tensor seed_mps_cpu = seed_mps.to(DeviceType::kCPU);
        std::cout << "seed MPS: ";
        for (size_t i = 0; i < N; ++i) std::cout << seed_mps_cpu.data_read<float>()[i] << " ";
        std::cout << std::endl;
    }

    AutoGrad::backward(y_cpu.getRelatedNode(), false);
    AutoGrad::backward(y_mps.getRelatedNode(), false);

    MPS_flush_wait(true);
    Tensor g_mps_cpu = x_mps.grad().to(DeviceType::kCPU);

    std::cout << "CPU grad: ";
    for (size_t i = 0; i < N; ++i) std::cout << x_cpu.grad().data_read<float>()[i] << " ";
    std::cout << std::endl;

    std::cout << "MPS grad: ";
    for (size_t i = 0; i < N; ++i) std::cout << g_mps_cpu.data_read<float>()[i] << " ";
    std::cout << std::endl;

    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        if (std::fabs(x_cpu.grad().data_read<float>()[i] - g_mps_cpu.data_read<float>()[i]) > 1e-4f) ok = false;
    }
    std::cout << "ReLU backward CPU vs MPS: " << (ok ? "MATCH" : "MISMATCH") << std::endl;
    return ok ? 0 : 1;
}
