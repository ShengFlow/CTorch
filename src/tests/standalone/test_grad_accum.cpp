#include "Tensor.h"
#include "AutoGrad/Nodes/GradAccumulator.h"
#include "CtorchError.h"
#include "CtorchScheduler.h"
#include "src/kernels/kernels.h"
#include "test_mps_flush.h"
#include <iostream>
#include <iomanip>
#include <cmath>

static bool near(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

static bool test_mps_copy_before_flush() {
    const size_t n = 4;
    Tensor a(ShapeTag{}, {n}, DType::kFloat, DeviceType::kMPS);
    Tensor b(ShapeTag{}, {n}, DType::kFloat, DeviceType::kMPS);

    float* a_data = a.data_write<float>();
    float* b_data = b.data_write<float>();
    float expected[4];
    for (size_t i = 0; i < n; ++i) {
        a_data[i] = static_cast<float>(i + 1) * 0.1f;
        b_data[i] = static_cast<float>(i + 1) * 0.3f;
        expected[i] = a_data[i] + b_data[i];
    }

    Tensor c = Add_MPS_kernel(a, b);
    // 在 flush 之前先深拷贝 c
    Tensor d = c;
    test_mps_flush_wait();

    Tensor d_cpu = d.to(DeviceType::kCPU);
    const float* d_data = d_cpu.data_read<float>();

    std::cout << "Copy before flush result: ";
    bool ok = true;
    for (size_t i = 0; i < n; ++i) {
        std::cout << std::setprecision(4) << d_data[i] << " ";
        if (!near(d_data[i], expected[i])) ok = false;
    }
    std::cout << std::endl;
    return ok;
}

static bool test_add_mps_directly() {
    const size_t n = 4;
    Tensor a(ShapeTag{}, {n}, DType::kFloat, DeviceType::kMPS);
    Tensor b(ShapeTag{}, {n}, DType::kFloat, DeviceType::kMPS);

    float* a_data = a.data_write<float>();
    float* b_data = b.data_write<float>();
    float expected[4];
    for (size_t i = 0; i < n; ++i) {
        a_data[i] = static_cast<float>(i + 1) * 0.1f;
        b_data[i] = static_cast<float>(i + 1) * 0.3f;
        expected[i] = a_data[i] + b_data[i];
    }

    Tensor c = Add_MPS_kernel(a, b);
    test_mps_flush_wait();

    Tensor c_cpu = c.to(DeviceType::kCPU);
    const float* c_data = c_cpu.data_read<float>();

    std::cout << "Add_MPS result: ";
    bool ok = true;
    for (size_t i = 0; i < n; ++i) {
        std::cout << std::setprecision(4) << c_data[i] << " ";
        if (!near(c_data[i], expected[i])) ok = false;
    }
    std::cout << std::endl;
    return ok;
}

static bool test_device(DeviceType device, const std::string& name) {
    const size_t n = 4;

    Tensor t(ShapeTag{}, {n}, DType::kFloat, device);
    t.requires_grad(true);

    Tensor g1(ShapeTag{}, {n}, DType::kFloat, device);
    Tensor g2(ShapeTag{}, {n}, DType::kFloat, device);

    float* g1_data = g1.data_write<float>();
    float* g2_data = g2.data_write<float>();
    float expected[4];
    for (size_t i = 0; i < n; ++i) {
        g1_data[i] = static_cast<float>(i + 1) * 0.1f;
        g2_data[i] = static_cast<float>(i + 1) * 0.3f;
        expected[i] = g1_data[i] + g2_data[i];
    }

    GradAccumulator acc(t.getWeakPtr());
    acc.backward({g1, g2});

    if (device == DeviceType::kMPS) {
        test_mps_flush_wait();
    }

    Tensor grad = t.grad();
    if (grad.numel() != n) {
        std::cout << name << ": grad shape mismatch" << std::endl;
        return false;
    }

    Tensor grad_cpu = (device == DeviceType::kMPS) ? grad.to(DeviceType::kCPU) : grad;
    const float* grad_data = grad_cpu.data_read<float>();

    std::cout << name << " grad: ";
    bool ok = true;
    for (size_t i = 0; i < n; ++i) {
        std::cout << std::setprecision(4) << grad_data[i] << " ";
        if (!near(grad_data[i], expected[i])) ok = false;
    }
    std::cout << std::endl;

    return ok;
}

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    // 先初始化调度器，确保 MPSAllocator 已注册（MNIST 同样做法）
    CtorchScheduler::getInstance();

    std::cout << "=== Add_MPS direct ===" << std::endl;
    bool add_mps_ok = test_add_mps_directly();
    std::cout << "Add_MPS direct: " << (add_mps_ok ? "PASS" : "FAIL") << std::endl;

    std::cout << "=== CPU GradAccumulator ===" << std::endl;
    bool cpu_ok = test_device(DeviceType::kCPU, "CPU");

    std::cout << "=== MPS GradAccumulator ===" << std::endl;
    bool mps_ok = test_device(DeviceType::kMPS, "MPS");

    std::cout << "=== Compare ===" << std::endl;
    std::cout << "CPU: " << (cpu_ok ? "PASS" : "FAIL") << std::endl;
    std::cout << "MPS: " << (mps_ok ? "PASS" : "FAIL") << std::endl;

    if (cpu_ok && mps_ok) {
        std::cout << "All GradAccumulator checks passed." << std::endl;
        return 0;
    }
    std::cout << "GradAccumulator checks failed." << std::endl;
    return 1;
}
