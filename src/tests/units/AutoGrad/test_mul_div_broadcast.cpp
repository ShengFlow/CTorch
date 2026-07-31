#include "AutoGrad.h"
#include "Tensor.h"
#include "CtorchError.h"
#include "CtorchScheduler.h"
#include <iostream>
#include <cmath>

namespace {

DeviceType g_device = DeviceType::kCPU;

const char* deviceName(DeviceType dev) {
    switch (dev) {
        case DeviceType::kCPU: return "CPU";
        case DeviceType::kCUDA: return "CUDA";
        case DeviceType::kMPS: return "MPS";
        case DeviceType::kAMX: return "AMX";
        case DeviceType::kSIMD: return "SIMD";
        case DeviceType::kUNKNOWN: return "Unknown";
        case DeviceType::kGENERAL: return "General";
        case DeviceType::kCount: return "Count";
        default: return "Unknown";
    }
}

void syncDevice(DeviceType dev) {
#ifdef __APPLE__
    if (dev == DeviceType::kMPS) {
        MPS_flush_wait(true);
    }
#endif
}

bool near(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) <= eps;
}

bool test_mul_broadcast_scalar_vector() {
    try {
        std::cout << "=== [" << deviceName(g_device) << "] 测试 MulNode 广播: 标量 × 向量 ===" << std::endl;

        Tensor a(2.0f, g_device);
        Tensor b(ShapeTag{}, {3}, DType::kFloat, g_device);
        b.data_write<float>()[0] = 1.0f;
        b.data_write<float>()[1] = 2.0f;
        b.data_write<float>()[2] = 3.0f;

        a.requires_grad(true);
        b.requires_grad(true);

        Tensor c = a * b;
        syncDevice(g_device);

        if (!near(c.data_read<float>()[0], 2.0f) ||
            !near(c.data_read<float>()[1], 4.0f) ||
            !near(c.data_read<float>()[2], 6.0f)) {
            std::cout << "❌ 前向计算错误" << std::endl;
            return false;
        }

        AutoGrad::backward(c.getRelatedNode(), false);
        syncDevice(g_device);

        Tensor grad_a = a.grad();
        Tensor grad_b = b.grad();

        float expected_grad_a = 1.0f + 2.0f + 3.0f;
        if (!near(grad_a.data_read<float>()[0], expected_grad_a)) {
            std::cout << "❌ grad_a 错误: 期望 " << expected_grad_a << ", 实际 " << grad_a.data_read<float>()[0] << std::endl;
            return false;
        }

        if (!near(grad_b.data_read<float>()[0], 2.0f) ||
            !near(grad_b.data_read<float>()[1], 2.0f) ||
            !near(grad_b.data_read<float>()[2], 2.0f)) {
            std::cout << "❌ grad_b 错误" << std::endl;
            return false;
        }

        std::cout << "✅ MulNode 标量×向量广播测试通过!" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cout << "!!! 测试异常: " << e.what() << std::endl;
        return false;
    }
}

bool test_mul_broadcast_scalar_matrix() {
    try {
        std::cout << "\n=== [" << deviceName(g_device) << "] 测试 MulNode 广播: 标量 × 矩阵 ===" << std::endl;

        Tensor a(2.0f, g_device);
        Tensor b(ShapeTag{}, {2, 3}, DType::kFloat, g_device);
        b.data_write<float>()[0] = 1.0f; b.data_write<float>()[1] = 2.0f; b.data_write<float>()[2] = 3.0f;
        b.data_write<float>()[3] = 4.0f; b.data_write<float>()[4] = 5.0f; b.data_write<float>()[5] = 6.0f;

        a.requires_grad(true);
        b.requires_grad(true);

        Tensor c = a * b;
        syncDevice(g_device);

        for (int i = 0; i < 6; ++i) {
            if (!near(c.data_read<float>()[i], static_cast<float>(i + 1) * 2.0f)) {
                std::cout << "❌ 前向计算错误" << std::endl;
                return false;
            }
        }

        AutoGrad::backward(c.getRelatedNode(), false);
        syncDevice(g_device);

        Tensor grad_a = a.grad();
        Tensor grad_b = b.grad();

        float expected_grad_a = 1.0f + 2.0f + 3.0f + 4.0f + 5.0f + 6.0f;
        if (!near(grad_a.data_read<float>()[0], expected_grad_a)) {
            std::cout << "❌ grad_a 错误: 期望 " << expected_grad_a << ", 实际 " << grad_a.data_read<float>()[0] << std::endl;
            return false;
        }

        for (int i = 0; i < 6; i++) {
            if (!near(grad_b.data_read<float>()[i], 2.0f)) {
                std::cout << "❌ grad_b[" << i << "] 错误: 期望 2.0, 实际 " << grad_b.data_read<float>()[i] << std::endl;
                return false;
            }
        }

        std::cout << "✅ MulNode 标量×矩阵广播测试通过!" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cout << "!!! 测试异常: " << e.what() << std::endl;
        return false;
    }
}

bool test_div_broadcast_scalar_vector() {
    try {
        std::cout << "\n=== [" << deviceName(g_device) << "] 测试 DivNode 广播: 向量 ÷ 标量 ===" << std::endl;

        Tensor a(ShapeTag{}, {3}, DType::kFloat, g_device);
        a.data_write<float>()[0] = 6.0f;
        a.data_write<float>()[1] = 8.0f;
        a.data_write<float>()[2] = 10.0f;

        Tensor b(2.0f, g_device);

        a.requires_grad(true);
        b.requires_grad(true);

        Tensor c = a / b;
        syncDevice(g_device);

        if (!near(c.data_read<float>()[0], 3.0f) ||
            !near(c.data_read<float>()[1], 4.0f) ||
            !near(c.data_read<float>()[2], 5.0f)) {
            std::cout << "❌ 前向计算错误" << std::endl;
            return false;
        }

        AutoGrad::backward(c.getRelatedNode(), false);
        syncDevice(g_device);

        Tensor grad_a = a.grad();
        Tensor grad_b = b.grad();

        float expected_grad_b = -(6.0f/(2.0f*2.0f) + 8.0f/(2.0f*2.0f) + 10.0f/(2.0f*2.0f));

        for (int i = 0; i < 3; i++) {
            if (!near(grad_a.data_read<float>()[i], 0.5f)) {
                std::cout << "❌ grad_a[" << i << "] 错误: 期望 0.5, 实际 " << grad_a.data_read<float>()[i] << std::endl;
                return false;
            }
        }

        if (!near(grad_b.data_read<float>()[0], expected_grad_b)) {
            std::cout << "❌ grad_b 错误: 期望 " << expected_grad_b << ", 实际 " << grad_b.data_read<float>()[0] << std::endl;
            return false;
        }

        std::cout << "✅ DivNode 向量÷标量广播测试通过!" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cout << "!!! 测试异常: " << e.what() << std::endl;
        return false;
    }
}

bool test_div_broadcast_vector_scalar() {
    try {
        std::cout << "\n=== [" << deviceName(g_device) << "] 测试 DivNode 广播: 标量 ÷ 向量 ===" << std::endl;

        Tensor a(12.0f, g_device);
        Tensor b(ShapeTag{}, {3}, DType::kFloat, g_device);
        b.data_write<float>()[0] = 2.0f;
        b.data_write<float>()[1] = 3.0f;
        b.data_write<float>()[2] = 4.0f;

        a.requires_grad(true);
        b.requires_grad(true);

        Tensor c = a / b;
        syncDevice(g_device);

        if (!near(c.data_read<float>()[0], 6.0f) ||
            !near(c.data_read<float>()[1], 4.0f) ||
            !near(c.data_read<float>()[2], 3.0f)) {
            std::cout << "❌ 前向计算错误" << std::endl;
            return false;
        }

        AutoGrad::backward(c.getRelatedNode(), false);
        syncDevice(g_device);

        Tensor grad_a = a.grad();
        Tensor grad_b = b.grad();

        float expected_grad_a = 1.0f/2.0f + 1.0f/3.0f + 1.0f/4.0f;

        if (!near(grad_a.data_read<float>()[0], expected_grad_a)) {
            std::cout << "❌ grad_a 错误: 期望 " << expected_grad_a << ", 实际 " << grad_a.data_read<float>()[0] << std::endl;
            return false;
        }

        if (!near(grad_b.data_read<float>()[0], -12.0f/(2.0f*2.0f)) ||
            !near(grad_b.data_read<float>()[1], -12.0f/(3.0f*3.0f)) ||
            !near(grad_b.data_read<float>()[2], -12.0f/(4.0f*4.0f))) {
            std::cout << "❌ grad_b 错误" << std::endl;
            return false;
        }

        std::cout << "✅ DivNode 标量÷向量广播测试通过!" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cout << "!!! 测试异常: " << e.what() << std::endl;
        return false;
    }
}

bool run_all_tests() {
    int passed = 0;
    int total = 4;

    if (test_mul_broadcast_scalar_vector()) ++passed;
    if (test_mul_broadcast_scalar_matrix()) ++passed;
    if (test_div_broadcast_scalar_vector()) ++passed;
    if (test_div_broadcast_vector_scalar()) ++passed;

    std::cout << "\n[" << deviceName(g_device) << "] 测试结果: " << passed << "/" << total << " 通过" << std::endl;
    return passed == total;
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    std::cout << "🚀 开始 MulNode 和 DivNode 跨后端广播梯度测试" << std::endl;
    std::cout << "====================================" << std::endl;

    CtorchScheduler::getInstance();

    const DeviceType devices[] = {
        DeviceType::kCPU,
        DeviceType::kSIMD,
        DeviceType::kAMX,
        DeviceType::kMPS
    };

    int device_passed = 0;
    int device_total = 0;

    for (DeviceType dev : devices) {
        if (!CtorchScheduler::isDeviceAvailable(dev)) {
            std::cout << "[SKIP] 设备不可用: " << deviceName(dev) << std::endl;
            continue;
        }
        g_device = dev;
        std::cout << "\n--- 设备: " << deviceName(dev) << " ---" << std::endl;
        ++device_total;
        if (run_all_tests()) {
            ++device_passed;
        }
    }

    std::cout << "\n====================================" << std::endl;
    std::cout << "设备通过: " << device_passed << "/" << device_total << std::endl;

    if (device_passed == device_total) {
        std::cout << "🎉 所有可用后端测试通过!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ 部分后端测试失败" << std::endl;
        return 1;
    }
}
