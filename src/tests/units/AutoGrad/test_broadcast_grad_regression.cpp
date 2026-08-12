/**
 * @file test_broadcast_grad_regression.cpp
 * @brief 广播梯度回归测试
 * @details 覆盖 Add/Sub/Mul/Div 在标量/向量/矩阵广播场景下的前向与反向梯度正确性，
 *          在 CPU/SIMD/AMX/MPS 可用后端上参数化运行。用于防止广播维度规约错误回归。
 */

#include "AutoGrad.h"
#include "Tensor.h"
#include "CtorchError.h"
#include "CtorchScheduler.h"
#include "C3/C3Cleanup.h"
#include <iostream>
#include <cmath>
#include <vector>

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

Tensor makeScalar(float v) {
    return Tensor(v, g_device);
}

Tensor makeVector(const std::vector<float>& vals) {
    Tensor t(ShapeTag{}, {vals.size()}, DType::kFloat, g_device);
    float* p = t.data_write<float>();
    for (size_t i = 0; i < vals.size(); ++i) {
        p[i] = vals[i];
    }
    return t;
}

Tensor makeMatrix(const std::vector<float>& vals, size_t rows, size_t cols) {
    Tensor t(ShapeTag{}, {rows, cols}, DType::kFloat, g_device);
    float* p = t.data_write<float>();
    for (size_t i = 0; i < vals.size(); ++i) {
        p[i] = vals[i];
    }
    return t;
}

Tensor makeTensor3D(const std::vector<float>& vals, size_t d0, size_t d1, size_t d2) {
    Tensor t(ShapeTag{}, {d0, d1, d2}, DType::kFloat, g_device);
    float* p = t.data_write<float>();
    for (size_t i = 0; i < vals.size(); ++i) {
        p[i] = vals[i];
    }
    return t;
}

// ---------------- Add ----------------

bool test_add_vector_scalar_broadcast() {
    try {
        std::cout << "=== [" << deviceName(g_device) << "] Add: 向量 + 标量 ===" << std::endl;
        Tensor a = makeVector({1.0f, 2.0f, 3.0f});
        Tensor b = makeScalar(10.0f);
        a.requires_grad(true);
        b.requires_grad(true);

        Tensor c = a + b;
        syncDevice(g_device);
        const float* cp = c.data_read<float>();
        if (!near(cp[0], 11.0f) || !near(cp[1], 12.0f) || !near(cp[2], 13.0f)) {
            std::cout << "❌ 前向错误" << std::endl;
            return false;
        }

        AutoGrad::backward(c.getRelatedNode(), false);
        syncDevice(g_device);

        const float* ga = a.grad().data_read<float>();
        const float* gb = b.grad().data_read<float>();
        if (!near(ga[0], 1.0f) || !near(ga[1], 1.0f) || !near(ga[2], 1.0f)) {
            std::cout << "❌ grad_a 错误" << std::endl;
            return false;
        }
        if (!near(gb[0], 3.0f)) {
            std::cout << "❌ grad_b 错误: 期望 3.0, 实际 " << gb[0] << std::endl;
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        std::cout << "!!! 异常: " << e.what() << std::endl;
        return false;
    }
}

bool test_add_matrix_vector_broadcast() {
    try {
        std::cout << "=== [" << deviceName(g_device) << "] Add: 矩阵 + 向量 ===" << std::endl;
        Tensor a = makeMatrix({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, 2, 3);
        Tensor b = makeVector({10.0f, 20.0f, 30.0f});
        a.requires_grad(true);
        b.requires_grad(true);

        Tensor c = a + b;
        syncDevice(g_device);
        const float* cp = c.data_read<float>();
        float expected_c[] = {11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f};
        for (int i = 0; i < 6; ++i) {
            if (!near(cp[i], expected_c[i])) {
                std::cout << "❌ 前向错误 c[" << i << "]" << std::endl;
                return false;
            }
        }

        AutoGrad::backward(c.getRelatedNode(), false);
        syncDevice(g_device);

        const float* ga = a.grad().data_read<float>();
        const float* gb = b.grad().data_read<float>();
        for (int i = 0; i < 6; ++i) {
            if (!near(ga[i], 1.0f)) {
                std::cout << "❌ grad_a[" << i << "] 错误" << std::endl;
                return false;
            }
        }
        // grad_b 沿被广播的维度 0 求和
        if (!near(gb[0], 2.0f) || !near(gb[1], 2.0f) || !near(gb[2], 2.0f)) {
            std::cout << "❌ grad_b 错误: 期望 [2,2,2], 实际 [" << gb[0] << "," << gb[1] << "," << gb[2] << "]" << std::endl;
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        std::cout << "!!! 异常: " << e.what() << std::endl;
        return false;
    }
}

// ---------------- Sub ----------------

bool test_sub_vector_scalar_broadcast() {
    try {
        std::cout << "=== [" << deviceName(g_device) << "] Sub: 向量 - 标量 ===" << std::endl;
        Tensor a = makeVector({5.0f, 7.0f, 9.0f});
        Tensor b = makeScalar(2.0f);
        a.requires_grad(true);
        b.requires_grad(true);

        Tensor c = a - b;
        syncDevice(g_device);
        const float* cp = c.data_read<float>();
        if (!near(cp[0], 3.0f) || !near(cp[1], 5.0f) || !near(cp[2], 7.0f)) {
            std::cout << "❌ 前向错误" << std::endl;
            return false;
        }

        AutoGrad::backward(c.getRelatedNode(), false);
        syncDevice(g_device);

        const float* ga = a.grad().data_read<float>();
        const float* gb = b.grad().data_read<float>();
        if (!near(ga[0], 1.0f) || !near(ga[1], 1.0f) || !near(ga[2], 1.0f)) {
            std::cout << "❌ grad_a 错误" << std::endl;
            return false;
        }
        if (!near(gb[0], -3.0f)) {
            std::cout << "❌ grad_b 错误: 期望 -3.0, 实际 " << gb[0] << std::endl;
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        std::cout << "!!! 异常: " << e.what() << std::endl;
        return false;
    }
}

bool test_sub_scalar_vector_broadcast() {
    try {
        std::cout << "=== [" << deviceName(g_device) << "] Sub: 标量 - 向量 ===" << std::endl;
        Tensor a = makeScalar(10.0f);
        Tensor b = makeVector({1.0f, 2.0f, 3.0f});
        a.requires_grad(true);
        b.requires_grad(true);

        Tensor c = a - b;
        syncDevice(g_device);
        const float* cp = c.data_read<float>();
        if (!near(cp[0], 9.0f) || !near(cp[1], 8.0f) || !near(cp[2], 7.0f)) {
            std::cout << "❌ 前向错误" << std::endl;
            return false;
        }

        AutoGrad::backward(c.getRelatedNode(), false);
        syncDevice(g_device);

        const float* ga = a.grad().data_read<float>();
        const float* gb = b.grad().data_read<float>();
        if (!near(ga[0], 3.0f)) {
            std::cout << "❌ grad_a 错误: 期望 3.0, 实际 " << ga[0] << std::endl;
            return false;
        }
        if (!near(gb[0], -1.0f) || !near(gb[1], -1.0f) || !near(gb[2], -1.0f)) {
            std::cout << "❌ grad_b 错误" << std::endl;
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        std::cout << "!!! 异常: " << e.what() << std::endl;
        return false;
    }
}

// ---------------- Mul ----------------

bool test_mul_matrix_vector_broadcast() {
    try {
        std::cout << "=== [" << deviceName(g_device) << "] Mul: 矩阵 × 向量 ===" << std::endl;
        Tensor a = makeMatrix({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, 2, 3);
        Tensor b = makeVector({2.0f, 3.0f, 4.0f});
        a.requires_grad(true);
        b.requires_grad(true);

        Tensor c = a * b;
        syncDevice(g_device);
        const float* cp = c.data_read<float>();
        float expected_c[] = {2.0f, 6.0f, 12.0f, 8.0f, 15.0f, 24.0f};
        for (int i = 0; i < 6; ++i) {
            if (!near(cp[i], expected_c[i])) {
                std::cout << "❌ 前向错误 c[" << i << "]" << std::endl;
                return false;
            }
        }

        AutoGrad::backward(c.getRelatedNode(), false);
        syncDevice(g_device);

        const float* ga = a.grad().data_read<float>();
        const float* gb = b.grad().data_read<float>();
        float expected_ga[] = {2.0f, 3.0f, 4.0f, 2.0f, 3.0f, 4.0f};
        for (int i = 0; i < 6; ++i) {
            if (!near(ga[i], expected_ga[i])) {
                std::cout << "❌ grad_a[" << i << "] 错误" << std::endl;
                return false;
            }
        }
        // grad_b 沿被广播维度 0 求和: [1+4, 2+5, 3+6]
        if (!near(gb[0], 5.0f) || !near(gb[1], 7.0f) || !near(gb[2], 9.0f)) {
            std::cout << "❌ grad_b 错误: 期望 [5,7,9], 实际 [" << gb[0] << "," << gb[1] << "," << gb[2] << "]" << std::endl;
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        std::cout << "!!! 异常: " << e.what() << std::endl;
        return false;
    }
}

// ---------------- Div ----------------

bool test_div_matrix_vector_broadcast() {
    try {
        std::cout << "=== [" << deviceName(g_device) << "] Div: 矩阵 ÷ 向量 ===" << std::endl;
        Tensor a = makeMatrix({2.0f, 6.0f, 12.0f, 20.0f, 30.0f, 42.0f}, 2, 3);
        Tensor b = makeVector({2.0f, 3.0f, 4.0f});
        a.requires_grad(true);
        b.requires_grad(true);

        Tensor c = a / b;
        syncDevice(g_device);
        const float* cp = c.data_read<float>();
        float expected_c[] = {1.0f, 2.0f, 3.0f, 10.0f, 10.0f, 10.5f};
        for (int i = 0; i < 6; ++i) {
            if (!near(cp[i], expected_c[i])) {
                std::cout << "❌ 前向错误 c[" << i << "]" << std::endl;
                return false;
            }
        }

        AutoGrad::backward(c.getRelatedNode(), false);
        syncDevice(g_device);

        const float* ga = a.grad().data_read<float>();
        const float* gb = b.grad().data_read<float>();
        float expected_ga[] = {0.5f, 1.0f/3.0f, 0.25f, 0.5f, 1.0f/3.0f, 0.25f};
        for (int i = 0; i < 6; ++i) {
            if (!near(ga[i], expected_ga[i])) {
                std::cout << "❌ grad_a[" << i << "] 错误" << std::endl;
                return false;
            }
        }
        // grad_b_i = -sum_j a_ji / b_i^2
        if (!near(gb[0], -(2.0f + 20.0f) / 4.0f) ||
            !near(gb[1], -(6.0f + 30.0f) / 9.0f) ||
            !near(gb[2], -(12.0f + 42.0f) / 16.0f)) {
            std::cout << "❌ grad_b 错误" << std::endl;
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        std::cout << "!!! 异常: " << e.what() << std::endl;
        return false;
    }
}

// ---------------- Combined ----------------

bool test_combined_broadcast_chain() {
    try {
        std::cout << "=== [" << deviceName(g_device) << "] 组合: (向量 + 标量) × 标量 ===" << std::endl;
        Tensor a = makeVector({1.0f, 2.0f, 3.0f});
        Tensor b = makeScalar(10.0f);
        Tensor s = makeScalar(2.0f);
        a.requires_grad(true);
        b.requires_grad(true);
        s.requires_grad(true);

        Tensor c = (a + b) * s;
        syncDevice(g_device);
        const float* cp = c.data_read<float>();
        if (!near(cp[0], 22.0f) || !near(cp[1], 24.0f) || !near(cp[2], 26.0f)) {
            std::cout << "❌ 前向错误" << std::endl;
            return false;
        }

        AutoGrad::backward(c.getRelatedNode(), false);
        syncDevice(g_device);

        const float* ga = a.grad().data_read<float>();
        const float* gb = b.grad().data_read<float>();
        const float* gs = s.grad().data_read<float>();
        if (!near(ga[0], 2.0f) || !near(ga[1], 2.0f) || !near(ga[2], 2.0f)) {
            std::cout << "❌ grad_a 错误" << std::endl;
            return false;
        }
        if (!near(gb[0], 6.0f)) {
            std::cout << "❌ grad_b 错误: 期望 6.0, 实际 " << gb[0] << std::endl;
            return false;
        }
        if (!near(gs[0], 11.0f + 12.0f + 13.0f)) {
            std::cout << "❌ grad_s 错误: 期望 36.0, 实际 " << gs[0] << std::endl;
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        std::cout << "!!! 异常: " << e.what() << std::endl;
        return false;
    }
}

bool test_add_3d_broadcast() {
    try {
        std::cout << "=== [" << deviceName(g_device) << "] Add: 3D 广播 {2,3,4} + {1,1,4} ===" << std::endl;
        // a: {2,3,4}, b: {1,1,4} -> c: {2,3,4}
        Tensor a = makeTensor3D({
            1,2,3,4, 5,6,7,8, 9,10,11,12,
            13,14,15,16, 17,18,19,20, 21,22,23,24
        }, 2, 3, 4);
        Tensor b = makeVector({1.0f, 2.0f, 3.0f, 4.0f});
        b = b.reshape({1, 1, 4});
        a.requires_grad(true);
        b.requires_grad(true);

        Tensor c = a + b;
        syncDevice(g_device);
        const float* cp = c.data_read<float>();
        // c[i,j,k] = a[i,j,k] + b[k]
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                for (size_t k = 0; k < 4; ++k) {
                    float expected = static_cast<float>(i*12 + j*4 + k + 1) + static_cast<float>(k + 1);
                    if (!near(cp[i*12 + j*4 + k], expected)) {
                        std::cout << "❌ 前向错误 c[" << i << "," << j << "," << k << "]" << std::endl;
                        return false;
                    }
                }
            }
        }

        AutoGrad::backward(c.getRelatedNode(), false);
        syncDevice(g_device);

        const float* ga = a.grad().data_read<float>();
        for (size_t i = 0; i < 24; ++i) {
            if (!near(ga[i], 1.0f)) {
                std::cout << "❌ grad_a[" << i << "] 错误" << std::endl;
                return false;
            }
        }
        // grad_b 沿维度 0,1 求和：每个 k 被加 2*3=6 次
        const float* gb = b.grad().data_read<float>();
        for (size_t k = 0; k < 4; ++k) {
            if (!near(gb[k], 6.0f)) {
                std::cout << "❌ grad_b[" << k << "] 错误: 期望 6.0, 实际 " << gb[k] << std::endl;
                return false;
            }
        }
        return true;
    } catch (const std::exception& e) {
        std::cout << "!!! 异常: " << e.what() << std::endl;
        return false;
    }
}

bool test_illegal_broadcast_raises() {
    try {
        std::cout << "=== [" << deviceName(g_device) << "] 非法广播：input 维度 > grad 维度应抛异常 ===" << std::endl;
        // 构造 input shape {2,3}，但反向时 grad shape {3}，不是合法广播对
        Tensor a = makeMatrix({1.0f,2.0f,3.0f,4.0f,5.0f,6.0f}, 2, 3);
        Tensor b = makeVector({1.0f, 1.0f, 1.0f});
        a.requires_grad(true);
        b.requires_grad(true);

        Tensor c = a + b; // c shape {2,3}
        syncDevice(g_device);

        // 手动构造非法的下游梯度 shape {3} 而不是 {2,3}
        Tensor illegal_grad(ShapeTag{}, {3}, DType::kFloat, g_device);
        float* p = illegal_grad.data_write<float>();
        p[0] = 1.0f; p[1] = 1.0f; p[2] = 1.0f;

        // 通过 getRelatedNode 拿到 AddNode，手动调用 backward 并传入非法 grad
        std::shared_ptr<Node> node = c.getRelatedNode();
        node->backward({illegal_grad});

        std::cout << "❌ 未抛出异常" << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cout << "✅ 正确抛出异常: " << e.what() << std::endl;
        return true;
    }
}

bool run_all_tests() {
    int passed = 0;
    int total = 10;

    if (test_add_vector_scalar_broadcast()) ++passed;
    if (test_add_matrix_vector_broadcast()) ++passed;
    if (test_add_3d_broadcast()) ++passed;
    if (test_illegal_broadcast_raises()) ++passed;
    if (test_sub_vector_scalar_broadcast()) ++passed;
    if (test_sub_scalar_vector_broadcast()) ++passed;
    if (test_mul_matrix_vector_broadcast()) ++passed;
    if (test_div_matrix_vector_broadcast()) ++passed;
    if (test_combined_broadcast_chain()) ++passed;

    // Mul scalar×向量 / Div 向量÷标量 / Div 标量÷向量 已在 test_mul_div_broadcast 覆盖，
    // 这里补充一个 Add/Sub 直接对矩阵广播的回归用例。
    try {
        std::cout << "=== [" << deviceName(g_device) << "] Add/Sub 矩阵×标量混合 ===" << std::endl;
        Tensor a = makeMatrix({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2);
        Tensor b = makeScalar(5.0f);
        a.requires_grad(true);
        b.requires_grad(true);

        Tensor c = (a + b) - b;
        syncDevice(g_device);
        const float* cp = c.data_read<float>();
        for (int i = 0; i < 4; ++i) {
            if (!near(cp[i], static_cast<float>(i + 1))) {
                std::cout << "❌ 前向错误 c[" << i << "]" << std::endl;
                throw std::runtime_error("forward mismatch");
            }
        }

        AutoGrad::backward(c.getRelatedNode(), false);
        syncDevice(g_device);

        const float* ga = a.grad().data_read<float>();
        const float* gb = b.grad().data_read<float>();
        for (int i = 0; i < 4; ++i) {
            if (!near(ga[i], 1.0f)) {
                std::cout << "❌ grad_a[" << i << "] 错误" << std::endl;
                throw std::runtime_error("grad_a mismatch");
            }
        }
        // d((a+5)-5)/d5 = 1 - 1 = 0（按广播求和后）
        if (!near(gb[0], 0.0f)) {
            std::cout << "❌ grad_b 错误: 期望 0.0, 实际 " << gb[0] << std::endl;
            throw std::runtime_error("grad_b mismatch");
        }
        ++passed;
    } catch (const std::exception& e) {
        std::cout << "!!! 组合测试异常: " << e.what() << std::endl;
    }

    std::cout << "\n[" << deviceName(g_device) << "] 测试结果: " << passed << "/" << total << " 通过" << std::endl;
    return passed == total;
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    std::cout << "🚀 开始广播梯度回归测试（Add/Sub/Mul/Div）" << std::endl;
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

    // 优雅清理 C3，避免静态析构期的 recursive_mutex lock failed
    ct::c3::shutdownAll();

    if (device_passed == device_total) {
        std::cout << "🎉 所有可用后端广播梯度回归测试通过!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ 部分后端测试失败" << std::endl;
        return 1;
    }
}
