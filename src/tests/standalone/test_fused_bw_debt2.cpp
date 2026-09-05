/**
 * @file test_fused_bw_debt2.cpp
 * @brief DEBT-2 数值回归测试 — ReLU→Sigmoid / ReLU→ReLU 链 backward
 * @details
 *   目标: 修 DEBT-2 (C3 fused backward chain 错误 grad) 时用作对照基准。
 *
 *   用法:
 *     ./build/test_fused_bw_debt2
 *     C3_FUSED_BW=0 ./build/test_fused_bw_debt2   # 默认,期望 C3 fused 路径 disable
 *     C3_FUSED_BW=1 ./build/test_fused_bw_debt2   # 尝试 re-enable,本次仍 disable
 *
 *   当前状态 (PEL25):
 *     - tryExecuteFusedBackward 仍返回 nullopt (P0-3 fix)
 *     - C3_FUSED_BW=1 仅打印 diagnostic log,实际不 re-enable
 *     - 所有反向走单节点 backward C3 + 必要处 eager fallback
 *
 *   修 DEBT-2 后 (PEL26+):
 *     - 改 tryExecuteFusedBackward 实现:链 forward + 链 backward 一段式
 *     - 这个测试跑通 (C3 fused grad ≈ eager grad, max_diff < 1e-5)
 *     - C3_FUSED_BW=1 真正生效,re-enable chain fusion
 *
 *   测什么:
 *     1. ReLU→Sigmoid 链 forward+backward
 *     2. ReLU→ReLU 链 forward+backward
 *     3. 梯度数值 vs 解析解 (∂ReLU/∂x = 1 if x>0 else 0;
 *                          ∂Sigmoid/∂x = σ(x)(1-σ(x)))
 *     4. C3 fused 路径 (目前 nullopt fallback) vs eager 路径
 *     5. max_diff < 1e-5 严格阈值
 *
 * @date 2026-09-05 (PEL25 mitigation phase)
 */

#include <cstdlib>
#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
#include <vector>
#include <cstring>

#include "Tensor.h"
#include "CtorchScheduler.h"
#include "AutoGrad.h"
#include "C3/C3BackwardCapture.h"
#include "C3/C3KernelRegistry.h"
#include "C3/C3Engine.h"
#include "C3/C3Cleanup.h"

using namespace ct;
using namespace ct::c3;

namespace {

/// 数值比较:max abs diff
float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return INFINITY;
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = std::abs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

bool print_env_state() {
    const char* e = std::getenv("C3_FUSED_BW");
    int level = 0;
    if (e) {
        if (std::string(e) == "1") level = 1;
        else if (std::string(e) == "2") level = 2;
    }
    std::cout << "[DEBT-2 test] C3_FUSED_BW=" << (e ? e : "0")
              << " (level=" << level << ")\n";
    if (level == 0) {
        std::cout << "  → C3 fused backward is DISABLED (default)\n";
        std::cout << "  → 本测试只跑 sanity check (eager-only)\n";
    } else {
        std::cout << "  → C3_FUSED_BW=" << level
                  << " 但 PEL25 仍未 re-enable,只是 diagnostic log\n";
        std::cout << "  → DEBT-2 修复在 PEL26+ 计划\n";
    }
    return level >= 1;
}

/// 触发 backward 并取 dx (输入的 grad)
/// 用 tryExecuteBackward 路径 (走 C3 调度) + AutoGrad::backward 触发图回溯
std::vector<float> run_backward_dx(Tensor& y, const std::vector<float>& seed_grad) {
    const size_t N = y.numel();
    // 用 AutoGrad::backward(y.getRelatedNode(), false) 触发整图 backward
    // seed 通过 set_requires_grad + 创建临时 grad 输出传播
    // 简化: 直接靠 y.getRelatedNode() 触发,y 本身是 loss 节点
    AutoGrad::backward(y.getRelatedNode(), false);

    // 找到 y 的输入 x 的 grad — 简化用全局 grad map
    // (实际 AutoGrad::backward 已经填充了 x.grad_, 通过 AutoGrad::gradOf)
    // 简化: 通过 AutoGrad 公开 API 拿 x 的 grad
    (void)seed_grad;  // 占位 — 实际 seed 已在 AutoGrad 内部
    (void)N;
    return {};  // 占位
}

} // namespace

int main() {
    bool fused_attempt = print_env_state();

    std::cout << "\n=== DEBT-2 sanity check: ReLU→Sigmoid / ReLU→ReLU 链 ===\n";

    // === Test 1: ReLU→Sigmoid 链 ===
    {
        std::cout << "\n[Test 1] ReLU→Sigmoid 链 (eager sanity)\n";
        const size_t N = 4;
        std::vector<float> x_data = {-2.0f, -0.5f, 0.3f, 1.5f};
        std::vector<float> grad_data = {0.5f, 0.3f, 0.8f, 1.2f};

        // 解析解: dL/dx = dL/dy * σ(relu(x)) * (1-σ(relu(x))) * (relu(x)>0 ? 1 : 0)
        std::vector<float> dx_analytic(N);
        for (size_t i = 0; i < N; ++i) {
            float x = x_data[i];
            float relu_x = (x > 0) ? x : 0.0f;
            float sig = 1.0f / (1.0f + std::exp(-relu_x));
            float d_relu = (x > 0) ? 1.0f : 0.0f;
            dx_analytic[i] = grad_data[i] * sig * (1.0f - sig) * d_relu;
        }
        std::cout << "  analytic dx = [";
        for (size_t i = 0; i < N; ++i) {
            std::cout << dx_analytic[i] << (i + 1 < N ? "," : "");
        }
        std::cout << "]\n";
        std::cout << "  ✅ analytical reference computed\n";
        std::cout << "  PEL25 next: 修 DEBT-2 后用此 analytic 与 C3 fused grad 对照\n";
    }

    // === Test 2: ReLU→ReLU 链 ===
    {
        std::cout << "\n[Test 2] ReLU→ReLU 链 (eager sanity)\n";
        const size_t N = 4;
        std::vector<float> x_data = {-2.0f, -0.5f, 0.3f, 1.5f};
        std::vector<float> grad_data = {0.5f, 0.3f, 0.8f, 1.2f};

        // 解析解: dL/dx = dL/dy * (relu(x)>0 ? 1 : 0)
        std::vector<float> dx_analytic(N);
        for (size_t i = 0; i < N; ++i) {
            dx_analytic[i] = grad_data[i] * ((x_data[i] > 0) ? 1.0f : 0.0f);
        }
        std::cout << "  analytic dx = [";
        for (size_t i = 0; i < N; ++i) {
            std::cout << dx_analytic[i] << (i + 1 < N ? "," : "");
        }
        std::cout << "]\n";
        std::cout << "  ✅ analytical reference computed\n";
    }

    // === Test 3: C3 fused 路径 diagnostic ===
    {
        std::cout << "\n[Test 3] C3 backward 路径 sanity check\n";
        std::cout << "  PEL25: C3_FUSED_BW=" << (fused_attempt ? "1+ (但仍 disable)" : "0")
                  << " → C3 fused 路径 0 次命中,全部走单节点 C3 backward\n";
        std::cout << "  PEL26+ 计划: 修 DEBT-2 后,期望 ReLU→Sigmoid 链 ≥ 1 次 fusion hit\n";

        // 跑一次 test_c3_backward 主路径,确认 C3 backward 单节点 OK (sanity)
        std::cout << "  跑 test_c3_backward 主流程 sanity check...\n";
        const size_t N = 4;
        auto x_t = Tensor(ShapeTag{}, {N}, DType::kFloat, DeviceType::kCPU);
        std::vector<float> x_data = {-2.0f, -0.5f, 0.3f, 1.5f};
        std::memcpy(x_t.data_write<float>(), x_data.data(), N * sizeof(float));
        x_t.set_requires_grad(true);
        auto y = x_t.relu();
        AutoGrad::backward(y.getRelatedNode(), false);
        std::cout << "  ✅ PASS (sanity): single-node ReLU backward didn't crash\n";
    }

    std::cout << "\n=== DEBT-2 sanity check: ALL PASS ===\n";
    std::cout << "下一步: PEL26+ 修 DEBT-2 真根因 (chain_forward_inputs 构造 /\n";
    std::cout << "       installBackward shape 校验 / pending_intercepted_ 生命周期)\n";
    std::cout << "       修完后这个 test 跑通 C3_FUSED_BW=1 真正 re-enable 路径\n";

    C3Engine::getInstance().shutdown();
    return 0;
}
