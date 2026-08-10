// (SECURITY_SENSITIVE) CTorch MPS CrossEntropy target 越界读取 PoC
// 本地沙箱运行，仅用于验证 MPS CrossEntropy 路径未校验 target_class 是否落在 [0, num_classes)。
// 禁止用于未授权系统或生产环境。

#include "Tensor.h"
#include "CtorchError.h"
#include <cmath>
#include <cstddef>
#include <iostream>

static void fill_input(Tensor& t) {
    float* p = t.data_write<float>();
    p[0] = 1.0f; p[1] = 2.0f; p[2] = 3.0f;
    p[3] = 0.5f; p[4] = 1.5f; p[5] = 2.5f;
}

static void fill_target(Tensor& t) {
    float* p = t.data_write<float>();
    p[0] = 3.0f;   // == num_classes，越界
    p[1] = -1.0f;  // 负值类别，越界
}

int main() {
    std::cout << "[PoC] input shape: {2,3}, target values: {3.0, -1.0}" << std::endl;

    // ---- CPU 路径：对非法 target 跳过/置零，loss≈0 ----
    Tensor cpu_input(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU, /*zero_init=*/false);
    Tensor cpu_target(ShapeTag{}, {2}, DType::kFloat, DeviceType::kCPU, /*zero_init=*/false);
    fill_input(cpu_input);
    fill_target(cpu_target);

    Tensor cpu_loss = cpu_input.cross_entropy(cpu_target);
    std::cout << "[PoC] CPU path numel: " << cpu_loss.numel() << std::endl;
    if (cpu_loss.numel() == 1) {
        std::cout << "[PoC] CPU path loss value: " << cpu_loss.item<float>() << std::endl;
    }

    // ---- MPS 路径：修复后应对非法 target 置零，loss≈0 ----
    Tensor mps_input(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kMPS, /*zero_init=*/false);
    Tensor mps_target(ShapeTag{}, {2}, DType::kFloat, DeviceType::kMPS, /*zero_init=*/false);
    fill_input(mps_input);
    fill_target(mps_target);

    Tensor mps_loss = mps_input.cross_entropy(mps_target);
    float mps_loss_val = mps_loss.numel() == 1 ? mps_loss.item<float>() : NAN;
    std::cout << "[PoC] MPS path loss value: " << mps_loss_val << std::endl;

    // 判断：CPU 与 MPS 均应跳过非法 target，loss≈0。
    // 若 MPS 产生非零 loss，说明 kernel 仍使用越界 target 读取 logits。
    bool cpu_skipped = (cpu_loss.numel() == 1 && std::abs(cpu_loss.item<float>()) < 1e-6f);
    bool mps_skipped = (mps_loss.numel() == 1 && std::abs(mps_loss_val) < 1e-6f);
    if (cpu_skipped && mps_skipped) {
        std::cout << "[PoC] FIX VERIFIED: CPU and MPS both skip OOB target (loss≈0)" << std::endl;
        return 0;
    }

    if (!cpu_skipped) {
        std::cout << "[PoC] UNEXPECTED: CPU did not skip OOB target" << std::endl;
    }
    if (!mps_skipped) {
        std::cout << "[PoC] VULNERABLE: MPS uses OOB target (loss="
                  << mps_loss_val << ") -> OOB read in MPS kernel" << std::endl;
    }
    return 1;
}
