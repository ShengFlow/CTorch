// (SECURITY_SENSITIVE) CTorch MPS storage_offset / view Buffer lookup PoC
// 本地沙箱运行，验证修复后 MPS kernel 能正确处理 storage_offset > 0 的 view 张量。
// 禁止用于未授权系统或生产环境。

#include "Tensor.h"
#include "CtorchError.h"
#include <cmath>
#include <cstddef>
#include <iostream>

int main() {
    // ---- CPU 参考路径 ----
    Tensor cpu_base(ShapeTag{}, {8}, DType::kFloat, DeviceType::kCPU, /*zero_init=*/false);
    float* cpu_data = cpu_base.data_write<float>();
    for (size_t i = 0; i < 8; ++i) cpu_data[i] = static_cast<float>(i);

    Tensor cpu_view = cpu_base[5];  // _storage_offset = 5, shape {1}, value = 5.0
    Tensor cpu_one(1.0f, DeviceType::kCPU);
    Tensor cpu_result = cpu_view + cpu_one;
    float cpu_val = cpu_result.item<float>();
    std::cout << "[PoC] CPU view[5] + 1 = " << cpu_val << std::endl;

    // ---- MPS 修复后路径 ----
    Tensor mps_base(ShapeTag{}, {8}, DType::kFloat, DeviceType::kMPS, /*zero_init=*/false);
    float* mps_data = mps_base.data_write<float>();
    for (size_t i = 0; i < 8; ++i) mps_data[i] = static_cast<float>(i);

    Tensor mps_view = mps_base[5];  // _storage_offset = 5, shape {1}, value = 5.0
    std::cout << "[PoC] MPS base numel: " << mps_base.numel()
              << ", view numel: " << mps_view.numel()
              << ", view storage_offset: " << mps_view.storage_offset() << std::endl;

    Tensor mps_one(1.0f, DeviceType::kMPS);
    try {
        Tensor mps_result = mps_view + mps_one;
        float mps_val = mps_result.item<float>();
        std::cout << "[PoC] MPS view[5] + 1 = " << mps_val << std::endl;

        if (std::abs(cpu_val - 6.0f) < 1e-5f && std::abs(mps_val - 6.0f) < 1e-5f) {
            std::cout << "[PoC] FIX VERIFIED: MPS view with storage_offset works correctly"
                      << std::endl;
            return 0;
        }
        std::cout << "[PoC] UNEXPECTED: result mismatch (expected 6.0)" << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cout << "[PoC] VULNERABLE: view + one threw MPS buffer lookup error: " << e.what()
                  << std::endl;
        return 1;
    }
}
