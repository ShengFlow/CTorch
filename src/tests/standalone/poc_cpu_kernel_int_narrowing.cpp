// (SECURITY_SENSITIVE) CTorch CPU kernel int elem_count narrowing PoC
// 本地沙箱运行，仅用于验证漏洞存在与修复有效性。
// 禁止用于未授权系统或生产环境。
//
// 触发原理：
//   CPU-BASIC/SIMD kernel 将 size_t 类型的 a.numel() 赋值给 int elem_count。
//   当 numel() > INT_MAX 时，int 窄化转换导致 elem_count 为负值（实现定义，通常为模 2^32）。
//   BASIC 路径中 for (int i=0; i<elem_count; ++i) 不会执行，结果张量未被写入。
//   SIMD 路径中 for (size_t i=0; i+7 < elem_count; i+=8) 因 elem_count 转 size_t 后极大，
//   会导致越界读写。
//
// 注意：触发本缺陷需要分配约 8.6 GiB 以上的 float32 存储（两张量约 17 GiB），
//       本地内存不足时只会观察到分配失败，不会真正进入 kernel 的异常路径。
//       即使无法物理执行，源码语义已足够证明缺陷存在。

#include "Tensor.h"
#include <climits>
#include <cstddef>
#include <iostream>
#include <limits>

int main() {
    // INT_MAX = 2147483647，加 1 后 numel() 无法在 int 中保存。
    size_t big_numel = static_cast<size_t>(std::numeric_limits<int>::max()) + 1ULL;
    std::vector<size_t> shape = {big_numel};

    std::cout << "[PoC] requested numel: " << big_numel
              << " (> INT_MAX=" << std::numeric_limits<int>::max() << ")" << std::endl;
    std::cout << "[PoC] single tensor bytes: " << (big_numel * sizeof(float)) << std::endl;
    std::cout << "[PoC] two tensors bytes:   " << (2 * big_numel * sizeof(float)) << std::endl;

    try {
        Tensor a(ShapeTag{}, shape, DType::kFloat, DeviceType::kCPU, /*zero_init=*/false);
        Tensor b(ShapeTag{}, shape, DType::kFloat, DeviceType::kCPU, /*zero_init=*/false);
        std::cout << "[PoC] allocation succeeded, a.numel()=" << a.numel() << std::endl;

        // 触发 CPU kernel（调度器默认优先 SIMD/AMX，最终仍落入使用 int elem_count 的循环）
        Tensor c = a + b;
        std::cout << "[PoC] kernel returned, c.numel()=" << c.numel() << std::endl;

        // 若 BASIC/SIMD 路径未正确写入，结果应全为 0 或未初始化。
        std::cout << "[PoC] first element of result: " << c.data_read<float>()[0] << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[PoC] exception (likely allocation failure): " << e.what() << std::endl;
    }

    return 0;
}
