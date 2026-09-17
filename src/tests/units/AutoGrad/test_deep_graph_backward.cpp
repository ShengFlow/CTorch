/**
 * @file test_deep_graph_backward.cpp
 * @author 苏璃珞
 * @brief 深链计算图的反向传播回归测试
 *
 * @details 背景：CTorch 原有的训练图（MNIST 的 FC、LLaMA 的 FFN）都是**宽而浅**的
 *          —— 几百个节点、深度十几层。而「可微仿真」这类应用构造的是**深链图**：
 *          一条轨迹每个时间步展开数十个算子，几十步就有上万层。此时图遍历的
 *          递归实现会耗尽线程栈。
 *
 *          本测试用最省的方式构造深链（每层一个 MulNode），把深度做成可配置参数，
 *          用于：
 *           1. 复现「深度超过阈值后 backward 崩溃」；
 *           2. 作为修复后的回归 —— 深度加大到远超阈值仍必须通过。
 *
 * 判据不只看「不崩溃」：梯度值也必须正确（∂loss/∂x0 = 1，每层乘数都是 1）。
 *
 * 用法：
 *   ./test_deep_graph_backward            # 默认深度（回归用）
 *   ./test_deep_graph_backward 200000     # 指定链长，用于探测栈上限
 *
 * @date 2026/9/17
 **/

#include "Tensor.h"
#include "AutoGrad.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    // 默认深度取在「递归实现必然栈溢出」的量级之上：单帧约百字节量级，
    // 主线程栈通常 8 MB，五万层已接近上限，这里留出安全余量。
    std::size_t depth = 100000;
    if (argc > 1) {
        depth = static_cast<std::size_t>(std::stoul(argv[1]));
    }

    std::cout << "========================================\n";
    std::cout << "深链计算图反向传播\n";
    std::cout << "链长 = " << depth << "\n";
    std::cout << "========================================\n";

    Tensor x(ShapeTag{}, {1});
    x.data_write<float>()[0] = 2.0f;
    x.requires_grad(true);

    // 每层 y = y * 1.0：一个 MulNode，链长即图深度（外加一个 SumNode）
    Tensor y = x * 1.0f;
    for (std::size_t i = 0; i < depth; ++i) {
        y = y * 1.0f;
    }

    std::cout << "  建图完成，开始反向..." << std::endl;

    Tensor loss = y.sum();
    AutoGrad::backward(loss.getRelatedNode(), false);

    std::cout << "  反向完成" << std::endl;

    int failed = 0;
    const float *g = x.grad_ptr();
    if (g == nullptr) {
        std::cout << "  [FAIL] 叶子张量未收到梯度\n";
        failed = 1;
    } else {
        const double got = static_cast<double>(g[0]);
        const bool ok = std::fabs(got - 1.0) < 1e-5;
        if (!ok) {
            ++failed;
        }
        std::cout << "  [" << (ok ? " ok " : "FAIL") << "] ∂loss/∂x0 = " << got
                  << "（期望 1）\n";
    }

    // 前向值也应保持正确（每层乘 1）
    const float fwd = y.data<float>()[0];
    const bool fwd_ok = std::fabs(static_cast<double>(fwd) - 2.0) < 1e-5;
    if (!fwd_ok) {
        ++failed;
    }
    std::cout << "  [" << (fwd_ok ? " ok " : "FAIL") << "] 前向值 = " << fwd
              << "（期望 2）\n";

    std::cout << "========================================\n";
    if (failed == 0) {
        std::cout << "PASSED\n";
    } else {
        std::cout << failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return failed == 0 ? 0 : 1;
}
