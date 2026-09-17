/**
 * @file test_repeated_backward.cpp
 * @author 苏璃珞
 * @brief 重复构建-反向-释放同一图形态的回归测试
 *
 * @details 背景：可微仿真（OpenInspire3 的六自由度动力学）按「滚动窗口」工作 ——
 *          每个优化步重建一张形状相同的计算图、反传、释放、再重建。这与 CTorch
 *          原有训练的图形态不同：训练是一个大图反复 forward/backward（参数不变），
 *          而这里是**图结构反复重建**，每个 epoch 都有一批节点被回收、地址被复用。
 *
 *          本测试复刻该形态，并把算子组合对齐可微动力学的实际用法
 *          （slice 取分量 -> 逐元素运算 -> concat 拼装 -> reshape），
 *          用于复现「重复反向若干轮后崩溃」并作为修复后的回归。
 *
 * 判据：每一轮的梯度都必须正确（∂loss/∂x0 = 1），任何一轮出问题即失败。
 *
 * 用法：./test_repeated_backward [轮数] [每轮链长]
 *
 * @date 2026/9/17
 **/

#include "Tensor.h"
#include "AutoGrad.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void fillSeq(Tensor &t, float base) {
    float *p = t.data_write<float>();
    for (size_t i = 0; i < t.numel(); ++i) {
        p[i] = base + static_cast<float>(i);
    }
}

} // namespace

int main(int argc, char **argv) {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    int epochs = 10;
    int chain = 40;
    if (argc > 1) {
        epochs = std::atoi(argv[1]);
    }
    if (argc > 2) {
        chain = std::atoi(argv[2]);
    }

    std::cout << "========================================\n";
    std::cout << "重复构建-反向-释放\n";
    std::cout << "轮数 = " << epochs << "，每轮链长 = " << chain << "\n";
    std::cout << "========================================\n";

    int failed = 0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // 参数（持久叶子）：不能被覆盖，否则其 GradAccumulator 的弱引用失效
        Tensor x(ShapeTag{}, {4});
        fillSeq(x, 1.0f);
        x.requires_grad(true);

        // 状态经一次派生进入链，链上反复做 slice -> concat -> reshape
        Tensor y = x * 1.0f;
        for (int i = 0; i < chain; ++i) {
            Tensor a = y.slice(0, 0, 2);
            Tensor b = y.slice(0, 2, 2);
            Tensor c = a.concat(b, 0);
            y = c.reshape({4}) * 1.0f;
        }

        Tensor loss = y.sum();
        AutoGrad::backward(loss.getRelatedNode(), false);

        const float *g = x.grad_ptr();
        bool ok = (g != nullptr);
        if (ok) {
            for (int i = 0; i < 4; ++i) {
                if (std::fabs(g[i] - 1.0f) > 1e-5f) {
                    ok = false;
                    std::cout << "      轮 " << epoch << " 元素 " << i << " 梯度 " << g[i]
                              << " 期望 1\n";
                }
            }
        }
        if (!ok) {
            ++failed;
        }
        std::cout << "  轮 " << epoch << (ok ? " ok" : " FAIL") << std::endl;
    }

    std::cout << "========================================\n";
    if (failed == 0) {
        std::cout << "PASSED\n";
    } else {
        std::cout << failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return failed == 0 ? 0 : 1;
}
