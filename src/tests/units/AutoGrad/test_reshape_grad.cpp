/**
 * @file test_reshape_grad.cpp
 * @author 苏璃珞
 * @brief reshape 梯度回归测试
 *
 * @details 判据是「重塑之后梯度还能回到原张量」，而不是「前向形状正确」。
 *          旧实现（拷贝构造 + 改 _shape）前向完全正常、梯度恒为零且不报错。
 *
 * 覆盖：
 *  1. 前向形状与元素线性顺序不变
 *  2. 反向按输入形状还原
 *  3. 与数值梯度（有限差分）对照
 *  4. 非连续输入被显式拒绝（旧实现会静默返回布局错乱的张量）
 *  5. reshapeNoGrad 不建立反向路径
 *
 * @date 2026/9/17
 **/

#include "Tensor.h"
#include "AutoGrad.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

int g_checks = 0;
int g_failed = 0;

void checkTrue(const char *name, bool cond) {
    ++g_checks;
    if (!cond) {
        ++g_failed;
    }
    std::cout << "  [" << (cond ? " ok " : "FAIL") << "] " << name << "\n";
}

void fillSeq(Tensor &t) {
    float *p = t.data_write<float>();
    for (size_t i = 0; i < t.numel(); ++i) {
        p[i] = static_cast<float>(i);
    }
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    std::cout << "========================================\n";
    std::cout << "reshape 梯度回归\n";
    std::cout << "========================================\n";

    // ---- 1. 前向 ----
    std::cout << "\n[1] 前向形状与元素顺序\n";
    {
        Tensor x(ShapeTag{}, {2, 6});
        fillSeq(x);

        Tensor r = x.reshape(std::vector<size_t>{3, 4});
        checkTrue("形状变为 {3,4}",
                  r.sizes().size() == 2 && r.sizes()[0] == 3 && r.sizes()[1] == 4);

        const float *rp = r.data<float>();
        bool order_ok = true;
        for (int i = 0; i < 12; ++i) {
            if (std::fabs(rp[i] - static_cast<float>(i)) > 1e-6f) {
                order_ok = false;
            }
        }
        checkTrue("元素线性顺序不变", order_ok);
    }

    // ---- 2. 反向形状还原 ----
    std::cout << "\n[2] 反向按输入形状还原（旧实现此处梯度为 0）\n";
    {
        Tensor x(ShapeTag{}, {2, 6});
        fillSeq(x);
        x.requires_grad(true);

        Tensor r = x.reshape(std::vector<size_t>{3, 4});
        Tensor loss = r.sum();
        AutoGrad::backward(loss.getRelatedNode(), false);

        const float *g = x.grad_ptr();
        checkTrue("原张量拿到梯度指针", g != nullptr);
        if (g != nullptr) {
            bool ok = true;
            for (int i = 0; i < 12; ++i) {
                if (std::fabs(g[i] - 1.0f) > 1e-6f) {
                    ok = false;
                }
            }
            checkTrue("梯度全为 1（sum 对每个元素导数为 1）", ok);
        }
    }

    // ---- 3. 与数值梯度对照 ----
    std::cout << "\n[3] 与数值梯度对照\n";
    {
        const float init[6] = {0.5f, -1.5f, 2.0f, 3.5f, -4.0f, 1.25f};

        auto f = [&](const std::vector<float> &v) {
            Tensor t(ShapeTag{}, {2, 3});
            float *p = t.data_write<float>();
            for (size_t i = 0; i < v.size(); ++i) {
                p[i] = v[i];
            }
            Tensor r = t.reshape(std::vector<size_t>{3, 2});
            return r.square().sum().data<float>()[0];
        };

        Tensor y(ShapeTag{}, {2, 3});
        {
            float *p = y.data_write<float>();
            for (int i = 0; i < 6; ++i) {
                p[i] = init[i];
            }
        }
        y.requires_grad(true);
        Tensor r = y.reshape(std::vector<size_t>{3, 2});
        Tensor loss = r.square().sum();
        AutoGrad::backward(loss.getRelatedNode(), false);
        const float *g = y.grad_ptr();

        std::vector<float> base(init, init + 6);
        const double h = 1e-3;
        double max_err = 0.0;
        for (int i = 0; i < 6; ++i) {
            std::vector<float> vp = base, vm = base;
            vp[i] += static_cast<float>(h);
            vm[i] -= static_cast<float>(h);
            const double num = (f(vp) - f(vm)) / (2.0 * h);
            max_err = std::max(max_err, std::fabs(num - static_cast<double>(g[i])));
        }
        std::cout << "    最大偏差 = " << max_err << "\n";
        checkTrue("解析梯度与中心差分一致（< 1e-2）", max_err < 1e-2);
    }

    // ---- 4. 非连续输入被拒绝 ----
    std::cout << "\n[4] 非连续输入\n";
    {
        Tensor x(ShapeTag{}, {4, 3});
        fillSeq(x);

        // 切片产生非连续视图（storage_offset 非零）
        Tensor s = x.slice_dim0NoGrad(1, 2);
        checkTrue("切片结果是连续的吗（本题应为是，因行内紧凑）", s.is_contiguous());

        // 转置产生真正的非连续视图
        Tensor t = x.transposeNoGrad(0, 1);
        checkTrue("转置结果非连续", !t.is_contiguous());

        bool rejected = false;
        try {
            Tensor bad = t.reshape(std::vector<size_t>{12});
            (void)bad;
        } catch (const std::exception &) {
            rejected = true;
        }
        checkTrue("对非连续输入 reshape 被显式拒绝（旧实现静默返回错乱布局）", rejected);
    }

    // ---- 5. reshapeNoGrad 不建立反向路径 ----
    std::cout << "\n[5] NoGrad 版本\n";
    {
        Tensor x(ShapeTag{}, {2, 4});
        fillSeq(x);
        x.requires_grad(true);

        Tensor r = x.reshapeNoGrad(std::vector<size_t>{4, 2});
        Tensor loss = r.sum();
        AutoGrad::backward(loss.getRelatedNode(), false);

        const float *g = x.grad_ptr();
        bool no_path = true;
        if (g != nullptr) {
            for (int i = 0; i < 8; ++i) {
                if (std::fabs(g[i]) > 1e-6f) {
                    no_path = false;
                }
            }
        }
        checkTrue("NoGrad 重塑不建立反向路径（原张量梯度为零）", no_path);

        Tensor r2 = x.reshape(std::vector<size_t>{4, 2});
        Tensor loss2 = r2.sum();
        AutoGrad::backward(loss2.getRelatedNode(), false);
        const float *g2 = x.grad_ptr();
        checkTrue("带梯度版本建立了反向路径", g2 != nullptr && std::fabs(g2[0]) > 1e-6f);
    }

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
