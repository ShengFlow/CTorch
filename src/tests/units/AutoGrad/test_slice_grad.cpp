/**
 * @file test_slice_grad.cpp
 * @author 苏璃珞
 * @brief slice_dim0 梯度回归测试
 *
 * @details 判据是「切片之后梯度还能回到原张量」，而不是「前向数值正确」。
 *          旧实现（拷贝构造 + 改元数据）前向完全正常、梯度恒为零且不报错，
 *          因此仅比较前向数值的测试无法发现该缺陷。
 *
 * 覆盖：
 *  1. 前向选取正确（按行对照）
 *  2. 反向散射正确（选中行梯度为 1、未选中行为 0）
 *  3. 与数值梯度（有限差分）对照
 *  4. 链式切片（非连续下游梯度）
 *  5. slice_dim0NoGrad 不建图
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

void checkNear(const char *name, double got, double want, double tol) {
    const double err = std::fabs(got - want);
    const bool ok = err <= tol;
    ++g_checks;
    if (!ok) {
        ++g_failed;
    }
    std::cout << "  [" << (ok ? " ok " : "FAIL") << "] " << name << "  (got " << got
              << ", want " << want << ", |err| = " << err << ")\n";
}

/// 用固定数据填充，便于人工核对
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
    std::cout << "slice_dim0 梯度回归\n";
    std::cout << "========================================\n";

    // ---- 1. 前向选取正确 ----
    std::cout << "\n[1] 前向选取\n";
    {
        Tensor x(ShapeTag{}, {5, 3});
        fillSeq(x); // 0..14

        Tensor s = x.slice_dim0(1, 2); // 行 1、2 → 值 3..8
        checkTrue("切片形状为 {2,3}",
                  s.sizes().size() == 2 && s.sizes()[0] == 2 && s.sizes()[1] == 3);

        const float *sp = s.data<float>();
        bool value_ok = true;
        const float expect[6] = {3, 4, 5, 6, 7, 8};
        for (int i = 0; i < 6; ++i) {
            if (std::fabs(sp[i] - expect[i]) > 1e-6f) {
                value_ok = false;
            }
        }
        checkTrue("切片数值等于原张量第 1..2 行", value_ok);
    }

    // ---- 2. 反向散射正确 ----
    std::cout << "\n[2] 反向散射（旧实现此处梯度为 0）\n";
    {
        Tensor x(ShapeTag{}, {5, 3});
        fillSeq(x);
        x.requires_grad(true);

        Tensor s = x.slice_dim0(1, 2);
        Tensor loss = s.sum();
        AutoGrad::backward(loss.getRelatedNode(), false);

        const float *g = x.grad_ptr();
        checkTrue("原张量拿到梯度指针", g != nullptr);
        if (g != nullptr) {
            bool ok = true;
            const float expect[15] = {0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0};
            for (int i = 0; i < 15; ++i) {
                if (std::fabs(g[i] - expect[i]) > 1e-6f) {
                    ok = false;
                    std::cout << "      位置 " << i << " 梯度 " << g[i] << " 期望 "
                              << expect[i] << "\n";
                }
            }
            checkTrue("选中行梯度为 1、未选中行为 0", ok);
        }
    }

    // ---- 3. 与数值梯度对照 ----
    std::cout << "\n[3] 与数值梯度对照\n";
    {
        Tensor x(ShapeTag{}, {4, 2});
        float *xp = x.data_write<float>();
        const float init[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
        for (int i = 0; i < 8; ++i) {
            xp[i] = init[i];
        }

        // 标量目标函数 f(x) = Σ (slice(2,2) 的平方)，梯度解析解 = 2*x_i（仅选中行）
        auto f = [&](const std::vector<float> &v) {
            Tensor t(ShapeTag{}, {4, 2});
            float *p = t.data_write<float>();
            for (size_t i = 0; i < v.size(); ++i) {
                p[i] = v[i];
            }
            Tensor s = t.slice_dim0(2, 2);
            return s.square().sum().data<float>()[0];
        };

        Tensor y(ShapeTag{}, {4, 2});
        {
            float *p = y.data_write<float>();
            for (int i = 0; i < 8; ++i) {
                p[i] = init[i];
            }
        }
        y.requires_grad(true);
        Tensor s = y.slice_dim0(2, 2);
        Tensor loss = s.square().sum();
        AutoGrad::backward(loss.getRelatedNode(), false);
        const float *g = y.grad_ptr();

        std::vector<float> base(init, init + 8);
        const double h = 1e-3;
        double max_err = 0.0;
        for (int i = 0; i < 8; ++i) {
            std::vector<float> vp = base, vm = base;
            vp[i] += static_cast<float>(h);
            vm[i] -= static_cast<float>(h);
            const double num = (f(vp) - f(vm)) / (2.0 * h);
            max_err = std::max(max_err, std::fabs(num - static_cast<double>(g[i])));
        }
        std::cout << "    最大偏差 = " << max_err << "\n";
        checkTrue("解析梯度与中心差分一致（< 1e-2）", max_err < 1e-2);
    }

    // ---- 4. 链式切片（下游梯度为非连续视图）----
    std::cout << "\n[4] 链式切片\n";
    {
        Tensor x(ShapeTag{}, {6, 2});
        fillSeq(x);
        x.requires_grad(true);

        Tensor a = x.slice_dim0(1, 4); // 行 1..4
        Tensor b = a.slice_dim0(1, 2); // 原张量的行 2..3
        Tensor loss = b.sum();
        AutoGrad::backward(loss.getRelatedNode(), false);

        const float *g = x.grad_ptr();
        bool ok = true;
        if (g == nullptr) {
            ok = false;
        } else {
            const float expect[12] = {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0};
            for (int i = 0; i < 12; ++i) {
                if (std::fabs(g[i] - expect[i]) > 1e-6f) {
                    ok = false;
                    std::cout << "      位置 " << i << " 梯度 " << g[i] << " 期望 "
                              << expect[i] << "\n";
                }
            }
        }
        checkTrue("两级切片的梯度散射到正确位置", ok);
    }

    // ---- 5. slice_dim0NoGrad 不建立反向路径 ----
    std::cout << "\n[5] NoGrad 版本\n";
    {
        Tensor x(ShapeTag{}, {4, 2});
        fillSeq(x);
        x.requires_grad(true);

        // 注意：NoGrad 版本走的是拷贝构造，而拷贝构造会给副本创建**新的**
        // GradAccumulator，所以 getRelatedNode() 并不为 nullptr —— 只是该节点
        // 不指向上游。因此判据必须是行为层面的（原张量是否收到梯度），
        // 而不是「节点指针是否为空」。
        Tensor s = x.slice_dim0NoGrad(1, 2);
        Tensor loss = s.sum();
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
        checkTrue("NoGrad 切片不建立反向路径（原张量梯度为零）", no_path);

        Tensor s2 = x.slice_dim0(1, 2);
        Tensor loss2 = s2.sum();
        AutoGrad::backward(loss2.getRelatedNode(), false);
        const float *g2 = x.grad_ptr();
        checkTrue("带梯度版本建立了反向路径（原张量梯度非零）",
                  g2 != nullptr && std::fabs(g2[2]) > 1e-6f);
    }

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
