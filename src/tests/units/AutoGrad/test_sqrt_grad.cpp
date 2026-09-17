/**
 * @file test_sqrt_grad.cpp
 * @author 苏璃珞
 * @brief Sqrt 算子前向数值与梯度回归
 *
 * @details 覆盖：
 *  1. 前向数值（与 std::sqrt 逐元素对照）
 *  2. 反向公式 dy/dx = 1/(2·sqrt(x))
 *  3. 与中心差分对照
 *  4. 链式调用（sqrt 之后再参与其他算子）
 *  5. 边界：x = 0 处前向为 0（梯度发散属预期，不做截断）
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

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    std::cout << "========================================\n";
    std::cout << "Sqrt 算子回归\n";
    std::cout << "========================================\n";

    // ---- 1. 前向数值 ----
    std::cout << "\n[1] 前向数值\n";
    {
        Tensor x(ShapeTag{}, {5});
        const float in[5] = {0.0f, 1.0f, 4.0f, 9.0f, 2.25f};
        float *p = x.data_write<float>();
        for (int i = 0; i < 5; ++i) {
            p[i] = in[i];
        }

        Tensor y = x.sqrt();
        const float *yp = y.data<float>();
        bool ok = true;
        for (int i = 0; i < 5; ++i) {
            if (std::fabs(yp[i] - std::sqrt(in[i])) > 1e-6f) {
                ok = false;
                std::cout << "      输入 " << in[i] << " 得到 " << yp[i] << "，期望 "
                          << std::sqrt(in[i]) << "\n";
            }
        }
        checkTrue("sqrt 前向与 std::sqrt 一致", ok);
        checkTrue("形状保持 {5}",
                  y.sizes().size() == 1 && y.sizes()[0] == 5);
    }

    // ---- 2. 反向公式 ----
    std::cout << "\n[2] 反向 dy/dx = 1/(2*sqrt(x))\n";
    {
        Tensor x(ShapeTag{}, {3});
        const float in[3] = {1.0f, 4.0f, 16.0f};
        float *p = x.data_write<float>();
        for (int i = 0; i < 3; ++i) {
            p[i] = in[i];
        }
        x.requires_grad(true);

        Tensor y = x.sqrt();
        Tensor loss = y.sum();
        AutoGrad::backward(loss.getRelatedNode(), false);

        const float *g = x.grad_ptr();
        checkTrue("原张量拿到梯度指针", g != nullptr);
        if (g != nullptr) {
            bool ok = true;
            for (int i = 0; i < 3; ++i) {
                const double want = 1.0 / (2.0 * std::sqrt(static_cast<double>(in[i])));
                if (std::fabs(g[i] - want) > 1e-5) {
                    ok = false;
                    std::cout << "      x=" << in[i] << " 梯度 " << g[i] << "，期望 " << want
                              << "\n";
                }
            }
            checkTrue("梯度等于解析解", ok);
        }
    }

    // ---- 3. 与中心差分对照 ----
    std::cout << "\n[3] 与中心差分对照\n";
    {
        const float init[4] = {0.25f, 1.0f, 2.5f, 7.0f};

        auto f = [&](const std::vector<float> &v) {
            Tensor t(ShapeTag{}, {4});
            float *p = t.data_write<float>();
            for (size_t i = 0; i < v.size(); ++i) {
                p[i] = v[i];
            }
            return t.sqrt().square().sum().data<float>()[0];
        };

        Tensor y(ShapeTag{}, {4});
        {
            float *p = y.data_write<float>();
            for (int i = 0; i < 4; ++i) {
                p[i] = init[i];
            }
        }
        y.requires_grad(true);
        Tensor loss = y.sqrt().square().sum(); // = Σ x
        AutoGrad::backward(loss.getRelatedNode(), false);
        const float *g = y.grad_ptr();

        std::vector<float> base(init, init + 4);
        const double h = 1e-3;
        double max_err = 0.0;
        for (int i = 0; i < 4; ++i) {
            std::vector<float> vp = base, vm = base;
            vp[i] += static_cast<float>(h);
            vm[i] -= static_cast<float>(h);
            const double num = (f(vp) - f(vm)) / (2.0 * h);
            max_err = std::max(max_err, std::fabs(num - static_cast<double>(g[i])));
        }
        std::cout << "    最大偏差 = " << max_err << "\n";
        checkTrue("sqrt(x)^2 的梯度与中心差分一致（< 1e-2）", max_err < 1e-2);
        // 注意：sqrt(x)^2 = x（x>=0），故梯度应全为 1 —— 这是对消去性的强校验
        checkNear("sqrt(x)^2 对 x 的梯度为 1（首个元素）",
                  g != nullptr ? g[0] : 0.0, 1.0, 1e-3);
    }

    // ---- 4. 链式调用 ----
    std::cout << "\n[4] 链式调用\n";
    {
        Tensor x(ShapeTag{}, {2});
        float *p = x.data_write<float>();
        p[0] = 4.0f;
        p[1] = 9.0f;
        x.requires_grad(true);

        // f = sqrt(x) * 2 + 1  =>  df/dx = 1/sqrt(x)
        Tensor y = x.sqrt() * 2.0f + 1.0f;
        Tensor loss = y.sum();
        AutoGrad::backward(loss.getRelatedNode(), false);

        const float *g = x.grad_ptr();
        bool ok = (g != nullptr);
        if (ok) {
            const double want0 = 1.0 / std::sqrt(4.0);
            const double want1 = 1.0 / std::sqrt(9.0);
            ok = std::fabs(g[0] - want0) < 1e-5 && std::fabs(g[1] - want1) < 1e-5;
        }
        checkTrue("链式调用下梯度正确（df/dx = 1/sqrt(x)）", ok);
    }

    // ---- 5. 边界 ----
    std::cout << "\n[5] 边界 x = 0\n";
    {
        Tensor x(ShapeTag{}, {1});
        x.data_write<float>()[0] = 0.0f;

        Tensor y = x.sqrt();
        checkNear("sqrt(0) = 0", y.data<float>()[0], 0.0, 1e-9);
        std::cout << "    注：x=0 处导数发散，与 PyTorch 一致不做截断；\n"
                     "        调用方若在可能取零处使用应自行加 eps\n";
    }

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
