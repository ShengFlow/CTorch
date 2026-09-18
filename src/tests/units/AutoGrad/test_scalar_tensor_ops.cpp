/**
 * @file test_scalar_tensor_ops.cpp
 * @author 苏璃珞
 * @brief 0 维标量张量的运算、梯度与广播反向回归测试
 *
 * @details 背景：`sum()` / `dot()` 这类全归约算子的结果是 **0 维张量**（`_shape`
 *          为空、numel()=1、strides 为空）。CTorch 既有测试几乎都在 ≥1 维张量上，
 *          0 维张量参与的运算缺少覆盖 —— 而实际使用中很常见：
 *          `loss = (a*a).sum() + (b*b).sum() * 0.5f` 就是两个 0 维张量相加。
 *
 * 覆盖：
 *  A. 0 维张量的前向数值（+ - * /）与与一维的广播
 *  B. 0 维张量的梯度
 *  C. 同形状除法的梯度（最常见情形，必须正确）
 *  D. **逐分量除法**：比值型归一化的稳妥写法（先 slice 取分量，各自除以标量再
 *     concat），梯度正确 —— 这是 OpenInspire3 可微六自由度动力学采用的写法。
 *  E. 反复构建-反向-释放（200 轮）的稳定性
 *
 * @par 回归覆盖：形状不同的广播除法反向（曾静默错值，已修）
 *
 * 历史缺陷：`{N} / 标量` 与 `标量 / {N}` 的反向只累加被广播张量的**第一个元素**
 * 而非其和。例如 `loss = Σ(c_j/Σc²)` 的解析梯度是 `1/S − 2c_i·Σc_j/S²`（S = Σc²），
 * 当时实测得到 `1/S − 2c_i/S²`，`Σc_j` 这段贡献整体丢失；前向数值完全正常，
 * 只有梯度错且不报错。
 *
 * 根因与修法（2026-09-17）：
 *  - 症状在 C3 反向融合路径 —— eager 的 `DivNode::backward` 逐段复刻数值正确，
 *    但反向期间它**从未被调用**（函数入口日志确认零次进入）；
 *  - 入口 `tryExecuteBackward` 的白名单短路写成 `_n == 1 && !supportsNodeType`，
 *    只拦单输入节点；而 `supportsNodeType` 的名单本就是单输入 unary element-wise
 *    类型集合，其注释也写明「多输入节点仍按注释回退 eager」—— 设计意图是多输入
 *    一律回退，入口却放行了它们，于是多输入节点命中按同形状假设的 element-wise
 *    反向 kernel，遇到广播即出错；
 *  - 修法：判据对齐设计意图，名单之外无论几个输入都不进入 C3 反向路径（与 §4.98
 *    leaky_relu 同源，那次同样只覆盖了单输入）。
 *
 * 打开这条 eager 路径后又暴露出第二处缺陷：`compute_broadcast_reduce_dims` 把
 * 「input 维度数多于 grad」一律判为非法广播对，而 CTorch 内部存在两种标量形状 ——
 * `sum()`/`dot()` 返回 0 维 `{}`，标量运算路径常构造 `{1}` —— 混用时就会命中该分支
 * 并抛异常中断反向。已修为：input 多出的**前导维度全为 1** 时视为等价标量，不归约。
 *
 * 本文件 [F] 段是该场景的常规回归（判据为与解析解一致）。
 *
 * @date 2026/9/17
 **/

#include "Tensor.h"
#include "AutoGrad.h"

#include <cmath>
#include <cstdlib>
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
    const bool ok = std::isfinite(got) && err <= tol;
    ++g_checks;
    if (!ok) {
        ++g_failed;
    }
    std::cout << "  [" << (ok ? " ok " : "FAIL") << "] " << name << "  (got " << got
              << ", want " << want << ", |err| = " << err << ")\n";
}

void fillSeq(Tensor &t, float base) {
    float *p = t.data_write<float>();
    for (size_t i = 0; i < t.numel(); ++i) {
        p[i] = base + static_cast<float>(i);
    }
}

std::vector<float> gradOf(Tensor &leaf) {
    const float *g = leaf.grad_ptr();
    std::vector<float> v(leaf.numel(), 0.0f);
    if (g != nullptr) {
        for (size_t i = 0; i < leaf.numel(); ++i) {
            v[i] = g[i];
        }
    }
    return v;
}

void checkGrad(const char *name, const std::vector<float> &got,
               const std::vector<double> &want, double tol = 1e-4) {
    bool ok = (got.size() == want.size());
    if (ok) {
        for (size_t i = 0; i < got.size(); ++i) {
            if (!std::isfinite(static_cast<double>(got[i])) ||
                std::fabs(static_cast<double>(got[i]) - want[i]) > tol) {
                ok = false;
            }
        }
    }
    ++g_checks;
    if (!ok) {
        ++g_failed;
    }
    std::cout << "  [" << (ok ? " ok " : "FAIL") << "] " << name << "\n";
    std::cout << "        实得 [";
    for (size_t i = 0; i < got.size(); ++i) {
        std::cout << got[i] << (i + 1 < got.size() ? ", " : "");
    }
    std::cout << "]，期望 [";
    for (size_t i = 0; i < want.size(); ++i) {
        std::cout << want[i] << (i + 1 < want.size() ? ", " : "");
    }
    std::cout << "]\n";
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    std::cout << "========================================\n";
    std::cout << "0 维标量张量运算与梯度\n";
    std::cout << "========================================\n";

    // ---- A. 0 维张量前向运算 ----
    std::cout << "\n[A] 0 维张量前向\n";
    {
        Tensor a(ShapeTag{}, {3});
        Tensor b(ShapeTag{}, {3});
        fillSeq(a, 1.0f); // 1,2,3
        fillSeq(b, 2.0f); // 2,3,4

        Tensor sa = (a * a).sum(); // 14
        Tensor sb = (b * b).sum(); // 29

        checkTrue("全归约结果是 0 维张量", sa.sizes().empty() && sa.numel() == 1);
        checkNear("Σa² = 14", sa.data<float>()[0], 14.0, 1e-5);
        checkNear("Σb² = 29", sb.data<float>()[0], 29.0, 1e-5);
        checkNear("0 维 + 0 维", (sa + sb).data<float>()[0], 43.0, 1e-5);
        checkNear("0 维 - 0 维", (sb - sa).data<float>()[0], 15.0, 1e-5);
        checkNear("0 维 * 0 维", (sa * sb).data<float>()[0], 406.0, 1e-5);
        checkNear("0 维 / 0 维", (sb / sa).data<float>()[0], 29.0 / 14.0, 1e-5);
        checkNear("实际 loss 形态 Σa² + 0.5Σb²", (sa + sb * 0.5f).data<float>()[0], 28.5,
                  1e-5);

        // 与一维的广播
        Tensor c(ShapeTag{}, {3});
        fillSeq(c, 1.0f); // 1,2,3
        Tensor scaled = c / sa; // {3} / {} → {3}
        checkTrue("0 维作除数时结果形状为 {3}",
                  scaled.sizes().size() == 1 && scaled.sizes()[0] == 3);
        checkNear("(c / Σa²)[2] = 3/14", scaled.data<float>()[2], 3.0 / 14.0, 1e-5);
        checkNear("(c + Σa²)[0] = 15", (c + sa).data<float>()[0], 15.0, 1e-5);
    }

    // ---- B. 0 维张量运算的梯度 ----
    std::cout << "\n[B] 0 维张量运算的梯度\n";
    {
        Tensor a(ShapeTag{}, {3});
        Tensor b(ShapeTag{}, {3});
        fillSeq(a, 1.0f);
        fillSeq(b, 2.0f);
        a.requires_grad(true);
        b.requires_grad(true);

        Tensor loss = (a * a).sum() + (b * b).sum() * 0.5f;
        AutoGrad::backward(loss.getRelatedNode(), false);

        // ∂(Σa²)/∂a_i = 2a_i ；∂(0.5Σb²)/∂b_i = b_i
        checkGrad("∂loss/∂a = 2a", gradOf(a), {2.0, 4.0, 6.0});
        checkGrad("∂loss/∂b = b", gradOf(b), {2.0, 3.0, 4.0});
    }

    // ---- C. 同形状除法的梯度 ----
    std::cout << "\n[C] 同形状除法梯度（{3} / {3}）\n";
    {
        Tensor d(ShapeTag{}, {3});
        Tensor e(ShapeTag{}, {3});
        fillSeq(d, 1.0f); // 1,2,3
        fillSeq(e, 2.0f); // 2,3,4
        d.requires_grad(true);
        e.requires_grad(true);

        Tensor loss = (d / e).sum();
        AutoGrad::backward(loss.getRelatedNode(), false);

        // ∂loss/∂d_i = 1/e_i ；∂loss/∂e_i = -d_i/e_i²
        checkGrad("∂loss/∂d = 1/e", gradOf(d), {0.5, 1.0 / 3.0, 0.25});
        checkGrad("∂loss/∂e = -d/e²", gradOf(e), {-1.0 / 4.0, -2.0 / 9.0, -3.0 / 16.0});
    }

    // ---- D. 逐分量除法（比值型归一化的稳妥写法）----
    std::cout << "\n[D] 逐分量除法梯度\n";
    {
        // 复刻 OpenInspire3 四元数归一化的写法：
        //   n = sqrt(qw² + qx² + qy² + qz²)      —— 全部是 {1} 之间的运算
        //   输出 = concat(qw/n, qx/n, qy/n, qz/n) —— 全是同形状（{1}/{1}）除法
        // 全程不出现形状不同的广播，因而不经过那条有缺陷的融合路径。
        Tensor q(ShapeTag{}, {4});
        fillSeq(q, 1.0f); // 1,2,3,4
        q.requires_grad(true);

        Tensor qw = q.slice(0, 0, 1);
        Tensor qx = q.slice(0, 1, 1);
        Tensor qy = q.slice(0, 2, 1);
        Tensor qz = q.slice(0, 3, 1);

        Tensor n = (qw * qw + qx * qx + qy * qy + qz * qz).sqrt(); // sqrt(30)
        Tensor normalized = (qw / n).concat(qx / n, 0).concat(qy / n, 0).concat(qz / n, 0);

        checkTrue("归一化结果形状为 {4}",
                  normalized.sizes().size() == 1 && normalized.sizes()[0] == 4);
        double norm = 0.0;
        for (int i = 0; i < 4; ++i) {
            const double v = normalized.data<float>()[i];
            norm += v * v;
        }
        checkNear("归一化后模长为 1", std::sqrt(norm), 1.0, 1e-5);

        Tensor loss = normalized.sum();
        AutoGrad::backward(loss.getRelatedNode(), false);

        // 解析梯度：y_i = q_i/N，N = sqrt(Σq²)
        // ∂(Σy)/∂q_i = 1/N - (Σq_j)·q_i / N³
        const double N = std::sqrt(30.0);
        const double T = 10.0; // Σq_j = 1+2+3+4
        std::vector<double> want;
        for (int i = 0; i < 4; ++i) {
            want.push_back(1.0 / N - T * (1.0 + i) / (N * N * N));
        }
        checkGrad("∂(Σ q/‖q‖)/∂q 与解析解一致", gradOf(q), want, 1e-4);
    }

    // ---- E. 反复构建-反向-释放 ----
    std::cout << "\n[E] 反复构建-反向-释放（200 轮）\n";
    {
        const int EPOCHS = 200;
        int bad = -1;
        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            Tensor a(ShapeTag{}, {3});
            Tensor b(ShapeTag{}, {3});
            fillSeq(a, 1.0f);
            fillSeq(b, 2.0f);
            a.requires_grad(true);
            b.requires_grad(true);

            Tensor loss = (a * a).sum() + (b * b).sum() * 0.5f;
            AutoGrad::backward(loss.getRelatedNode(), false);

            const float *ga = a.grad_ptr();
            const float *gb = b.grad_ptr();
            bool ok = (ga != nullptr) && (gb != nullptr);
            if (ok) {
                for (int i = 0; i < 3; ++i) {
                    if (!std::isfinite(static_cast<double>(ga[i])) ||
                        !std::isfinite(static_cast<double>(gb[i])) ||
                        std::fabs(static_cast<double>(ga[i]) - 2.0 * (1.0 + i)) > 1e-5 ||
                        std::fabs(static_cast<double>(gb[i]) - (2.0 + i)) > 1e-5) {
                        ok = false;
                    }
                }
            }
            if (!ok && bad < 0) {
                bad = epoch;
            }
        }
        if (bad >= 0) {
            std::cout << "      首次异常出现在第 " << bad << " 轮\n";
        }
        checkTrue("200 轮反复反向全部梯度正确", bad < 0);
    }

    // ---- F. 形状不同的广播除法反向（曾静默错值，见文件头）----
    std::cout << "\n[F] 形状不同的广播除法反向\n";
    {
        // 分母侧：loss = Σ(d_j / s) ⇒ ∂loss/∂s = -Σd_j / s²
        Tensor d(ShapeTag{}, {3});
        fillSeq(d, 1.0f); // 1,2,3
        d.requires_grad(true);
        Tensor s0(ShapeTag{}, {1});
        s0.data_write<float>()[0] = 14.0f;
        s0.requires_grad(true);

        Tensor loss = (d / s0).sum();
        AutoGrad::backward(loss.getRelatedNode(), false);

        const double want_s = -6.0 / (14.0 * 14.0); // -Σd_j / s²
        const float *gs = s0.grad_ptr();
        checkNear("∂loss/∂(标量分母) = -Σd/s²",
                  (gs == nullptr) ? 0.0 : static_cast<double>(gs[0]), want_s, 1e-5);
    }
    {
        // 分子侧：loss = Σ(s / d_j) ⇒ ∂loss/∂s = Σ 1/d_j
        Tensor d(ShapeTag{}, {3});
        fillSeq(d, 1.0f); // 1,2,3
        d.requires_grad(true);
        Tensor s0(ShapeTag{}, {1});
        s0.data_write<float>()[0] = 2.0f;
        s0.requires_grad(true);

        Tensor loss = (s0 / d).sum();
        AutoGrad::backward(loss.getRelatedNode(), false);

        const double want_s = 1.0 + 1.0 / 2.0 + 1.0 / 3.0;
        const float *gs = s0.grad_ptr();
        checkNear("∂loss/∂(标量分子) = Σ1/d",
                  (gs == nullptr) ? 0.0 : static_cast<double>(gs[0]), want_s, 1e-5);
    }
    {
        // 高维侧：loss = Σ(c_j / Σc²) ⇒ ∂loss/∂c_i = 1/S − 2c_i·Σc_j/S²
        Tensor c(ShapeTag{}, {3});
        fillSeq(c, 1.0f); // 1,2,3
        c.requires_grad(true);

        Tensor n = (c * c).sum(); // {0 维} = 14
        Tensor loss = (c / n).sum();
        AutoGrad::backward(loss.getRelatedNode(), false);

        const double S = 14.0, T = 6.0;
        std::vector<double> want;
        for (int i = 0; i < 3; ++i) {
            want.push_back(1.0 / S - 2.0 * (1.0 + i) * T / (S * S));
        }
        checkGrad("∂(Σ c/Σc²)/∂c 与解析解一致", gradOf(c), want, 1e-4);
    }

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
