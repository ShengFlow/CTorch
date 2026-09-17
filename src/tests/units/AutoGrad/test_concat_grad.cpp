/**
 * @file test_concat_grad.cpp
 * @author 苏璃珞
 * @brief concat 梯度回归测试
 *
 * @details 判据是「拼接之后梯度还能分别回到两个输入」，而不是「前向数值正确」。
 *          与 slice 同类：前向是纯数据搬运，若只补前向而漏掉反向节点，
 *          数值完全正常而梯度恒为零且不报错 —— 仅比前向数值的测试无法发现。
 *
 * 覆盖：
 *  1. dim=0 前向数值正确
 *  2. dim=0 反向：梯度切回两段（散射位置正确）
 *  3. dim=1 前向 + 反向
 *  4. 三维张量 dim=1（外层/内层块结构）
 *  5. 与数值梯度（中心差分）对照
 *  6. 只有一个输入需要梯度时，另一侧不参与（upStream 占位 nullptr）
 *  7. 非连续输入（转置视图）拼接
 *  8. 错误路径：形状/维度/dtype 不匹配
 *  9. concatNoGrad 不建立反向路径
 *
 * @date 2026/9/17
 **/

#include "Tensor.h"
#include "AutoGrad.h"

#include <cmath>
#include <functional>
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
void fillSeq(Tensor &t, float base = 0.0f) {
    float *p = t.data_write<float>();
    for (size_t i = 0; i < t.numel(); ++i) {
        p[i] = base + static_cast<float>(i);
    }
}

/// 逐元素比对连续张量
bool matchContig(const Tensor &t, const float *expect, size_t n) {
    if (!t.is_contiguous()) {
        return false;
    }
    const float *p = t.data<float>();
    if (p == nullptr) {
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        if (std::fabs(p[i] - expect[i]) > 1e-6f) {
            std::cout << "      位置 " << i << " 实为 " << p[i] << " 期望 " << expect[i]
                      << "\n";
            return false;
        }
    }
    return true;
}

/// 逐元素比对梯度（带打印）
bool matchGrad(const float *g, const float *expect, size_t n) {
    if (g == nullptr) {
        return false;
    }
    bool ok = true;
    for (size_t i = 0; i < n; ++i) {
        if (std::fabs(g[i] - expect[i]) > 1e-6f) {
            ok = false;
            std::cout << "      位置 " << i << " 梯度 " << g[i] << " 期望 " << expect[i]
                      << "\n";
        }
    }
    return ok;
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    std::cout << "========================================\n";
    std::cout << "concat 梯度回归\n";
    std::cout << "========================================\n";

    // ---- 1. dim=0 前向 ----
    std::cout << "\n[1] dim=0 前向\n";
    {
        Tensor a(ShapeTag{}, {2, 3});
        Tensor b(ShapeTag{}, {3, 3});
        fillSeq(a, 0.0f);  // 0..5
        fillSeq(b, 10.0f); // 10..18

        Tensor c = a.concat(b, 0);
        checkTrue("形状为 {5,3}",
                  c.sizes().size() == 2 && c.sizes()[0] == 5 && c.sizes()[1] == 3);
        const float expect[15] = {0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18};
        checkTrue("按行首尾相接", matchContig(c, expect, 15));
    }

    // ---- 2. dim=0 反向 ----
    std::cout << "\n[2] dim=0 反向（漏反向节点时梯度恒为 0）\n";
    {
        Tensor a(ShapeTag{}, {2, 3});
        Tensor b(ShapeTag{}, {3, 3});
        fillSeq(a);
        fillSeq(b, 10.0f);
        a.requires_grad(true);
        b.requires_grad(true);

        Tensor c = a.concat(b, 0);
        Tensor loss = c.sum();
        AutoGrad::backward(loss.getRelatedNode(), false);

        const float *ga = a.grad_ptr();
        const float *gb = b.grad_ptr();
        checkTrue("第一个输入拿到梯度", ga != nullptr);
        checkTrue("第二个输入拿到梯度", gb != nullptr);
        if (ga != nullptr) {
            const float ea[6] = {1, 1, 1, 1, 1, 1};
            checkTrue("∂L/∂a 全为 1（6 个元素）", matchGrad(ga, ea, 6));
        }
        if (gb != nullptr) {
            const float eb[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
            checkTrue("∂L/∂b 全为 1（9 个元素）", matchGrad(gb, eb, 9));
        }
    }

    // ---- 3. dim=1 前向 + 反向 ----
    std::cout << "\n[3] dim=1 前向与反向\n";
    {
        Tensor a(ShapeTag{}, {2, 2});
        Tensor b(ShapeTag{}, {2, 3});
        fillSeq(a, 0.0f);  // [[0,1],[2,3]]
        fillSeq(b, 10.0f); // [[10,11,12],[13,14,15]]
        a.requires_grad(true);
        b.requires_grad(true);

        Tensor c = a.concat(b, 1);
        checkTrue("形状为 {2,5}",
                  c.sizes().size() == 2 && c.sizes()[0] == 2 && c.sizes()[1] == 5);
        const float expect[10] = {0, 1, 10, 11, 12, 2, 3, 13, 14, 15};
        checkTrue("逐行左段接右段", matchContig(c, expect, 10));

        Tensor loss = c.square().sum();
        AutoGrad::backward(loss.getRelatedNode(), false);
        const float *ga = a.grad_ptr();
        const float *gb = b.grad_ptr();
        if (ga != nullptr) {
            const float ea[4] = {0, 2, 4, 6}; // 2*x
            checkTrue("∂L/∂a = 2a（仅左段）", matchGrad(ga, ea, 4));
        } else {
            checkTrue("∂L/∂a = 2a（仅左段）", false);
        }
        if (gb != nullptr) {
            const float eb[6] = {20, 22, 24, 26, 28, 30}; // 2*(10..15)
            checkTrue("∂L/∂b = 2b（仅右段）", matchGrad(gb, eb, 6));
        } else {
            checkTrue("∂L/∂b = 2b（仅右段）", false);
        }
    }

    // ---- 4. 三维张量 dim=1（外层/内层块结构）----
    std::cout << "\n[4] 三维张量 dim=1\n";
    {
        // a: {2,2,2} = 0..7, b: {2,1,2} = 100..103
        // c[o] = [ a[o][0], a[o][1], b[o][0] ]，每个外层块 3*2 = 6 个元素
        //
        // requires_grad 必须在 concat **之前**置位：前向注册节点的判据是「任一侧
        // 需要梯度」，若前向时两侧都还不需要梯度，ConcatNode 不会被建立，之后
        // 再 requires_grad(true) 也补不回来（反向起点因此为 nullptr）。
        Tensor a(ShapeTag{}, {2, 2, 2});
        Tensor b(ShapeTag{}, {2, 1, 2});
        fillSeq(a, 0.0f);
        fillSeq(b, 100.0f);
        a.requires_grad(true);
        b.requires_grad(true);

        Tensor c = a.concat(b, 1);
        checkTrue("形状为 {2,3,2}",
                  c.sizes().size() == 3 && c.sizes()[0] == 2 && c.sizes()[1] == 3 &&
                      c.sizes()[2] == 2);
        const float expect[12] = {0, 1, 2, 3, 100, 101, 4, 5, 6, 7, 102, 103};
        checkTrue("外层块各自拼接（dim=1 不是简单续接）", matchContig(c, expect, 12));

        Tensor loss = c.sum();
        AutoGrad::backward(loss.getRelatedNode(), false);
        const float *ga = a.grad_ptr();
        const float *gb = b.grad_ptr();
        if (ga != nullptr) {
            const float ea[8] = {1, 1, 1, 1, 1, 1, 1, 1};
            checkTrue("∂L/∂a 全 1（8 个元素）", matchGrad(ga, ea, 8));
        } else {
            checkTrue("∂L/∂a 全 1（8 个元素）", false);
        }
        if (gb != nullptr) {
            const float eb[4] = {1, 1, 1, 1};
            checkTrue("∂L/∂b 全 1（4 个元素）", matchGrad(gb, eb, 4));
        } else {
            checkTrue("∂L/∂b 全 1（4 个元素）", false);
        }
    }

    // ---- 5. 数值梯度对照 ----
    std::cout << "\n[5] 与数值梯度对照\n";
    {
        const float init_a[6] = {0.5f, -1.5f, 2.0f, -0.25f, 3.0f, 1.25f};
        const float init_b[4] = {1.0f, -2.0f, 0.75f, 4.0f};

        // f(a,b) = Σ (concat(a,b,0))²   ⇒ ∂f/∂a = 2a, ∂f/∂b = 2b
        auto f = [&](const std::vector<float> &va, const std::vector<float> &vb) {
            Tensor ta(ShapeTag{}, {3, 2});
            Tensor tb(ShapeTag{}, {2, 2});
            float *pa = ta.data_write<float>();
            float *pb = tb.data_write<float>();
            for (size_t i = 0; i < va.size(); ++i) {
                pa[i] = va[i];
            }
            for (size_t i = 0; i < vb.size(); ++i) {
                pb[i] = vb[i];
            }
            Tensor tc = ta.concat(tb, 0);
            return tc.square().sum().data<float>()[0];
        };

        Tensor a(ShapeTag{}, {3, 2});
        Tensor b(ShapeTag{}, {2, 2});
        {
            float *pa = a.data_write<float>();
            float *pb = b.data_write<float>();
            for (int i = 0; i < 6; ++i) {
                pa[i] = init_a[i];
            }
            for (int i = 0; i < 4; ++i) {
                pb[i] = init_b[i];
            }
        }
        a.requires_grad(true);
        b.requires_grad(true);
        Tensor c = a.concat(b, 0);
        Tensor loss = c.square().sum();
        AutoGrad::backward(loss.getRelatedNode(), false);
        const float *ga = a.grad_ptr();
        const float *gb = b.grad_ptr();

        const double h = 1e-3;
        double max_err = 0.0;
        std::vector<float> va(init_a, init_a + 6), vb(init_b, init_b + 4);
        for (int i = 0; i < 6; ++i) {
            std::vector<float> p = va, m = va;
            p[i] += static_cast<float>(h);
            m[i] -= static_cast<float>(h);
            const double num = (f(p, vb) - f(m, vb)) / (2.0 * h);
            max_err = std::max(max_err, std::fabs(num - static_cast<double>(ga[i])));
        }
        for (int i = 0; i < 4; ++i) {
            std::vector<float> p = vb, m = vb;
            p[i] += static_cast<float>(h);
            m[i] -= static_cast<float>(h);
            const double num = (f(va, p) - f(va, m)) / (2.0 * h);
            max_err = std::max(max_err, std::fabs(num - static_cast<double>(gb[i])));
        }
        std::cout << "    最大偏差 = " << max_err << "\n";
        checkTrue("两侧解析梯度与中心差分一致（< 1e-2）", max_err < 1e-2);
    }

    // ---- 6. 仅一个输入需要梯度 ----
    std::cout << "\n[6] 仅一个输入需要梯度\n";
    {
        Tensor a(ShapeTag{}, {2, 2});
        Tensor b(ShapeTag{}, {2, 2});
        fillSeq(a);
        fillSeq(b, 10.0f);
        a.requires_grad(true); // b 不需要

        Tensor c = a.concat(b, 0);
        Tensor loss = c.sum();
        AutoGrad::backward(loss.getRelatedNode(), false);

        const float *ga = a.grad_ptr();
        const float ones4[4] = {1, 1, 1, 1};
        checkTrue("需要梯度的一侧拿到梯度（2x2 全 1）",
                  ga != nullptr && matchGrad(ga, ones4, 4));
        checkTrue("不需要梯度的一侧保持无梯度", b.grad_ptr() == nullptr);
    }

    // ---- 7. 非连续输入 ----
    std::cout << "\n[7] 非连续输入（转置视图）\n";
    {
        Tensor x(ShapeTag{}, {3, 4});
        fillSeq(x); // 0..11
        Tensor t = x.transpose(0, 1); // {4,3} 非连续
        checkTrue("转置结果非连续", !t.is_contiguous());

        Tensor b(ShapeTag{}, {2, 3});
        fillSeq(b, 100.0f);
        Tensor c = t.concat(b, 0); // {6,3}

        checkTrue("形状为 {6,3}",
                  c.sizes().size() == 2 && c.sizes()[0] == 6 && c.sizes()[1] == 3);

        // t[i][j] = x[j][i] ⇒ t 的第 0 行为 x 的第 0 列 = {0,4,8}
        // t 的第 3 行为 x 的第 3 列 = {3,7,11}；b 的两行接在其后
        const float expect[18] = {0, 4,  8,  1, 5,  9,  2, 6,  10,
                                  3, 7,  11, 100, 101, 102, 103, 104, 105};
        checkTrue("非连续输入按逻辑索引搬运正确", matchContig(c, expect, 18));

        // 反向：非连续侧的梯度需先物化再回传，这里只断言「不需要梯度的那侧不崩、
        // 需要梯度的 b 侧梯度正确」——非连续侧回传给 TransposeNode 受 CTorch 既有
        // 梯度布局约定限制（见 test_slice_grad [7]），不在本测试内断言。
        x.requires_grad(true);
        Tensor t2 = x.transpose(0, 1);
        Tensor b2(ShapeTag{}, {2, 3});
        fillSeq(b2, 100.0f);
        b2.requires_grad(true);
        Tensor c2 = t2.concat(b2, 0);
        Tensor loss = c2.sum();
        AutoGrad::backward(loss.getRelatedNode(), false);
        const float *gb = b2.grad_ptr();
        const float ones6[6] = {1, 1, 1, 1, 1, 1};
        checkTrue("非连续侧参与前向时，另一侧梯度仍正确",
                  gb != nullptr && matchGrad(gb, ones6, 6));
    }

    // ---- 8. 错误路径 ----
    std::cout << "\n[8] 错误路径\n";
    {
        auto expectThrow = [&](const char *name, const std::function<void()> &fn) {
            bool threw = false;
            try {
                fn();
            } catch (const std::exception &) {
                threw = true;
            }
            checkTrue(name, threw);
        };

        Tensor a(ShapeTag{}, {2, 3});
        Tensor wrong_cols(ShapeTag{}, {2, 4});
        Tensor wrong_rank(ShapeTag{}, {2, 3, 1});
        expectThrow("除拼接维外形状不一致时抛异常",
                    [&] { (void)a.concat(wrong_cols, 0); });
        expectThrow("维度数不一致时抛异常", [&] { (void)a.concat(wrong_rank, 0); });
        expectThrow("维度越界时抛异常", [&] { (void)a.concat(wrong_cols, 5); });

        Tensor same(ShapeTag{}, {2, 3});
        Tensor c = a.concat(same, 1);
        checkTrue("同形状 dim=1 拼接形状为 {2,6}",
                  c.sizes().size() == 2 && c.sizes()[0] == 2 && c.sizes()[1] == 6);
    }

    // ---- 9. concatNoGrad 不建立反向路径 ----
    std::cout << "\n[9] NoGrad 版本\n";
    {
        Tensor a(ShapeTag{}, {2, 2});
        Tensor b(ShapeTag{}, {2, 2});
        fillSeq(a);
        fillSeq(b, 10.0f);
        a.requires_grad(true);
        b.requires_grad(true);

        // NoGrad 版本不注册任何节点，因此它**连图都不存在** —— 反向起点为空。
        // 这里不调用 backward(nullptr)（那属于调用方错误，本测试不制造该错误），
        // 而是直接断言图确实没有建起来。
        Tensor c = a.concatNoGrad(b, 0);
        checkTrue("concatNoGrad 结果没有 autograd 节点", c.getRelatedNode() == nullptr);

        Tensor loss = c.sum();
        checkTrue("其下游同样不建图（loss 无节点）", loss.getRelatedNode() == nullptr);

        // 行为层面的对照：带梯度版本必须建图并回传梯度。
        Tensor c2 = a.concat(b, 0);
        Tensor loss2 = c2.sum();
        AutoGrad::backward(loss2.getRelatedNode(), false);
        const float *ga2 = a.grad_ptr();
        checkTrue("带梯度版本建立了反向路径（梯度非零）",
                  ga2 != nullptr && std::fabs(ga2[0]) > 1e-6f);
    }

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
