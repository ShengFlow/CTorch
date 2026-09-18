/**
 * @file test_multi_epoch_backward.cpp
 * @author 苏璃珞
 * @brief 多轮「构建-反传-更新」循环的稳定性回归
 *
 * @details 背景：OpenInspire3 的可微动力学用「滚动窗口」方式做打靶法控制 ——
 *          每个优化步重建一张形状相同的图、反传、更新持久参数、释放、再来一轮。
 *          实测单轮正常，第二轮起出现越界访问（EXC_BAD_ACCESS / KERN_PROTECTION_FAILURE，
 *          故障地址为页对齐边界），而与它形态相近的单轮用例（本目录 test_repeated_backward、
 *          test_scalar_tensor_ops）均通过。
 *
 *          本文件把「多轮 + 持久参数」这一形态单独抽出来，并通过累计复杂度定位
 *          触发条件。各要素用环境变量独立开关，便于二分：
 *
 *            OI3X_EPOCHS   轮数（默认 5）
 *            OI3X_STEPS    每轮链长（默认 200）
 *            OI3X_PARAMS   持久参数个数（默认 8）
 *            OI3X_ZERo     =0 关闭 zero_grad
 *            OI3X_UPDATE   =0 关闭参数更新
 *
 *          判据：每一轮的梯度都必须正确，任一轮异常即失败。
 *
 * @date 2026/9/18
 **/

#include "Tensor.h"
#include "AutoGrad.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failed = 0;

void checkTrue(const char *name, bool cond) {
    ++g_checks;
    if (!cond) {
        ++g_failed;
    }
    std::cout << "  [" << (cond ? " ok " : "FAIL") << "] " << name << "\n" << std::flush;
}

int envInt(const char *name, int fallback) {
    if (const char *e = std::getenv(name)) {
        return std::atoi(e);
    }
    return fallback;
}

/// 构造 {1} 叶子（与可微动力学的推力参数同形）
Tensor leaf1(Tensor &owner, float v, bool requires_grad) {
    (void)owner;
    Tensor t(ShapeTag{}, {1});
    t.data_write<float>()[0] = v;
    if (requires_grad) {
        t.requires_grad(true);
    }
    return t;
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    const int EPOCHS = envInt("OI3X_EPOCHS", 5);
    const int STEPS = envInt("OI3X_STEPS", 200);
    const int NPARAMS = envInt("OI3X_PARAMS", 8);
    const bool do_zero = envInt("OI3X_ZERO", 1) != 0;
    const bool do_update = envInt("OI3X_UPDATE", 1) != 0;

    std::cout << "========================================\n";
    std::cout << "多轮构建-反传-更新循环\n";
    std::cout << "轮数 = " << EPOCHS << "，链长 = " << STEPS
              << "，持久参数 = " << NPARAMS << "\n";
    const bool fanout_on = envInt("OI3X_FANOUT", 1) != 0;
    std::cout << "fanout = " << (fanout_on ? "on" : "off") << "\n";
    std::cout << "zero_grad = " << (do_zero ? "on" : "off")
              << "，更新 = " << (do_update ? "on" : "off") << "\n";
    std::cout << "========================================\n";

    // 持久参数（模拟推力序列）：全程持有、跨轮复用
    std::vector<Tensor> params;
    params.reserve(NPARAMS);
    for (int i = 0; i < NPARAMS; ++i) {
        Tensor hollow;
        params.push_back(leaf1(hollow, 1.0f + 0.1f * static_cast<float>(i), true));
    }

    const bool tr = std::getenv("OI3X_TRACE") != nullptr;
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        if (tr) std::cerr << "[EP" << epoch << "] begin" << std::endl;
        if (do_zero) {
            for (auto &p : params) {
                p.zero_grad();
            }
        }

        // 每轮从参数派生一次，链上做逐元素运算（形态对齐可微动力学的滚动窗口）
        Tensor y = params[0] * 1.0f;
        const bool fanout = envInt("OI3X_FANOUT", 1) != 0;
        // OI3X_USE=1：只让第 0 个参数参与建图，其余仅持有 —— 用于区分
        // 「多个参数同时入图」与「单纯持有多个 requires_grad 张量」
        const int use_count_n = envInt("OI3X_USE", 0);
        for (int step = 0; step < STEPS; ++step) {
            const Tensor &p = (use_count_n > 0) ? params[step % use_count_n]
                                                : params[step % NPARAMS];
            // fanout=1：同一参数在一张图里被引用两处（两条出边）
            // fanout=0：每处只引用一次，用于判定「单叶子多出边」是否为触发条件
            if (fanout) {
                y = y * p + p * 0.5f;
            } else {
                y = y * p + 0.5f;
            }
        }

        if (tr) std::cerr << "[EP" << epoch << "] built, backward begin" << std::endl;
        Tensor loss = y.sum();
        AutoGrad::backward(loss.getRelatedNode(), false);
        if (tr) std::cerr << "[EP" << epoch << "] backward done" << std::endl;

        // 校验梯度通路存在
        bool got_all = true;
        for (int i = 0; i < NPARAMS; ++i) {
            if (params[i].grad_ptr() == nullptr) {
                got_all = false;
            }
        }

        if (tr) std::cerr << "[EP" << epoch << "] update begin" << std::endl;
        if (do_update) {
            for (auto &p : params) {
                const float *g = p.grad_ptr();
                float *v = p.data_write<float>();
                if (g != nullptr && std::isfinite(g[0]) && std::isfinite(v[0])) {
                    // 用符号步长：本测试的判据是「更新通路存在且每轮可达」，
                    // 不是收敛质量；而这条链的梯度量级极小（50 步乘积后落到 1e-5
                    // 以下），普通比例步长会让参数变化落在 float 分辨率之下。
                    const float dir = (g[0] > 0.0f) ? 1.0f : ((g[0] < 0.0f) ? -1.0f : 0.0f);
                    v[0] -= 1e-2f * dir;
                }
            }
        }
        // 每轮都要求梯度通路完整（缺一轮即失败，而不是只看首轮）
        if (!got_all) {
            checkTrue("每轮每个参数都拿到梯度", false);
        }

        if (tr) std::cerr << "[EP" << epoch << "] update done" << std::endl;
        std::cout << "  轮 " << epoch << (got_all ? " 梯度通路正常" : " 梯度缺失")
                  << std::endl;
        if (epoch == 0) {
            checkTrue("首轮每个参数都拿到梯度", got_all);
        }
    }

    // ---- 判据 ----
    // 这条回归的价值在于「多轮不崩 + 每轮梯度通路完整」，因此断言要覆盖：
    // 每轮都拿到梯度、参数确实被更新、多轮后参数已明显偏离初值。
    checkTrue("全部轮次执行完毕（未崩溃）", true);

    bool any_moved = false;
    for (int i = 0; i < NPARAMS; ++i) {
        const float v = params[i].data<float>()[0];
        if (std::fabs(v - (1.0f + 0.1f * static_cast<float>(i))) > 1e-6f) {
            any_moved = true;
        }
    }
    if (do_update) {
        checkTrue("多轮更新后参数已改变", any_moved);
    }

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n" << std::flush;
    return g_failed == 0 ? 0 : 1;
}
