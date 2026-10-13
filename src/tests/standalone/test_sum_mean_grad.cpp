/**
 * @file test_sum_mean_grad.cpp
 * @brief sum()/mean() 反向传播梯度回归测试
 * @details 修复 2026-09-06: sum() 经 dot(ones) 挂 DotNode(从未实现) / mean() 裸循环无节点,
 *          两者 backward 均静默断链。修复后 sum()/mean() 挂 SumNode/MeanNode。
 *          本测试验证: 全 reduce sum/mean 的梯度正确广播、链式传播正常。
 * @date 2026-09-06
 */
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "Tensor.h"
#include "AutoGrad.h"

using namespace ct;

static int g_fails = 0;
#define CHECK(cond, msg)                                    \
    do {                                                    \
        if (!(cond)) {                                      \
            std::cerr << "FAIL: " << msg << "\n";           \
            ++g_fails;                                      \
        } else {                                            \
            std::cout << "PASS: " << msg << "\n";           \
        }                                                   \
    } while (0)

int main() {
    // Test 1: sum 前向值 + 梯度全 1
    {
        Tensor x(ShapeTag{}, {3, 4}, DType::kFloat, DeviceType::kCPU);
        float* p = x.data_write<float>();
        for (int i = 0; i < 12; ++i) p[i] = i * 0.5f;  // sum = 0..5.5 = 33
        x.requires_grad(true);
        Tensor l = x.sum();
        CHECK(std::abs(l.item<float>() - 33.0f) < 1e-4f, "sum 前向值 = 33");
        AutoGrad::backward(l.getRelatedNode(), false);
        const float* g = x.grad_ptr();
        CHECK(g != nullptr, "sum backward 填了梯度");
        bool ok = (g != nullptr);
        for (int i = 0; ok && i < 12; ++i) ok = (g[i] == 1.0f);
        CHECK(ok, "sum 梯度全 1");
    }

    // Test 2: mean 梯度全 1/n
    {
        Tensor x(ShapeTag{}, {3, 4}, DType::kFloat, DeviceType::kCPU);
        float* p = x.data_write<float>();
        for (int i = 0; i < 12; ++i) p[i] = 1.0f + i;
        x.requires_grad(true);
        Tensor l = x.mean();
        AutoGrad::backward(l.getRelatedNode(), false);
        const float* g = x.grad_ptr();
        CHECK(g != nullptr, "mean backward 填了梯度");
        bool ok = (g != nullptr);
        for (int i = 0; ok && i < 12; ++i) ok = (std::abs(g[i] - 1.0f / 12.0f) < 1e-6f);
        CHECK(ok, "mean 梯度全 1/12");
    }

    // Test 3: 链式 relu→sum, 梯度 = relu mask
    {
        Tensor x(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
        float* p = x.data_write<float>();
        const float vals[6] = {-2.0f, -0.5f, 0.3f, 1.5f, -1.0f, 2.0f};
        for (int i = 0; i < 6; ++i) p[i] = vals[i];
        x.requires_grad(true);
        Tensor y = x.relu();
        Tensor l = y.sum();
        AutoGrad::backward(l.getRelatedNode(), false);
        const float* g = x.grad_ptr();
        CHECK(g != nullptr, "链式 relu→sum 梯度存在");
        bool ok = (g != nullptr);
        for (int i = 0; ok && i < 6; ++i) {
            float expect = (vals[i] > 0.0f) ? 1.0f : 0.0f;
            if (std::abs(g[i] - expect) > 1e-6f) ok = false;
        }
        CHECK(ok, "relu→sum 梯度 = relu mask");
    }

    // Test 4: sum 作 loss 后参数 SGD 更新真实发生(grad 非空且可更新)
    {
        Tensor w(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU);
        float* wp = w.data_write<float>();
        for (int i = 0; i < 4; ++i) wp[i] = 1.0f;
        w.requires_grad(true);
        Tensor x(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU);
        float* xp = x.data_write<float>();
        for (int i = 0; i < 4; ++i) xp[i] = 2.0f;
        Tensor out = x * w;          // MulNode, 梯度流到 w
        Tensor l = out.sum();
        AutoGrad::backward(l.getRelatedNode(), false);
        float* gp = w.grad_ptr();
        CHECK(gp != nullptr, "w 收到梯度(x=2 全 → grad 全 2)");
        bool ok = (gp != nullptr);
        for (int i = 0; ok && i < 4; ++i) ok = (std::abs(gp[i] - 2.0f) < 1e-6f);
        CHECK(ok, "w 梯度全 2");
    }

    // Test 5: sum(dim=1) 前向值 + 梯度全 1
    {
        Tensor x(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
        float* p = x.data_write<float>();
        for (int i = 0; i < 6; ++i) p[i] = i + 1.0f;  // [[1,2,3],[4,5,6]]
        x.requires_grad(true);
        Tensor l = x.sum(1);
        CHECK(l.numel() == 2 && std::abs(l.data_read<float>()[0] - 6.0f) < 1e-5f &&
                  std::abs(l.data_read<float>()[1] - 15.0f) < 1e-5f,
              "sum(dim=1) 前向 = [6,15]");
        AutoGrad::backward(l.getRelatedNode(), false);
        const float* g = x.grad_ptr();
        bool ok = (g != nullptr);
        for (int i = 0; ok && i < 6; ++i) ok = (g[i] == 1.0f);
        CHECK(ok, "sum(dim=1) 梯度全 1");
    }

    // Test 6: mean(dim=0) 前向值 + 梯度全 1/n
    {
        Tensor x(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
        float* p = x.data_write<float>();
        for (int i = 0; i < 6; ++i) p[i] = i + 1.0f;  // [[1,2,3],[4,5,6]]
        x.requires_grad(true);
        Tensor l = x.mean(0);
        const float* lv = l.data_read<float>();
        CHECK(l.numel() == 3 && std::abs(lv[0] - 2.5f) < 1e-5f && std::abs(lv[1] - 3.5f) < 1e-5f &&
                  std::abs(lv[2] - 4.5f) < 1e-5f,
              "mean(dim=0) 前向 = [2.5,3.5,4.5]");
        AutoGrad::backward(l.getRelatedNode(), false);
        const float* g = x.grad_ptr();
        bool ok = (g != nullptr);
        for (int i = 0; ok && i < 6; ++i) ok = (std::abs(g[i] - 0.5f) < 1e-6f);
        CHECK(ok, "mean(dim=0) 梯度全 1/2");
    }

    // Test 7: 链式 relu→sum(dim=1), 梯度 = relu mask
    {
        Tensor x(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
        float* p = x.data_write<float>();
        const float vals[6] = {-2.0f, 0.5f, 1.0f, 3.0f, -1.0f, 0.0f};
        for (int i = 0; i < 6; ++i) p[i] = vals[i];
        x.requires_grad(true);
        Tensor y = x.relu();
        Tensor l = y.sum(1);
        AutoGrad::backward(l.getRelatedNode(), false);
        const float* g = x.grad_ptr();
        bool ok = (g != nullptr);
        for (int i = 0; ok && i < 6; ++i) {
            float expect = (vals[i] > 0.0f) ? 1.0f : 0.0f;
            if (std::abs(g[i] - expect) > 1e-6f) ok = false;
        }
        CHECK(ok, "relu→sum(dim=1) 梯度 = relu mask");
    }

    // Test 8: sum(dims={0,1}) 前向 = 全和, 梯度全 1 (多级 DimReduceNode 链)
    {
        Tensor x(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
        float* p = x.data_write<float>();
        for (int i = 0; i < 6; ++i) p[i] = i + 1.0f;  // 总和 21
        x.requires_grad(true);
        Tensor l = x.sum(std::vector<int>{0, 1});
        CHECK(std::abs(l.item<float>() - 21.0f) < 1e-5f, "sum({0,1}) 前向 = 21");
        AutoGrad::backward(l.getRelatedNode(), false);
        const float* g = x.grad_ptr();
        bool ok = (g != nullptr);
        for (int i = 0; ok && i < 6; ++i) ok = (g[i] == 1.0f);
        CHECK(ok, "sum({0,1}) 梯度全 1");
    }

    // Test 9: mean(dims={0,1}) 前向 = 全均值, 梯度全 1/6
    {
        Tensor x(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
        float* p = x.data_write<float>();
        for (int i = 0; i < 6; ++i) p[i] = i + 1.0f;  // 均值 3.5
        x.requires_grad(true);
        Tensor l = x.mean(std::vector<int>{0, 1});
        CHECK(std::abs(l.item<float>() - 3.5f) < 1e-5f, "mean({0,1}) 前向 = 3.5");
        AutoGrad::backward(l.getRelatedNode(), false);
        const float* g = x.grad_ptr();
        bool ok = (g != nullptr);
        for (int i = 0; ok && i < 6; ++i) ok = (std::abs(g[i] - 1.0f / 6.0f) < 1e-6f);
        CHECK(ok, "mean({0,1}) 梯度全 1/6");
    }

    // Test 10: 双输入都 requires_grad 时 mul→sum 是否断链(FFN bench 断链根因定位)
    {
        Tensor x(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU);
        float* xp = x.data_write<float>();
        for (int i = 0; i < 4; ++i) xp[i] = 1.0f + i;
        x.requires_grad(true);
        Tensor w(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU);
        float* wp = w.data_write<float>();
        for (int i = 0; i < 4; ++i) wp[i] = 2.0f + i;
        w.requires_grad(true);
        Tensor out = x * w;
        Tensor l = out.sum();
        AutoGrad::backward(l.getRelatedNode(), false);
        const float* gx = x.grad_ptr();
        const float* gw = w.grad_ptr();
        CHECK(gx != nullptr, "双 requires_grad mul→sum: x 收到梯度");
        bool okx = (gx != nullptr);
        for (int i = 0; okx && i < 4; ++i) okx = (std::abs(gx[i] - (2.0f + i)) < 1e-6f);
        CHECK(okx, "x 梯度 = w 值");
        bool okw = (gw != nullptr);
        for (int i = 0; okw && i < 4; ++i) okw = (std::abs(gw[i] - (1.0f + i)) < 1e-6f);
        CHECK(okw, "w 梯度 = x 值");
    }

    // Test: dot 反向断链回归(§4.97 ⑥)
    {
        AutoGrad::EnableGrad = true;
        Tensor x(ShapeTag{}, {3}, DType::kFloat, DeviceType::kCPU);
        Tensor w(ShapeTag{}, {3}, DType::kFloat, DeviceType::kCPU);
        for (int i = 0; i < 3; ++i) {
            x.data_write<float>()[i] = (float)(i + 1);
            w.data_write<float>()[i] = 10.0f * (float)(i + 1);
        }
        x.requires_grad(true);
        w.requires_grad(true);
        Tensor y = x.dot(w);
        CHECK(std::abs(y.item<float>() - (10.0f + 40.0f + 90.0f)) < 1e-4f, "dot 前向 = 140");
        CHECK(y.getRelatedNode() != nullptr, "dot 挂节点(修复前为 null, 反向断链)");
        AutoGrad::backward(y.getRelatedNode(), false);
        const float* gx = x.grad_ptr();
        const float* gw = w.grad_ptr();
        CHECK(gx != nullptr, "dot 后 x 收到梯度(修复前无)");
        bool okx = (gx != nullptr);
        for (int i = 0; okx && i < 3; ++i) okx = (std::abs(gx[i] - 10.0f * (float)(i + 1)) < 1e-5f);
        CHECK(okx, "dot grad_x = w");
        bool okw = (gw != nullptr);
        for (int i = 0; okw && i < 3; ++i) okw = (std::abs(gw[i] - (float)(i + 1)) < 1e-5f);
        CHECK(okw, "dot grad_w = x");
    }

    // Test 11: R1 非连续视图 sum(1) 与 sum(0) 前向数值精确度及 backward 梯度校验
    {
        AutoGrad::EnableGrad = true;
        Tensor t(ShapeTag{}, {10, 4}, DType::kFloat, DeviceType::kCPU);
        float* p = t.data_write<float>();
        for (int i = 0; i < 40; ++i) p[i] = static_cast<float>(i + 1);

        Tensor tt = t.transpose(0, 1); // shape {4, 10}, non-contiguous
        CHECK(!tt.is_contiguous(), "t(10, 4).transpose(0, 1) 为非连续视图");

        // sum(1): 归约维度 1 (长度 10), 输出形状 {4}
        // 参考值: sum1[i] = t 的第 i 列之和 = sum_{r=0}^9 (r * 4 + i + 1)
        // sum1 = [190, 200, 210, 220]
        Tensor s1 = tt.sum(1);
        CHECK(s1.numel() == 4, "tt.sum(1) 元素数 = 4");
        const float* s1_ptr = s1.data_read<float>();
        const float ref_s1[4] = {190.0f, 200.0f, 210.0f, 220.0f};
        bool s1_ok = true;
        for (int i = 0; i < 4; ++i) {
            if (std::abs(s1_ptr[i] - ref_s1[i]) > 1e-4f) s1_ok = false;
        }
        CHECK(s1_ok, "tt.sum(1) 前向值匹配参考值 [190, 200, 210, 220]");

        // sum(0): 归约维度 0 (长度 4), 输出形状 {10}
        // 参考值: sum0[j] = t 的第 j 行之和 = sum_{c=0}^3 (j * 4 + c + 1)
        // sum0 = [10, 26, 42, 58, 74, 90, 106, 122, 138, 154]
        Tensor s0 = tt.sum(0);
        CHECK(s0.numel() == 10, "tt.sum(0) 元素数 = 10");
        const float* s0_ptr = s0.data_read<float>();
        const float ref_s0[10] = {10.0f, 26.0f, 42.0f, 58.0f, 74.0f, 90.0f, 106.0f, 122.0f, 138.0f, 154.0f};
        bool s0_ok = true;
        for (int i = 0; i < 10; ++i) {
            if (std::abs(s0_ptr[i] - ref_s0[i]) > 1e-4f) s0_ok = false;
        }
        CHECK(s0_ok, "tt.sum(0) 前向值匹配参考值 [10, 26, ..., 154]");

        // backward 校验 for tt.sum(1)
        {
            Tensor t_grad(ShapeTag{}, {10, 4}, DType::kFloat, DeviceType::kCPU);
            float* tp = t_grad.data_write<float>();
            for (int i = 0; i < 40; ++i) tp[i] = static_cast<float>(i + 1);
            t_grad.requires_grad(true);

            Tensor tt_grad = t_grad.transpose(0, 1);
            Tensor loss1 = tt_grad.sum(1).sum();
            AutoGrad::backward(loss1.getRelatedNode(), false);
            const float* g = t_grad.grad_ptr();
            CHECK(g != nullptr, "tt.sum(1).sum() backward 填了梯度");
            bool g_ok = (g != nullptr);
            for (int i = 0; g_ok && i < 40; ++i) {
                if (std::abs(g[i] - 1.0f) > 1e-5f) g_ok = false;
            }
            CHECK(g_ok, "tt.sum(1) backward 叶子梯度全 1.0");
        }

        // backward 校验 for tt.sum(0)
        {
            Tensor t_grad(ShapeTag{}, {10, 4}, DType::kFloat, DeviceType::kCPU);
            float* tp = t_grad.data_write<float>();
            for (int i = 0; i < 40; ++i) tp[i] = static_cast<float>(i + 1);
            t_grad.requires_grad(true);

            Tensor tt_grad = t_grad.transpose(0, 1);
            Tensor loss0 = tt_grad.sum(0).sum();
            AutoGrad::backward(loss0.getRelatedNode(), false);
            const float* g = t_grad.grad_ptr();
            CHECK(g != nullptr, "tt.sum(0).sum() backward 填了梯度");
            bool g_ok = (g != nullptr);
            for (int i = 0; g_ok && i < 40; ++i) {
                if (std::abs(g[i] - 1.0f) > 1e-5f) g_ok = false;
            }
            CHECK(g_ok, "tt.sum(0) backward 叶子梯度全 1.0");
        }
    }

    // Test 12: R1 非连续视图 mean(1) 与 mean(0) 前向数值精确度及 backward 梯度校验
    {
        AutoGrad::EnableGrad = true;
        Tensor t(ShapeTag{}, {10, 4}, DType::kFloat, DeviceType::kCPU);
        float* p = t.data_write<float>();
        for (int i = 0; i < 40; ++i) p[i] = static_cast<float>(i + 1);

        Tensor tt = t.transpose(0, 1); // shape {4, 10}

        // mean(1): 归约维度 1 (长度 10), 输出形状 {4}
        // 参考值: mean1[i] = ref_s1[i] / 10.0 = [19, 20, 21, 22]
        Tensor m1 = tt.mean(1);
        CHECK(m1.numel() == 4, "tt.mean(1) 元素数 = 4");
        const float* m1_ptr = m1.data_read<float>();
        const float ref_m1[4] = {19.0f, 20.0f, 21.0f, 22.0f};
        bool m1_ok = true;
        for (int i = 0; i < 4; ++i) {
            if (std::abs(m1_ptr[i] - ref_m1[i]) > 1e-4f) m1_ok = false;
        }
        CHECK(m1_ok, "tt.mean(1) 前向值匹配参考值 [19, 20, 21, 22]");

        // mean(0): 归约维度 0 (长度 4), 输出形状 {10}
        // 参考值: mean0[j] = ref_s0[j] / 4.0 = [2.5, 6.5, 10.5, 14.5, 18.5, 22.5, 26.5, 30.5, 34.5, 38.5]
        Tensor m0 = tt.mean(0);
        CHECK(m0.numel() == 10, "tt.mean(0) 元素数 = 10");
        const float* m0_ptr = m0.data_read<float>();
        const float ref_m0[10] = {2.5f, 6.5f, 10.5f, 14.5f, 18.5f, 22.5f, 26.5f, 30.5f, 34.5f, 38.5f};
        bool m0_ok = true;
        for (int i = 0; i < 10; ++i) {
            if (std::abs(m0_ptr[i] - ref_m0[i]) > 1e-4f) m0_ok = false;
        }
        CHECK(m0_ok, "tt.mean(0) 前向值匹配参考值 [2.5, 6.5, ..., 38.5]");

        // backward 校验 for tt.mean(1)
        {
            Tensor t_grad(ShapeTag{}, {10, 4}, DType::kFloat, DeviceType::kCPU);
            float* tp = t_grad.data_write<float>();
            for (int i = 0; i < 40; ++i) tp[i] = static_cast<float>(i + 1);
            t_grad.requires_grad(true);

            Tensor tt_grad = t_grad.transpose(0, 1);
            Tensor loss1 = tt_grad.mean(1).sum();
            AutoGrad::backward(loss1.getRelatedNode(), false);
            const float* g = t_grad.grad_ptr();
            CHECK(g != nullptr, "tt.mean(1).sum() backward 填了梯度");
            bool g_ok = (g != nullptr);
            for (int i = 0; g_ok && i < 40; ++i) {
                if (std::abs(g[i] - 0.1f) > 1e-5f) g_ok = false;
            }
            CHECK(g_ok, "tt.mean(1) backward 叶子梯度全 1/10 = 0.1");
        }

        // backward 校验 for tt.mean(0)
        {
            Tensor t_grad(ShapeTag{}, {10, 4}, DType::kFloat, DeviceType::kCPU);
            float* tp = t_grad.data_write<float>();
            for (int i = 0; i < 40; ++i) tp[i] = static_cast<float>(i + 1);
            t_grad.requires_grad(true);

            Tensor tt_grad = t_grad.transpose(0, 1);
            Tensor loss0 = tt_grad.mean(0).sum();
            AutoGrad::backward(loss0.getRelatedNode(), false);
            const float* g = t_grad.grad_ptr();
            CHECK(g != nullptr, "tt.mean(0).sum() backward 填了梯度");
            bool g_ok = (g != nullptr);
            for (int i = 0; g_ok && i < 40; ++i) {
                if (std::abs(g[i] - 0.25f) > 1e-5f) g_ok = false;
            }
            CHECK(g_ok, "tt.mean(0) backward 叶子梯度全 1/4 = 0.25");
        }
    }

    // Test 13: R1 非连续视图 keepdim=true 测试
    {
        Tensor t(ShapeTag{}, {10, 4}, DType::kFloat, DeviceType::kCPU);
        float* p = t.data_write<float>();
        for (int i = 0; i < 40; ++i) p[i] = static_cast<float>(i + 1);

        Tensor tt = t.transpose(0, 1);

        Tensor sk1 = tt.sum(1, true);
        CHECK(sk1.shape().size() == 2 && sk1.shape()[0] == 4 && sk1.shape()[1] == 1,
              "tt.sum(1, keepdim=true) 形状为 {4, 1}");
        CHECK(std::abs(sk1.data_read<float>()[0] - 190.0f) < 1e-4f, "tt.sum(1, keepdim=true)[0] = 190");

        Tensor sk0 = tt.sum(0, true);
        CHECK(sk0.shape().size() == 2 && sk0.shape()[0] == 1 && sk0.shape()[1] == 10,
              "tt.sum(0, keepdim=true) 形状为 {1, 10}");
        CHECK(std::abs(sk0.data_read<float>()[0] - 10.0f) < 1e-4f, "tt.sum(0, keepdim=true)[0] = 10");

        Tensor mk1 = tt.mean(1, true);
        CHECK(mk1.shape().size() == 2 && mk1.shape()[0] == 4 && mk1.shape()[1] == 1,
              "tt.mean(1, keepdim=true) 形状为 {4, 1}");
        CHECK(std::abs(mk1.data_read<float>()[0] - 19.0f) < 1e-4f, "tt.mean(1, keepdim=true)[0] = 19");

        Tensor mk0 = tt.mean(0, true);
        CHECK(mk0.shape().size() == 2 && mk0.shape()[0] == 1 && mk0.shape()[1] == 10,
              "tt.mean(0, keepdim=true) 形状为 {1, 10}");
        CHECK(std::abs(mk0.data_read<float>()[0] - 2.5f) < 1e-4f, "tt.mean(0, keepdim=true)[0] = 2.5");
    }

    // Test 14: R1 Double 类型非连续视图 sum 与 mean 前向数值精确度
    {
        Tensor td(ShapeTag{}, {10, 4}, DType::kDouble, DeviceType::kCPU);
        double* p = td.data_write<double>();
        for (int i = 0; i < 40; ++i) p[i] = static_cast<double>(i + 1);

        Tensor ttd = td.transpose(0, 1);
        CHECK(!ttd.is_contiguous(), "td.transpose(0, 1) 为非连续 Double 张量");

        Tensor sd1 = ttd.sum(1);
        const double* sd1_p = sd1.data_read<double>();
        bool sd1_ok = (sd1.numel() == 4) &&
                      (std::abs(sd1_p[0] - 190.0) < 1e-9) &&
                      (std::abs(sd1_p[1] - 200.0) < 1e-9) &&
                      (std::abs(sd1_p[2] - 210.0) < 1e-9) &&
                      (std::abs(sd1_p[3] - 220.0) < 1e-9);
        CHECK(sd1_ok, "Double ttd.sum(1) 前向值精确匹配 [190, 200, 210, 220]");

        Tensor sd0 = ttd.sum(0);
        const double* sd0_p = sd0.data_read<double>();
        bool sd0_ok = (sd0.numel() == 10) &&
                      (std::abs(sd0_p[0] - 10.0) < 1e-9) &&
                      (std::abs(sd0_p[9] - 154.0) < 1e-9);
        CHECK(sd0_ok, "Double ttd.sum(0) 前向值精确匹配 [10, ..., 154]");

        Tensor md1 = ttd.mean(1);
        const double* md1_p = md1.data_read<double>();
        bool md1_ok = (md1.numel() == 4) &&
                      (std::abs(md1_p[0] - 19.0) < 1e-9) &&
                      (std::abs(md1_p[3] - 22.0) < 1e-9);
        CHECK(md1_ok, "Double ttd.mean(1) 前向值精确匹配 [19, ..., 22]");

        Tensor md0 = ttd.mean(0);
        const double* md0_p = md0.data_read<double>();
        bool md0_ok = (md0.numel() == 10) &&
                      (std::abs(md0_p[0] - 2.5) < 1e-9) &&
                      (std::abs(md0_p[9] - 38.5) < 1e-9);
        CHECK(md0_ok, "Double ttd.mean(0) 前向值精确匹配 [2.5, ..., 38.5]");
    }

    // Test 15 [Acceptance Criterion Deep Verification]:
    // t(10, 4).transpose(0, 1).sum(1) and .sum(0) exact values matching dense references
    {
        Tensor t(ShapeTag{}, {10, 4}, DType::kFloat, DeviceType::kCPU);
        float* p = t.data_write<float>();
        for (int i = 0; i < 40; ++i) {
            p[i] = (i + 1) * 2.5f - 13.0f;
        }

        Tensor tt = t.transpose(0, 1); // shape {4, 10}
        CHECK(!tt.is_contiguous(), "t(10, 4).transpose(0, 1) is non-contiguous view");

        // Dense reference 4x10 matrix
        float dense_ref[4][10];
        for (int r = 0; r < 10; ++r) {
            for (int c = 0; c < 4; ++c) {
                dense_ref[c][r] = p[r * 4 + c];
            }
        }

        // sum(1): reduce axis 1 -> shape {4}
        Tensor s1 = tt.sum(1);
        CHECK(s1.shape().size() == 1 && s1.shape()[0] == 4, "tt.sum(1) shape {4}");
        const float* s1_ptr = s1.data_read<float>();
        bool s1_match = true;
        for (int r = 0; r < 4; ++r) {
            float exp_val = 0.0f;
            for (int c = 0; c < 10; ++c) exp_val += dense_ref[r][c];
            if (std::abs(s1_ptr[r] - exp_val) > 1e-4f) s1_match = false;
        }
        CHECK(s1_match, "tt.sum(1) mathematically exact matching dense reference");

        // sum(0): reduce axis 0 -> shape {10}
        Tensor s0 = tt.sum(0);
        CHECK(s0.shape().size() == 1 && s0.shape()[0] == 10, "tt.sum(0) shape {10}");
        const float* s0_ptr = s0.data_read<float>();
        bool s0_match = true;
        for (int c = 0; c < 10; ++c) {
            float exp_val = 0.0f;
            for (int r = 0; r < 4; ++r) exp_val += dense_ref[r][c];
            if (std::abs(s0_ptr[c] - exp_val) > 1e-4f) s0_match = false;
        }
        CHECK(s0_match, "tt.sum(0) mathematically exact matching dense reference");

        // mean(1) and mean(0)
        Tensor m1 = tt.mean(1);
        const float* m1_ptr = m1.data_read<float>();
        bool m1_match = (m1.shape().size() == 1 && m1.shape()[0] == 4);
        for (int r = 0; m1_match && r < 4; ++r) {
            float exp_val = 0.0f;
            for (int c = 0; c < 10; ++c) exp_val += dense_ref[r][c];
            if (std::abs(m1_ptr[r] - (exp_val / 10.0f)) > 1e-4f) m1_match = false;
        }
        CHECK(m1_match, "tt.mean(1) mathematically exact matching dense reference");

        Tensor m0 = tt.mean(0);
        const float* m0_ptr = m0.data_read<float>();
        bool m0_match = (m0.shape().size() == 1 && m0.shape()[0] == 10);
        for (int c = 0; m0_match && c < 10; ++c) {
            float exp_val = 0.0f;
            for (int r = 0; r < 4; ++r) exp_val += dense_ref[r][c];
            if (std::abs(m0_ptr[c] - (exp_val / 4.0f)) > 1e-4f) m0_match = false;
        }
        CHECK(m0_match, "tt.mean(0) mathematically exact matching dense reference");
    }

    // Test 16 [Edge Boundary Memory Safety]: Zero-offset, Buffer-end, and Sub-slices
    {
        Tensor t(ShapeTag{}, {8, 6}, DType::kFloat, DeviceType::kCPU);
        float* p = t.data_write<float>();
        for (int i = 0; i < 48; ++i) p[i] = static_cast<float>(i + 1);

        // Sub-slice touching the very last float in buffer (index 47)
        Tensor last_elem = t.slice(0, 7, 1).slice(1, 5, 1);
        Tensor st_last = last_elem.transpose(0, 1);
        Tensor s0_last = st_last.sum(0);
        Tensor s1_last = st_last.sum(1);
        CHECK(std::abs(s0_last.item<float>() - 48.0f) < 1e-4f, "Buffer-end last element sum(0) exact = 48.0");
        CHECK(std::abs(s1_last.item<float>() - 48.0f) < 1e-4f, "Buffer-end last element sum(1) exact = 48.0");
        CHECK(std::abs(st_last.mean(0).item<float>() - 48.0f) < 1e-4f, "Buffer-end last element mean(0) exact = 48.0");
        CHECK(std::abs(st_last.mean(1).item<float>() - 48.0f) < 1e-4f, "Buffer-end last element mean(1) exact = 48.0");

        // Sub-slice at beginning
        Tensor first_elem = t.slice(0, 0, 1).slice(1, 0, 1);
        Tensor st_first = first_elem.transpose(0, 1);
        CHECK(std::abs(st_first.sum(0).item<float>() - 1.0f) < 1e-4f, "Buffer-start first element sum(0) exact = 1.0");
        CHECK(std::abs(st_first.sum(1).item<float>() - 1.0f) < 1e-4f, "Buffer-start first element sum(1) exact = 1.0");

        // 1x1 Tensor
        Tensor t1(ShapeTag{}, {1, 1}, DType::kFloat, DeviceType::kCPU);
        t1.data_write<float>()[0] = 42.0f;
        Tensor t1_t = t1.transpose(0, 1);
        CHECK(std::abs(t1_t.sum(0).item<float>() - 42.0f) < 1e-4f, "1x1 transposed sum(0) = 42.0");
        CHECK(std::abs(t1_t.sum(1).item<float>() - 42.0f) < 1e-4f, "1x1 transposed sum(1) = 42.0");

        // Internal sub-slice transposed: shape {4, 3} from {8, 6}
        Tensor sub = t.slice(0, 2, 4).slice(1, 1, 3); // rows 2..5, cols 1..3
        Tensor sub_t = sub.transpose(0, 1);           // shape {3, 4}
        Tensor sub_s1 = sub_t.sum(1);                 // shape {3}
        // row 0 of sub_t is col 1 of sub: t[2..5, 1] = {14, 20, 26, 32} -> sum = 92
        // row 1 of sub_t is col 2 of sub: t[2..5, 2] = {15, 21, 27, 33} -> sum = 96
        // row 2 of sub_t is col 3 of sub: t[2..5, 3] = {16, 22, 28, 34} -> sum = 100
        const float* sub_s1_p = sub_s1.data_read<float>();
        bool sub_ok = (sub_s1.numel() == 3) &&
                      (std::abs(sub_s1_p[0] - 92.0f) < 1e-4f) &&
                      (std::abs(sub_s1_p[1] - 96.0f) < 1e-4f) &&
                      (std::abs(sub_s1_p[2] - 100.0f) < 1e-4f);
        CHECK(sub_ok, "Internal sub-slice transposed sum(1) exact values [92, 96, 100]");

        Tensor sub_s0 = sub_t.sum(0);                 // shape {4}
        // col 0: 14 + 15 + 16 = 45
        // col 1: 20 + 21 + 22 = 63
        // col 2: 26 + 27 + 28 = 81
        // col 3: 32 + 33 + 34 = 99
        const float* sub_s0_p = sub_s0.data_read<float>();
        bool sub0_ok = (sub_s0.numel() == 4) &&
                       (std::abs(sub_s0_p[0] - 45.0f) < 1e-4f) &&
                       (std::abs(sub_s0_p[1] - 63.0f) < 1e-4f) &&
                       (std::abs(sub_s0_p[2] - 81.0f) < 1e-4f) &&
                       (std::abs(sub_s0_p[3] - 99.0f) < 1e-4f);
        CHECK(sub0_ok, "Internal sub-slice transposed sum(0) exact values [45, 63, 81, 99]");
    }

    // Test 17 [SIMD Vector Boundary & Prime Strides]:
    // Non-multiples of 4 to stress SIMD tail branches
    {
        // Shape {13, 17} -> transpose(0, 1) -> {17, 13}
        Tensor t(ShapeTag{}, {13, 17}, DType::kFloat, DeviceType::kCPU);
        float* p = t.data_write<float>();
        for (int i = 0; i < 13 * 17; ++i) p[i] = static_cast<float>(i + 1);

        Tensor tt = t.transpose(0, 1); // shape {17, 13}
        Tensor s1 = tt.sum(1);         // shape {17}
        const float* s1_p = s1.data_read<float>();
        bool s1_ok = (s1.numel() == 17);
        for (int r = 0; s1_ok && r < 17; ++r) {
            float exp_val = 0.0f;
            for (int c = 0; c < 13; ++c) {
                exp_val += static_cast<float>(c * 17 + r + 1);
            }
            if (std::abs(s1_p[r] - exp_val) > 1e-3f) s1_ok = false;
        }
        CHECK(s1_ok, "Prime dimension {17, 13} sum(1) (tail SIMD loop) exact");

        Tensor s0 = tt.sum(0);         // shape {13}
        const float* s0_p = s0.data_read<float>();
        bool s0_ok = (s0.numel() == 13);
        for (int c = 0; s0_ok && c < 13; ++c) {
            float exp_val = 0.0f;
            for (int r = 0; r < 17; ++r) {
                exp_val += static_cast<float>(c * 17 + r + 1);
            }
            if (std::abs(s0_p[c] - exp_val) > 1e-3f) s0_ok = false;
        }
        CHECK(s0_ok, "Prime dimension {17, 13} sum(0) exact");

        // Double precision prime dimension {7, 11} -> {11, 7}
        Tensor td(ShapeTag{}, {7, 11}, DType::kDouble, DeviceType::kCPU);
        double* pd = td.data_write<double>();
        for (int i = 0; i < 7 * 11; ++i) pd[i] = static_cast<double>(i + 1);

        Tensor ttd = td.transpose(0, 1);
        Tensor sd1 = ttd.sum(1);
        const double* sd1_p = sd1.data_read<double>();
        bool sd1_ok = (sd1.numel() == 11);
        for (int r = 0; sd1_ok && r < 11; ++r) {
            double exp_val = 0.0;
            for (int c = 0; c < 7; ++c) {
                exp_val += static_cast<double>(c * 11 + r + 1);
            }
            if (std::abs(sd1_p[r] - exp_val) > 1e-6) sd1_ok = false;
        }
        CHECK(sd1_ok, "Double precision prime dimension {11, 7} sum(1) exact");

        // Single row/column transpose
        Tensor t_row(ShapeTag{}, {1, 15}, DType::kFloat, DeviceType::kCPU);
        for (int i = 0; i < 15; ++i) t_row.data_write<float>()[i] = static_cast<float>(i + 1);
        Tensor t_row_t = t_row.transpose(0, 1); // shape {15, 1}
        CHECK(std::abs(t_row_t.sum(0).item<float>() - 120.0f) < 1e-4f, "Shape {15, 1} sum(0) exact = 120.0");
        CHECK(t_row_t.sum(1).numel() == 15, "Shape {15, 1} sum(1) numel = 15");
    }

    // Test 18 [High-Dimensional Non-Contiguous Tensors]: 4D & 5D Views
    {
        // 4D: {2, 3, 4, 5} -> transpose(0, 3) -> {5, 3, 4, 2}
        Tensor t4(ShapeTag{}, {2, 3, 4, 5}, DType::kFloat, DeviceType::kCPU);
        float* p4 = t4.data_write<float>();
        for (int i = 0; i < 120; ++i) p4[i] = static_cast<float>(i + 1);

        Tensor tt4 = t4.transpose(0, 3);
        CHECK(!tt4.is_contiguous(), "4D transposed tensor is non-contiguous");

        // Dense equivalent for comparison
        Tensor dense4 = tt4.contiguous();
        for (int axis = 0; axis < 4; ++axis) {
            Tensor red_tt = tt4.sum(axis);
            Tensor red_dense = dense4.sum(axis);
            CHECK(red_tt.numel() == red_dense.numel(), "4D sum axis numel match");
            const float* p_tt = red_tt.data_read<float>();
            const float* p_de = red_dense.data_read<float>();
            bool axis_ok = true;
            for (size_t k = 0; axis_ok && k < red_tt.numel(); ++k) {
                if (std::abs(p_tt[k] - p_de[k]) > 1e-3f) axis_ok = false;
            }
            CHECK(axis_ok, "4D sum axis exact match against dense reference");

            Tensor mean_tt = tt4.mean(axis);
            Tensor mean_dense = dense4.mean(axis);
            const float* pm_tt = mean_tt.data_read<float>();
            const float* pm_de = mean_dense.data_read<float>();
            bool mean_ok = true;
            for (size_t k = 0; mean_ok && k < mean_tt.numel(); ++k) {
                if (std::abs(pm_tt[k] - pm_de[k]) > 1e-4f) mean_ok = false;
            }
            CHECK(mean_ok, "4D mean axis exact match against dense reference");
        }

        // 5D: {2, 2, 3, 2, 2} (numel 48) -> transpose(1, 4) -> {2, 2, 3, 2, 2}
        Tensor t5(ShapeTag{}, {2, 2, 3, 2, 2}, DType::kFloat, DeviceType::kCPU);
        float* p5 = t5.data_write<float>();
        for (int i = 0; i < 48; ++i) p5[i] = static_cast<float>(i + 1);

        Tensor tt5 = t5.transpose(1, 4);
        CHECK(!tt5.is_contiguous(), "5D transposed tensor is non-contiguous");
        Tensor dense5 = tt5.contiguous();

        Tensor s_mid5 = tt5.sum(2);
        Tensor sm_de5 = dense5.sum(2);
        const float* pm5 = s_mid5.data_read<float>();
        const float* pd5 = sm_de5.data_read<float>();
        bool s5_ok = (s_mid5.numel() == sm_de5.numel());
        for (size_t k = 0; s5_ok && k < s_mid5.numel(); ++k) {
            if (std::abs(pm5[k] - pd5[k]) > 1e-4f) s5_ok = false;
        }
        CHECK(s5_ok, "5D middle axis sum exact match against dense reference");
    }

    // Test 19 [Autograd Backward on Boundary Slices and Chained Transposes]
    {
        AutoGrad::EnableGrad = true;
        Tensor t(ShapeTag{}, {6, 8}, DType::kFloat, DeviceType::kCPU);
        float* p = t.data_write<float>();
        for (int i = 0; i < 48; ++i) p[i] = static_cast<float>(i + 1);
        t.requires_grad(true);

        Tensor s = t.slice(0, 1, 4).slice(1, 2, 5); // shape {4, 5}
        Tensor st = s.transpose(0, 1);               // shape {5, 4}
        Tensor loss = st.sum(1).sum();              // sum over all elements of slice
        AutoGrad::backward(loss.getRelatedNode(), false);

        const float* g = t.grad_ptr();
        CHECK(g != nullptr, "Sliced transposed chain backward populated gradient");
        bool grad_ok = (g != nullptr);
        // Elements within row 1..4, col 2..6 must have grad = 1.0; all others 0.0
        for (int r = 0; grad_ok && r < 6; ++r) {
            for (int c = 0; grad_ok && c < 8; ++c) {
                float expected = (r >= 1 && r < 5 && c >= 2 && c < 7) ? 1.0f : 0.0f;
                float actual = g[r * 8 + c];
                if (std::abs(actual - expected) > 1e-5f) grad_ok = false;
            }
        }
        CHECK(grad_ok, "Leaf tensor gradient exactly reflects sliced transpose reduction");
    }

    // Test 20 [Negative Dim Indices on Non-Contiguous Views]
    {
        Tensor t(ShapeTag{}, {10, 4}, DType::kFloat, DeviceType::kCPU);
        for (int i = 0; i < 40; ++i) t.data_write<float>()[i] = static_cast<float>(i + 1);
        Tensor tt = t.transpose(0, 1); // shape {4, 10}

        Tensor sn1 = tt.sum(-1); // should equal tt.sum(1)
        Tensor s1 = tt.sum(1);
        bool n1_ok = (sn1.numel() == s1.numel());
        for (size_t i = 0; n1_ok && i < s1.numel(); ++i) {
            if (std::abs(sn1.data_read<float>()[i] - s1.data_read<float>()[i]) > 1e-5f) n1_ok = false;
        }
        CHECK(n1_ok, "tt.sum(-1) matches tt.sum(1)");

        Tensor sn2 = tt.sum(-2); // should equal tt.sum(0)
        Tensor s0 = tt.sum(0);
        bool n2_ok = (sn2.numel() == s0.numel());
        for (size_t i = 0; n2_ok && i < s0.numel(); ++i) {
            if (std::abs(sn2.data_read<float>()[i] - s0.data_read<float>()[i]) > 1e-5f) n2_ok = false;
        }
        CHECK(n2_ok, "tt.sum(-2) matches tt.sum(0)");
    }

    std::cout << (g_fails == 0 ? "=== ALL PASS ===" : "=== HAS FAIL ===") << "\n";
    return g_fails;
}
