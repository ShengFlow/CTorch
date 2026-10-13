/**
 * @file test_core_semantics.cpp
 * @brief Storage / Tensor / Node 核心数据结构 copy/move/lifetime 语义回归测试
 * @details 覆盖深/浅拷贝、移动后状态、_grad 独立性、跨设备 to()、shared_storage 视图。
 *          修改核心数据结构语义时必须运行本测试，防止类似 `_grad` 浅拷贝事故重演。
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <utility>
#include "Tensor.h"
#include "Storage.h"
#include "AutoGrad.h"
#include "AutoGrad/Node.h"
#include "AutoGrad/Nodes/GradAccumulator.h"
#include "CtorchScheduler.h"
#include "C3/Graph.h"
#include "C3/JITCache.h"
#include "kernels/kernels.h"
#include "ThreadPool.h"

static std::vector<float> make_values(size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = static_cast<float>(i) - static_cast<float>(n) / 2.0f;
    }
    return v;
}

static void fill_tensor(Tensor& t, const std::vector<float>& values) {
    float* p = t.data_write<float>();
    for (size_t i = 0; i < values.size(); ++i) {
        p[i] = values[i];
    }
}

// ======================= Storage 语义 =======================

TEST(StorageSemantics, DefaultConstruct) {
    Storage s;
    EXPECT_EQ(s.size(), 0u);
    EXPECT_EQ(s.dtype(), DType::kFloat);
}

TEST(StorageSemantics, ReadWrite) {
    Storage s(4, DType::kFloat, DeviceType::kCPU);
    float* p = s.data<float>();
    p[0] = 1.0f;
    p[3] = 4.0f;
    EXPECT_EQ(p[0], 1.0f);
    EXPECT_EQ(p[3], 4.0f);
}

TEST(StorageSemantics, CopySharesData) {
    Storage s(4, DType::kFloat, DeviceType::kCPU);
    s.data<float>()[0] = 42.0f;

    Storage s2(s);
    EXPECT_EQ(s2.data<float>()[0], 42.0f);

    s2.data<float>()[0] = 99.0f;
    EXPECT_EQ(s.data<float>()[0], 99.0f);
}

TEST(StorageSemantics, CopyAssignmentSharesData) {
    Storage s(4, DType::kFloat, DeviceType::kCPU);
    s.data<float>()[0] = 7.0f;

    Storage s2(1, DType::kFloat, DeviceType::kCPU);
    s2 = s;
    EXPECT_EQ(s2.size(), 4u);
    EXPECT_EQ(s2.data<float>()[0], 7.0f);

    s2.data<float>()[0] = 8.0f;
    EXPECT_EQ(s.data<float>()[0], 8.0f);
}

TEST(StorageSemantics, CloneIsIndependent) {
    Storage s(4, DType::kFloat, DeviceType::kCPU);
    s.data<float>()[0] = 1.0f;

    Storage s2 = s.clone();
    s2.data<float>()[0] = 2.0f;
    EXPECT_EQ(s.data<float>()[0], 1.0f);
}

TEST(StorageSemantics, MoveConstructTransfers) {
    Storage s(4, DType::kFloat, DeviceType::kCPU);
    s.data<float>()[0] = 5.0f;
    void* old_ptr = s.data<float>();

    Storage s2(std::move(s));
    EXPECT_EQ(s2.data<float>(), old_ptr);
    EXPECT_EQ(s2.data<float>()[0], 5.0f);
    EXPECT_EQ(s.size(), 0u);
}

TEST(StorageSemantics, MoveAssignmentTransfers) {
    Storage s(4, DType::kFloat, DeviceType::kCPU);
    s.data<float>()[0] = 6.0f;
    void* old_ptr = s.data<float>();

    Storage s2(1, DType::kFloat, DeviceType::kCPU);
    s2 = std::move(s);
    EXPECT_EQ(s2.data<float>(), old_ptr);
    EXPECT_EQ(s2.size(), 4u);
    EXPECT_EQ(s.size(), 0u);
}

TEST(StorageSemantics, DestructAfterMove) {
    Storage s(4, DType::kFloat, DeviceType::kCPU);
    {
        Storage s2(std::move(s));
        (void)s2;
    }
    // s 已为空，析构不应崩溃
}

// ======================= Tensor 拷贝/移动语义 =======================

TEST(TensorSemantics, CopySharesStorage) {
    auto values = make_values(4);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(a, values);

    Tensor b(a);
    b.data_write<float>()[0] = 999.0f;
    EXPECT_EQ(a.data_read<float>()[0], 999.0f);
}

TEST(TensorSemantics, CopyAssignmentSharesStorage) {
    auto values = make_values(4);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(a, values);

    Tensor c(ShapeTag{}, {1}, DType::kFloat, DeviceType::kCPU);
    c = a;
    c.data_write<float>()[0] = 888.0f;
    EXPECT_EQ(a.data_read<float>()[0], 888.0f);
}

TEST(TensorSemantics, CopyDoesNotCopyGrad) {
    // 现代语义验证：Tensor 拷贝构造共享 storage，独立 autograd 弱引用，
    // 不深拷贝 grad（_autograd_meta._grad 为 nullptr，grad() 返回与 shape 一致的零张量）。
    Tensor a(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU);
    a.requires_grad(true);
    fill_tensor(a, {1.0f, 2.0f, 3.0f, 4.0f});

    Tensor b = a + 0.0f;
    AutoGrad::backward(b.getRelatedNode(), false);

    // a 的 grad 应为全 1
    EXPECT_EQ(a.grad().data_read<float>()[0], 1.0f);

    Tensor a_copy(a);
    // a_copy 未执行自身 backward，grad() 返回零张量
    EXPECT_EQ(a_copy.grad().data_read<float>()[0], 0.0f);

    // 清零 a 的 grad，两者相互独立
    a.zero_grad();
    EXPECT_EQ(a.grad().data_read<float>()[0], 0.0f);
    EXPECT_EQ(a_copy.grad().data_read<float>()[0], 0.0f);
}

TEST(TensorSemantics, MoveConstructInvalidatesSource) {
    auto values = make_values(4);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(a, values);

    Tensor b(std::move(a));
    // 空 shape 在 numel() 中按标量返回 1，因此这里检查 shape 已清空且 data 为空
    EXPECT_TRUE(a.shape().empty());
    EXPECT_EQ(a.data_read<float>(), nullptr);
    EXPECT_EQ(b.numel(), 4u);
    EXPECT_EQ(b.data_read<float>()[0], values[0]);
}

TEST(TensorSemantics, MoveAssignmentInvalidatesSource) {
    auto values = make_values(4);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(a, values);

    Tensor c(ShapeTag{}, {1}, DType::kFloat, DeviceType::kCPU);
    c = std::move(a);
    EXPECT_TRUE(a.shape().empty());
    EXPECT_EQ(a.data_read<float>(), nullptr);
    EXPECT_EQ(c.numel(), 4u);
}

TEST(TensorSemantics, ViewModificationReflectsOnBase) {
    auto values = make_values(6);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(a, values);

    // 通过拷贝构造创建视图（共享 storage）
    Tensor view(a);
    view.data_write<float>()[2] = 123.0f;
    EXPECT_EQ(a.data_read<float>()[2], 123.0f);
}

TEST(TensorSemantics, CloneIsIndependent) {
    auto values = make_values(4);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(a, values);

    Tensor b = a.clone();
    b.data_write<float>()[0] = 777.0f;
    EXPECT_NE(a.data_read<float>()[0], 777.0f);
}

TEST(TensorSemantics, GradIndependenceAfterCopy) {
    Tensor a(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU);
    a.requires_grad(true);
    fill_tensor(a, {1.0f, 2.0f, 3.0f, 4.0f});

    Tensor b = a + 0.0f;
    AutoGrad::backward(b.getRelatedNode(), false);
    EXPECT_EQ(a.grad().data_read<float>()[0], 1.0f);

    Tensor a_copy(a);
    // 对 a_copy 做新的前向 + backward，不应影响 a 已有的 grad
    Tensor c = a_copy + 0.0f;
    AutoGrad::backward(c.getRelatedNode(), false);

    // a 的 grad 保持为 1（来自 b 的 backward）
    EXPECT_EQ(a.grad().data_read<float>()[0], 1.0f);
    // a_copy 的 grad 累加为 1（自身独立累计）
    EXPECT_EQ(a_copy.grad().data_read<float>()[0], 1.0f);
}

TEST(TensorSemantics, CopyMoveChain) {
    auto values = make_values(4);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(a, values);

    Tensor b(a);              // copy
    Tensor c(std::move(b));   // move from copy
    Tensor d(c);              // copy from moved-to
    d.data_write<float>()[0] = 111.0f;

    EXPECT_EQ(a.data_read<float>()[0], 111.0f);
    EXPECT_EQ(c.data_read<float>()[0], 111.0f);
    EXPECT_TRUE(b.shape().empty());
    EXPECT_EQ(b.data_read<float>(), nullptr);
}

// ======================= Tensor 设备迁移语义 =======================

TEST(TensorSemantics, ToSameDeviceReturnsAliasLikeCopy) {
    auto values = make_values(4);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(a, values);

    Tensor b = a.to(DeviceType::kCPU);
    EXPECT_EQ(b.device(), DeviceType::kCPU);
    EXPECT_EQ(b.data_read<float>()[0], values[0]);
}

TEST(TensorSemantics, ToCrossDeviceAndBack) {
    if (!CtorchScheduler::isDeviceAvailable(DeviceType::kMPS)) {
        GTEST_SKIP() << "MPS not available";
    }
    auto values = make_values(4);
    Tensor cpu_t(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(cpu_t, values);

    Tensor mps_t = cpu_t.to(DeviceType::kMPS);
    MPS_flush_wait(true);
    EXPECT_EQ(mps_t.device(), DeviceType::kMPS);

    Tensor cpu_t2 = mps_t.to(DeviceType::kCPU);
    MPS_flush_wait(true);
    EXPECT_EQ(cpu_t2.device(), DeviceType::kCPU);
    EXPECT_EQ(cpu_t2.data_read<float>()[0], values[0]);
}

TEST(TensorSemantics, NonContiguousReductionSafety) {
    AutoGrad::EnableGrad = true;
    Tensor t(ShapeTag{}, {10, 4}, DType::kFloat, DeviceType::kCPU);
    float* p = t.data_write<float>();
    for (int i = 0; i < 40; ++i) p[i] = static_cast<float>(i + 1);

    Tensor tt = t.transpose(0, 1);
    EXPECT_FALSE(tt.is_contiguous());

    // Forward checks
    Tensor s1 = tt.sum(1);
    EXPECT_EQ(s1.numel(), 4u);
    EXPECT_NEAR(s1.data_read<float>()[0], 190.0f, 1e-4f);
    EXPECT_NEAR(s1.data_read<float>()[1], 200.0f, 1e-4f);
    EXPECT_NEAR(s1.data_read<float>()[2], 210.0f, 1e-4f);
    EXPECT_NEAR(s1.data_read<float>()[3], 220.0f, 1e-4f);

    Tensor s0 = tt.sum(0);
    EXPECT_EQ(s0.numel(), 10u);
    EXPECT_NEAR(s0.data_read<float>()[0], 10.0f, 1e-4f);
    EXPECT_NEAR(s0.data_read<float>()[9], 154.0f, 1e-4f);

    Tensor m1 = tt.mean(1);
    EXPECT_EQ(m1.numel(), 4u);
    EXPECT_NEAR(m1.data_read<float>()[0], 19.0f, 1e-4f);
    EXPECT_NEAR(m1.data_read<float>()[3], 22.0f, 1e-4f);

    Tensor m0 = tt.mean(0);
    EXPECT_EQ(m0.numel(), 10u);
    EXPECT_NEAR(m0.data_read<float>()[0], 2.5f, 1e-4f);
    EXPECT_NEAR(m0.data_read<float>()[9], 38.5f, 1e-4f);

    // Backward checks
    {
        Tensor tg(ShapeTag{}, {10, 4}, DType::kFloat, DeviceType::kCPU);
        for (int i = 0; i < 40; ++i) tg.data_write<float>()[i] = static_cast<float>(i + 1);
        tg.requires_grad(true);
        Tensor ttg = tg.transpose(0, 1);
        Tensor loss = ttg.sum(1).sum();
        AutoGrad::backward(loss.getRelatedNode(), false);
        ASSERT_NE(tg.grad_ptr(), nullptr);
        for (int i = 0; i < 40; ++i) {
            EXPECT_NEAR(tg.grad_ptr()[i], 1.0f, 1e-5f);
        }
    }
    {
        Tensor tg(ShapeTag{}, {10, 4}, DType::kFloat, DeviceType::kCPU);
        for (int i = 0; i < 40; ++i) tg.data_write<float>()[i] = static_cast<float>(i + 1);
        tg.requires_grad(true);
        Tensor ttg = tg.transpose(0, 1);
        Tensor loss = ttg.sum(0).sum();
        AutoGrad::backward(loss.getRelatedNode(), false);
        ASSERT_NE(tg.grad_ptr(), nullptr);
        for (int i = 0; i < 40; ++i) {
            EXPECT_NEAR(tg.grad_ptr()[i], 1.0f, 1e-5f);
        }
    }
    {
        Tensor tg(ShapeTag{}, {10, 4}, DType::kFloat, DeviceType::kCPU);
        for (int i = 0; i < 40; ++i) tg.data_write<float>()[i] = static_cast<float>(i + 1);
        tg.requires_grad(true);
        Tensor ttg = tg.transpose(0, 1);
        Tensor loss = ttg.mean(1).sum();
        AutoGrad::backward(loss.getRelatedNode(), false);
        ASSERT_NE(tg.grad_ptr(), nullptr);
        for (int i = 0; i < 40; ++i) {
            EXPECT_NEAR(tg.grad_ptr()[i], 0.1f, 1e-5f);
        }
    }
    {
        Tensor tg(ShapeTag{}, {10, 4}, DType::kFloat, DeviceType::kCPU);
        for (int i = 0; i < 40; ++i) tg.data_write<float>()[i] = static_cast<float>(i + 1);
        tg.requires_grad(true);
        Tensor ttg = tg.transpose(0, 1);
        Tensor loss = ttg.mean(0).sum();
        AutoGrad::backward(loss.getRelatedNode(), false);
        ASSERT_NE(tg.grad_ptr(), nullptr);
        for (int i = 0; i < 40; ++i) {
            EXPECT_NEAR(tg.grad_ptr()[i], 0.25f, 1e-5f);
        }
    }
}

// ======================= Node / GradPack 生命周期 =======================

TEST(NodeSemantics, GradPackConstruct) {
    GradPack pack;
    pack._targetNode = nullptr;
    pack._idx = 0;
    EXPECT_EQ(pack._targetNode, nullptr);
}

TEST(NodeSemantics, GradAccumulatorLifetime) {
    Tensor t(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU);
    t.requires_grad(true);
    {
        auto node = std::make_shared<GradAccumulator>(t.getWeakPtr());
        EXPECT_NE(node, nullptr);
    }
}

// ======================= R3: SIMD 广播跨步与 Over-Read 防御回归测试 =======================

TEST(SIMDBroadcastSemantics, ColumnBroadcastAddMul) {
    // 形状 [10, 1] 与 [10, 8] 列广播加法、乘法
    const size_t M = 10;
    const size_t N = 8;
    Tensor a(ShapeTag{}, {M, 1}, DType::kFloat, DeviceType::kCPU);
    Tensor b(ShapeTag{}, {M, N}, DType::kFloat, DeviceType::kCPU);

    for (size_t i = 0; i < M; ++i) {
        a.data_write<float>()[i] = static_cast<float>(i + 1); // a[i, 0] = i + 1
    }
    for (size_t i = 0; i < M * N; ++i) {
        b.data_write<float>()[i] = static_cast<float>(i + 10);
    }

    // 1. a + b
    Tensor c_add = a + b;
    ASSERT_EQ(c_add.shape(), (std::vector<size_t>{M, N}));
    const float* c_add_data = c_add.data_read<float>();
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float expected = static_cast<float>(i + 1) + static_cast<float>(i * N + j + 10);
            EXPECT_NEAR(c_add_data[i * N + j], expected, 1e-5f);
        }
    }

    // 2. b + a (反向操作数顺序)
    Tensor c_add_rev = b + a;
    ASSERT_EQ(c_add_rev.shape(), (std::vector<size_t>{M, N}));
    const float* c_add_rev_data = c_add_rev.data_read<float>();
    for (size_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(c_add_rev_data[i], c_add_data[i], 1e-5f);
    }

    // 3. a * b
    Tensor c_mul = a * b;
    ASSERT_EQ(c_mul.shape(), (std::vector<size_t>{M, N}));
    const float* c_mul_data = c_mul.data_read<float>();
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float expected = static_cast<float>(i + 1) * static_cast<float>(i * N + j + 10);
            EXPECT_NEAR(c_mul_data[i * N + j], expected, 1e-5f);
        }
    }

    // 4. b * a
    Tensor c_mul_rev = b * a;
    ASSERT_EQ(c_mul_rev.shape(), (std::vector<size_t>{M, N}));
    const float* c_mul_rev_data = c_mul_rev.data_read<float>();
    for (size_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(c_mul_rev_data[i], c_mul_data[i], 1e-5f);
    }
}

TEST(SIMDBroadcastSemantics, ColumnBroadcastMaxMin) {
    // 形状 [10, 1] 与 [10, 8] 列广播最大值、最小值
    const size_t M = 10;
    const size_t N = 8;
    Tensor a(ShapeTag{}, {M, 1}, DType::kFloat, DeviceType::kCPU);
    Tensor b(ShapeTag{}, {M, N}, DType::kFloat, DeviceType::kCPU);

    for (size_t i = 0; i < M; ++i) {
        a.data_write<float>()[i] = static_cast<float>(i * 5 + 3);
    }
    for (size_t i = 0; i < M * N; ++i) {
        b.data_write<float>()[i] = static_cast<float>(i * 2 + 1);
    }

    // 1. Max_SIMD_kernel(a, b) 与 Max_SIMD_kernel(b, a)
    Tensor c_max = Max_SIMD_kernel(a, b);
    ASSERT_EQ(c_max.shape(), (std::vector<size_t>{M, N}));
    const float* c_max_data = c_max.data_read<float>();
    for (size_t i = 0; i < M; ++i) {
        float a_val = static_cast<float>(i * 5 + 3);
        for (size_t j = 0; j < N; ++j) {
            float b_val = static_cast<float>((i * N + j) * 2 + 1);
            float expected = std::max(a_val, b_val);
            EXPECT_NEAR(c_max_data[i * N + j], expected, 1e-5f);
        }
    }

    Tensor c_max_rev = Max_SIMD_kernel(b, a);
    ASSERT_EQ(c_max_rev.shape(), (std::vector<size_t>{M, N}));
    const float* c_max_rev_data = c_max_rev.data_read<float>();
    for (size_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(c_max_rev_data[i], c_max_data[i], 1e-5f);
    }

    // 2. Min_SIMD_kernel(a, b) 与 Min_SIMD_kernel(b, a)
    Tensor c_min = Min_SIMD_kernel(a, b);
    ASSERT_EQ(c_min.shape(), (std::vector<size_t>{M, N}));
    const float* c_min_data = c_min.data_read<float>();
    for (size_t i = 0; i < M; ++i) {
        float a_val = static_cast<float>(i * 5 + 3);
        for (size_t j = 0; j < N; ++j) {
            float b_val = static_cast<float>((i * N + j) * 2 + 1);
            float expected = std::min(a_val, b_val);
            EXPECT_NEAR(c_min_data[i * N + j], expected, 1e-5f);
        }
    }

    Tensor c_min_rev = Min_SIMD_kernel(b, a);
    ASSERT_EQ(c_min_rev.shape(), (std::vector<size_t>{M, N}));
    const float* c_min_rev_data = c_min_rev.data_read<float>();
    for (size_t i = 0; i < M * N; ++i) {
        EXPECT_NEAR(c_min_rev_data[i], c_min_data[i], 1e-5f);
    }
}

TEST(SIMDBroadcastSemantics, OddShapesAndOverReadSafetyASan) {
    // 非 SIMD lane 整数倍与奇数尺寸 (如 M=7, N=11 和 M=5, N=17)
    // 专门测试尾行 SIMD splat 防御，确保 ASan 下零越界
    std::vector<std::pair<size_t, size_t>> test_cases = {
        {7, 11},
        {5, 17},
        {3, 5},
        {13, 15},
        {1, 9}
    };

    for (const auto& [M, N] : test_cases) {
        Tensor a(ShapeTag{}, {M, 1}, DType::kFloat, DeviceType::kCPU);
        Tensor b(ShapeTag{}, {M, N}, DType::kFloat, DeviceType::kCPU);

        for (size_t i = 0; i < M; ++i) {
            a.data_write<float>()[i] = static_cast<float>(i + 1) * 1.5f;
        }
        for (size_t i = 0; i < M * N; ++i) {
            b.data_write<float>()[i] = static_cast<float>(i) * 0.5f - 2.0f;
        }

        // Add: [M, 1] + [M, N] and [M, N] + [M, 1]
        {
            Tensor c1 = a + b;
            Tensor c2 = b + a;
            const float* c1_ptr = c1.data_read<float>();
            const float* c2_ptr = c2.data_read<float>();
            for (size_t i = 0; i < M; ++i) {
                float a_val = a.data_read<float>()[i];
                for (size_t j = 0; j < N; ++j) {
                    float b_val = b.data_read<float>()[i * N + j];
                    float exp_val = a_val + b_val;
                    EXPECT_NEAR(c1_ptr[i * N + j], exp_val, 1e-5f);
                    EXPECT_NEAR(c2_ptr[i * N + j], exp_val, 1e-5f);
                }
            }
        }

        // Mul: [M, 1] * [M, N] and [M, N] * [M, 1]
        {
            Tensor c1 = a * b;
            Tensor c2 = b * a;
            const float* c1_ptr = c1.data_read<float>();
            const float* c2_ptr = c2.data_read<float>();
            for (size_t i = 0; i < M; ++i) {
                float a_val = a.data_read<float>()[i];
                for (size_t j = 0; j < N; ++j) {
                    float b_val = b.data_read<float>()[i * N + j];
                    float exp_val = a_val * b_val;
                    EXPECT_NEAR(c1_ptr[i * N + j], exp_val, 1e-5f);
                    EXPECT_NEAR(c2_ptr[i * N + j], exp_val, 1e-5f);
                }
            }
        }

        // Max: Max_SIMD_kernel([M, 1], [M, N]) and Max_SIMD_kernel([M, N], [M, 1])
        {
            Tensor c1 = Max_SIMD_kernel(a, b);
            Tensor c2 = Max_SIMD_kernel(b, a);
            const float* c1_ptr = c1.data_read<float>();
            const float* c2_ptr = c2.data_read<float>();
            for (size_t i = 0; i < M; ++i) {
                float a_val = a.data_read<float>()[i];
                for (size_t j = 0; j < N; ++j) {
                    float b_val = b.data_read<float>()[i * N + j];
                    float exp_val = std::max(a_val, b_val);
                    EXPECT_NEAR(c1_ptr[i * N + j], exp_val, 1e-5f);
                    EXPECT_NEAR(c2_ptr[i * N + j], exp_val, 1e-5f);
                }
            }
        }

        // Min: Min_SIMD_kernel([M, 1], [M, N]) and Min_SIMD_kernel([M, N], [M, 1])
        {
            Tensor c1 = Min_SIMD_kernel(a, b);
            Tensor c2 = Min_SIMD_kernel(b, a);
            const float* c1_ptr = c1.data_read<float>();
            const float* c2_ptr = c2.data_read<float>();
            for (size_t i = 0; i < M; ++i) {
                float a_val = a.data_read<float>()[i];
                for (size_t j = 0; j < N; ++j) {
                    float b_val = b.data_read<float>()[i * N + j];
                    float exp_val = std::min(a_val, b_val);
                    EXPECT_NEAR(c1_ptr[i * N + j], exp_val, 1e-5f);
                    EXPECT_NEAR(c2_ptr[i * N + j], exp_val, 1e-5f);
                }
            }
        }
    }
}

TEST(SIMDBroadcastSemantics, MultiDimDecoupledBroadcastStride) {
    // 3D 广播 [M, 1, K] 与 [M, N, K] 检验 stride 解耦
    const size_t M = 3;
    const size_t N = 4;
    const size_t K = 5;
    Tensor a(ShapeTag{}, {M, 1, K}, DType::kFloat, DeviceType::kCPU);
    Tensor b(ShapeTag{}, {M, N, K}, DType::kFloat, DeviceType::kCPU);

    for (size_t i = 0; i < M * 1 * K; ++i) {
        a.data_write<float>()[i] = static_cast<float>(i + 1);
    }
    for (size_t i = 0; i < M * N * K; ++i) {
        b.data_write<float>()[i] = static_cast<float>(i * 2 + 5);
    }

    Tensor c = a + b;
    ASSERT_EQ(c.shape(), (std::vector<size_t>{M, N, K}));
    const float* c_ptr = c.data_read<float>();
    const float* a_ptr = a.data_read<float>();
    const float* b_ptr = b.data_read<float>();

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            for (size_t k = 0; k < K; ++k) {
                size_t c_idx = m * (N * K) + n * K + k;
                size_t a_idx = m * (1 * K) + 0 * K + k;
                size_t b_idx = m * (N * K) + n * K + k;
                EXPECT_NEAR(c_ptr[c_idx], a_ptr[a_idx] + b_ptr[b_idx], 1e-5f);
            }
        }
    }
}

// ======================= R4: C3 JIT Cache Graph Key 差异化回归测试 =======================

TEST(C3GraphSerialization, AttributeAndDTypeKeyDifferentiation) {
    using namespace ct::c3;

    auto desc_float = TensorDesc::fromShape({4, 4}, DType::kFloat, DeviceType::kCPU);
    auto desc_double = TensorDesc::fromShape({4, 4}, DType::kDouble, DeviceType::kCPU);

    // 1. SumReduce(axis=0) vs SumReduce(axis=1)
    Graph g_red0, g_red1;
    size_t in0 = g_red0.addInput(desc_float);
    g_red0.addNode(SumReduceNode{desc_float, 0, false}, {in0}, TensorDesc::fromShape({4}, DType::kFloat));
    size_t in1 = g_red1.addInput(desc_float);
    g_red1.addNode(SumReduceNode{desc_float, 1, false}, {in1}, TensorDesc::fromShape({4}, DType::kFloat));

    EXPECT_NE(g_red0.toString(), g_red1.toString());
    EXPECT_NE(JITCache::makeKey(g_red0.toString(), 2), JITCache::makeKey(g_red1.toString(), 2));

    // 2. SumReduce(keepdim=false) vs SumReduce(keepdim=true)
    Graph g_keep_f, g_keep_t;
    size_t in_kf = g_keep_f.addInput(desc_float);
    g_keep_f.addNode(SumReduceNode{desc_float, 1, false}, {in_kf}, TensorDesc::fromShape({4}, DType::kFloat));
    size_t in_kt = g_keep_t.addInput(desc_float);
    g_keep_t.addNode(SumReduceNode{desc_float, 1, true}, {in_kt}, TensorDesc::fromShape({4, 1}, DType::kFloat));

    EXPECT_NE(g_keep_f.toString(), g_keep_t.toString());
    EXPECT_NE(JITCache::makeKey(g_keep_f.toString(), 2), JITCache::makeKey(g_keep_t.toString(), 2));

    // 3. Softmax(axis=0) vs Softmax(axis=1)
    Graph g_sm0, g_sm1;
    size_t in_sm0 = g_sm0.addInput(desc_float);
    g_sm0.addNode(ct::c3::SoftmaxNode{desc_float, 0}, {in_sm0}, desc_float);
    size_t in_sm1 = g_sm1.addInput(desc_float);
    g_sm1.addNode(ct::c3::SoftmaxNode{desc_float, 1}, {in_sm1}, desc_float);

    EXPECT_NE(g_sm0.toString(), g_sm1.toString());
    EXPECT_NE(JITCache::makeKey(g_sm0.toString(), 2), JITCache::makeKey(g_sm1.toString(), 2));

    // 4. Transpose(0, 1) vs Transpose(1, 2)
    auto desc_3d = TensorDesc::fromShape({2, 3, 4}, DType::kFloat);
    Graph g_tr01, g_tr12;
    size_t in_tr01 = g_tr01.addInput(desc_3d);
    g_tr01.addNode(ct::c3::TransposeNode{desc_3d, 0, 1}, {in_tr01}, TensorDesc::fromShape({3, 2, 4}, DType::kFloat));
    size_t in_tr12 = g_tr12.addInput(desc_3d);
    g_tr12.addNode(ct::c3::TransposeNode{desc_3d, 1, 2}, {in_tr12}, TensorDesc::fromShape({2, 4, 3}, DType::kFloat));

    EXPECT_NE(g_tr01.toString(), g_tr12.toString());
    EXPECT_NE(JITCache::makeKey(g_tr01.toString(), 2), JITCache::makeKey(g_tr12.toString(), 2));

    // 5. DType Float vs Double on identical topology
    Graph g_fl, g_db;
    size_t in_fl = g_fl.addInput(desc_float);
    g_fl.addNode(ct::c3::NegNode{desc_float}, {in_fl}, desc_float);
    size_t in_db = g_db.addInput(desc_double);
    g_db.addNode(ct::c3::NegNode{desc_double}, {in_db}, desc_double);

    EXPECT_NE(g_fl.toString(), g_db.toString());
    EXPECT_NE(JITCache::makeKey(g_fl.toString(), 2), JITCache::makeKey(g_db.toString(), 2));
}

// ======================= P1 回归测试套件 =======================

// 1. D1: 非紧凑/非零 offset 视图上的 in-place 与 zero/ones/rand 不会静默污染相邻内存
TEST(TensorViewSemantics, InplaceUnaryAndZeroOnStridedView) {
    Tensor t(ShapeTag{}, {4, 4}, DType::kFloat, DeviceType::kCPU);
    float* p = t.data_write<float>();
    for (int i = 0; i < 16; ++i) p[i] = static_cast<float>(i + 1);

    Tensor sub = t.slice(0, 1, 2).slice(1, 1, 2);
    EXPECT_FALSE(sub.is_contiguous());
    EXPECT_EQ(sub.storage_offset(), 5u);

    sub.zero();

    Tensor sub_contig = sub.contiguous();
    const float* sc_p = sub_contig.data_read<float>();
    EXPECT_FLOAT_EQ(sc_p[0], 0.0f);
    EXPECT_FLOAT_EQ(sc_p[1], 0.0f);
    EXPECT_FLOAT_EQ(sc_p[2], 0.0f);
    EXPECT_FLOAT_EQ(sc_p[3], 0.0f);

    const float* tp = t.data_read<float>();
    EXPECT_FLOAT_EQ(tp[0], 1.0f);   // (0,0) 未破坏
    EXPECT_FLOAT_EQ(tp[4], 5.0f);   // (1,0) 未破坏
    EXPECT_FLOAT_EQ(tp[5], 0.0f);   // (1,1) 被修改
    EXPECT_FLOAT_EQ(tp[6], 0.0f);   // (1,2) 被修改
    EXPECT_FLOAT_EQ(tp[7], 8.0f);   // (1,3) 未破坏
    EXPECT_FLOAT_EQ(tp[8], 9.0f);   // (2,0) 未破坏
    EXPECT_FLOAT_EQ(tp[9], 0.0f);   // (2,1) 被修改
    EXPECT_FLOAT_EQ(tp[10], 0.0f);  // (2,2) 被修改
    EXPECT_FLOAT_EQ(tp[11], 12.0f); // (2,3) 未破坏
    EXPECT_FLOAT_EQ(tp[15], 16.0f); // (3,3) 未破坏

    Tensor tr = t.transpose(0, 1);
    tr.neg_();
    EXPECT_FLOAT_EQ(t.data_read<float>()[0], -1.0f);
    EXPECT_FLOAT_EQ(t.data_read<float>()[7], -8.0f);
}

// 2. D2: operator[] 生成的 view 正确注册 autograd 节点
TEST(TensorViewSemantics, OperatorIndexAutograd) {
    Tensor x(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU);
    x.requires_grad(true);
    float* p = x.data_write<float>();
    for (int i = 0; i < 4; ++i) p[i] = 1.0f;

    Tensor elem = x[1];
    EXPECT_TRUE(elem.requires_grad());

    Tensor loss = elem + 0.0f;
    AutoGrad::backward(loss.getRelatedNode(), false);

    const float* gp = x.grad().data_read<float>();
    EXPECT_FLOAT_EQ(gp[0], 0.0f);
    EXPECT_FLOAT_EQ(gp[1], 1.0f);
    EXPECT_FLOAT_EQ(gp[2], 0.0f);
    EXPECT_FLOAT_EQ(gp[3], 0.0f);
}

// 3. D3: detach() 保留 strides 与 storage_offset
TEST(TensorViewSemantics, DetachPreservesMetadata) {
    Tensor x(ShapeTag{}, {4, 4}, DType::kFloat, DeviceType::kCPU);
    x.requires_grad(true);
    Tensor sl = x.slice(0, 1, 2).slice(1, 1, 2);
    Tensor d = sl.detach();

    EXPECT_FALSE(d.requires_grad());
    EXPECT_EQ(d.shape(), sl.shape());
    EXPECT_EQ(d.strides(), sl.strides());
    EXPECT_EQ(d.storage_offset(), sl.storage_offset());
}

// 4. D4: broadcast_to 对非紧凑视图安全物化
TEST(TensorViewSemantics, BroadcastToFromView) {
    Tensor x(ShapeTag{}, {4, 4}, DType::kFloat, DeviceType::kCPU);
    float* p = x.data_write<float>();
    for (int i = 0; i < 16; ++i) p[i] = static_cast<float>(i);

    Tensor col = x.slice(1, 2, 1);
    Tensor b = col.broadcast_to({4, 3});
    EXPECT_EQ(b.shape(), (std::vector<size_t>{4, 3}));
    const float* bp = b.data_read<float>();
    for (size_t r = 0; r < 4; ++r) {
        float expected = static_cast<float>(r * 4 + 2);
        for (size_t c = 0; c < 3; ++c) {
            EXPECT_FLOAT_EQ(bp[r * 3 + c], expected);
        }
    }
}

// 5. D5: to(DType) 与 to(DeviceType) 从非紧凑/带 offset 视图正确规范化拷贝
TEST(TensorViewSemantics, ToDeviceFromStridedView) {
    Tensor x(ShapeTag{}, {3, 3}, DType::kFloat, DeviceType::kCPU);
    float* p = x.data_write<float>();
    for (int i = 0; i < 9; ++i) p[i] = static_cast<float>(i + 1);

    Tensor tr = x.transpose(0, 1);
    Tensor tr_double = tr.to(DType::kDouble);

    EXPECT_TRUE(tr_double.is_contiguous());
    EXPECT_EQ(tr_double.storage_offset(), 0u);
    const double* dp = tr_double.data_read<double>();
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(dp[i * 3 + j], static_cast<double>(j * 3 + i + 1));
        }
    }

    if (CtorchScheduler::isDeviceAvailable(DeviceType::kMPS)) {
        Tensor tr_mps = tr.to(DeviceType::kMPS);
        Tensor tr_back = tr_mps.to(DeviceType::kCPU);
        EXPECT_TRUE(tr_back.is_contiguous());
        const float* fbp = tr_back.data_read<float>();
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                EXPECT_FLOAT_EQ(fbp[i * 3 + j], static_cast<float>(j * 3 + i + 1));
            }
        }
    }
}

// 6. E1: 多叶共享恒等梯度不发生别名双写
TEST(AutoGradCorrectness, MultiLeafSharedGradIndependent) {
    Tensor a(ShapeTag{}, {2, 2}, DType::kFloat, DeviceType::kCPU);
    Tensor b(ShapeTag{}, {2, 2}, DType::kFloat, DeviceType::kCPU);
    a.requires_grad(true);
    b.requires_grad(true);
    for (int i = 0; i < 4; ++i) {
        a.data_write<float>()[i] = 1.0f;
        b.data_write<float>()[i] = 2.0f;
    }

    Tensor c = a + b;
    Tensor loss = c.sum();
    AutoGrad::backward(loss.getRelatedNode(), false);

    EXPECT_NE(a.grad().data_read<float>(), b.grad().data_read<float>());

    a.grad().data_write<float>()[0] = 999.0f;
    EXPECT_FLOAT_EQ(b.grad().data_read<float>()[0], 1.0f);
}

// 7. E2: MSE 与 MAE loss autograd 正确性
TEST(AutoGradCorrectness, MseMaeLossBackward) {
    Tensor pred(ShapeTag{}, {2, 2}, DType::kFloat, DeviceType::kCPU);
    Tensor target(ShapeTag{}, {2, 2}, DType::kFloat, DeviceType::kCPU);
    pred.requires_grad(true);
    float p_init[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float t_init[4] = {2.0f, 2.0f, 1.0f, 6.0f};
    for (int i = 0; i < 4; ++i) {
        pred.data_write<float>()[i] = p_init[i];
        target.data_write<float>()[i] = t_init[i];
    }

    Tensor mse = pred.mse_loss(target);
    AutoGrad::backward(mse.getRelatedNode(), false);

    const float* gp = pred.grad().data_read<float>();
    for (int i = 0; i < 4; ++i) {
        float expected_grad = 2.0f / 4.0f * (p_init[i] - t_init[i]);
        EXPECT_NEAR(gp[i], expected_grad, 1e-5f);
    }

    Tensor pred2(ShapeTag{}, {2, 2}, DType::kFloat, DeviceType::kCPU);
    pred2.requires_grad(true);
    for (int i = 0; i < 4; ++i) pred2.data_write<float>()[i] = p_init[i];
    Tensor mae = pred2.mae_loss(target);
    AutoGrad::backward(mae.getRelatedNode(), false);

    const float* gp2 = pred2.grad().data_read<float>();
    for (int i = 0; i < 4; ++i) {
        float diff = p_init[i] - t_init[i];
        float sgn = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
        float expected_grad = sgn / 4.0f;
        EXPECT_NEAR(gp2[i], expected_grad, 1e-5f);
    }
}

// 8. E3: DivNode 遇分母为 0 抛出异常而非静默返回空梯度
TEST(AutoGradCorrectness, DivNodeThrowsOnZero) {
    Tensor a(ShapeTag{}, {2}, DType::kFloat, DeviceType::kCPU);
    Tensor b(ShapeTag{}, {2}, DType::kFloat, DeviceType::kCPU);
    a.requires_grad(true);
    b.requires_grad(true);
    a.data_write<float>()[0] = 1.0f; a.data_write<float>()[1] = 1.0f;
    b.data_write<float>()[0] = 0.0f; b.data_write<float>()[1] = 1.0f;

    Tensor c = a / b;
    EXPECT_THROW(AutoGrad::backward(c.getRelatedNode(), false), std::runtime_error);
}

// 9. F1: ThreadPool 线程安全提交与单例执行
TEST(ConcurrencySafety, ThreadPoolSingletonExecution) {
    auto& pool = ThreadPool::getInstance();
    auto f = pool.addTask([](int val) { return val * 2; }, 21);
    EXPECT_EQ(f.get(), 42);
}

// 10. P2: Scalar sum and mean forward & backward
TEST(P2DefectRemediation, ScalarSumAndMean) {
    Tensor s(3.5f);
    EXPECT_FLOAT_EQ(s.sum().item<float>(), 3.5f);
    EXPECT_FLOAT_EQ(s.mean().item<float>(), 3.5f);

    Tensor s1(3.5f);
    s1.requires_grad(true);
    Tensor loss1 = s1.sum();
    AutoGrad::backward(loss1.getRelatedNode(), false);
    EXPECT_FLOAT_EQ(s1.grad().item<float>(), 1.0f);

    Tensor s2(3.5f);
    s2.requires_grad(true);
    Tensor loss2 = s2.mean();
    AutoGrad::backward(loss2.getRelatedNode(), false);
    EXPECT_FLOAT_EQ(s2.grad().item<float>(), 1.0f);
}

// 11. P2: contiguous() 支持多数据类型 (int32, int64, bool)
TEST(P2DefectRemediation, ContiguousMultiDType) {
    Tensor t_int(ShapeTag{}, {2, 3}, DType::kInt, DeviceType::kCPU);
    int32_t* pi = t_int.data_write<int32_t>();
    for (int i = 0; i < 6; ++i) pi[i] = i + 10;
    Tensor t_int_t = t_int.transpose(0, 1);
    EXPECT_FALSE(t_int_t.is_contiguous());
    Tensor t_int_c = t_int_t.contiguous();
    EXPECT_TRUE(t_int_c.is_contiguous());
    const int32_t* pic = t_int_c.data_read<int32_t>();
    EXPECT_EQ(pic[0], 10);
    EXPECT_EQ(pic[1], 13);
    EXPECT_EQ(pic[2], 11);
    EXPECT_EQ(pic[3], 14);

    Tensor t_long(ShapeTag{}, {2, 3}, DType::kLong, DeviceType::kCPU);
    int64_t* pl = t_long.data_write<int64_t>();
    for (int i = 0; i < 6; ++i) pl[i] = i + 1000;
    Tensor t_long_t = t_long.transpose(0, 1);
    Tensor t_long_c = t_long_t.contiguous();
    EXPECT_TRUE(t_long_c.is_contiguous());
    const int64_t* plc = t_long_c.data_read<int64_t>();
    EXPECT_EQ(plc[0], 1000);
    EXPECT_EQ(plc[1], 1003);

    Tensor t_bool(ShapeTag{}, {2, 2}, DType::kBool, DeviceType::kCPU);
    bool* pb = t_bool.data_write<bool>();
    pb[0] = true; pb[1] = false; pb[2] = false; pb[3] = true;
    Tensor t_bool_t = t_bool.transpose(0, 1);
    Tensor t_bool_c = t_bool_t.contiguous();
    EXPECT_TRUE(t_bool_c.is_contiguous());
    const bool* pbc = t_bool_c.data_read<bool>();
    EXPECT_EQ(pbc[0], true);
    EXPECT_EQ(pbc[1], false);
    EXPECT_EQ(pbc[2], false);
    EXPECT_EQ(pbc[3], true);
}

// 12. P2: view 与维度归约 (max/min/var/std)
TEST(P2DefectRemediation, ViewAndReductions) {
    Tensor t(ShapeTag{}, {2, 3}, DType::kFloat, DeviceType::kCPU);
    float* p = t.data_write<float>();
    // [1, 2, 3]
    // [4, 5, 6]
    for (int i = 0; i < 6; ++i) p[i] = static_cast<float>(i + 1);

    Tensor v1 = t.view({3, 2});
    EXPECT_EQ(v1.shape(), (std::vector<size_t>{3, 2}));
    Tensor v2 = t.view(std::vector<size_t>{6});
    EXPECT_EQ(v2.shape(), (std::vector<size_t>{6}));

    // max(dim=0): [4, 5, 6]
    Tensor m0 = t.max(0, false);
    EXPECT_EQ(m0.numel(), 3u);
    EXPECT_FLOAT_EQ(m0.data_read<float>()[0], 4.0f);
    EXPECT_FLOAT_EQ(m0.data_read<float>()[1], 5.0f);
    EXPECT_FLOAT_EQ(m0.data_read<float>()[2], 6.0f);

    // min(dim=1): [1, 4]
    Tensor min1 = t.min(1, false);
    EXPECT_EQ(min1.numel(), 2u);
    EXPECT_FLOAT_EQ(min1.data_read<float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(min1.data_read<float>()[1], 4.0f);

    // var and std
    Tensor vr = t.var();
    EXPECT_GT(vr.item<float>(), 0.0f);
    Tensor st = t.std();
    EXPECT_NEAR(st.item<float>(), std::sqrt(vr.item<float>()), 1e-5f);
}

// 13. P2: 拓展数学与统一接口 (tan, asin, acos, atan, sinh, cosh, log_softmax, matmul_unified)
TEST(P2DefectRemediation, UnaryMathAndUnifiedMatMul) {
    Tensor t(ShapeTag{}, {2}, DType::kFloat, DeviceType::kCPU);
    t.data_write<float>()[0] = 0.5f;
    t.data_write<float>()[1] = -0.5f;

    Tensor tn = t.tan();
    EXPECT_NEAR(tn.data_read<float>()[0], std::tan(0.5f), 1e-5f);

    Tensor as = t.asin();
    EXPECT_NEAR(as.data_read<float>()[0], std::asin(0.5f), 1e-5f);

    Tensor ac = t.acos();
    EXPECT_NEAR(ac.data_read<float>()[0], std::acos(0.5f), 1e-5f);

    Tensor at = t.atan();
    EXPECT_NEAR(at.data_read<float>()[0], std::atan(0.5f), 1e-5f);

    Tensor sh = t.sinh();
    EXPECT_NEAR(sh.data_read<float>()[0], std::sinh(0.5f), 1e-5f);

    Tensor ch = t.cosh();
    EXPECT_NEAR(ch.data_read<float>()[0], std::cosh(0.5f), 1e-5f);

    Tensor ls = t.log_softmax(0);
    EXPECT_EQ(ls.numel(), 2u);

    Tensor A(ShapeTag{}, {2, 2}, DType::kFloat, DeviceType::kCPU);
    Tensor B(ShapeTag{}, {2, 2}, DType::kFloat, DeviceType::kCPU);
    A.data_write<float>()[0] = 1.0f; A.data_write<float>()[1] = 2.0f;
    A.data_write<float>()[2] = 3.0f; A.data_write<float>()[3] = 4.0f;
    B.data_write<float>()[0] = 1.0f; B.data_write<float>()[1] = 0.0f;
    B.data_write<float>()[2] = 0.0f; B.data_write<float>()[3] = 1.0f;

    Tensor C = A.matmul_unified(B);
    EXPECT_FLOAT_EQ(C.data_read<float>()[0], 1.0f);
    EXPECT_FLOAT_EQ(C.data_read<float>()[1], 2.0f);
    EXPECT_FLOAT_EQ(C.data_read<float>()[2], 3.0f);
    EXPECT_FLOAT_EQ(C.data_read<float>()[3], 4.0f);
}

// 14. P2: Ctorch_Stats 线程安全初始化与 level 原子性
TEST(P2DefectRemediation, CtorchStatsThreadSafety) {
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([]() {
            auto& stats = Ctorch_Stats::getInstance();
            stats.level = PrintLevel::MINIUM;
            EXPECT_EQ(stats.level, PrintLevel::MINIUM);
            stats.level = PrintLevel::FULL;
        });
    }
    for (auto& th : threads) {
        th.join();
    }
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
