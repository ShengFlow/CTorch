/**
 * @file test_core_semantics.cpp
 * @brief Storage / Tensor / Node 核心数据结构 copy/move/lifetime 语义回归测试
 * @details 覆盖深/浅拷贝、移动后状态、_grad 独立性、跨设备 to()、shared_storage 视图。
 *          修改核心数据结构语义时必须运行本测试，防止类似 `_grad` 浅拷贝事故重演。
 */

#include <gtest/gtest.h>
#include "Tensor.h"
#include "Storage.h"
#include "AutoGrad.h"
#include "AutoGrad/Node.h"
#include "AutoGrad/Nodes/GradAccumulator.h"
#include "CtorchScheduler.h"

static std::vector<float> make_values(size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = static_cast<float>(i) - static_cast<float>(n) / 2.0f;
    }
    return v;
}

static void fill_tensor(Tensor& t, const std::vector<float>& values) {
    float* p = t.data<float>();
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
    b.data<float>()[0] = 999.0f;
    EXPECT_EQ(a.data<float>()[0], 999.0f);
}

TEST(TensorSemantics, CopyAssignmentSharesStorage) {
    auto values = make_values(4);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(a, values);

    Tensor c(ShapeTag{}, {1}, DType::kFloat, DeviceType::kCPU);
    c = a;
    c.data<float>()[0] = 888.0f;
    EXPECT_EQ(a.data<float>()[0], 888.0f);
}

TEST(TensorSemantics, CopyDeepCopiesGrad) {
    // 通过简单运算产生 grad，验证拷贝后 grad 独立
    Tensor a(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU);
    a.requires_grad(true);
    fill_tensor(a, {1.0f, 2.0f, 3.0f, 4.0f});

    Tensor b = a + 0.0f;
    AutoGrad::backward(b.getRelatedNode(), false);

    // a 的 grad 应为全 1
    EXPECT_EQ(a.grad().data<float>()[0], 1.0f);

    Tensor a_copy(a);
    EXPECT_EQ(a_copy.grad().data<float>()[0], 1.0f);

    // 清零 a 的 grad，若 a_copy.grad 是深拷贝则不应受影响
    a.zero_grad();
    EXPECT_EQ(a.grad().data<float>()[0], 0.0f);
    EXPECT_EQ(a_copy.grad().data<float>()[0], 1.0f);
}

TEST(TensorSemantics, MoveConstructInvalidatesSource) {
    auto values = make_values(4);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(a, values);

    Tensor b(std::move(a));
    // 空 shape 在 numel() 中按标量返回 1，因此这里检查 shape 已清空且 data 为空
    EXPECT_TRUE(a.shape().empty());
    EXPECT_EQ(a.data<float>(), nullptr);
    EXPECT_EQ(b.numel(), 4u);
    EXPECT_EQ(b.data<float>()[0], values[0]);
}

TEST(TensorSemantics, MoveAssignmentInvalidatesSource) {
    auto values = make_values(4);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(a, values);

    Tensor c(ShapeTag{}, {1}, DType::kFloat, DeviceType::kCPU);
    c = std::move(a);
    EXPECT_TRUE(a.shape().empty());
    EXPECT_EQ(a.data<float>(), nullptr);
    EXPECT_EQ(c.numel(), 4u);
}

TEST(TensorSemantics, ViewModificationReflectsOnBase) {
    auto values = make_values(6);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(a, values);

    // 通过拷贝构造创建视图（共享 storage）
    Tensor view(a);
    view.data<float>()[2] = 123.0f;
    EXPECT_EQ(a.data<float>()[2], 123.0f);
}

TEST(TensorSemantics, CloneIsIndependent) {
    auto values = make_values(4);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(a, values);

    Tensor b = a.clone();
    b.data<float>()[0] = 777.0f;
    EXPECT_NE(a.data<float>()[0], 777.0f);
}

TEST(TensorSemantics, GradIndependenceAfterCopy) {
    Tensor a(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU);
    a.requires_grad(true);
    fill_tensor(a, {1.0f, 2.0f, 3.0f, 4.0f});

    Tensor b = a + 0.0f;
    AutoGrad::backward(b.getRelatedNode(), false);
    EXPECT_EQ(a.grad().data<float>()[0], 1.0f);

    Tensor a_copy(a);
    // 对 a_copy 做新的前向 + backward，不应影响 a 已有的 grad
    Tensor c = a_copy + 0.0f;
    AutoGrad::backward(c.getRelatedNode(), false);

    // a 的 grad 保持为 1（来自 b 的 backward）
    EXPECT_EQ(a.grad().data<float>()[0], 1.0f);
    // a_copy 的 grad 累加为 2（b 的 grad 深拷贝 + c 的 backward）
    EXPECT_EQ(a_copy.grad().data<float>()[0], 2.0f);
}

TEST(TensorSemantics, CopyMoveChain) {
    auto values = make_values(4);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(a, values);

    Tensor b(a);              // copy
    Tensor c(std::move(b));   // move from copy
    Tensor d(c);              // copy from moved-to
    d.data<float>()[0] = 111.0f;

    EXPECT_EQ(a.data<float>()[0], 111.0f);
    EXPECT_EQ(c.data<float>()[0], 111.0f);
    EXPECT_TRUE(b.shape().empty());
    EXPECT_EQ(b.data<float>(), nullptr);
}

// ======================= Tensor 设备迁移语义 =======================

TEST(TensorSemantics, ToSameDeviceReturnsAliasLikeCopy) {
    auto values = make_values(4);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(a, values);

    Tensor b = a.to(DeviceType::kCPU);
    EXPECT_EQ(b.device(), DeviceType::kCPU);
    EXPECT_EQ(b.data<float>()[0], values[0]);
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
    EXPECT_EQ(cpu_t2.data<float>()[0], values[0]);
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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
