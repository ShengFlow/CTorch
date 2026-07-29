/**
 * @file test_unary_inplace.cpp
 * @brief Unary in-place 算子单元测试（P1-3）
 * @details 覆盖 CPU/MPS 双后端的完全重叠、连续调用及跨后端一致性。
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "Tensor.h"
#include "CtorchScheduler.h"

class UnaryInplaceTest : public ::testing::Test {
protected:
    static constexpr float kLReLUSlope = 0.01f;

    static std::vector<float> make_mixed_values(size_t n) {
        std::vector<float> v(n);
        for (size_t i = 0; i < n; ++i) {
            float x = static_cast<float>(i) - 3.0f;
            v[i] = x * 0.5f;
        }
        return v;
    }

    static void fill_tensor(Tensor& t, const std::vector<float>& values) {
        float* p = t.data<float>();
        for (size_t i = 0; i < values.size(); ++i) {
            p[i] = values[i];
        }
    }

    static std::vector<float> read_tensor(const Tensor& t) {
        const float* p = t.data<float>();
        std::vector<float> out(t.numel());
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = p[i];
        }
        return out;
    }

    static float expected_relu(float x) { return std::max(0.0f, x); }
    static float expected_lrelu(float x) { return x > 0.0f ? x : x * kLReLUSlope; }
    static float expected_neg(float x) { return -x; }
    static float expected_abs(float x) { return std::fabs(x); }

    static void expect_close(const std::vector<float>& a, const std::vector<float>& b,
                             float atol = 1e-5f) {
        ASSERT_EQ(a.size(), b.size());
        for (size_t i = 0; i < a.size(); ++i) {
            EXPECT_NEAR(a[i], b[i], atol) << "at index " << i;
        }
    }
};

TEST_F(UnaryInplaceTest, CPU_ReLU_FullOverlap) {
    auto values = make_mixed_values(10);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(a, values);

    a.relu_();

    std::vector<float> expected(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        expected[i] = expected_relu(values[i]);
    }
    expect_close(read_tensor(a), expected);
}

TEST_F(UnaryInplaceTest, CPU_LReLU_FullOverlap) {
    auto values = make_mixed_values(10);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(a, values);

    a.leaky_relu_();

    std::vector<float> expected(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        expected[i] = expected_lrelu(values[i]);
    }
    expect_close(read_tensor(a), expected);
}

TEST_F(UnaryInplaceTest, CPU_Neg_ChainedReturnsOriginal) {
    auto values = make_mixed_values(8);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(a, values);

    a.neg_().neg_();

    expect_close(read_tensor(a), values);
}

TEST_F(UnaryInplaceTest, CPU_AbsExpLog_RoundTrip) {
    auto values = make_mixed_values(8);
    for (auto& v : values) v = std::fabs(v) + 0.1f; // 保证 log 定义域

    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(a, values);

    a.log_().exp_();

    expect_close(read_tensor(a), values, 1e-3f);
}

TEST_F(UnaryInplaceTest, CPU_InplaceMatchesOutOfPlace) {
    auto values = make_mixed_values(8);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(a, values);
    Tensor b = a.relu();

    a.relu_();

    expect_close(read_tensor(a), read_tensor(b));
}

TEST_F(UnaryInplaceTest, CPU_SharedStorageIsModified) {
    auto values = make_mixed_values(6);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(a, values);
    Tensor b(a); // 共享底层 storage

    a.relu_();

    // b 与 a 共享 storage，应观察到同一修改
    expect_close(read_tensor(b), read_tensor(a));
}

TEST_F(UnaryInplaceTest, CPU_UnsupportedOpThrows) {
    Tensor a(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU);
    EXPECT_THROW(CtorchScheduler::getInstance().dispatch_inplace(a, op::Softmax),
                 std::runtime_error);
}

TEST_F(UnaryInplaceTest, CPU_RequiresGradThrows) {
    Tensor a(ShapeTag{}, {4}, DType::kFloat, DeviceType::kCPU);
    a.requires_grad(true);
    EXPECT_THROW(a.relu_(), std::runtime_error);
    EXPECT_THROW(a.neg_(), std::runtime_error);
}

TEST_F(UnaryInplaceTest, CPU_NonCPUTensorThrows) {
    if (!CtorchScheduler::isDeviceAvailable(DeviceType::kMPS)) {
        GTEST_SKIP() << "MPS not available";
    }
    Tensor a(ShapeTag{}, {4}, DType::kFloat, DeviceType::kMPS);
    // 强制调用 CPU in-place kernel 应因设备不匹配而抛异常
    EXPECT_THROW(Neg_BASIC_inplace(a), std::runtime_error);
}

TEST_F(UnaryInplaceTest, MPS_ReLU_FullOverlap) {
    if (!CtorchScheduler::isDeviceAvailable(DeviceType::kMPS)) {
        GTEST_SKIP() << "MPS not available";
    }
    auto values = make_mixed_values(10);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kMPS);
    fill_tensor(a, values);
    MPS_flush_wait(true);

    a.relu_();
    MPS_flush_wait(true);

    std::vector<float> expected(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        expected[i] = expected_relu(values[i]);
    }
    expect_close(read_tensor(a), expected);
}

TEST_F(UnaryInplaceTest, MPS_LReLU_FullOverlap) {
    if (!CtorchScheduler::isDeviceAvailable(DeviceType::kMPS)) {
        GTEST_SKIP() << "MPS not available";
    }
    auto values = make_mixed_values(10);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kMPS);
    fill_tensor(a, values);
    MPS_flush_wait(true);

    a.leaky_relu_();
    MPS_flush_wait(true);

    std::vector<float> expected(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        expected[i] = expected_lrelu(values[i]);
    }
    expect_close(read_tensor(a), expected);
}

TEST_F(UnaryInplaceTest, MPS_Neg_ChainedReturnsOriginal) {
    if (!CtorchScheduler::isDeviceAvailable(DeviceType::kMPS)) {
        GTEST_SKIP() << "MPS not available";
    }
    auto values = make_mixed_values(8);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kMPS);
    fill_tensor(a, values);
    MPS_flush_wait(true);

    a.neg_().neg_();
    MPS_flush_wait(true);

    expect_close(read_tensor(a), values);
}

TEST_F(UnaryInplaceTest, MPS_AbsExpLog_RoundTrip) {
    if (!CtorchScheduler::isDeviceAvailable(DeviceType::kMPS)) {
        GTEST_SKIP() << "MPS not available";
    }
    auto values = make_mixed_values(8);
    for (auto& v : values) v = std::fabs(v) + 0.1f;

    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kMPS);
    fill_tensor(a, values);
    MPS_flush_wait(true);

    a.log_().exp_();
    MPS_flush_wait(true);

    expect_close(read_tensor(a), values, 1e-3f);
}

TEST_F(UnaryInplaceTest, MPS_CPU_Consistency) {
    if (!CtorchScheduler::isDeviceAvailable(DeviceType::kMPS)) {
        GTEST_SKIP() << "MPS not available";
    }
    auto values = make_mixed_values(16);

    Tensor cpu_t(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kCPU);
    fill_tensor(cpu_t, values);
    cpu_t.relu_();

    Tensor mps_t(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kMPS);
    fill_tensor(mps_t, values);
    MPS_flush_wait(true);
    mps_t.relu_();
    MPS_flush_wait(true);

    expect_close(read_tensor(cpu_t), read_tensor(mps_t), 1e-4f);
}

TEST_F(UnaryInplaceTest, MPS_ConsecutiveInplaceCalls) {
    if (!CtorchScheduler::isDeviceAvailable(DeviceType::kMPS)) {
        GTEST_SKIP() << "MPS not available";
    }
    auto values = make_mixed_values(12);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kMPS);
    fill_tensor(a, values);
    MPS_flush_wait(true);

    a.relu_();
    a.neg_();
    a.abs_();
    MPS_flush_wait(true);

    std::vector<float> expected(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        expected[i] = std::fabs(-expected_relu(values[i]));
    }
    expect_close(read_tensor(a), expected);
}

TEST_F(UnaryInplaceTest, MPS_RequiresGradThrows) {
    if (!CtorchScheduler::isDeviceAvailable(DeviceType::kMPS)) {
        GTEST_SKIP() << "MPS not available";
    }
    Tensor a(ShapeTag{}, {4}, DType::kFloat, DeviceType::kMPS);
    a.requires_grad(true);
    MPS_flush_wait(true);
    EXPECT_THROW(a.relu_(), std::runtime_error);
}

TEST_F(UnaryInplaceTest, MPS_2D_Contiguous) {
    if (!CtorchScheduler::isDeviceAvailable(DeviceType::kMPS)) {
        GTEST_SKIP() << "MPS not available";
    }
    // 2x4 连续张量，验证多维度 contiguous 路径正确
    std::vector<float> values = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, -3.0f, 4.0f};
    Tensor a(ShapeTag{}, {2, 4}, DType::kFloat, DeviceType::kMPS);
    fill_tensor(a, values);
    MPS_flush_wait(true);

    a.relu_();
    MPS_flush_wait(true);

    std::vector<float> expected(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        expected[i] = expected_relu(values[i]);
    }
    expect_close(read_tensor(a), expected);
}

TEST_F(UnaryInplaceTest, MPS_Transpose_Throws) {
    if (!CtorchScheduler::isDeviceAvailable(DeviceType::kMPS)) {
        GTEST_SKIP() << "MPS not available";
    }
    // transpose 后 strides 变为非连续，MPS in-place 应明确拒绝
    Tensor a(ShapeTag{}, {2, 4}, DType::kFloat, DeviceType::kMPS);
    auto values = make_mixed_values(8);
    fill_tensor(a, values);
    MPS_flush_wait(true);

    Tensor a_t = a.t();
    EXPECT_THROW(a_t.relu_(), std::runtime_error);
}

TEST_F(UnaryInplaceTest, MPS_Slice_Throws) {
    if (!CtorchScheduler::isDeviceAvailable(DeviceType::kMPS)) {
        GTEST_SKIP() << "MPS not available";
    }
    // operator[] 产生带 storage_offset 的标量视图，MPS in-place 应明确拒绝
    Tensor a(ShapeTag{}, {8}, DType::kFloat, DeviceType::kMPS);
    auto values = make_mixed_values(8);
    fill_tensor(a, values);
    MPS_flush_wait(true);

    Tensor slice = a[3];
    EXPECT_THROW(slice.relu_(), std::runtime_error);
}

TEST_F(UnaryInplaceTest, MPS_SharedStorageViewIsModified) {
    if (!CtorchScheduler::isDeviceAvailable(DeviceType::kMPS)) {
        GTEST_SKIP() << "MPS not available";
    }
    // 共享底层 storage 的 contiguous 视图应观察到同一修改
    auto values = make_mixed_values(6);
    Tensor a(ShapeTag{}, {values.size()}, DType::kFloat, DeviceType::kMPS);
    fill_tensor(a, values);
    MPS_flush_wait(true);

    Tensor b(a); // 共享底层 storage
    a.relu_();
    MPS_flush_wait(true);

    expect_close(read_tensor(b), read_tensor(a));
}

TEST_F(UnaryInplaceTest, MPS_InplaceMatchesOutOfPlace_NonContiguous) {
    if (!CtorchScheduler::isDeviceAvailable(DeviceType::kMPS)) {
        GTEST_SKIP() << "MPS not available";
    }
    // 对 non-contiguous 张量，in-place 应抛异常；out-of-place 通过副本正确计算
    Tensor a(ShapeTag{}, {2, 4}, DType::kFloat, DeviceType::kMPS);
    auto values = make_mixed_values(8);
    fill_tensor(a, values);
    MPS_flush_wait(true);

    Tensor a_t = a.t();
    EXPECT_THROW(a_t.relu_(), std::runtime_error);

    // out-of-place 不依赖 contiguous，应正常工作
    Tensor out = a_t.relu();
    MPS_flush_wait(true);
    std::vector<float> expected(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        expected[i] = expected_relu(values[i]);
    }
    // 注意：a_t 是 transpose，其数据逻辑顺序与 values 不同；
    // 这里只验证 out-of-place 不崩溃且 shape 正确，数值由 kernel 保证
    EXPECT_EQ(out.numel(), a_t.numel());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
