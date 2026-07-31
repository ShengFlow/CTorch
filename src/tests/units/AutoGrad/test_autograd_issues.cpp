#include <gtest/gtest.h>
#include "AutoGrad.h"
#include "Tensor.h"

constexpr float kEps = 1e-5f;

TEST(AutoDiffTest, LogGradRegistered) {
    AutoGrad::EnableGrad = true;
    Tensor a({2.0f, 3.0f});
    a.requires_grad(true);
    Tensor b = a.log();
    
    EXPECT_TRUE(b.requires_grad()) << "Log result should require grad";
    EXPECT_NE(b.getRelatedNode(), nullptr) << "Log should create graph node";
}

TEST(AutoDiffTest, ExpGradRegistered) {
    AutoGrad::EnableGrad = true;
    Tensor a({2.0f, 3.0f});
    a.requires_grad(true);
    Tensor b = a.exp();
    
    EXPECT_TRUE(b.requires_grad()) << "Exp result should require grad";
    EXPECT_NE(b.getRelatedNode(), nullptr) << "Exp should create graph node";
}

TEST(AutoDiffTest, AbsGradRegistered) {
    AutoGrad::EnableGrad = true;
    Tensor a({2.0f, -3.0f});
    a.requires_grad(true);
    Tensor b = a.abs();
    
    EXPECT_TRUE(b.requires_grad()) << "Abs result should require grad";
    EXPECT_NE(b.getRelatedNode(), nullptr) << "Abs should create graph node";
}

TEST(AutoDiffTest, LogGradCorrectness) {
    AutoGrad::EnableGrad = true;
    Tensor a({2.0f, 4.0f});
    a.requires_grad(true);
    Tensor b = a.log();
    
    ASSERT_NE(b.getRelatedNode(), nullptr) << "Log should have related node";
    
    AutoGrad::backward(b.getRelatedNode(), false);
    Tensor ga = a.grad();
    
    EXPECT_NEAR(ga.data_read<float>()[0], 1.0f / 2.0f, kEps) << "Log grad for 2.0 should be 0.5";
    EXPECT_NEAR(ga.data_read<float>()[1], 1.0f / 4.0f, kEps) << "Log grad for 4.0 should be 0.25";
}

TEST(AutoDiffTest, ExpGradCorrectness) {
    AutoGrad::EnableGrad = true;
    Tensor a({1.0f, 2.0f});
    a.requires_grad(true);
    Tensor b = a.exp();
    
    ASSERT_NE(b.getRelatedNode(), nullptr) << "Exp should have related node";
    
    AutoGrad::backward(b.getRelatedNode(), false);
    Tensor ga = a.grad();
    
    EXPECT_NEAR(ga.data_read<float>()[0], std::exp(1.0f), kEps) << "Exp grad for 1.0 should be e";
    EXPECT_NEAR(ga.data_read<float>()[1], std::exp(2.0f), kEps) << "Exp grad for 2.0 should be e^2";
}

TEST(AutoDiffTest, AbsGradCorrectness) {
    AutoGrad::EnableGrad = true;
    Tensor a({2.0f, -3.0f});
    a.requires_grad(true);
    Tensor b = a.abs();
    
    ASSERT_NE(b.getRelatedNode(), nullptr) << "Abs should have related node";
    
    AutoGrad::backward(b.getRelatedNode(), false);
    Tensor ga = a.grad();
    
    EXPECT_NEAR(ga.data_read<float>()[0], 1.0f, kEps) << "Abs grad for positive should be 1";
    EXPECT_NEAR(ga.data_read<float>()[1], -1.0f, kEps) << "Abs grad for negative should be -1";
}

TEST(AutoDiffTest, SharedParameterGradAccumulation) {
    AutoGrad::EnableGrad = true;
    Tensor w({2.0f});
    w.requires_grad(true);
    
    Tensor x1({3.0f});
    Tensor x2({4.0f});
    
    Tensor y1 = w * x1;
    Tensor y2 = w * x2;
    Tensor z = y1 + y2;
    
    AutoGrad::backward(z.getRelatedNode(), false);
    Tensor gw = w.grad();
    
    EXPECT_NEAR(gw.data_read<float>()[0], 3.0f + 4.0f, kEps) 
        << "Shared param grad should accumulate: 3 + 4 = 7";
}

TEST(AutoDiffTest, RetainGraphMultipleBackward) {
    AutoGrad::EnableGrad = true;
    Tensor a({2.0f});
    a.requires_grad(true);
    Tensor b = a * a;
    
    AutoGrad::backward(b.getRelatedNode(), true);
    Tensor ga1 = a.grad();
    
    a.setGrad(std::make_shared<Tensor>(Tensor(ShapeTag{}, a.shape(), a.dtype(), a.device(), true)));
    
    AutoGrad::backward(b.getRelatedNode(), false);
    Tensor ga2 = a.grad();
    
    EXPECT_NEAR(ga1.data_read<float>()[0], 4.0f, kEps) << "First backward should give 4";
    EXPECT_NEAR(ga2.data_read<float>()[0], 4.0f, kEps) << "Second backward should also give 4";
}

TEST(AutoDiffTest, ChainRuleComplex) {
    AutoGrad::EnableGrad = true;
    Tensor x({2.0f});
    x.requires_grad(true);
    
    Tensor y = x * x;
    Tensor z = y.sin();
    Tensor w = z.exp();
    
    AutoGrad::backward(w.getRelatedNode(), false);
    Tensor gx = x.grad();
    
    float y_val = 4.0f;
    float dy_dx = 4.0f;
    float dz_dy = std::cos(y_val);
    float dw_dz = std::exp(std::sin(y_val));
    float expected_grad = dw_dz * dz_dy * dy_dx;
    
    EXPECT_NEAR(gx.data_read<float>()[0], expected_grad, kEps) << "Chain rule should give correct gradient";
}

TEST(AutoDiffTest, MinGradCorrectness) {
    AutoGrad::EnableGrad = true;
    Tensor a({3.0f, 1.0f});
    Tensor b({2.0f, 4.0f});
    a.requires_grad(true);
    b.requires_grad(true);
    
    Tensor c = a.min(b);
    
    ASSERT_NE(c.getRelatedNode(), nullptr) << "Min should have related node";
    
    AutoGrad::backward(c.getRelatedNode(), false);
    Tensor ga = a.grad();
    Tensor gb = b.grad();
    
    EXPECT_NEAR(ga.data_read<float>()[0], 0.0f, kEps) << "a[0]=3 > b[0]=2, so grad_a[0]=0";
    EXPECT_NEAR(ga.data_read<float>()[1], 1.0f, kEps) << "a[1]=1 < b[1]=4, so grad_a[1]=1";
    EXPECT_NEAR(gb.data_read<float>()[0], 1.0f, kEps) << "b[0]=2 < a[0]=3, so grad_b[0]=1";
    EXPECT_NEAR(gb.data_read<float>()[1], 0.0f, kEps) << "b[1]=4 > a[1]=1, so grad_b[1]=0";
}

TEST(AutoDiffTest, MaxGradCorrectness) {
    AutoGrad::EnableGrad = true;
    Tensor a({3.0f, 1.0f});
    Tensor b({2.0f, 4.0f});
    a.requires_grad(true);
    b.requires_grad(true);
    
    Tensor c = a.max(b);
    
    ASSERT_NE(c.getRelatedNode(), nullptr) << "Max should have related node";
    
    AutoGrad::backward(c.getRelatedNode(), false);
    Tensor ga = a.grad();
    Tensor gb = b.grad();
    
    EXPECT_NEAR(ga.data_read<float>()[0], 1.0f, kEps) << "a[0]=3 > b[0]=2, so grad_a[0]=1";
    EXPECT_NEAR(ga.data_read<float>()[1], 0.0f, kEps) << "a[1]=1 < b[1]=4, so grad_a[1]=0";
    EXPECT_NEAR(gb.data_read<float>()[0], 0.0f, kEps) << "b[0]=2 < a[0]=3, so grad_b[0]=0";
    EXPECT_NEAR(gb.data_read<float>()[1], 1.0f, kEps) << "b[1]=4 > a[1]=1, so grad_b[1]=1";
}

TEST(AutoDiffTest, SoftmaxGradCorrectness) {
    AutoGrad::EnableGrad = true;
    Tensor a({1.0f, 2.0f, 3.0f});
    a.requires_grad(true);
    Tensor b = a.softmax(-1);
    
    ASSERT_NE(b.getRelatedNode(), nullptr) << "Softmax should have related node";
    
    AutoGrad::backward(b.getRelatedNode(), false);
    Tensor ga = a.grad();
    
    EXPECT_NEAR(ga.data_read<float>()[0], 0.0f, 1e-4f) << "Softmax grad for sum loss should be 0";
    EXPECT_NEAR(ga.data_read<float>()[1], 0.0f, 1e-4f) << "Softmax grad for sum loss should be 0";
    EXPECT_NEAR(ga.data_read<float>()[2], 0.0f, 1e-4f) << "Softmax grad for sum loss should be 0";
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}