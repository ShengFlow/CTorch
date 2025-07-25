#include "Tensor.h"

int main() {
    // 创建自动微分上下文
    AutoDiff ctx;

    auto x = Tensor({1.0f, 2.0f,3.0f, 4.0f},{2,2});
    x.set_autograd_ctx(&ctx);
    x.requires_grad(true);

    // 使用标量乘法
    Tensor y = 2.0f * x;

    // 使用比较运算符
    Tensor mask = x > 2.0f;

    // 使用广播加法
    Tensor z = x + Tensor({5.0f});

    // 使用矩阵乘法
    Tensor w = Tensor({0.5f, -0.2f,1.0f, 0.8f},{2,2});
    w.set_autograd_ctx(&ctx);
    w.requires_grad(true);
    Tensor mat_result = matMul(x, w);

    // 使用求和
    Tensor sum_all = x.sum();
    Tensor sum_dim = x.sum({0}, true);

    // 反向传播
    mat_result.backward();

    std::cout << "x gradient:\n" << x.grad().toString() << std::endl;
    std::cout << "w gradient:\n" << w.grad().toString() << std::endl;

    return 0;
}