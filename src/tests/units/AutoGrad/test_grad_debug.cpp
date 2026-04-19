#include "../include/Tensor.h"
#include "../include/AutoGrad.h"
#include "../include/CtorchError.h"
#include <iostream>

int main() {
    std::cout << "=== 测试：梯度是否正确传播到函数参数 ===" << std::endl;

    AutoGrad::EnableGrad = true;

    Tensor a({2, 3}, DType::kFloat32, DeviceType::kCPU);
    a.randn_(0, 1);
    a.setRequiresGrad(true);

    Tensor b({3, 2}, DType::kFloat32, DeviceType::kCPU);
    b.randn_(0, 1);
    b.setRequiresGrad(true);

    std::cout << "a 的地址: " << &a << std::endl;
    std::cout << "b 的地址: " << &b << std::endl;
    std::cout << "a._node before: " << a.getRelatedNode().get() << std::endl;
    std::cout << "b._node before: " << b.getRelatedNode().get() << std::endl;

    Tensor c = a * b;

    std::cout << "c 的地址: " << &c << std::endl;
    std::cout << "a._node afterMul: " << a.getRelatedNode().get() << std::endl;
    std::cout << "b._node afterMul: " << b.getRelatedNode().get() << std::endl;
    std::cout << "c._node: " << c.getRelatedNode().get() << std::endl;

    Tensor loss = c.sum();
    loss.backward();

    std::cout << "\n反向传播后:" << std::endl;
    std::cout << "a.grad 有效: " << (a.grad().isNull() ? "false" : "true") << std::endl;
    std::cout << "b.grad 有效: " << (b.grad().isNull() ? "false" : "true") << std::endl;

    if (!a.grad().isNull()) {
        std::cout << "a.grad norm: " << a.grad().norm().item<float>() << std::endl;
    }
    if (!b.grad().isNull()) {
        std::cout << "b.grad norm: " << b.grad().norm().item<float>() << std::endl;
    }

    std::cout << "\n=== 测试：通过引用传递权重 ===" << std::endl;

    auto test_with_ref = [](Tensor& w1, Tensor& w2) {
        std::cout << "w1 的地址: " << &w1 << std::endl;
        std::cout << "w2 的地址: " << &w2 << std::endl;
        std::cout << "w1._node before: " << w1.getRelatedNode().get() << std::endl;
        std::cout << "w2._node before: " << w2.getRelatedNode().get() << std::endl;

        Tensor out = w1 * w2;

        std::cout << "w1._node afterMul: " << w1.getRelatedNode().get() << std::endl;
        std::cout << "w2._node afterMul: " << w2.getRelatedNode().get() << std::endl;

        out.sum().backward();

        std::cout << "w1.grad 有效: " << (w1.grad().isNull() ? "false" : "true") << std::endl;
        std::cout << "w2.grad 有效: " << (w2.grad().isNull() ? "false" : "true") << std::endl;
        if (!w1.grad().isNull()) {
            std::cout << "w1.grad norm: " << w1.grad().norm().item<float>() << std::endl;
        }
        if (!w2.grad().isNull()) {
            std::cout << "w2.grad norm: " << w2.grad().norm().item<float>() << std::endl;
        }
    };

    Tensor w1({2, 2}, DType::kFloat32, DeviceType::kCPU);
    w1.ones_();
    w1.setRequiresGrad(true);

    Tensor w2({2, 2}, DType::kFloat32, DeviceType::kCPU);
    w2.ones_();
    w2.setRequiresGrad(true);

    test_with_ref(w1, w2);

    return 0;
}