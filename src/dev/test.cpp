#include "Tensor.h"
#include <iostream>

int main() {
    // 1. 创建上下文
    AutoDiff ctx;
    AutoDiffContext::Guard guard(&ctx);
    Ctorch_Error::setPrintLevel(PrintLevel::MINIUM);
    // 2. 创建张量
    Tensor x(2.0f);
    Tensor y(3.0f);
    x.requires_grad(true);
    y.requires_grad(true);

    // 3. 前向计算
    Tensor z = x * x + y * y;  // z = x² + y²

    // 4. 反向传播
    backward(z);

    // 5. 获取结果
    std::cout << "z = " << z.item<float>() << std::endl;      // 输出: 13
    std::cout << "∂z/∂x = " << grad(x).item<float>() << std::endl;  // 输出: 4
    std::cout << "∂z/∂y = " << grad(y).item<float>() << std::endl;  // 输出: 6

    return 0;
}