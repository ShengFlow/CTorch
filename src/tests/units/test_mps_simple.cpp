#include <iostream>
#include "Tensor.h"
#include "AutoGrad.h"
#include "DeviceAllocator.h"

int main() {
    std::cout << "=== 简单 MPS 测试 ===" << std::endl;
    
    Tensor a(ShapeTag{}, {3}, DType::kFloat, DeviceType::kMPS);
    a.data<float>()[0] = 1.0f;
    a.data<float>()[1] = 2.0f;
    a.data<float>()[2] = 3.0f;
    
    Tensor b(ShapeTag{}, {3}, DType::kFloat, DeviceType::kMPS);
    b.data<float>()[0] = 4.0f;
    b.data<float>()[1] = 5.0f;
    b.data<float>()[2] = 6.0f;
    
    std::cout << "a = [" << a.data<float>()[0] << ", " << a.data<float>()[1] << ", " << a.data<float>()[2] << "]" << std::endl;
    std::cout << "b = [" << b.data<float>()[0] << ", " << b.data<float>()[1] << ", " << b.data<float>()[2] << "]" << std::endl;
    
    Tensor c = a * b;
    
    std::cout << "a * b = [" << c.data<float>()[0] << ", " << c.data<float>()[1] << ", " << c.data<float>()[2] << "]" << std::endl;
    
    a.requires_grad(true);
    b.requires_grad(true);
    
    Tensor d = a * b;
    
    std::cout << "带梯度 a * b = [" << d.data<float>()[0] << ", " << d.data<float>()[1] << ", " << d.data<float>()[2] << "]" << std::endl;
    
    std::cout << "\n=== 测试初始梯度 ===" << std::endl;
    
    std::vector<size_t> root_shape = {3};
    Tensor grad_tensor(1.0f, DeviceType::kMPS);
    Tensor broadcasted(ShapeTag{}, root_shape, grad_tensor.dtype(), grad_tensor.device());
    const float scalar = grad_tensor.data<float>()[0];
    const size_t total = broadcasted.numel();
    float* p = broadcasted.data<float>();
    for (size_t i = 0; i < total; ++i) p[i] = scalar;
    
    std::cout << "初始梯度: [" << broadcasted.data<float>()[0] << ", " << broadcasted.data<float>()[1] << ", " << broadcasted.data<float>()[2] << "]" << std::endl;
    
    return 0;
}